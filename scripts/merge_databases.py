"""Offline schema-v16 merge. Inputs are read-only; outputs are never overwritten.

The gzip JSONL archive is also the classification manifest. ``kind=source_row``
preserves exact SQLite values discarded by canonical precedence or regeneration;
BLOBs use {"__sqlite_blob_base64__": "..."}. ``kind=classification`` accounts
for every source match. A successful report is the publication readiness marker.

Crawl queue entries match on PUUID (legacy rows without one: Riot ID and region);
the target wins and imported entries receive fresh ids. Source spatial embeddings
are archived, never imported: they are derived data, legacy rows have no recorded
encoder, and the staged output can be recomputed with the pinned encoder.

match_facts are derived from the raw match. Their availability (``ingested_at``,
``fetched_by_puuid``) follows the match: the target's record wins, and an imported
match keeps the source's record when it is coherent (ingested at or after
creation, fetched by a participant); otherwise it becomes NULL (unknown) and the
original row is archived. Collection provenance (anchor_selections,
match_sightings) is append-only and unioned. Trainer state (trained_records) is
never imported, and imported training rows use the ingestor's pre-match features.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import ExitStack, closing
import gzip
import hashlib
import json
import logging
from pathlib import Path
import sqlite3

from core.collection.anchors import SOURCES as ANCHOR_SOURCES
from core.collection.ingestor import PRE_MATCH_FEATURE_EXTRACTOR
from core.collection.regions import TAG_TO_REGION
from core.database import SIGHTING_DISPOSITIONS, _facts_row
from core.feature_labels import extract_labels, extract_timeline_labels
from core.json_utils import extract_pre_match_features
from core.pre_match_dataset import _decode, _valid_features, _valid_labels, _valid_timestamp


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 16
TABLES = ("schema_version", "matches", "match_facts", "training_dataset", "trained_records",
          "match_participants", "match_timelines", "match_sightings", "players", "player_aliases",
          "anchor_selections", "champions", "crawl_queue", "player_spatial_embeddings")
KEYS = {"matches": ("match_id",), "match_facts": ("match_id",), "training_dataset": ("match_id",),
        "trained_records": ("training_id",),
        "match_participants": ("match_id", "puuid"), "match_timelines": ("match_id",),
        "match_sightings": ("match_id", "fetched_by_puuid", "seen_at", "disposition"),
        "players": ("puuid",), "player_aliases": ("alias_puuid",), "anchor_selections": ("id",),
        "champions": ("name",), "crawl_queue": ("id",),
        "player_spatial_embeddings": ("match_id", "puuid")}
FACT_COLUMNS = ("match_id", "queue_id", "game_start_at", "game_end_at", "game_version", "platform",
                "ingested_at", "fetched_by_puuid")
REQUIRED = {
    "schema_version": {"version"},
    "matches": {"match_id", "raw_json", "game_mode", "game_creation_time", "game_duration", "season_id"},
    "match_facts": set(FACT_COLUMNS),
    "training_dataset": {"id", "match_id", "feature_json", "winner_label", "labels_json", "feature_extractor"},
    "trained_records": {"training_id", "trained_at"},
    "match_participants": {"match_id", "puuid", "champion_name", "team_position", "win",
        "kills", "deaths", "assists", "total_cs", "gold_earned", "game_creation_time",
        "game_duration", "game_mode", "items_json", "team_id"},
    "match_timelines": {"match_id", "timeline_json"},
    "match_sightings": {"match_id", "fetched_by_puuid", "seen_at", "disposition"},
    "players": {"puuid", "game_name", "tag_line", "updated_at"},
    "player_aliases": {"alias_puuid", "canonical_puuid"},
    "anchor_selections": {"id", "puuid", "region", "source", "selected_at", "rank_json"},
    "champions": {"name"},
    "crawl_queue": {"id", "puuid", "region", "game_name", "tag_line", "priority", "is_processed",
        "created_at", "processed_at"},
    "player_spatial_embeddings": {"match_id", "puuid", "game_creation_time", "embedding_blob", "encoder_id"},
}
REGIONS = frozenset(TAG_TO_REGION.values())


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=lambda x: {"__sqlite_blob_base64__": base64.b64encode(x).decode()})


def _read_only(path):
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-65536")
    # Hold a consistent logical snapshot, including committed WAL content.
    conn.execute("BEGIN")
    return conn


def _schema(conn):
    try:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version")]
        if versions != [SCHEMA_VERSION]:
            raise ValueError(f"only schema version {SCHEMA_VERSION} is supported; "
                             "migrate a copy of older databases with core.database.init_db first")
        for table, fields in REQUIRED.items():
            actual = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            if actual != fields:
                raise ValueError(f"incompatible schema for {table}: {sorted(actual)}")
        actual_tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if actual_tables != set(TABLES):
            raise ValueError("unsupported additional/missing schema tables")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"invalid schema: {exc}") from exc


def _counts(conn):
    return {table: conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] for table in TABLES}


def _file_state(path):
    state = {}
    for suffix in ("", "-wal"):
        item = Path(str(path) + suffix)
        if item.exists():
            stat = item.stat()
            state[suffix or "database"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        else:
            state[suffix] = None
    return state


def _check_unchanged(initial):
    for path, state in initial.items():
        if _file_state(path) != state:
            raise ValueError(f"input changed during merge: {path}")


def _fingerprint(conn, path):
    logger.info("Fingerprinting %s", path)
    hashes = {}
    for table in TABLES:
        logger.info("Fingerprinting table %s", table)
        # Scan payloads sequentially, sorting only fixed-size row digests. At the
        # supported ~2M participant scale this uses <160MiB, not whole raw JSONs.
        row_digests = [hashlib.sha256(_json(dict(row)).encode()).digest()
                       for row in conn.execute(f'SELECT * FROM "{table}" NOT INDEXED')]
        row_digests.sort()
        digest = hashlib.sha256()
        for row_digest in row_digests:
            digest.update(row_digest)
        hashes[table] = digest.hexdigest()
        logger.info("Fingerprinted %s: %s rows", table, len(row_digests))
        del row_digests
    wal = Path(str(path) + "-wal")
    return {"algorithm": "sha256-sorted-row-sha256-v1", "path": str(path), "logical_sha256": hashlib.sha256(_json(hashes).encode()).hexdigest(),
            "tables": hashes, "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
            "database_bytes": path.stat().st_size, "wal_present": wal.exists(),
            "wal_bytes": wal.stat().st_size if wal.exists() else 0,
            "snapshot_includes_committed_wal": True}


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value.upper() not in {
        "UNKNOWN", "UNDEFINED", "NULL", "NONE", "N/A"}


def _puuid(value):
    return _text(value) and not value.lower().startswith(("synthetic", "unknown", "placeholder", "fake_", "__unknown_", "__tombstone__"))


def _validate(row):
    data = _decode(row["raw_json"])
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        return None, "malformed_raw_json"
    info = data["info"]
    metadata = data.get("metadata")
    if not _text(row["match_id"]) or not isinstance(metadata, dict) or metadata.get("matchId") != row["match_id"]:
        return None, "incoherent_match_id"
    if info.get("gameMode") != "CLASSIC":
        return None, "non_classic"
    if not _valid_timestamp(info.get("gameCreation")):
        return None, "invalid_timestamp"
    if type(info.get("gameDuration")) is not int or info["gameDuration"] < 0:
        return None, "invalid_duration"
    participants = info.get("participants")
    if not isinstance(participants, list) or len(participants) != 10 or any(not isinstance(p, dict) for p in participants):
        return None, "invalid_participants"
    puuids = [p.get("puuid") for p in participants]
    if any(not _puuid(p) for p in puuids) or len(set(puuids)) != 10:
        return None, "invalid_puuids"
    if any(type(p.get("teamId")) is not int for p in participants) or Counter(p.get("teamId") for p in participants) != {100: 5, 200: 5}:
        return None, "invalid_teams"
    teams = info.get("teams")
    if not isinstance(teams, list) or len(teams) != 2 or any(not isinstance(t, dict) for t in teams):
        return None, "invalid_winner"
    if any(type(t.get("teamId")) is not int for t in teams) or {t.get("teamId") for t in teams} != {100, 200} or any(type(t.get("win")) is not bool for t in teams):
        return None, "invalid_winner"
    winners = {t["teamId"]: t["win"] for t in teams}
    if sum(winners.values()) != 1 or any(type(p.get("win")) is not bool or p["win"] != winners[p["teamId"]] for p in participants):
        return None, "invalid_winner"
    for p in participants:
        if not _text(p.get("championName")) or not isinstance(p.get("teamPosition", ""), str) or not isinstance(p.get("individualPosition", ""), str):
            return None, "invalid_participant_identity"
        for key in ("kills", "deaths", "assists", "goldEarned", "totalMinionsKilled", "neutralMinionsKilled"):
            if type(p.get(key, 0)) is not int or not 0 <= p.get(key, 0) <= 10_000_000:
                return None, "invalid_participant_metrics"
    try:
        features, labels = extract_pre_match_features(data), extract_labels(data)
        if not _valid_features(features) or not labels or not _valid_labels(labels, labels["winner"]):
            return None, "invalid_extracted_training"
    except (ValueError, TypeError, OverflowError):
        return None, "invalid_extracted_training"
    return (data, features, labels), None


def _facts(match_id, info, source_facts):
    """match_facts derived from the raw match, keeping the source's availability when coherent.

    Coherent: ingested at or after creation, fetched by one of its participants.
    """
    ingested = source_facts["ingested_at"] if source_facts else None
    if not _valid_timestamp(ingested) or ingested < info["gameCreation"]:
        ingested = None
    fetched = source_facts["fetched_by_puuid"] if source_facts else None
    return dict(zip(FACT_COLUMNS, _facts_row(match_id, info, ingested, fetched)))


def _insert(conn, table, row):
    cols = list(row)
    return conn.execute(f'INSERT INTO "{table}" ({",".join(cols)}) VALUES ({",".join("?" for _ in cols)})',
                        tuple(row[c] for c in cols))


def _lookup(conn, table, row):
    fields = KEYS[table]
    found = conn.execute(f'SELECT * FROM "{table}" WHERE ' + " AND ".join(f'"{k}" IS ?' for k in fields),
                         tuple(row[k] for k in fields)).fetchone()
    return dict(found) if found else None


class _Archive:
    def __init__(self, path):
        self.path, self.counts = path, Counter()
        self.handle = gzip.open(path, "xt", encoding="utf-8", compresslevel=6)

    def write(self, entry):
        self.handle.write(_json(entry) + "\n")
        self.counts[entry.get("table", entry["kind"])] += 1

    def row(self, table, row, reason):
        self.write({"kind": "source_row", "table": table, "reason": reason, "row": dict(row)})

    def close(self):
        self.handle.close()

    def report(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(self.path), "sha256": digest.hexdigest(),
                "records": sum(self.counts.values()), "counts": dict(self.counts)}


def _participants(row, data):
    info = data["info"]
    for p in info["participants"]:
        yield {"match_id": row["match_id"], "puuid": p["puuid"],
            "champion_name": p["championName"],
            "team_position": p.get("teamPosition") or p.get("individualPosition", "UNKNOWN"),
            "win": int(p["win"]), "kills": p.get("kills", 0), "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            "total_cs": p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
            "gold_earned": p.get("goldEarned", 0), "game_creation_time": info["gameCreation"],
            "game_duration": info.get("gameDuration"), "game_mode": "CLASSIC",
            "items_json": json.dumps([p.get(f"item{i}", 0) for i in range(7)]), "team_id": p["teamId"]}


def _compatible_timeline(row, match):
    data, raw = _decode(row["timeline_json"]), _decode(match["raw_json"])
    if not isinstance(data, dict) or not isinstance(raw, dict):
        return False
    if not isinstance(data.get("metadata"), dict) or data["metadata"].get("matchId") != row["match_id"]:
        return False
    try:
        expected = {p["puuid"] for p in raw["info"]["participants"]}
        actual = {p["puuid"] for p in data["info"]["participants"]}
        return len(expected) == 10 and actual == expected and extract_timeline_labels(data) is not None
    except (KeyError, TypeError):
        return False


def _merge_matches(source, target, out, archive):
    counts, reasons = Counter(existing=0, imported=0, archived=0), Counter()
    logger.info("Classifying and merging source matches")
    for index, sqlite_row in enumerate(source.execute("SELECT * FROM matches NOT INDEXED"), 1):
        if index % 10000 == 0:
            logger.info("Source matches processed: %s", index)
        row = dict(sqlite_row)
        existing = _lookup(target, "matches", row)
        if existing:
            category, reason = "existing", "canonical_precedence"
            if row != existing:
                archive.row("matches", row, reason)
            same_raw = _decode(row["raw_json"]) == _decode(existing["raw_json"])
        else:
            validated, reason = _validate(row)
            category, same_raw = ("imported", True) if validated else ("archived", False)
            if not validated:
                archive.row("matches", row, reason)
            else:
                data, features, labels = validated
                info = data["info"]
                normalized = dict(row, game_mode="CLASSIC", game_creation_time=info["gameCreation"],
                                  game_duration=info.get("gameDuration"), season_id=features.get("season_id"))
                if normalized != row:
                    archive.row("matches", row, "normalized_metadata")
                _insert(out, "matches", normalized)
                _insert(out, "match_facts", _facts(row["match_id"], info, _lookup(source, "match_facts", row)))
                for participant in _participants(row, data):
                    _insert(out, "match_participants", participant)
                training = {"match_id": row["match_id"], "feature_json": _json(features),
                    "winner_label": labels["winner"], "labels_json": _json(labels),
                    "feature_extractor": PRE_MATCH_FEATURE_EXTRACTOR}
                _insert(out, "training_dataset", training)
        counts[category] += 1
        if reason:
            reasons[reason] += 1
        out.execute("INSERT INTO merge_status VALUES (?,?,?)", (row["match_id"], category, int(same_raw)))
        archive.write({"kind": "classification", "match_id": row["match_id"], "classification": category, "reason": reason})
    return dict(counts), dict(reasons)


def _merge_facts(row, status, existing, out, archive):
    """True when the source facts row is accounted for (kept, recomputed or archived)."""
    if status[0] == "imported":
        # Recomputed by _merge_matches; the differing original is kept in the archive.
        archive.row("match_facts", row, "normalized_metadata")
        return True
    if status[0] == "existing" and existing is None and status[1]:
        # The target never recorded this match's availability: the source's fills it.
        data = _decode(_lookup(out, "matches", row)["raw_json"])
        info = data.get("info") if isinstance(data, dict) else None
        if isinstance(info, dict) and _valid_timestamp(info.get("gameCreation")) \
                and isinstance(info.get("participants"), list) and all(isinstance(p, dict) for p in info["participants"]):
            facts = _facts(row["match_id"], info, row)
            _insert(out, "match_facts", facts)
            if facts != row:
                archive.row("match_facts", row, "normalized_metadata")
            return True
    if status[0] == "existing":
        archive.row("match_facts", row, "canonical_precedence")
        return True
    return False


def _merge_relationships(source, out, archive):
    logger.info("Merging and archiving source relationships")
    for table in ("match_timelines", "match_facts", "training_dataset", "match_participants",
                  "player_spatial_embeddings"):
        for item in source.execute(f'SELECT * FROM "{table}"'):
            row = dict(item)
            status = out.execute("SELECT category,same_raw FROM merge_status WHERE match_id=?", (row["match_id"],)).fetchone()
            if status is None:
                archive.row(table, row, "orphan_relationship")
                continue
            existing = _lookup(out, table, row)
            if existing == row:
                continue
            if table == "match_facts" and _merge_facts(row, status, existing, out, archive):
                continue
            if table == "match_timelines" and existing is None and status and status[0] != "archived" and status[1]:
                match = _lookup(out, "matches", row)
                if _compatible_timeline(row, match):
                    _insert(out, table, row)
                    if status[0] == "imported":
                        training = _lookup(out, "training_dataset", row)
                        labels = _decode(training["labels_json"])
                        labels.update(extract_timeline_labels(_decode(row["timeline_json"])))
                        out.execute("UPDATE training_dataset SET labels_json=? WHERE match_id=?", (_json(labels), row["match_id"]))
                    continue
            reason = ("derived_embedding_not_merged" if table == "player_spatial_embeddings"
                      else "excluded_or_regenerated_relationship")
            archive.row(table, row, reason)


def _merge_identities(source, out, archive):
    logger.info("Resolving player identities and rebuilding aggregate statistics")
    # Disk-backed temporary identity candidates bound memory regardless of match count.
    out.execute("CREATE TEMP TABLE merge_names (puuid TEXT PRIMARY KEY, game_name TEXT, tag_line TEXT, name_date INTEGER, tag_date INTEGER)")
    for row in out.execute("SELECT raw_json FROM matches"):
        data = _decode(row[0])
        if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
            continue
        info = data["info"]
        date = info.get("gameCreation")
        if not _valid_timestamp(date):
            continue
        participants = info.get("participants")
        if not isinstance(participants, list):
            continue
        for p in participants:
            if not isinstance(p, dict) or not _puuid(p.get("puuid")):
                continue
            name, tag = p.get("riotIdGameName"), p.get("riotIdTagline")
            out.execute("""INSERT INTO merge_names VALUES (?,?,?,?,?) ON CONFLICT(puuid)
                DO UPDATE SET
                game_name=CASE WHEN excluded.name_date > merge_names.name_date THEN excluded.game_name ELSE merge_names.game_name END,
                name_date=MAX(excluded.name_date,merge_names.name_date),
                tag_line=CASE WHEN excluded.tag_date > merge_names.tag_date THEN excluded.tag_line ELSE merge_names.tag_line END,
                tag_date=MAX(excluded.tag_date,merge_names.tag_date)""",
                (p["puuid"], name if _text(name) else None, tag if _text(tag) else None,
                 date if _text(name) else 0, date if _text(tag) else 0))
    for item in out.execute("SELECT * FROM merge_names"):
        row = dict(item)
        existing = _lookup(out, "players", row)
        if existing is None:
            _insert(out, "players", {"puuid": row["puuid"], "game_name": row["game_name"], "tag_line": row["tag_line"]})
        else:
            out.execute("UPDATE players SET game_name=?,tag_line=? WHERE puuid=?", (
                existing["game_name"] if _text(existing["game_name"]) else row["game_name"],
                existing["tag_line"] if _text(existing["tag_line"]) else row["tag_line"], row["puuid"]))
    # Participants can outlive raw identity data; their real PUUID still gets statistics.
    for row in out.execute("SELECT DISTINCT puuid FROM match_participants"):
        if _puuid(row[0]):
            out.execute("INSERT OR IGNORE INTO players (puuid) VALUES (?)", (row[0],))
    for item in source.execute("SELECT * FROM players"):
        row = dict(item)
        if not _puuid(row["puuid"]):
            archive.row("players", row, "invalid_puuid")
            continue
        existing = _lookup(out, "players", row)
        if existing is None:
            _insert(out, "players", row)
        else:
            out.execute("UPDATE players SET game_name=?,tag_line=? WHERE puuid=?", (
                existing["game_name"] if _text(existing["game_name"]) else row["game_name"],
                existing["tag_line"] if _text(existing["tag_line"]) else row["tag_line"], row["puuid"]))
        # Precedence may discard old metadata: preserve the original.
        archive.row("players", row, "identity_metadata_provenance")
    # One edge per alias. The graph is small; avoid any repeated full SQL scans.
    edges = {r[0]: r[1] for r in out.execute("SELECT alias_puuid,canonical_puuid FROM player_aliases")}
    for item in source.execute("SELECT * FROM player_aliases ORDER BY alias_puuid"):
        row = dict(item)
        alias, canonical = row["alias_puuid"], row["canonical_puuid"]
        if alias in edges and edges[alias] == canonical:
            continue
        reason = None
        if alias in edges:
            reason = "canonical_alias_precedence"
        elif not _puuid(alias) or not _puuid(canonical) or alias == canonical:
            reason = "invalid_alias"
        elif any(out.execute("SELECT 1 FROM players WHERE puuid=?", (p,)).fetchone() is None for p in (alias, canonical)):
            reason = "orphan_alias"
        else:
            seen, cursor = {alias}, canonical
            while cursor in edges:
                if cursor in seen:
                    reason = "alias_cycle"
                    break
                seen.add(cursor)
                cursor = edges[cursor]
            if cursor in seen:
                reason = "alias_cycle"
        if reason:
            archive.row("player_aliases", row, reason)
        else:
            _insert(out, "player_aliases", row)
            edges[alias] = canonical


def _queue_entry(conn, row):
    if row["puuid"] is not None:
        found = conn.execute("SELECT * FROM crawl_queue WHERE puuid IS ?", (row["puuid"],)).fetchone()
    else:
        found = conn.execute("SELECT * FROM crawl_queue WHERE puuid IS NULL AND game_name IS ? "
                             "AND tag_line IS ? AND region IS ? ORDER BY id LIMIT 1",
                             (row["game_name"], row["tag_line"], row["region"])).fetchone()
    return dict(found) if found else None


def _valid_queue_entry(row):
    if row["puuid"] is not None:
        identified = _puuid(row["puuid"])
    else:
        identified = _text(row["game_name"]) and _text(row["tag_line"])
    return (identified and row["region"] in REGIONS
            and type(row["priority"]) is int and row["is_processed"] in (0, 1))


def _valid_rank_json(value):
    if value is None:
        return True
    try:
        return isinstance(json.loads(value), dict)
    except (TypeError, ValueError):
        return False


def _merge_provenance(source, out, archive):
    """Union the append-only collection provenance; the target's rows never change."""
    logger.info("Merging collection provenance")
    for item in source.execute("SELECT * FROM anchor_selections ORDER BY id"):
        row = dict(item)
        content = {k: v for k, v in row.items() if k != "id"}
        if out.execute("SELECT 1 FROM anchor_selections WHERE " + " AND ".join(f'"{k}" IS ?' for k in content),
                       tuple(content.values())).fetchone():
            continue
        if not (_puuid(row["puuid"]) and row["source"] in ANCHOR_SOURCES and _valid_timestamp(row["selected_at"])
                and (row["region"] is None or row["region"] in REGIONS) and _valid_rank_json(row["rank_json"])):
            archive.row("anchor_selections", row, "invalid_anchor_selection")
            continue
        if _insert(out, "anchor_selections", content).lastrowid != row["id"]:
            archive.row("anchor_selections", row, "regenerated_anchor_id")
    for item in source.execute("SELECT * FROM match_sightings"):
        row = dict(item)
        if _lookup(out, "match_sightings", row):
            continue
        if not (_text(row["match_id"]) and _valid_timestamp(row["seen_at"])
                and row["disposition"] in SIGHTING_DISPOSITIONS
                and (row["fetched_by_puuid"] is None or _puuid(row["fetched_by_puuid"]))):
            archive.row("match_sightings", row, "invalid_sighting")
            continue
        _insert(out, "match_sightings", row)
    for item in source.execute("SELECT * FROM trained_records"):
        # Source training ids are not the staged output's: trainer state is never imported.
        archive.row("trained_records", dict(item), "trainer_state_not_merged")


def _merge_registries(source, out, archive):
    for item in source.execute("SELECT * FROM champions"):
        row = dict(item)
        if _text(row["name"]):
            out.execute("INSERT OR IGNORE INTO champions VALUES (?)", (row["name"],))
        else:
            archive.row("champions", row, "invalid_champion")
    for row in out.execute("SELECT DISTINCT champion_name FROM match_participants"):
        if _text(row[0]):
            out.execute("INSERT OR IGNORE INTO champions VALUES (?)", (row[0],))
    for item in source.execute("SELECT * FROM crawl_queue ORDER BY id"):
        row = dict(item)
        existing = _queue_entry(out, row)
        if existing == row:
            continue
        if not _valid_queue_entry(row):
            archive.row("crawl_queue", row, "invalid_queue_entry")
        elif existing is not None:
            archive.row("crawl_queue", row, "canonical_queue_precedence")
        else:
            # Source ids collide with the target's; AUTOINCREMENT assigns fresh ones.
            inserted = _insert(out, "crawl_queue", {k: v for k, v in row.items() if k != "id"})
            if inserted.lastrowid != row["id"]:
                archive.row("crawl_queue", row, "regenerated_queue_id")


def _preservation(out, target_path):
    out.execute("ATTACH DATABASE ? AS canonical", (target_path.as_uri() + "?mode=ro",))
    violations = {}
    for table in TABLES:
        if table == "schema_version":
            continue
        keys = KEYS[table]
        fields = REQUIRED[table]
        if table == "players":
            fields = {"puuid", "updated_at"}
        join = " AND ".join(f't."{k}" IS m."{k}"' for k in keys)
        changed = " OR ".join(f't."{k}" IS NOT m."{k}"' for k in sorted(fields))
        if table == "players":
            for k in ("game_name", "tag_line"):
                changed += f' OR (merge_valid_identity(t."{k}") AND t."{k}" IS NOT m."{k}")'
        count = out.execute(f'SELECT count(*) FROM canonical."{table}" t LEFT JOIN main."{table}" m ON {join} '
                            f'WHERE m."{keys[0]}" IS NULL OR ({changed})').fetchone()[0]
        violations[table] = count
    return {"ok": not any(violations.values()), "changed_or_missing_rows": violations,
            "players_exception": "missing identity fields filled"}


def merge_databases(*, source, target, output, report, dry_run=False):
    """Build an independently reviewable merge without modifying either input."""
    source, target, output, report = (Path(p).resolve() for p in (source, target, output, report))
    archive_path = Path(str(report) + ".archive.jsonl.gz")
    paths = (source, target, output, report, archive_path)
    if len({str(p).casefold() for p in paths}) != len(paths):
        raise ValueError("source, target, output, report and archive paths must be distinct")
    for path in (source, target):
        if not path.is_file():
            raise FileNotFoundError(path)
    reserved = {str(Path(str(p) + suffix)).casefold() for p in (source, target)
                for suffix in ("-wal", "-shm", "-journal")}
    if any(str(p).casefold() in reserved for p in (output, report, archive_path)):
        raise ValueError("outputs cannot replace input SQLite sidecars")
    if source.samefile(target):
        raise ValueError("source and target must be distinct files")
    for path in (output, report, archive_path):
        if path.exists() or any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise FileExistsError(path)
        if not path.parent.is_dir():
            raise ValueError(f"output parent does not exist: {path.parent}")
    with ExitStack() as stack:
        src = stack.enter_context(closing(_read_only(source)))
        dst = stack.enter_context(closing(_read_only(target)))
        initial_states = {p: _file_state(p) for p in (source, target)}
        _schema(src)
        _schema(dst)
        result = {"schema_version": SCHEMA_VERSION, "dry_run": bool(dry_run), "output": str(output),
            "inputs": {"source": _fingerprint(src, source), "target": _fingerprint(dst, target)},
            "before": {"source": _counts(src), "target": _counts(dst)},
            "sqlite_cache_kib": {"input": 65536, "output": 262144, "temp": 65536},
            "match_scan": "sequential_table_scan"}
        if dry_run:
            counts, reasons = Counter(existing=0, imported=0, archived=0), Counter()
            for item in src.execute("SELECT * FROM matches"):
                row = dict(item)
                if _lookup(dst, "matches", row):
                    counts["existing"] += 1
                else:
                    validated, reason = _validate(row)
                    counts["imported" if validated else "archived"] += 1
                    if reason:
                        reasons[reason] += 1
            _check_unchanged(initial_states)
            result["inputs_unchanged"] = True
            result.update(classification=dict(counts), reasons=dict(reasons),
                note="Read-only classification preview. Preservation, relationship merge, and integrity checks run on the staged output during a real merge.")
            return result
        # Reserve with exclusive creation; never replace an existing artifact.
        created = []
        try:
            with output.open("xb"):
                pass
            created.append(output)
            out = stack.enter_context(closing(sqlite3.connect(output.as_uri() + "?mode=rw", uri=True)))
            out.row_factory = sqlite3.Row
            out.create_function("merge_valid_identity", 1, _text, deterministic=True)
            out.execute("PRAGMA cache_size=-262144")
            out.execute("PRAGMA temp.cache_size=-65536")
            logger.info("Backing up canonical target into %s", output)
            dst.backup(out)
            out.execute("PRAGMA cache_size=-262144")
            out.execute("PRAGMA temp.cache_size=-65536")
            out.execute("PRAGMA journal_mode=DELETE")
            out.execute("PRAGMA foreign_keys=ON")
            out.execute("PRAGMA temp_store=FILE")
            out.execute("CREATE TEMP TABLE merge_status (match_id TEXT PRIMARY KEY,category TEXT,same_raw INTEGER)")
            archive = _Archive(archive_path)
            created.append(archive_path)
            stack.callback(archive.close)
            result["classification"], result["reasons"] = _merge_matches(src, dst, out, archive)
            _merge_relationships(src, out, archive)
            _merge_identities(src, out, archive)
            _merge_registries(src, out, archive)
            _merge_provenance(src, out, archive)
            out.commit()
            logger.info("Verifying integrity and canonical row preservation")
            result["after"] = _counts(out)
            result["target_preservation"] = _preservation(out, target)
            result["verification"] = {
                "integrity_check": [r[0] for r in out.execute("PRAGMA main.integrity_check")],
                "foreign_key_check": [list(r) for r in out.execute("PRAGMA main.foreign_key_check")]}
            if (sum(result["classification"].values()) != result["before"]["source"]["matches"]
                    or not result["target_preservation"]["ok"]
                    or result["verification"]["integrity_check"] != ["ok"]
                    or result["verification"]["foreign_key_check"]):
                raise ValueError("merge verification failed: " + _json({"verification": result["verification"], "preservation": result["target_preservation"]}))
            archive.close()
            result["archive"] = archive.report()
            result["classification_manifest"] = {"path": str(archive_path), "kind": "classification"}
            _check_unchanged(initial_states)
            result["inputs_unchanged"] = True
            result["input_file_states"] = {str(p): state for p, state in initial_states.items()}
            with report.open("x", encoding="utf-8") as handle:
                created.append(report)
                json.dump(result, handle, ensure_ascii=False, indent=2)
            return result
        except BaseException:
            # Only artifacts exclusively created by this invocation are removed.
            stack.close()
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "target", "output", "report"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        result = merge_databases(**vars(args))
    except (ValueError, OSError, sqlite3.DatabaseError) as exc:
        parser.exit(1, f"Merge refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
