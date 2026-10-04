"""Point-in-time chronologically streaming dataset loader for pre-match modeling.

Streams matches and participants in strict creation time order, computing
leak-free Bayesian player priors before updating state with match outcomes.
"""
from __future__ import annotations

from contextlib import closing
import base64
import gzip
import json
import logging
from pathlib import Path
import sqlite3
import zlib
from typing import Any

from core.features.timeline_metrics import extract_timeline_lane_metrics
from core.player_priors import PlayerPriorsTracker
from core.pre_match_dataset import _decode, _valid_timestamp, _valid_labels

logger = logging.getLogger("point_in_time_dataset")


def _parse_timeline_json(raw: Any) -> dict[str, Any] | None:
    """Safely decode and parse raw or gzipped/base64 timeline JSON."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        if isinstance(raw, (bytes, bytearray)):
            try:
                return json.loads(gzip.decompress(raw).decode("utf-8"))
            except Exception:
                return json.loads(raw.decode("utf-8"))
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                return json.loads(stripped)
            decoded = base64.b64decode(stripped)
            try:
                return json.loads(gzip.decompress(decoded).decode("utf-8"))
            except Exception:
                return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    return None


def load_result_end_times(
    db_path: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Read explicit Riot result ends in a separate, read-only SQLite connection.

    Missing or unusable metadata is excluded. A database/cache creation mismatch
    and database access or schema errors are raised to the caller.
    """
    expected = {row['match_id']: row['game_creation_time'] for row in rows}
    ids = list(expected)
    ends: dict[str, int] = {}
    counts = {'requested': len(rows), 'found': 0, 'invalid': 0}
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        raise FileNotFoundError(f"Database not found: {db_file}")
    uri = db_file.as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        for offset in range(0, len(ids), 400):
            chunk = ids[offset:offset + 400]
            sql = ('SELECT match_id, game_creation_time, raw_json FROM matches '
                   'WHERE match_id IN (' + ','.join('?' for _ in chunk) + ')')
            for mid, creation, raw in conn.execute(sql, chunk):
                counts['found'] += 1
                if creation != expected[mid]:
                    raise ValueError(f'Database/cache creation timestamp mismatch: {mid}')
                try:
                    payload = raw if raw.startswith('{') else gzip.decompress(base64.b64decode(raw))
                    info = json.loads(payload)['info']
                    start, end = info.get('gameStartTimestamp'), info.get('gameEndTimestamp')
                    valid = (type(start) is int and type(end) is int and
                             creation <= start < end and end - creation <= 24 * 3600 * 1000)
                except (TypeError, ValueError, KeyError, OSError, EOFError, AttributeError, zlib.error):
                    valid = False
                if valid:
                    ends[mid] = end
                else:
                    counts['invalid'] += 1
    counts.update(valid=len(ends), missing=len(ids) - counts['found'])
    return ends, counts


def load_enriched_records(
    db_path: str,
    limit: int = 15000,
    warmup_matches: int = 3000,
    include_timeline: bool = False,
) -> list[dict[str, Any]]:
    """Load enriched pre-match records with leak-free point-in-time player priors.

    1. Reads matches and participants chronologically from SQLite using indexes.
    2. Updates an in-memory PlayerPriorsTracker sequentially.
    3. Attaches point-in-time player priors to each participant dict before outcome update.
    4. Discards initial `warmup_matches` to ensure training on informed player priors.
    5. Optionally incorporates timeline early-game lane metrics when include_timeline=True.
    """
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(warmup_matches, int) or warmup_matches < 0:
        raise ValueError("warmup_matches must be a non-negative integer")

    total_needed = limit + warmup_matches
    db_file = Path(db_path).resolve()
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file}")

    uri = db_file.as_uri() + "?mode=ro"
    records: list[dict[str, Any]] = []
    tracker = PlayerPriorsTracker()

    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")

        # Check if match_participants table exists
        has_participants = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_participants'"
        ).fetchone() is not None

        if not has_participants:
            logger.warning("match_participants table not found; falling back to load_records with default priors.")
            from core.pre_match_dataset import load_records
            return load_records(db_path, limit=limit)

        # Check if match_timelines table exists
        has_timelines = False
        if include_timeline:
            has_timelines = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_timelines'"
            ).fetchone() is not None

        # 1. Fetch chronological match IDs with training dataset info
        match_cursor = conn.execute(
            """
            SELECT m.match_id, m.game_creation_time, t.feature_json, t.labels_json, t.winner_label
            FROM matches AS m
            JOIN training_dataset AS t ON t.match_id = m.match_id
            WHERE +m.game_mode = 'CLASSIC' AND m.game_creation_time > 0
                  AND typeof(m.game_creation_time) = 'integer'
            ORDER BY m.game_creation_time ASC
            LIMIT ?
            """,
            (total_needed,),
        )
        candidates = match_cursor.fetchall()
        logger.info("Found %d candidate matches for enriched streaming (target=%d, warmup=%d)",
                    len(candidates), limit, warmup_matches)

        # Check available columns in match_participants
        part_cols = [c[1] for c in conn.execute("PRAGMA table_info(match_participants)").fetchall()]
        has_combat = "kills" in part_cols
        cols_str = "rowid, match_id, puuid, champion_name, team_position, win, kills, deaths, assists, total_cs, gold_earned, game_duration" if has_combat else "rowid, match_id, puuid, champion_name, team_position, win"

        # 2. Batch pre-fetch participants in chunks to optimize SQLite throughput
        logger.info("Batch pre-fetching participants for %d matches (has_combat=%s)...", len(candidates), has_combat)
        parts_by_match: dict[str, list[Any]] = {}
        chunk_size = 2000
        candidate_mids = [row["match_id"] for row in candidates]
        for i in range(0, len(candidate_mids), chunk_size):
            chunk = candidate_mids[i : i + chunk_size]
            ph = ",".join("?" for _ in chunk)
            p_cursor = conn.execute(
                f"""
                SELECT {cols_str}
                FROM match_participants
                WHERE match_id IN ({ph})
                ORDER BY rowid ASC
                """,
                chunk,
            )
            for prow in p_cursor:
                parts_by_match.setdefault(prow["match_id"], []).append(prow)

        raw_timelines_by_match: dict[str, Any] = {}
        if include_timeline and has_timelines:
            logger.info("Batch pre-fetching timelines for %d candidate matches...", len(candidate_mids))
            for i in range(0, len(candidate_mids), chunk_size):
                chunk = candidate_mids[i : i + chunk_size]
                ph = ",".join("?" for _ in chunk)
                t_cursor = conn.execute(
                    f"""
                    SELECT match_id, timeline_json
                    FROM match_timelines
                    WHERE match_id IN ({ph})
                    """,
                    chunk,
                )
                for trow in t_cursor:
                    raw_tj = trow["timeline_json"]
                    if raw_tj is not None:
                        raw_timelines_by_match[trow["match_id"]] = raw_tj

        match_count = 0
        valid_post_warmup = 0

        for row in candidates:
            mid = row["match_id"]
            gtime = row["game_creation_time"]
            winner = row["winner_label"]

            labels = _decode(row["labels_json"])
            if not _valid_timestamp(gtime) or not _valid_labels(labels, winner):
                continue

            parts = parts_by_match.get(mid, [])
            if len(parts) != 10:
                continue

            team_a_parts = parts[:5]
            team_b_parts = parts[5:]

            # 1. Extract point-in-time priors BEFORE updating state
            team_a_clean = []
            for p in team_a_parts:
                puuid = p["puuid"]
                champ = p["champion_name"]
                pos = p["team_position"]
                priors = tracker.get_player_priors(
                    puuid, champ, pos, extended=True, include_timeline=include_timeline
                )
                team_a_clean.append({
                    "puuid": puuid,
                    "champion": champ,
                    "role": pos,
                    "priors": priors,
                })

            team_b_clean = []
            for p in team_b_parts:
                puuid = p["puuid"]
                champ = p["champion_name"]
                pos = p["team_position"]
                priors = tracker.get_player_priors(
                    puuid, champ, pos, extended=True, include_timeline=include_timeline
                )
                team_b_clean.append({
                    "puuid": puuid,
                    "champion": champ,
                    "role": pos,
                    "priors": priors,
                })

            # 2. Update tracker state AFTER feature extraction
            for p in parts:
                tracker.update(
                    puuid=p["puuid"],
                    champion=p["champion_name"],
                    role=p["team_position"],
                    win=p["win"],
                    kills=(p["kills"] or 0) if has_combat else 0,
                    deaths=(p["deaths"] or 0) if has_combat else 0,
                    assists=(p["assists"] or 0) if has_combat else 0,
                    total_cs=(p["total_cs"] or 0) if has_combat else 0,
                    gold_earned=(p["gold_earned"] or 0) if has_combat else 0,
                    game_duration=(p["game_duration"] or 0) if has_combat else 0,
                )

            # 3. Accumulate timeline metrics AFTER feature extraction
            if include_timeline and has_timelines:
                raw_tj = raw_timelines_by_match.get(mid)
                if raw_tj is not None:
                    t_data = _parse_timeline_json(raw_tj)
                    if t_data:
                        participant_meta = [
                            {"participantId": idx, "puuid": p["puuid"]}
                            for idx, p in enumerate(parts, start=1)
                        ]
                        lane_metrics = extract_timeline_lane_metrics(t_data, participants=participant_meta)
                        for p in parts:
                            p_puuid = p["puuid"]
                            if p_puuid in lane_metrics:
                                m = lane_metrics[p_puuid]
                                tracker.record_timeline_metrics(
                                    puuid=p_puuid,
                                    gd15=m["gd15"],
                                    csd15=m["csd15"],
                                    xpd15=m["xpd15"],
                                    fbi10=m["fbi10"],
                                )

            match_count += 1
            if match_count <= warmup_matches:
                continue  # Discard warmup matches

            feat = {
                "team_a": team_a_clean,
                "team_b": team_b_clean,
                "game_mode": "CLASSIC",
            }
            records.append({
                "match_id": mid,
                "game_creation_time": gtime,
                "feature_json": feat,
                "labels_json": labels,
                "winner_label": winner,
            })
            valid_post_warmup += 1
            if valid_post_warmup >= limit:
                break

    logger.info("Loaded %d enriched records after %d warmup matches.", len(records), warmup_matches)
    return records
