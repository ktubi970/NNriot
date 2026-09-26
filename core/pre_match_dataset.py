"""Read-only training records and chronological pre-match holdouts.

This module deliberately has no TensorFlow or writable database dependency.
Feature sanitation belongs to the training entry point, after loading.
"""

from __future__ import annotations

import base64
from contextlib import closing
import gzip
import json
import logging
import math
from pathlib import Path
import sqlite3
import zlib

from core.feature_labels import LABEL_KEYS, TIMELINE_LABEL_KEYS

logger = logging.getLogger(__name__)

_BINARY_KEYS = (
    "winner", "team_b_kill_lead", "kills_odd", "both_baron",
    "both_inhibitor", "both_dragon", "elder_dragon",
)
_THREE_CLASS_KEYS = ("first_blood", "first_baron", "first_inhibitor", "first_tower")
_COUNT_KEYS = (
    "total_kills", "team_a_kills", "team_b_kills", "total_barons",
    "total_dragons", "total_towers",
)
# A corruption guard, intentionally far above ordinary match totals.
_MAX_COUNT = 10_000


def _decode(blob):
    if not isinstance(blob, str) or not blob:
        return None
    try:
        text = blob.lstrip()
        if not text.startswith(("{", "[")):
            text = gzip.decompress(base64.b64decode(text, validate=True)).decode("utf-8")
        return json.loads(text)
    except (ValueError, TypeError, OSError, EOFError, zlib.error):
        return None


def _numeric_in_range(value, low, high):
    return (
        type(value) in (int, float)
        and low <= value <= high
        and math.isfinite(value)
        and value == int(value)
    )


def _valid_timestamp(value):
    return type(value) is int and value > 0


def _valid_labels(labels, winner):
    if not isinstance(labels, dict) or not set(LABEL_KEYS).issubset(labels):
        return False
    if not _numeric_in_range(winner, 0, 1) or labels["winner"] != winner:
        return False
    if any(not _numeric_in_range(labels[key], 0, 1) for key in _BINARY_KEYS):
        return False
    for key in (*_THREE_CLASS_KEYS, *TIMELINE_LABEL_KEYS):
        if key in labels and not _numeric_in_range(labels[key], 0, 2):
            return False
    return (
        all(_numeric_in_range(labels[key], 0, _MAX_COUNT) for key in _COUNT_KEYS)
        and _numeric_in_range(labels["kill_handicap"], -_MAX_COUNT, _MAX_COUNT)
    )


def _valid_features(features):
    if not isinstance(features, dict):
        return False
    for key in ("team_a", "team_b"):
        team = features.get(key)
        if not isinstance(team, list) or len(team) != 5:
            return False
        for player in team:
            if not isinstance(player, dict):
                return False
            champion = (
                player.get("champion") or player.get("championName")
                or player.get("champion_name")
            )
            if not isinstance(champion, str) or not champion.strip() or champion == "UNKNOWN":
                return False
    return True


def to_pre_match_features(feat: dict) -> dict:
    """Allowlist draft/context features; undated career aggregates are excluded."""
    if not isinstance(feat, dict):
        return {}
    if "participants" in feat or (
        isinstance(feat.get("info"), dict) and "participants" in feat["info"]
    ):
        from core import json_utils
        feat = json_utils.extract_pre_match_features(feat, region=feat.get("region"))
    clean = {
        "game_mode": feat.get("game_mode", "CLASSIC"),
        "game_version": feat.get("game_version", "UNKNOWN"),
        "season_id": feat.get("season_id"),
    }
    if feat.get("region"):
        clean["region"] = feat["region"]
    for team_key in ("team_a", "team_b"):
        clean[team_key] = [
            {
                "champion": p.get("champion") or p.get("championName")
                or p.get("champion_name", "UNKNOWN"),
                "role": p.get("role") or p.get("teamPosition") or "UNKNOWN",
            }
            for p in feat.get(team_key, [])
        ]
    return clean


def _decode_record(row) -> dict | None:
    record = dict(row)
    features = _decode(record["feature_json"])
    labels = _decode(record["labels_json"])
    if (
        not isinstance(record["match_id"], str)
        or not record["match_id"].strip()
        or not _valid_timestamp(record["game_creation_time"])
        or not _valid_features(features)
        or not _valid_labels(labels, record["winner_label"])
    ):
        return None
    record["feature_json"] = features
    record["labels_json"] = labels
    return record


def load_records(db_path: str, limit: int = 5000) -> list[dict]:
    """Read the latest ``limit`` candidates, retaining valid CLASSIC records.

    Selection and returned order use match creation time, never ingestion IDs.
    Invalid candidates are skipped without reading unbounded older rows. The
    returned count may therefore be below ``limit``; counts are logged.
    Missing database paths fail instead of creating a SQLite file.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    records = []
    candidates = 0
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            """
            SELECT t.id, t.match_id, m.game_creation_time, t.feature_json,
                   t.labels_json, t.winner_label
            FROM matches AS m
            JOIN training_dataset AS t ON t.match_id = m.match_id
            WHERE +m.game_mode = 'CLASSIC' AND m.game_creation_time > 0
                  AND typeof(m.game_creation_time) = 'integer'
            ORDER BY m.game_creation_time DESC, t.match_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            candidates += 1
            record = _decode_record(row)
            if record is None:
                continue
            records.append(record)
    logger.info(
        "Loaded %d valid records from %d candidates (limit=%d; skipped=%d).",
        len(records), candidates, limit, candidates - len(records),
    )
    return sorted(records, key=lambda row: (row["game_creation_time"], row["match_id"]))


def load_records_by_ids(db_path: str, match_ids: list[str]) -> list[dict]:
    """Read records for exactly the requested match IDs.

    Same eligibility and decoding rules as load_records.
    Fails if any ID is missing, invalid, or duplicated.
    """
    if not isinstance(match_ids, list) or len(match_ids) == 0:
        raise ValueError("match_ids must be a non-empty list of strings")
    seen_ids = set()
    for mid in match_ids:
        if not isinstance(mid, str) or not mid.strip():
            raise ValueError("every requested match_id must be a non-empty string")
        if mid in seen_ids:
            raise ValueError(f"duplicate requested match_id: {mid}")
        seen_ids.add(mid)

    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    selected = []
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        for start in range(0, len(match_ids), 500):
            batch = match_ids[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            sql = f"""SELECT t.id, t.match_id, m.game_creation_time, t.feature_json,
                             t.labels_json, t.winner_label
                      FROM training_dataset AS t
                      JOIN matches AS m ON m.match_id = t.match_id
                      WHERE m.game_mode = 'CLASSIC' AND t.match_id IN ({placeholders})"""
            for row in conn.execute(sql, batch):
                record = _decode_record(row)
                if record is None:
                    raise ValueError(f"Invalid frozen record: {row['match_id']}")
                selected.append(record)

    if len(selected) != len(match_ids):
        found_ids = {r["match_id"] for r in selected}
        missing = set(match_ids) - found_ids
        raise ValueError(f"Missing {len(missing)} requested match records: {missing}")

    if len({r["match_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate records returned from database query")

    return sorted(selected, key=lambda row: (row["game_creation_time"], row["match_id"]))


def chronological_split(records: list[dict]) -> dict[str, list[dict]]:
    """Split near 70/15/15 without sharing match IDs or timestamps.

    Timestamp groups stay whole. Boundaries are the nearest available cuts to
    70% and 85%, preferring the earlier cut for equal distances. At least 20
    records and three distinct timestamps are required for nonempty holdouts.
    The records themselves are returned unchanged.
    """
    if len(records) < 20:
        raise ValueError("chronological split requires at least 20 records")
    seen = set()
    for record in records:
        if not _valid_timestamp(record.get("game_creation_time")):
            raise ValueError("every record must have a positive integer timestamp")
        match_id = record.get("match_id")
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError("every record must have a nonempty match_id")
        if match_id in seen:
            raise ValueError(f"duplicate match_id: {match_id}")
        seen.add(match_id)

    ordered = sorted(records, key=lambda row: (row["game_creation_time"], row["match_id"]))
    cuts = [
        index for index in range(1, len(ordered))
        if ordered[index - 1]["game_creation_time"] != ordered[index]["game_creation_time"]
    ]
    if len(cuts) < 2:
        raise ValueError("three distinct timestamp groups are required for nonempty partitions")
    train_end = min(cuts[:-1], key=lambda index: (abs(index - len(ordered) * 0.70), index))
    validation_end = min(
        (index for index in cuts if index > train_end),
        key=lambda index: (abs(index - len(ordered) * 0.85), index),
    )
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def extract_major_patch(version: str) -> tuple[int, int]:
    """Extract (major, minor) integers from patch version string."""
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Invalid patch version: {version}")
    parts = version.strip().split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid patch version (expected major.minor): {version}")
    try:
        major = int(parts[0])
        minor = int(parts[1])
        if major < 0 or minor < 0:
            raise ValueError()
        return (major, minor)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid patch version (expected decimal integers): {version}")


def patches_by_split(splits: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Return sorted unique patch strings per split partition."""
    result = {}
    for name, rows in splits.items():
        unique_keys = set()
        for r in rows:
            feat = r.get("feature_json")
            ver = feat.get("game_version") if isinstance(feat, dict) else None
            try:
                major, minor = extract_major_patch(ver)
                unique_keys.add(((major, minor), f"{major}.{minor}"))
            except (ValueError, TypeError):
                unique_keys.add(((-1, -1), "UNKNOWN"))
        sorted_items = sorted(unique_keys, key=lambda x: (x[0][0], x[0][1], x[1]))
        result[name] = [item[1] for item in sorted_items]
    return result


def split_records(records: list[dict], strategy: str = "chronological") -> tuple[dict[str, list[dict]], dict]:
    """Split records according to the requested strategy (chronological or patch)."""
    if strategy not in ("chronological", "patch"):
        raise ValueError(f"Unsupported split strategy: {strategy}")

    chronological = chronological_split(records)

    if strategy == "chronological":
        meta = {
            "requested_strategy": "chronological",
            "effective_strategy": "chronological",
            "fallback_reason": None,
            "patches": patches_by_split(chronological),
        }
        return chronological, meta

    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        feat = r.get("feature_json")
        if not isinstance(feat, dict):
            raise ValueError("patch split strategy requires feature_json in every record")
        v = feat.get("game_version")
        if not v or v == "UNKNOWN":
            raise ValueError(f"patch split strategy requires valid patch version, got: {v}")
        patch = extract_major_patch(v)
        groups[patch].append(r)

    ordered_patches = sorted(groups)
    if len(ordered_patches) < 3:
        return chronological, {
            "requested_strategy": "patch",
            "effective_strategy": "chronological",
            "fallback_reason": "fewer_than_three_patches",
            "patches": patches_by_split(chronological),
        }

    splits = {
        "train": [r for p in ordered_patches[:-2] for r in groups[p]],
        "validation": list(groups[ordered_patches[-2]]),
        "test": list(groups[ordered_patches[-1]]),
    }
    for rows in splits.values():
        rows.sort(key=lambda r: (r["game_creation_time"], r["match_id"]))

    for left, right in (("train", "validation"), ("validation", "test")):
        if splits[left][-1]["game_creation_time"] >= splits[right][0]["game_creation_time"]:
            raise ValueError("patch time ranges overlap; choose an explicit chronological run")

    meta = {
        "requested_strategy": "patch",
        "effective_strategy": "patch",
        "fallback_reason": None,
        "patches": patches_by_split(splits),
    }
    return splits, meta
