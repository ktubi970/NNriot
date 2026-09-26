"""Point-in-time chronologically streaming dataset loader for pre-match modeling.

Streams matches and participants in strict creation time order, computing
leak-free Bayesian player priors before updating state with match outcomes.
"""
from __future__ import annotations

from contextlib import closing
import logging
from pathlib import Path
import sqlite3
from typing import Any

from core.player_priors import PlayerPriorsTracker
from core.pre_match_dataset import _decode, _valid_timestamp, _valid_labels

logger = logging.getLogger("point_in_time_dataset")


def load_enriched_records(
    db_path: str,
    limit: int = 15000,
    warmup_matches: int = 3000,
) -> list[dict[str, Any]]:
    """Load enriched pre-match records with leak-free point-in-time player priors.

    1. Reads matches and participants chronologically from SQLite using indexes.
    2. Updates an in-memory PlayerPriorsTracker sequentially.
    3. Attaches point-in-time player priors to each participant dict before outcome update.
    4. Discards initial `warmup_matches` to ensure training on informed player priors.
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

        # 2. Batch pre-fetch participants in chunks to optimize SQLite throughput
        logger.info("Batch pre-fetching participants for %d matches...", len(candidates))
        parts_by_match: dict[str, list[Any]] = {}
        chunk_size = 2000
        candidate_mids = [row["match_id"] for row in candidates]
        for i in range(0, len(candidate_mids), chunk_size):
            chunk = candidate_mids[i : i + chunk_size]
            ph = ",".join("?" for _ in chunk)
            p_cursor = conn.execute(
                f"""
                SELECT rowid, match_id, puuid, champion_name, team_position, win
                FROM match_participants
                WHERE match_id IN ({ph})
                ORDER BY rowid ASC
                """,
                chunk,
            )
            for prow in p_cursor:
                parts_by_match.setdefault(prow["match_id"], []).append(prow)

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
                priors = tracker.get_player_priors(puuid, champ, pos)
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
                priors = tracker.get_player_priors(puuid, champ, pos)
                team_b_clean.append({
                    "puuid": puuid,
                    "champion": champ,
                    "role": pos,
                    "priors": priors,
                })

            # 2. Update tracker state AFTER feature extraction
            for p in parts:
                tracker.update(p["puuid"], p["champion_name"], p["team_position"], p["win"])

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
