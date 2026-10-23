"""Batch compute and persist player spatial trajectory embeddings.

Reads match timelines from `match_timelines`, runs the pinned DualPhaseTrajectoryGRUEncoder,
and writes 32D embeddings with their `encoder_id` into `player_spatial_embeddings`.
Refuses to append to a database holding embeddings from other encoder weights;
`--recompute-all` replaces every stored embedding instead.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.database import DB_PATH, get_connection
from core.spatial_rnn import (
    MixedSpatialEncoderError,
    extract_dual_phase_trajectory,
    foreign_spatial_encoder,
    load_pinned_spatial_encoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compute_spatial_embeddings")

EMPTY_TRAJECTORY = "__EMPTY_TRAJECTORY__"


def compute_and_save_spatial_embeddings(
    db_path: str = DB_PATH,
    batch_size: int = 250,
    limit: int | None = None,
    recompute_all: bool = False,
) -> dict[str, int]:
    """Process timelines and store player trajectory embeddings in memory-efficient chunks."""
    encoder, encoder_id = load_pinned_spatial_encoder()
    logger.info("Loaded pinned DualPhaseTrajectoryGRUEncoder %s", encoder_id)

    processed_matches = 0
    saved_embeddings = 0
    t_start = time.time()

    with get_connection(db_path) as conn:
        if recompute_all:
            deleted = conn.execute("DELETE FROM player_spatial_embeddings").rowcount
            logger.warning("Deleted %d stored spatial embeddings before recomputing them with %s.", deleted, encoder_id)
        else:
            foreign = foreign_spatial_encoder(conn, encoder_id)
            if foreign:
                raise MixedSpatialEncoderError(
                    f"{db_path} already holds spatial embeddings from encoder {foreign}; appending {encoder_id} "
                    "would mix embedding spaces. Rerun with --recompute-all to replace every stored embedding."
                )

    # Query fast remaining matches count for progress reporting
    with get_connection(db_path) as conn:
        total_timelines = conn.execute(
            "SELECT COUNT(*) FROM match_timelines WHERE timeline_json IS NOT NULL"
        ).fetchone()[0]
        existing_matches = conn.execute(
            "SELECT COUNT(DISTINCT match_id) FROM player_spatial_embeddings"
        ).fetchone()[0]
        total_remaining = max(0, total_timelines - existing_matches)

    target_total = min(limit, total_remaining) if limit else total_remaining
    logger.info("Total matches requiring spatial embeddings: %d (already computed: %d)", target_total, existing_matches)

    if target_total == 0:
        logger.info("All matches already have spatial embeddings stored.")
        return {"matches": 0, "embeddings": 0}

    chunk_size = max(10, min(batch_size, 500))

    while True:
        # Determine remaining limit for this chunk
        current_limit = chunk_size
        if limit is not None:
            remaining_limit = limit - processed_matches
            if remaining_limit <= 0:
                break
            current_limit = min(chunk_size, remaining_limit)

        with get_connection(db_path) as conn:
            query = """
                SELECT t.match_id, t.timeline_json, m.game_creation_time
                FROM match_timelines t
                JOIN matches m ON m.match_id = t.match_id
                WHERE t.timeline_json IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM player_spatial_embeddings p WHERE p.match_id = t.match_id
                  )
                LIMIT ?
            """
            rows = conn.execute(query, (current_limit,)).fetchall()

        if not rows:
            break

        all_early = []
        all_mid = []
        # One key per encoded trajectory, in the same order as all_early/all_mid.
        trajectory_keys: list[tuple[str, str, int]] = []
        db_records: list[tuple[str, str, int, bytes, str]] = []
        valid_matches_in_chunk = 0

        for row in rows:
            mid = row["match_id"]
            tj = row["timeline_json"]
            creation_time = row["game_creation_time"] or 0

            try:
                timeline_data = json.loads(gzip.decompress(base64.b64decode(tj)).decode("utf-8"))
            except Exception as e:
                logger.warning("Failed to decompress timeline for %s: %s", mid, e)
                continue

            participants = timeline_data.get("info", {}).get("participants") or []
            if not participants:
                logger.warning("Timeline for %s has empty participants — marking handled.", mid)
                db_records.append((mid, EMPTY_TRAJECTORY, int(creation_time), b"\x00" * 128, encoder_id))
                continue

            match_puuids = []
            match_early = []
            match_mid = []

            for p in participants:
                puuid = p.get("puuid")
                pid = p.get("participantId")
                if not puuid or pid is None:
                    continue

                is_red = int(pid) > 5
                early_seq, mid_seq = extract_dual_phase_trajectory(
                    timeline_data, puuid=puuid, is_red_side=is_red
                )
                match_early.append(early_seq)
                match_mid.append(mid_seq)
                match_puuids.append(puuid)

            if match_puuids:
                all_early.extend(match_early)
                all_mid.extend(match_mid)
                for puuid in match_puuids:
                    trajectory_keys.append((mid, puuid, int(creation_time)))
                valid_matches_in_chunk += 1
            else:
                logger.warning("Timeline for %s has 0 valid participants (remake/aborted) — marking handled.", mid)
                db_records.append((mid, EMPTY_TRAJECTORY, int(creation_time), b"\x00" * 128, encoder_id))

        if all_early:
            early_tensor = np.stack(all_early, axis=0)
            mid_tensor = np.stack(all_mid, axis=0)
            emb_matrix = encoder([early_tensor, mid_tensor]).numpy().astype(np.float32)

            for (mid, puuid, ctime), emb_vec in zip(trajectory_keys, emb_matrix, strict=True):
                db_records.append((mid, puuid, ctime, emb_vec.tobytes(), encoder_id))

        if db_records:
            with get_connection(db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO player_spatial_embeddings
                    (match_id, puuid, game_creation_time, embedding_blob, encoder_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    db_records,
                )

            saved_embeddings += len(db_records)
        else:
            logger.info("No further records to extract. Stopping.")
            break

        processed_matches += len(rows)
        elapsed = time.time() - t_start
        rate = processed_matches / elapsed if elapsed > 0 else 0.0
        remaining_sec = (target_total - processed_matches) / rate if rate > 0 else 0.0

        logger.info(
            "Progress: %d/%d matches (%d embeddings) [%.1f matches/s, ETA: %.1f min]",
            processed_matches,
            target_total,
            saved_embeddings,
            rate,
            remaining_sec / 60.0,
        )

    logger.info("Complete! Processed %d matches, saved %d spatial embeddings.", processed_matches, saved_embeddings)
    return {"matches": processed_matches, "embeddings": saved_embeddings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store player trajectory embeddings.")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--recompute-all", action="store_true",
                        help="Delete every stored spatial embedding and recompute them with the pinned encoder")
    args = parser.parse_args()

    try:
        compute_and_save_spatial_embeddings(db_path=args.db, limit=args.limit, batch_size=args.batch,
                                            recompute_all=args.recompute_all)
    except MixedSpatialEncoderError as exc:
        parser.exit(1, f"Refused: {exc}\n")
