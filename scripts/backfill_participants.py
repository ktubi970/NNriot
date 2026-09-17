#!/usr/bin/env python3
"""
backfill_participants.py - High-throughput backfill of match_participants from matches.raw_json.
"""
import os
import sys
import time
import argparse
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_participants")

def run_backfill(batch_size: int = 500, max_matches: int = None, db_path: str = database.DB_PATH) -> int:
    start_time = time.time()
    total_backfilled = 0

    logger.info("Starting match_participants backfill on '%s' (batch_size=%d)...", db_path, batch_size)
    while True:
        limit = batch_size
        if max_matches is not None:
            limit = min(batch_size, max_matches - total_backfilled)
            if limit <= 0:
                break

        count = database.backfill_match_participants(limit=limit, db_path=db_path)
        if count == 0:
            break
        total_backfilled += count
        elapsed = time.time() - start_time
        rate = total_backfilled / max(1e-3, elapsed)
        logger.info("Backfilled %d matches so far (%.1f matches/s)...", total_backfilled, rate)

    total_time = time.time() - start_time
    logger.info("Finished backfill: %d total matches processed in %.2f seconds.", total_backfilled, total_time)
    return total_backfilled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill match_participants table")
    parser.add_argument("--batch-size", type=int, default=500, help="Matches per transaction batch")
    parser.add_argument("--max", type=int, default=None, help="Max matches to backfill")
    parser.add_argument("--db", type=str, default=database.DB_PATH, help="Path to database")
    args = parser.parse_args()

    run_backfill(batch_size=args.batch_size, max_matches=args.max, db_path=args.db)
