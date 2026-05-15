"""
cleanup_malformed.py - One-shot script to delete training_dataset rows whose
labels could not be extracted (`labels_json IS NULL`).

Most commonly these are CHERRY (Arena 2v2v2v2) and STRAWBERRY (Swarm PvE)
matches with non-standard team structures. After this script runs and the
EXCLUDED_GAME_MODES filter is in place in data_collector, no new such rows
should appear.

Idempotent. Safe to re-run.
"""

import argparse
import logging

import database

logger = logging.getLogger(__name__)


def cleanup_malformed(db_path: str = database.DB_PATH, dry_run: bool = True) -> dict:
    """
    Delete training_dataset rows with NULL labels_json.

    The corresponding `matches` rows are preserved (forensic value).

    Args:
        db_path: SQLite database path.
        dry_run: If True, only count and report. Default True for safety.

    Returns:
        Stats dict: {"would_delete": N} (dry-run) or {"deleted": N}.
    """
    with database.get_connection(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM training_dataset WHERE labels_json IS NULL"
        ).fetchone()[0]

    if n == 0:
        logger.info("No malformed training records to clean up.")
        return {"deleted": 0}

    if dry_run:
        logger.info("Dry run: %d training records have labels_json IS NULL. "
                    "Re-run with --apply to delete them.", n)
        return {"would_delete": n}

    with database.get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM training_dataset WHERE labels_json IS NULL")
        deleted = cur.rowcount

    logger.info("Deleted %d malformed training records.", deleted)
    return {"deleted": deleted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete training_dataset rows with NULL labels_json.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete. Without this flag, runs in dry-run mode.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = cleanup_malformed(dry_run=not args.apply)
    print(f"Result: {result}")
