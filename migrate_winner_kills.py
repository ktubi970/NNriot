"""
migrate_winner_kills.py — One-shot script to rename the `winner_kills` key
to `team_b_kill_lead` in all training_dataset.labels_json blobs.

Idempotent — re-running on already-migrated rows is a no-op.

Run after deploying the rename in the code. Backup the DB first.
"""

import argparse
import json
import gzip
import base64
import logging

import database

logger = logging.getLogger(__name__)

OLD_KEY = "winner_kills"
NEW_KEY = "team_b_kill_lead"


def _decode(blob: str) -> dict | None:
    if not blob:
        return None
    if blob.startswith("{") or blob.startswith("["):
        return json.loads(blob)
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))


def _encode(labels: dict) -> str:
    return base64.b64encode(
        gzip.compress(json.dumps(labels).encode("utf-8"))
    ).decode("ascii")


def migrate(db_path: str = database.DB_PATH, dry_run: bool = True, chunk_size: int = 500) -> dict:
    """
    Rename `winner_kills` → `team_b_kill_lead` in all training_dataset.labels_json blobs.

    Args:
        db_path: SQLite DB path.
        dry_run: If True (default), only count rows that would be updated.
        chunk_size: How many rows to process per transaction.

    Returns:
        Stats dict.
    """
    stats = {"total_seen": 0, "renamed": 0, "already_new": 0, "errors": 0}
    cumulative = 0

    while True:
        with database.get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, labels_json
                FROM training_dataset
                WHERE labels_json IS NOT NULL
                  AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (cumulative, chunk_size),
            ).fetchall()

        if not rows:
            break

        stats["total_seen"] += len(rows)
        updates = []
        for r in rows:
            cumulative = r["id"]
            try:
                labels = _decode(r["labels_json"])
                if labels is None:
                    continue
                if OLD_KEY not in labels:
                    stats["already_new"] += 1
                    continue
                if NEW_KEY in labels:
                    # Both present (partial migration from a crash mid-run):
                    # drop the legacy key without overwriting the new one.
                    labels.pop(OLD_KEY)
                else:
                    labels[NEW_KEY] = labels.pop(OLD_KEY)
                updates.append((_encode(labels), r["id"]))
            except Exception:
                stats["errors"] += 1
                logger.error("Failed to migrate id=%s", r["id"], exc_info=True)

        if updates and not dry_run:
            with database.get_connection(db_path) as conn:
                conn.executemany(
                    "UPDATE training_dataset SET labels_json = ? WHERE id = ?",
                    updates,
                )
            stats["renamed"] += len(updates)
        elif updates and dry_run:
            stats["renamed"] += len(updates)  # count only

        logger.info(
            "Progress: cumulative=%d renamed=%d already_new=%d errors=%d",
            cumulative, stats["renamed"], stats["already_new"], stats["errors"]
        )

    if dry_run:
        logger.info("Dry run — no rows were modified. Re-run with --apply to commit.")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename winner_kills → team_b_kill_lead in labels_json.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually update rows. Without this flag, runs in dry-run mode.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = migrate(dry_run=not args.apply)
    print(f"Result: {result}")
