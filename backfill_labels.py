"""
backfill_labels.py - One-shot script to populate labels_json for all existing
training records by re-decoding the corresponding raw match JSON.

Idempotent: re-running only processes rows where labels_json IS NULL.
"""

import json
import gzip
import base64
import logging

import database
from feature_labels import extract_labels

logger = logging.getLogger(__name__)


def backfill_labels(db_path: str = database.DB_PATH) -> dict:
    """
    For every training_dataset row with NULL labels_json, decode the
    corresponding matches.raw_json, run extract_labels, and update.

    Returns a stats dict.
    """
    database.init_db(db_path)

    with database.get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT td.id, td.match_id, m.raw_json
            FROM training_dataset td
            JOIN matches m ON td.match_id = m.match_id
            WHERE td.labels_json IS NULL
            """
        ).fetchall()

    stats = {"total": len(rows), "updated": 0, "skipped_malformed": 0, "errors": 0}
    if not rows:
        logger.info("No rows require backfill.")
        return stats

    logger.info("Backfilling labels for %d training records...", len(rows))
    updates = []
    for r in rows:
        try:
            rj = r["raw_json"]
            if rj.startswith("{") or rj.startswith("["):
                match_data = json.loads(rj)
            else:
                match_data = json.loads(
                    gzip.decompress(base64.b64decode(rj)).decode("utf-8")
                )

            labels = extract_labels(match_data)
            if labels is None:
                stats["skipped_malformed"] += 1
                continue

            compressed = base64.b64encode(
                gzip.compress(json.dumps(labels).encode("utf-8"))
            ).decode("ascii")
            updates.append((compressed, r["id"]))
        except Exception:
            stats["errors"] += 1
            logger.error(
                "Failed to backfill record id=%s match_id=%s",
                r["id"],
                r["match_id"],
                exc_info=True,
            )

    if updates:
        with database.get_connection(db_path) as conn:
            conn.executemany(
                "UPDATE training_dataset SET labels_json = ? WHERE id = ?",
                updates,
            )
        stats["updated"] = len(updates)
        logger.info("Backfilled %d records.", len(updates))

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    result = backfill_labels()
    print(f"Backfill result: {result}")
