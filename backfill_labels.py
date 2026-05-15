"""
backfill_labels.py - One-shot script to populate labels_json for all existing
training records by re-decoding the corresponding raw match JSON.

Idempotent: re-running only processes rows where labels_json IS NULL.
Streaming: reads/updates in chunks so memory usage stays bounded regardless
of total row count.
"""

import json
import gzip
import base64
import logging

import database
from feature_labels import extract_labels

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 500


def _decode_raw_json(rj: str) -> dict:
    """Decode a raw_json string (either plain JSON or gzip+base64)."""
    if rj.startswith("{") or rj.startswith("["):
        return json.loads(rj)
    return json.loads(gzip.decompress(base64.b64decode(rj)).decode("utf-8"))


def _compress_labels(labels: dict) -> str:
    """gzip+base64-encode a labels dict for storage."""
    return base64.b64encode(
        gzip.compress(json.dumps(labels).encode("utf-8"))
    ).decode("ascii")


def backfill_labels(db_path: str = database.DB_PATH, chunk_size: int = _CHUNK_SIZE) -> dict:
    """
    Stream-update training_dataset.labels_json for all rows where it IS NULL,
    in chunks of `chunk_size`. Memory usage stays bounded regardless of total
    row count.

    Idempotent. Re-runnable.
    """
    database.init_db(db_path)

    stats = {"total_seen": 0, "updated": 0, "skipped_malformed": 0, "errors": 0}
    cumulative = 0

    while True:
        with database.get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT td.id, td.match_id, m.raw_json
                FROM training_dataset td
                JOIN matches m ON td.match_id = m.match_id
                WHERE td.labels_json IS NULL
                LIMIT ?
                """,
                (chunk_size,),
            ).fetchall()

        if not rows:
            break

        stats["total_seen"] += len(rows)
        updates = []
        for r in rows:
            try:
                match_data = _decode_raw_json(r["raw_json"])
                labels = extract_labels(match_data)
                if labels is None:
                    stats["skipped_malformed"] += 1
                    continue
                updates.append((_compress_labels(labels), r["id"]))
            except Exception:
                stats["errors"] += 1
                logger.error("Failed to backfill record id=%s match_id=%s",
                             r["id"], r["match_id"], exc_info=True)

        if updates:
            with database.get_connection(db_path) as conn:
                conn.executemany(
                    "UPDATE training_dataset SET labels_json = ? WHERE id = ?",
                    updates,
                )
            stats["updated"] += len(updates)

        cumulative += len(rows)
        logger.info("Backfill progress: chunk processed (%d rows), cumulative %d, updated %d, skipped %d, errors %d",
                    len(rows), cumulative, stats["updated"], stats["skipped_malformed"], stats["errors"])

        # Safety: if a chunk yields ZERO updates AND zero skips AND zero errors,
        # something's wrong (probably NULL labels_json on a row whose raw_json
        # decoded successfully but extract_labels returned None — counted in
        # skipped_malformed). If updates were empty but we kept seeing rows,
        # we'd loop forever. Break to be safe.
        if not updates and stats["skipped_malformed"] == cumulative and stats["errors"] == 0:
            # All remaining NULL rows are malformed — no point looping.
            # But they ARE still in the DB with NULL. Tell the user.
            logger.warning(
                "All %d remaining NULL rows are malformed (extract_labels returned None). "
                "Run cleanup_malformed.py --apply to remove them.", stats["skipped_malformed"]
            )
            break

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = backfill_labels()
    print(f"Backfill result: {result}")
