"""CrawlQueueRepository for managing persistent player crawl queues."""

import logging
from typing import Iterable, Optional

from core import config, database
from core.collection.models import CrawlItem
from core.collection.provenance import LADDER_SEED_MIN_PRIORITY, parse_sqlite_timestamp_ms

logger = logging.getLogger(__name__)

# Stay below SQLite's historical 999 bound-parameter limit.
_LOOKUP_CHUNK = 500


class CrawlQueueRepository:
    """Repository encapsulating crawl queue persistence, priority scheduling, and deduplication."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = config.get_db_path(db_path, require_exists=False)

    def enqueue_puuids(self, puuids: list[str], region: str, priority: int = 0) -> int:
        """Insert multiple PUUIDs into crawl_queue with ON CONFLICT(puuid) DO NOTHING.

        Returns the number of rows successfully inserted.
        """
        cleaned = [p.strip() for p in puuids if p and p.strip()]
        if not cleaned:
            return 0
        with database.get_connection(self.db_path) as conn:
            cursor = conn.executemany(
                """
                INSERT INTO crawl_queue (puuid, region, priority)
                VALUES (?, ?, ?)
                ON CONFLICT(puuid) DO NOTHING
                """,
                [(p, region, priority) for p in cleaned],
            )
            return max(0, cursor.rowcount)

    def enqueue_riot_id(
        self,
        game_name: str,
        tag_line: str,
        region: str,
        priority: int = 10,
        puuid: Optional[str] = None,
    ) -> bool:
        """Enqueue a Riot ID.

        If PUUID provided, inserts or updates priority and name/tag.
        If no PUUID, inserts row with game_name, tag_line, region.
        """
        if not game_name or not tag_line or not region:
            return False
        try:
            with database.get_connection(self.db_path) as conn:
                if puuid:
                    conn.execute(
                        """
                        INSERT INTO crawl_queue (puuid, region, game_name, tag_line, priority)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(puuid) DO UPDATE SET
                            region = excluded.region,
                            game_name = excluded.game_name,
                            tag_line = excluded.tag_line,
                            priority = excluded.priority
                        """,
                        (puuid, region, game_name, tag_line, priority),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO crawl_queue (game_name, tag_line, region, priority)
                        VALUES (?, ?, ?, ?)
                        """,
                        (game_name, tag_line, region, priority),
                    )
                return True
        except Exception as e:
            logger.error("Failed to enqueue riot ID %s#%s: %s", game_name, tag_line, e)
            return False

    def pop_next(self, limit: int = 1) -> list[CrawlItem]:
        """Select next unprocessed items ordered by priority DESC, id ASC."""
        if limit <= 0:
            return []
        with database.get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, puuid, region, game_name, tag_line, priority, is_processed
                FROM crawl_queue
                WHERE is_processed = 0
                ORDER BY priority DESC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            CrawlItem(
                id=row["id"],
                puuid=row["puuid"],
                region=row["region"],
                game_name=row["game_name"],
                tag_line=row["tag_line"],
                priority=row["priority"],
                is_processed=row["is_processed"],
            )
            for row in rows
        ]

    def mark_processed(self, puuid: str, region: Optional[str] = None) -> None:
        """Set is_processed = 1, processed_at = CURRENT_TIMESTAMP WHERE puuid = ?."""
        if not puuid:
            return
        with database.get_connection(self.db_path) as conn:
            if region is not None:
                conn.execute(
                    """
                    UPDATE crawl_queue
                    SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                    WHERE puuid = ? AND region = ?
                    """,
                    (puuid, region),
                )
            else:
                conn.execute(
                    """
                    UPDATE crawl_queue
                    SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                    WHERE puuid = ?
                    """,
                    (puuid,),
                )

    def mark_processed_by_id(self, queue_id: int) -> None:
        """Set is_processed = 1, processed_at = CURRENT_TIMESTAMP WHERE id = ?."""
        with database.get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE crawl_queue
                SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (queue_id,),
            )

    def mark_pending_processed(self, puuids: list[str]) -> int:
        """Mark pending entries as crawled; entries crawled earlier keep their processed_at."""
        cleaned = [p for p in puuids if p]
        if not cleaned:
            return 0
        with database.get_connection(self.db_path) as conn:
            cursor = conn.executemany(
                """
                UPDATE crawl_queue
                SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                WHERE puuid = ? AND is_processed = 0
                """,
                [(p,) for p in cleaned],
            )
            return max(0, cursor.rowcount)

    def seed_entry_ms(self, puuids: Iterable[Optional[str]]) -> dict[str, int]:
        """Entry time (UTC epoch ms) of the ladder seeds among *puuids*.

        Seeds are entries with priority >= LADDER_SEED_MIN_PRIORITY, the same
        definition as ``core.collection.provenance``.
        """
        wanted = sorted({p for p in puuids if p})
        entries: dict[str, int] = {}
        if not wanted:
            return entries
        with database.get_connection(self.db_path) as conn:
            for start in range(0, len(wanted), _LOOKUP_CHUNK):
                chunk = wanted[start:start + _LOOKUP_CHUNK]
                rows = conn.execute(
                    f"""
                    SELECT puuid, created_at FROM crawl_queue
                    WHERE priority >= ? AND puuid IN ({",".join("?" for _ in chunk)})
                    """,
                    (LADDER_SEED_MIN_PRIORITY, *chunk),
                ).fetchall()
                for row in rows:
                    entry = parse_sqlite_timestamp_ms(row["created_at"])
                    if entry is not None:
                        entries[row["puuid"]] = entry
        return entries

    def get_stats(self) -> dict[str, int]:
        """Return queue metrics: total, processed, pending."""
        with database.get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN is_processed = 1 THEN 1 ELSE 0 END) AS processed
                FROM crawl_queue
                """
            ).fetchone()
            total = int(row["total"]) if row and row["total"] is not None else 0
            processed = int(row["processed"]) if row and row["processed"] is not None else 0
            return {
                "total": total,
                "processed": processed,
                "pending": total - processed,
            }
