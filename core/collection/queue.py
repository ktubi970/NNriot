"""CrawlQueueRepository for managing persistent player crawl queues."""

import logging
import time
from typing import Optional

from core import config, database
from core.collection.models import CrawlItem

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


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
        """Set is_processed = 1, processed_at = now (UTC ms) WHERE puuid = ?."""
        if not puuid:
            return
        with database.get_connection(self.db_path) as conn:
            if region is not None:
                conn.execute(
                    """
                    UPDATE crawl_queue
                    SET is_processed = 1, processed_at = ?
                    WHERE puuid = ? AND region = ?
                    """,
                    (_now_ms(), puuid, region),
                )
            else:
                conn.execute(
                    """
                    UPDATE crawl_queue
                    SET is_processed = 1, processed_at = ?
                    WHERE puuid = ?
                    """,
                    (_now_ms(), puuid),
                )

    def mark_processed_by_id(self, queue_id: int) -> None:
        """Set is_processed = 1, processed_at = now (UTC ms) WHERE id = ?."""
        with database.get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE crawl_queue
                SET is_processed = 1, processed_at = ?
                WHERE id = ?
                """,
                (_now_ms(), queue_id),
            )

    def unqueued(self, puuids: list[Optional[str]]) -> set[str]:
        """The PUUIDs among *puuids* that have no crawl_queue entry yet."""
        wanted = sorted({p for p in puuids if p})
        known: set[str] = set()
        with database.get_connection(self.db_path) as conn:
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                known.update(r[0] for r in conn.execute(
                    f"SELECT puuid FROM crawl_queue WHERE puuid IN ({','.join('?' for _ in chunk)})", chunk))
        return set(wanted) - known

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
