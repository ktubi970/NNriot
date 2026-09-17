"""
system_service.py - Service layer for System and DB health logic.
"""
import os
import logging
from core import database
from core.repository import SystemRepository

logger = logging.getLogger(__name__)


class SystemService:
    def __init__(self, repo=None):
        self.repo = repo or SystemRepository()

    def get_full_telemetry(self, db_path: str = None) -> dict:
        target_path = db_path or getattr(self.repo, 'db_path', None) or database.DB_PATH

        # If repo is provided and db_path was not explicitly passed, prefer repo.get_db_stats()
        stats = None
        if not db_path and self.repo and hasattr(self.repo, "get_db_stats"):
            try:
                stats = self.repo.get_db_stats()
            except Exception:
                stats = None
        if stats is None:
            stats = database.get_db_stats(target_path)

        try:
            crawl_stats = database.get_crawl_queue_stats(target_path)
        except Exception:
            crawl_stats = {}

        pending = 0
        if isinstance(crawl_stats, dict):
            pending = crawl_stats.get("pending", 0)

        size_mb = 0.0
        if target_path and os.path.exists(target_path):
            size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)

        return {
            "players": stats.get("players", 0) if isinstance(stats, dict) else 0,
            "matches": stats.get("matches", 0) if isinstance(stats, dict) else 0,
            "training_records": stats.get("training_records", 0) if isinstance(stats, dict) else 0,
            "trained_records": stats.get("trained_records", 0) if isinstance(stats, dict) else 0,
            "untrained_records": stats.get("untrained_records", 0) if isinstance(stats, dict) else 0,
            "trained_ratio": stats.get("trained_ratio", 0.0) if isinstance(stats, dict) else 0.0,
            "crawl_queue_pending": pending,
            "db_size_mb": size_mb,
            "schema_version": stats.get("schema_version", 0) if isinstance(stats, dict) else 0,
            "success": True,
            "ok": True,
        }

    def get_region_distribution(self, db_path: str = None) -> dict:
        target_path = db_path or getattr(self.repo, 'db_path', None) or database.DB_PATH
        try:
            with database.get_connection(target_path) as conn:
                regions = conn.execute(
                    """
                    SELECT
                        SUBSTR(match_id, 1, INSTR(match_id, '_') - 1) as region,
                        COUNT(*) as count
                    FROM matches
                    WHERE match_id LIKE '%_%'
                    GROUP BY region
                    ORDER BY count DESC
                    """
                ).fetchall()
                return {r[0]: r[1] for r in regions if r[0]}
        except Exception as e:
            logger.warning(f"Could not compute region distribution: {e}")
            return {}

    def get_db_health(self, db_path: str = None):
        return self.get_full_telemetry(db_path=db_path)

    def get_explorer_stats(self):
        stats = self.repo.get_db_stats()
        # Use repo.db_path or fallback to database.DB_PATH just in case
        db_path = getattr(self.repo, 'db_path', None)
        if not db_path:
            db_path = database.DB_PATH

        if db_path and os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            stats["db_size_mb"] = round(size_mb, 2)
        else:
            stats["db_size_mb"] = 0.0
        return stats
