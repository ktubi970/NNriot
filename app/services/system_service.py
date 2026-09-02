"""
system_service.py - Service layer for System and DB health logic.
"""
import os
import logging
from core.repository import SystemRepository

logger = logging.getLogger(__name__)

class SystemService:
    def __init__(self):
        self.repo = SystemRepository()
        
    def get_db_health(self):
        stats = self.repo.get_db_stats()
        return {
            "players": stats.get("players", 0),
            "matches": stats.get("matches", 0),
            "training_records": stats.get("training_records", 0),
            "trained_records": stats.get("trained_records", 0),
            "untrained_records": stats.get("untrained_records", 0),
            "trained_ratio": stats.get("trained_ratio", 0),
            "success": True,
            "ok": True,
        }, 200
        
    def get_explorer_stats(self):
        stats = self.repo.get_db_stats()
        # Use repo.db_path or fallback to database.DB_PATH just in case
        db_path = getattr(self.repo, 'db_path', None)
        if not db_path:
            import database
            db_path = database.DB_PATH
            
        if db_path and os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            stats["db_size_mb"] = round(size_mb, 2)
        else:
            stats["db_size_mb"] = 0.0
        return stats, 200
