"""
player_service.py - Service layer for Player related business logic.
"""
import random
import logging
from core.repository import PlayerRepository

logger = logging.getLogger(__name__)

class PlayerService:
    def __init__(self):
        self.repo = PlayerRepository()
        
    def get_all_aliases(self):
        return self.repo.get_all_aliases(), 200
        
    def unlink_accounts(self, alias_puuid: str):
        if not alias_puuid:
            return {"error": "alias_puuid is required"}, 400
        self.repo.unlink_accounts(alias_puuid)
        return {"success": True}, 200
        
    def search_players(self, query: str, limit: int = 10):
        if not query or len(query) < 2:
            return [], 200
        players = self.repo.search_players(query, limit=limit)
        return players, 200
        
    def get_random_players(self, limit: int = 10):
        pool_size = max(limit * 20, 200)
        pool = self.repo.get_top_players(limit=pool_size)
        sample = random.sample(pool, min(limit, len(pool)))
        return sample, 200
        
    def get_top_players(self, limit: int = 10):
        players = self.repo.get_top_players(limit=limit)
        return players, 200
        
    def link_accounts(self, canonical_puuid: str, alias_puuid: str):
        if not canonical_puuid or not alias_puuid:
            return {"error": "canonical_puuid and alias_puuid are required"}, 400
        self.repo.link_accounts(canonical_puuid, alias_puuid)
        return {"success": True, "message": "Accounts completely linked."}, 200
