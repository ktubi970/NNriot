"""
player_service.py - Service layer for Player related business logic.
"""
import random
import logging
from core.repository import PlayerRepository

logger = logging.getLogger(__name__)

class PlayerService:
    def __init__(self, repo=None):
        self.repo = repo or PlayerRepository()
        
    def get_all_aliases(self):
        return self.repo.get_all_aliases()
        
    def unlink_accounts(self, alias_puuid: str):
        if not alias_puuid:
            raise ValueError("alias_puuid is required")
        self.repo.unlink_accounts(alias_puuid)
        return {"success": True}
        
    def search_players(self, query: str, limit: int = 10):
        if not query or len(query) < 2:
            return []
        players = self.repo.search_players(query, limit=limit)
        return players
        
    def get_random_players(self, limit: int = 10):
        pool_size = max(limit * 20, 200)
        pool = self.repo.get_top_players(limit=pool_size)
        sample = random.sample(pool, min(limit, len(pool)))
        return sample
        
    def get_top_players(self, limit: int = 10):
        players = self.repo.get_top_players(limit=limit)
        return players
        
    def link_accounts(self, canonical_puuid: str, alias_puuid: str):
        if not canonical_puuid or not alias_puuid:
            raise ValueError("canonical_puuid and alias_puuid are required")
        self.repo.link_accounts(canonical_puuid, alias_puuid)
        return {"success": True, "message": "Accounts completely linked."}

