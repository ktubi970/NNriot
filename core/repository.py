"""
repository.py - Data Access Object (DAO) layer for NNriot

This module implements the Repository design pattern to abstract database 
operations away from the business logic. It provides a clean interface for 
data access, allowing the underlying storage mechanism (currently SQLite) 
to be changed in the future (e.g. to PostgreSQL) without affecting the 
application logic.
"""

from typing import List, Dict, Any, Optional
from core import database

class BaseRepository:
    """Base repository providing common database connection utilities."""
    def __init__(self, db_path: str = None):
        self.db_path = db_path or database.DB_PATH
        
    def get_connection(self):
        return database.get_connection(self.db_path)

class MatchRepository(BaseRepository):
    """Handles all database operations related to Matches."""
    
    def save_match(self, match_id: str, raw_data: dict) -> None:
        """Delegates to legacy database.py for now, can be overridden."""
        return database.save_match(match_id, raw_data, db_path=self.db_path)
        
    def save_matches_batch(self, matches: list) -> None:
        return database.save_matches_batch(matches, db_path=self.db_path)
        
    def save_match_timelines_batch(self, timelines: list) -> None:
        return database.save_match_timelines_batch(timelines, db_path=self.db_path)
        
    def match_exists(self, match_id: str) -> bool:
        return database.match_exists(match_id, db_path=self.db_path)
        
    def get_raw_match(self, match_id: str) -> Optional[dict]:
        return database.get_raw_match(match_id, db_path=self.db_path)
        
    def get_match_count(self) -> int:
        return database.get_match_count(db_path=self.db_path)

class PlayerRepository(BaseRepository):
    """Handles all database operations related to Players."""
    
    def save_player(self, puuid: str, name: str, tag: str) -> None:
        return database.save_player(puuid, name, tag, db_path=self.db_path)
        
    def increment_player_matches(self, puuid: str, count: int) -> None:
        return database.increment_player_matches(puuid, count, db_path=self.db_path)
        
    def search_players(self, query: str, limit: int = 10) -> List[Dict]:
        return database.search_players(query, limit=limit, db_path=self.db_path)

    def get_all_aliases(self) -> List[Dict]:
        return database.get_all_aliases(db_path=self.db_path)
        
    def link_accounts(self, main_puuid: str, smurf_puuid: str) -> None:
        return database.link_accounts(main_puuid, smurf_puuid, db_path=self.db_path)
        
    def unlink_accounts(self, alias_puuid: str) -> None:
        return database.unlink_accounts(alias_puuid, db_path=self.db_path)
        
    def get_top_players(self, limit: int = 10) -> List[Dict]:
        return database.get_top_players(limit=limit, db_path=self.db_path)

class SystemRepository(BaseRepository):
    """Handles global database status and health."""
    
    def get_db_stats(self) -> Dict:
        return database.get_db_stats(db_path=self.db_path)

class TrainingRepository(BaseRepository):
    """Handles all database operations related to ML Training Data."""
    
    def save_training_records_batch(self, records: list) -> None:
        return database.save_training_records_batch(records, db_path=self.db_path)
        
    def get_untrained_records(self, limit: int = 500) -> List[Dict]:
        return database.get_untrained_records(limit=limit, db_path=self.db_path)
        
    def get_random_trained_records(self, limit: int = 500) -> List[Dict]:
        return database.get_random_trained_records(limit=limit, db_path=self.db_path)
        
    def mark_as_trained(self, record_ids: List[int]) -> None:
        return database.mark_as_trained(record_ids, db_path=self.db_path)
        
    def get_training_count(self) -> int:
        return database.get_training_count(db_path=self.db_path)
        
class CrawlQueueRepository(BaseRepository):
    """Handles operations for the data collection crawl queue."""
    
    def add_to_crawl_queue(self, game_name: str, tag_line: str, priority: int = 0) -> None:
        return database.add_to_crawl_queue(game_name, tag_line, priority=priority, db_path=self.db_path)
        
    def get_next_from_crawl_queue(self, limit: int = 1) -> List[Dict]:
        return database.get_next_from_crawl_queue(limit=limit, db_path=self.db_path)
        
    def mark_as_processed(self, game_name: str, tag_line: str) -> None:
        return database.mark_as_processed(game_name, tag_line, db_path=self.db_path)
