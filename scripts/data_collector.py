"""Backward-compatibility adapter for data collection routines.

Delegates core collection and ingestion operations to `core.collection.CollectionService`
and `core.collection.ingestor.MatchIngestor`.
"""

import logging
import os
import sys
import threading
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import config, database, riot_api
from core.collection import CollectionService, resolve_region
from core.collection.ingestor import MatchIngestor
from core.collection.models import MatchQualityFilter
from core.collection.regions import TAG_TO_REGION
from core.repository import MatchRepository, PlayerRepository, TrainingRepository

logger = logging.getLogger(__name__)

# Preserved constants for backward compatibility
FETCH_TIMELINES = os.environ.get("NNRIOT_FETCH_TIMELINES", "1") == "1"
_TAG_TO_REGION = TAG_TO_REGION
EXCLUDED_GAME_MODES = frozenset({"CHERRY", "STRAWBERRY"})

DEFAULT_REGIONS = ["KR", "NA", "EUW", "EUN", "BR", "LA1", "LA2", "OCE", "JP", "RU", "TR"]
DEFAULT_PLAYERS_PER_REGION = 500
DEFAULT_MATCHES_PER_PLAYER = 20
DEFAULT_LOOKBACK_DAYS = 60

# Filter for legacy _build_training_record compatibility
_legacy_quality_filter = MatchQualityFilter(
    allowed_game_modes=frozenset({"CLASSIC"}),
    allowed_queue_ids=frozenset(),
    min_duration_seconds=0,
)


def _build_training_record(
    match_id: str, details: dict, region: Optional[str] = None
) -> Optional[tuple]:
    """Parse raw Riot match JSON into training tuple via MatchIngestor."""
    ingestor = MatchIngestor(quality_filter=_legacy_quality_filter)
    return ingestor.build_training_record(match_id, details, region=region)


def _repositories(db_path: Optional[str]):
    """Bind one collection invocation to a resolved database, including initialization."""
    path = config.get_db_path(db_path, require_exists=True)
    database.init_db(db_path=path)
    return MatchRepository(path), PlayerRepository(path), TrainingRepository(path)


def _make_service(db_path: Optional[str]) -> CollectionService:
    path = config.get_db_path(db_path, require_exists=True)
    ingestor = MatchIngestor(db_path=path, quality_filter=_legacy_quality_filter)
    return CollectionService(
        db_path=path,
        ingestor=ingestor,
        api_factory=lambda r: riot_api.RiotAPI(r),
    )


def collect_training_data(
    seed_players: list[tuple[str, str]],
    matches_per_player: int = 5,
    days_back: Optional[int] = None,
    *,
    db_path: Optional[str] = None,
):
    """Collect match data for training from seed players."""
    service = _make_service(db_path)
    return service.collect_seed_players(
        seed_players, matches_per_player=matches_per_player, days_back=days_back
    )


def collect_top_leagues_data(
    regions: list[str] = ["KR"],
    total_players_per_region: int = 100,
    matches_per_player: int = 10,
    days_back: int = 30,
    *,
    db_path: Optional[str] = None,
):
    """Fetch official top league players and collect their matches."""
    service = _make_service(db_path)
    return service.collect_top_leagues(
        regions=regions,
        total_players_per_region=total_players_per_region,
        matches_per_player=matches_per_player,
        days_back=days_back,
    )


def collect_by_puuid(
    accounts: list[dict],
    matches_per_player: int = 20,
    *,
    db_path: Optional[str] = None,
) -> dict:
    """Collect match history for accounts where PUUID is already known."""
    service = _make_service(db_path)
    return service.collect_by_puuid_accounts(
        accounts, matches_per_player=matches_per_player
    )


batch_status: dict[str, Any] = {
    "status": "idle",
    "total": 0,
    "current": 0,
    "player": "",
    "logs": [],
}
batch_lock = threading.Lock()


def add_batch_log(msg: str) -> None:
    with batch_lock:
        batch_status["logs"].append(msg)
        print(f"[BATCH] {msg}")


def collect_batch_with_smurfs(
    player_list: list[tuple[str, str]],
    sources: list[str],
    count: int = 50,
    *,
    db_path: Optional[str] = None,
):
    """Batch collection for player names with optional smurf lookup."""
    with batch_lock:
        batch_status["status"] = "running"
        batch_status["total"] = len(player_list)
        batch_status["current"] = 0
        batch_status["logs"] = []

    service = _make_service(db_path)
    res = service.collect_batch_with_smurfs(player_list, sources=sources, count=count)

    with batch_lock:
        batch_status["status"] = "completed"
        batch_status["current"] = len(player_list)
    add_batch_log("Batch collection finished.")
    return res


if __name__ == "__main__":
    regions = ["KR", "NA", "EUW", "EUN"]
    collect_top_leagues_data(
        regions=regions,
        total_players_per_region=1500,
        matches_per_player=200,
        days_back=60,
    )
