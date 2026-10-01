"""Background collector service for continuous match discovery."""
import logging
import os
import signal
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database
from core.collection import CollectionService
from scripts.data_collector import (
    resolve_region,
    EXCLUDED_GAME_MODES,
    FETCH_TIMELINES,
    _build_training_record,
)

logger = logging.getLogger(__name__)

# Limit in Bytes — defaults to 50 GB, overridable via NNRIOT_MAX_DB_SIZE_GB.
MAX_DB_SIZE = int(os.environ.get("NNRIOT_MAX_DB_SIZE_GB", "50")) * 1024 * 1024 * 1024


def get_db_size():
    if os.path.exists(database.DB_PATH):
        return os.path.getsize(database.DB_PATH)
    return 0


def run_collector():
    """Run continuous collection using CollectionService."""
    database.init_db()
    service = CollectionService()

    print(f"Background collector started. Target DB: {database.DB_PATH}")
    print(f"Size limit: {MAX_DB_SIZE / (1024**3):.1f} GB")

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        print("\nStopping background collector gracefully...")
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except (ValueError, AttributeError):
        pass

    # Seeds to ensure initial queue coverage
    seeds = [
        ("Hide on bush", "KR1"),
        ("Chovy", "KR1"),
        ("T1 Gumayusi", "KR1"),
        ("ShowMaker", "KR1"),
        ("Canyon", "KR1"),
        ("T1 Faker", "KR1"),
        ("T1 Keira", "KR1"),
        ("T1 Zeus", "KR1"),
        ("T1 Oner", "KR1"),
    ]
    service.collect_seed_players(seeds, matches_per_player=10)

    service.run_crawler(
        target_matches=100000,
        regions=["KR", "EUW", "NA"],
        batch_size=20,
        stop_event=stop_event,
    )


if __name__ == "__main__":
    run_collector()
