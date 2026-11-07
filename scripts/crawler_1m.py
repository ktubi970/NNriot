#!/usr/bin/env python3
"""High-throughput crawler for collecting recent League of Legends matches.

Delegates core crawl loop to `core.collection.CollectionService.run_crawler`.
"""

import argparse
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
from core.repository import MatchRepository, TrainingRepository

logger = logging.getLogger("crawler_1m")


def print_db_status(db_path: str | None = None):
    svc = CollectionService(db_path=db_path)
    stats = database.get_db_stats(db_path=db_path) if db_path else database.get_db_stats()
    q_stats = svc.queue_repo.get_stats()
    print("\n--- Crawler & Database Status ---")
    print(f"Total Matches:    {stats.get('matches', 0):,}")
    print(f"Training Records: {stats.get('training_records', 0):,}")
    print(f"Total Players:    {stats.get('players', 0):,}")
    print(f"Queue Total:      {q_stats['total']:,}")
    print(f"Queue Pending:    {q_stats['pending']:,}")
    print(f"Queue Processed:  {q_stats['processed']:,}")
    print("---------------------------------\n")


class Crawler1M:
    def __init__(
        self,
        target_new_matches: int = 1000,
        regions: list[str] = None,
        days_back: int = 14,
        batch_size: int = 25,
        db_path: str = None,
    ):
        self.target_new_matches = target_new_matches
        self.regions = regions or ["EUW", "KR", "NA"]
        self.days_back = days_back
        self.batch_size = batch_size
        self.db_path = db_path
        self.service = CollectionService(db_path=self.db_path)
        self.stop_event = threading.Event()

    def run(self):
        def handle_signal(sig, frame):
            print("\nStopping crawler gracefully...")
            self.stop_event.set()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except (ValueError, AttributeError):
            pass

        return self.service.run_crawler(
            target_matches=self.target_new_matches,
            regions=self.regions,
            days_back=self.days_back,
            batch_size=self.batch_size,
            stop_event=self.stop_event,
        )


def main():
    parser = argparse.ArgumentParser(
        description="High-throughput crawler for 1M recent LoL matches"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of new matches to collect in this run (default: 1000)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="EUW,KR,NA",
        help="Comma-separated list of regions to crawl (default: EUW,KR,NA)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=14,
        help="Lookback window in days for recent matches (default: 14)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of matches per database commit (default: 25)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current database stats and exit",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Target SQLite database path",
    )
    args = parser.parse_args()

    if args.status:
        print_db_status(args.db)
        return

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    crawler = Crawler1M(
        target_new_matches=args.limit,
        regions=regions,
        days_back=args.days_back,
        batch_size=args.batch_size,
        db_path=args.db,
    )
    crawler.run()


if __name__ == "__main__":
    main()
