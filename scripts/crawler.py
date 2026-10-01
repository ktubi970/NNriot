#!/usr/bin/env python3
"""Unified high-throughput crawler and match collection CLI."""
import argparse
import os
import signal
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database
from core.collection import CollectionService, resolve_region


def print_status(db_path: str | None = None):
    stats = database.get_db_stats(db_path=db_path) if db_path else database.get_db_stats()
    svc = CollectionService(db_path=db_path)
    q_stats = svc.queue_repo.get_stats()

    print("\n--- NNriot Unified Collection & Database Status ---")
    print(f"Total Matches:         {stats.get('matches', 0):,}")
    print(f"Training Dataset:      {stats.get('training_records', 0):,}")
    print(f"Total Players:         {stats.get('players', 0):,}")
    print(f"Queue Total:           {q_stats['total']:,}")
    print(f"Queue Pending:         {q_stats['pending']:,}")
    print(f"Queue Processed:       {q_stats['processed']:,}")
    print("----------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="NNriot Unified Collector & Crawler")
    parser.add_argument("--target", type=int, default=1000, help="Target new matches to collect")
    parser.add_argument("--regions", type=str, default="EUW,KR,NA", help="Comma-separated regions")
    parser.add_argument("--days-back", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--batch-size", type=int, default=25, help="Matches per player fetch")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated Name#Tag seeds")
    parser.add_argument("--top", action="store_true", help="Crawl Challenger & GM top leagues")
    parser.add_argument("--status", action="store_true", help="Display collection status and exit")
    parser.add_argument("--db", type=str, default=None, help="Target SQLite database path")
    args = parser.parse_args()

    if args.status:
        print_status(args.db)
        return

    service = CollectionService(db_path=args.db)
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]

    if args.seeds:
        seed_list = []
        for s in args.seeds.split(","):
            if "#" in s:
                name, tag = s.split("#", 1)
                seed_list.append((name.strip(), tag.strip()))
        print(f"Collecting {len(seed_list)} seed players...")
        service.collect_seed_players(seed_list, matches_per_player=args.batch_size, days_back=args.days_back)
        return

    if args.top:
        print(f"Collecting top league players in {regions}...")
        service.collect_top_leagues(regions=regions, total_players_per_region=100, matches_per_player=args.batch_size)
        return

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        print("\nStopping crawler gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting continuous crawler towards +{args.target} matches in {regions}...")
    service.run_crawler(
        target_matches=args.target,
        regions=regions,
        days_back=args.days_back,
        batch_size=args.batch_size,
        stop_event=stop_event,
    )


if __name__ == "__main__":
    main()
