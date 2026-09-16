#!/usr/bin/env python3
"""
crawler_1m.py - High-throughput resilient crawler for collecting recent League of Legends matches.

Features:
- Respects Riot Personal Dev Key limits (100 req / 120s, 20 req / 1s) with sliding-window protection.
- Automatically pauses and auto-resumes when Riot Dev Key expires (HTTP 401/403) and is updated in .env.
- Filters strictly for Ranked Solo/Duo (queue=420) on Summoner's Rift (CLASSIC, duration >= 5m).
- Extracts clean pre-match features (zero post-game leakage) and multi-output labels.
- Persists frontier queue to disk (crawler_frontier.json) for seamless resume across sessions.
- Keeps SQLite WAL compact via periodic checkpoints to protect disk space.
"""

import argparse
import collections
import json
import logging
import os
import signal
import sys
import time

# Ensure repo root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database
from core import json_utils
from core.feature_labels import extract_labels
from core.repository import MatchRepository, TrainingRepository
from core.riot_api import RiotAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crawler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("crawler_1m")

FRONTIER_FILE = os.path.join(REPO_ROOT, "crawler_frontier.json")
_match_repo = MatchRepository()
_train_repo = TrainingRepository()


class Crawler1M:
    def __init__(
        self,
        target_new_matches: int = 1000,
        regions: list[str] = None,
        days_back: int = 14,
        batch_size: int = 25,
    ):
        self.target_new_matches = target_new_matches
        self.regions = regions or ["EUW", "KR", "NA"]
        self.days_back = days_back
        self.batch_size = batch_size

        self.start_time_filter = int(time.time()) - (days_back * 86400)
        self.apis = {r: RiotAPI(r) for r in self.regions}

        self.frontier = collections.deque()  # elements: (puuid, region)
        self.visited_puuids = set()
        self.seen_matches = set()

        self.matches_collected = 0
        self.start_run_time = time.time()
        self.is_running = True

        # Register graceful exit
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\n[INFO] Stop signal received. Finishing pending batch and saving state...")
        self.is_running = False

    def load_state(self):
        """Load frontier from disk if available."""
        if os.path.exists(FRONTIER_FILE):
            try:
                with open(FRONTIER_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                frontier_items = data.get("frontier", [])
                visited = data.get("visited_puuids", [])
                for item in frontier_items:
                    self.frontier.append((item["puuid"], item["region"]))
                self.visited_puuids.update(visited)
                logger.info(
                    "Restored frontier from disk: %d players in queue, %d visited.",
                    len(self.frontier),
                    len(self.visited_puuids),
                )
            except Exception as e:
                logger.warning("Failed to load frontier state: %s. Starting fresh.", e)

    def save_state(self):
        """Save frontier and visited players to disk."""
        try:
            # Keep at most 20,000 frontier items on disk to save space
            items = [{"puuid": p, "region": r} for p, r in list(self.frontier)[:20000]]
            data = {
                "frontier": items,
                "visited_puuids": list(self.visited_puuids)[-50000:],
                "timestamp": time.time(),
            }
            temp_file = FRONTIER_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            if os.path.exists(FRONTIER_FILE):
                os.remove(FRONTIER_FILE)
            os.rename(temp_file, FRONTIER_FILE)
            logger.debug("Frontier state saved (%d items).", len(items))
        except Exception as e:
            logger.error("Failed to save frontier state: %s", e)

    def seed_top_players(self):
        """Fetch Challenger and GrandMaster players as seed roots."""
        logger.info("Seeding top players from regions: %s...", self.regions)
        for region in self.regions:
            api = self.apis[region]
            try:
                # 1. Challenger
                logger.info("[%s] Fetching Challenger league entries...", region)
                chal = api.get_top_league("challenger", queue="RANKED_SOLO_5x5")
                for entry in chal:
                    puuid = entry.get("puuid")
                    if puuid and puuid not in self.visited_puuids:
                        self.frontier.append((puuid, region))

                # 2. Grandmaster if needed
                if len(self.frontier) < 300:
                    logger.info("[%s] Fetching Grandmaster league entries...", region)
                    gm = api.get_top_league("grandmaster", queue="RANKED_SOLO_5x5")
                    for entry in gm:
                        puuid = entry.get("puuid")
                        if puuid and puuid not in self.visited_puuids:
                            self.frontier.append((puuid, region))
            except Exception as e:
                logger.warning("[%s] Error seeding league: %s", region, e)

        logger.info("Seeding complete. Initial frontier size: %d players.", len(self.frontier))

    def _checkpoint_wal(self):
        """Truncate SQLite WAL file to ensure it stays small on disk."""
        try:
            with database.get_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("WAL checkpoint truncated successfully.")
        except Exception as e:
            logger.debug("WAL checkpoint warning: %s", e)

    def run(self):
        database.migrate_db()
        self.load_state()

        if not self.frontier:
            self.seed_top_players()

        if not self.frontier:
            logger.error("Frontier is empty and seeding failed. Exiting.")
            return

        matches_batch = []
        training_batch = []
        last_progress_time = time.time()
        batch_counter = 0

        logger.info(
            "=== Starting crawl towards target: +%d new matches ===", self.target_new_matches
        )
        logger.info("Filtering: Queue=420 (Ranked Solo/Duo), Lookback=%d days", self.days_back)

        try:
            while self.is_running and self.matches_collected < self.target_new_matches:
                if not self.frontier:
                    logger.info("Frontier depleted. Re-seeding...")
                    self.seed_top_players()
                    if not self.frontier:
                        logger.error("Unable to seed new players. Exiting.")
                        break

                puuid, region = self.frontier.popleft()
                if puuid in self.visited_puuids:
                    continue
                self.visited_puuids.add(puuid)

                api = self.apis.get(region) or self.apis["EUW"]

                # Fetch recent Ranked Solo/Duo match IDs
                try:
                    match_ids = api.get_match_ids(
                        puuid,
                        count=20,
                        start_time=self.start_time_filter,
                        queue=420,
                        type="ranked",
                    )
                except Exception as e:
                    logger.warning("Error fetching match IDs for %s: %s", puuid[:10], e)
                    continue

                if not match_ids:
                    continue

                # Filter out existing matches
                new_mids = []
                for mid in match_ids:
                    if mid not in self.seen_matches and not _match_repo.match_exists(mid):
                        new_mids.append(mid)
                        self.seen_matches.add(mid)

                if not new_mids:
                    continue

                # Fetch match details for each new match
                for mid in new_mids:
                    if not self.is_running or self.matches_collected >= self.target_new_matches:
                        break

                    details = api.get_match_details(mid)
                    if not details or not isinstance(details, dict):
                        continue

                    info = details.get("info", {})
                    # Quality gates:
                    # 1. Mode CLASSIC
                    if (info.get("gameMode") or "").upper() != "CLASSIC":
                        continue
                    # 2. Queue 420
                    if info.get("queueId") != 420:
                        continue
                    # 3. Minimum duration (avoid remakes)
                    if info.get("gameDuration", 0) < 300:
                        continue

                    # 4. Winner validation
                    teams = info.get("teams", [])
                    winner = 0
                    team_won = False
                    for t in teams:
                        if t.get("win") and t.get("teamId") == 200:
                            winner = 1
                            team_won = True
                            break
                        elif t.get("win") and t.get("teamId") == 100:
                            winner = 0
                            team_won = True
                            break

                    if not team_won:
                        continue

                    # Extract pre-match features (ZERO post-match leakage)
                    try:
                        pre_match_feat = json_utils.extract_pre_match_features(
                            details, region=region
                        )
                    except Exception as e:
                        logger.debug("Failed pre-match feature extraction for %s: %s", mid, e)
                        continue

                    # Multi-output labels
                    labels = extract_labels(details)
                    if labels is None:
                        continue

                    # Valid match!
                    matches_batch.append((mid, details))
                    training_batch.append((mid, pre_match_feat, winner, labels))
                    self.matches_collected += 1

                    # Snowballing: add the other participants to frontier
                    participants = info.get("participants", [])
                    for p in participants:
                        p_puuid = p.get("puuid")
                        if p_puuid and p_puuid not in self.visited_puuids:
                            self.frontier.append((p_puuid, region))

                    # Flush batch when threshold reached
                    if (
                        len(matches_batch) >= self.batch_size
                        or self.matches_collected >= self.target_new_matches
                    ):
                        _match_repo.save_matches_batch(matches_batch)
                        _train_repo.save_training_records_batch(training_batch)
                        batch_counter += 1

                        matches_batch.clear()
                        training_batch.clear()

                        elapsed = time.time() - self.start_run_time
                        rate = (self.matches_collected / max(1.0, elapsed)) * 60.0
                        remaining = self.target_new_matches - self.matches_collected
                        eta_min = (remaining / max(0.1, rate)) if rate > 0 else 0

                        logger.info(
                            "[PROGRESS] %d/%d new matches collected (%.1f%%) | "
                            "Speed: %.1f matches/min | ETA: %.1f min | Frontier: %d players",
                            self.matches_collected,
                            self.target_new_matches,
                            (self.matches_collected / self.target_new_matches) * 100,
                            rate,
                            eta_min,
                            len(self.frontier),
                        )

                        # Periodically truncate WAL every 10 batches (250 matches)
                        if batch_counter % 10 == 0:
                            self._checkpoint_wal()
                            self.save_state()

        finally:
            # Commit any remaining records in buffer
            if matches_batch:
                _match_repo.save_matches_batch(matches_batch)
                _train_repo.save_training_records_batch(training_batch)
                logger.info("Committed final batch of %d matches.", len(matches_batch))

            self._checkpoint_wal()
            self.save_state()

            elapsed = time.time() - self.start_run_time
            logger.info(
                "=== Crawl Run Concluded: +%d matches collected in %.1f minutes ===",
                self.matches_collected,
                elapsed / 60.0,
            )


def print_db_status():
    stats = database.get_db_stats()
    db_size_mb = 0.0
    if os.path.exists(database.DB_PATH):
        db_size_mb = os.path.getsize(database.DB_PATH) / (1024 * 1024)

    print("\n--- NNriot Database Status ---")
    print(f"Total Matches:         {stats.get('matches', 0):,}")
    print(f"Training Dataset:      {stats.get('training_records', 0):,}")
    print(f"Total Players:         {stats.get('players', 0):,}")
    print(f"Champions Registry:    {stats.get('champions', 0):,}")
    print(f"Database File:         {os.path.abspath(database.DB_PATH)}")
    print(f"Database Size:         {db_size_mb:.2f} MB")
    print("------------------------------\n")


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
    args = parser.parse_args()

    if args.status:
        print_db_status()
        return

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    crawler = Crawler1M(
        target_new_matches=args.limit,
        regions=regions,
        days_back=args.days_back,
        batch_size=args.batch_size,
    )
    crawler.run()


if __name__ == "__main__":
    main()
