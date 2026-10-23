"""Orchestration Facade for unified LoL match data collection.

Coordinates:
- CrawlQueueRepository for frontier management
- MatchIngestor for leak-free match and timeline ingestion
- CollectionTracker for thread-safe progress monitoring
- PlayerRepository, MatchRepository, TrainingRepository for persistent storage
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

from core import config, database, riot_api
from core.collection.ingestor import MatchIngestor
from core.collection.models import CrawlItem, IngestionResult
from core.collection.queue import CrawlQueueRepository
from core.collection.regions import resolve_region
from core.collection.tracker import CollectionTracker, get_default_tracker
from core.repository import MatchRepository, PlayerRepository, TrainingRepository

logger = logging.getLogger(__name__)

DEFAULT_REGIONS = ["KR", "NA", "EUW", "EUN", "BR", "LA1", "LA2", "OCE", "JP", "RU", "TR"]
DEFAULT_PLAYERS_PER_REGION = 500
DEFAULT_MATCHES_PER_PLAYER = 20
DEFAULT_LOOKBACK_DAYS = 60


class CollectionService:
    """Central orchestration facade for all match data collection operations."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        tracker: Optional[CollectionTracker] = None,
        match_repo: Optional[MatchRepository] = None,
        player_repo: Optional[PlayerRepository] = None,
        train_repo: Optional[TrainingRepository] = None,
        queue_repo: Optional[CrawlQueueRepository] = None,
        ingestor: Optional[MatchIngestor] = None,
        api_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.db_path = config.get_db_path(db_path, require_exists=False)
        if not Path(self.db_path).is_file():
            database.init_db(self.db_path)

        self.match_repo = match_repo or MatchRepository(self.db_path)
        self.player_repo = player_repo or PlayerRepository(self.db_path)
        self.train_repo = train_repo or TrainingRepository(self.db_path)
        self.queue_repo = queue_repo or CrawlQueueRepository(self.db_path)
        self.ingestor = ingestor or MatchIngestor(
            db_path=self.db_path,
            match_repo=self.match_repo,
            train_repo=self.train_repo,
        )
        self.tracker = tracker or get_default_tracker()
        self.api_factory = api_factory or (lambda r: riot_api.RiotAPI(r))

    def collect_player_matches(
        self,
        puuid: str,
        region: str,
        count: int = 20,
        start_time: Optional[int] = None,
        queue: int = 420,
    ) -> IngestionResult:
        """Fetch and ingest recent matches for a single player by PUUID."""
        canon_region = resolve_region(region)
        try:
            api = self.api_factory(canon_region)
            match_ids = api.get_match_ids(
                puuid, count=count, start_time=start_time, queue=queue, type="ranked"
            )
            if not match_ids:
                return IngestionResult()

            result = self.ingestor.ingest_match_ids(match_ids, api=api, region=canon_region)
            if result.records_saved > 0:
                self.player_repo.increment_player_matches(puuid, result.records_saved)
            return result
        except Exception as e:
            logger.error("Error collecting matches for %s (%s): %s", puuid[:12] if puuid else "?", canon_region, e)
            res = IngestionResult()
            res.errors = 1
            return res

    def collect_seed_players(
        self,
        seed_players: list[tuple[str, str]],
        matches_per_player: int = 10,
        days_back: Optional[int] = None,
    ) -> IngestionResult:
        """Collect match history for a list of (name, tag) seeds."""
        total_res = IngestionResult()
        start_time = int(time.time()) - (days_back * 86400) if days_back else None
        self.tracker.set_running(total=len(seed_players), player="Seeds Batch")

        for name, tag in seed_players:
            self.tracker.set_player(f"{name}#{tag}")
            region = resolve_region(tag)
            try:
                api = self.api_factory(region)
                puuid = api.get_puuid(name, tag)
                if not puuid:
                    self.tracker.add_log(f"PUUID resolution failed for {name}#{tag}")
                    total_res.skipped += 1
                    continue

                self.player_repo.save_player(puuid, name, tag)
                sub_res = self.collect_player_matches(
                    puuid, region=region, count=matches_per_player, start_time=start_time
                )
                total_res.matches_saved += sub_res.matches_saved
                total_res.records_saved += sub_res.records_saved
                total_res.timelines_saved += sub_res.timelines_saved
                total_res.embeddings_computed += sub_res.embeddings_computed
                total_res.skipped += sub_res.skipped
                total_res.errors += sub_res.errors
                total_res.discovered_puuids.extend(sub_res.discovered_puuids)

                self.tracker.add_log(f"Saved {sub_res.matches_saved} matches for {name}#{tag}")
            except Exception as e:
                logger.error("Error collecting seed player %s#%s: %s", name, tag, e)
                total_res.errors += 1
                self.tracker.add_log(f"[ERROR] {name}#{tag}: {e}")
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def collect_by_puuid_accounts(
        self,
        accounts: list[dict],
        matches_per_player: int = 20,
    ) -> dict[str, int]:
        """Collect match history for accounts where PUUID is known (e.g. lolpros)."""
        stats = {"processed": 0, "saved": 0, "skipped": 0, "errors": 0}
        self.tracker.set_running(total=len(accounts), player="PUUID Accounts")

        for acc in accounts:
            puuid = (acc.get("puuid") or "").strip()
            server = (acc.get("server") or "").strip()
            name = acc.get("gamename") or acc.get("summoner_name") or "?"
            tag = acc.get("tagline") or server or "?"

            if not puuid:
                stats["skipped"] += 1
                self.tracker.step(1)
                continue

            region = resolve_region(server)
            self.tracker.set_player(f"{name}#{tag}")
            try:
                self.player_repo.save_player(puuid, name, tag)
                res = self.collect_player_matches(
                    puuid, region=region, count=matches_per_player
                )
                if res.errors > 0 and res.matches_saved == 0:
                    stats["errors"] += 1
                else:
                    stats["processed"] += 1
                stats["saved"] += res.records_saved
                self.tracker.add_log(f"[{region}] Saved {res.matches_saved} matches for {name}#{tag}")
            except Exception as e:
                logger.error("Error for %s#%s: %s", name, tag, e)
                stats["errors"] += 1
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return stats

    def collect_top_leagues(
        self,
        regions: list[str] = ["KR"],
        total_players_per_region: int = 100,
        matches_per_player: int = 10,
        days_back: int = 30,
    ) -> IngestionResult:
        """Fetch top league players (Challenger/GM/Master) across regions and collect their matches."""
        total_res = IngestionResult()
        all_entries: list[dict[str, Any]] = []

        for region in regions:
            canon = resolve_region(region)
            api = self.api_factory(canon)
            try:
                entries = api.get_top_players(total_count=total_players_per_region)
                for entry in entries:
                    if isinstance(entry, dict):
                        entry["_region"] = canon
                        all_entries.append(entry)
            except Exception as e:
                logger.error("Error fetching top league entries for %s: %s", canon, e)

        self.tracker.set_running(total=len(all_entries), player="Top Leagues")
        start_time = int(time.time()) - (days_back * 86400)

        for entry in all_entries:
            puuid = entry.get("puuid")
            region = entry.get("_region", "EUW")
            if not puuid:
                total_res.skipped += 1
                self.tracker.step(1)
                continue

            try:
                api = self.api_factory(region)
                account_info = api.get_account_by_puuid(puuid)
                game_name = account_info.get("game_name", "Unknown") if account_info else "Unknown"
                tag_line = account_info.get("tag_line", region) if account_info else region
                self.player_repo.save_player(puuid, game_name, tag_line)
                self.tracker.set_player(f"{game_name}#{tag_line} ({region})")

                sub_res = self.collect_player_matches(
                    puuid, region=region, count=matches_per_player, start_time=start_time
                )
                total_res.matches_saved += sub_res.matches_saved
                total_res.records_saved += sub_res.records_saved
                total_res.timelines_saved += sub_res.timelines_saved
                total_res.embeddings_computed += sub_res.embeddings_computed
                total_res.skipped += sub_res.skipped
                total_res.errors += sub_res.errors
                total_res.discovered_puuids.extend(sub_res.discovered_puuids)
            except Exception as e:
                logger.error("Error collecting top player %s (%s): %s", puuid, region, e)
                total_res.errors += 1
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def collect_batch_with_smurfs(
        self,
        player_list: list[tuple[str, str]],
        sources: list[str],
        count: int = 50,
    ) -> IngestionResult:
        """Collect batch for players and their smurf accounts."""
        self.tracker.set_running(total=len(player_list), player="Batch Smurfs")
        total_res = IngestionResult()

        for name, tag in player_list:
            self.tracker.set_player(f"{name}#{tag}")
            region = resolve_region(tag)
            try:
                api = self.api_factory(region)
                main_puuid = api.get_puuid(name, tag)
                if not main_puuid:
                    self.tracker.add_log(f"Could not resolve PUUID for {name}#{tag}")
                    total_res.skipped += 1
                    continue

                self.player_repo.save_player(main_puuid, name, tag)
                accounts = [{"puuid": main_puuid, "server": region, "gamename": name, "tagline": tag}]

                for acc in accounts:
                    sub_res = self.collect_player_matches(
                        acc["puuid"], region=acc["server"], count=count
                    )
                    total_res.matches_saved += sub_res.matches_saved
                    total_res.records_saved += sub_res.records_saved
                    total_res.timelines_saved += sub_res.timelines_saved
                    total_res.embeddings_computed += sub_res.embeddings_computed
                    total_res.skipped += sub_res.skipped
                    total_res.errors += sub_res.errors
                    total_res.discovered_puuids.extend(sub_res.discovered_puuids)
            except Exception as e:
                logger.error("Error in batch with smurfs for %s#%s: %s", name, tag, e)
                total_res.errors += 1
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def run_crawler(
        self,
        target_matches: int = 1000,
        regions: list[str] = ["EUW", "KR", "NA"],
        days_back: int = 14,
        batch_size: int = 25,
        stop_event: Optional[threading.Event] = None,
    ) -> IngestionResult:
        """High-throughput crawler running continuously until target matches or stopped."""
        total_res = IngestionResult()
        self.tracker.set_running(total=target_matches, player="Continuous Crawler")
        start_time = int(time.time()) - (days_back * 86400)
        batch_counter = 0
        depletion_retries = 0
        ladder_depth = 50

        try:
            # Seed initial queue if empty
            if self.queue_repo.get_stats()["pending"] == 0:
                for region in regions:
                    canon = resolve_region(region)
                    api = self.api_factory(canon)
                    try:
                        entries = api.get_top_players(total_count=ladder_depth)
                        puuids = [e["puuid"] for e in entries if isinstance(e, dict) and e.get("puuid")]
                        self.queue_repo.enqueue_puuids(puuids, region=canon, priority=10)
                    except Exception as e:
                        logger.warning("Failed seeding top players for %s: %s", canon, e)

            while (
                total_res.matches_saved < target_matches
                and not (stop_event and stop_event.is_set())
            ):
                items = self.queue_repo.pop_next(limit=1)
                if not items:
                    depletion_retries += 1
                    if depletion_retries > 5:
                        logger.warning("Frontier queue depleted after multiple deepening attempts. Exiting crawler loop.")
                        break

                    ladder_depth += 50
                    logger.info("Frontier queue depleted. Deepening seeds to top %d players...", ladder_depth)
                    for region in regions:
                        canon = resolve_region(region)
                        api = self.api_factory(canon)
                        try:
                            entries = api.get_top_players(total_count=ladder_depth)
                            puuids = [e["puuid"] for e in entries if isinstance(e, dict) and e.get("puuid")]
                            self.queue_repo.enqueue_puuids(puuids, region=canon, priority=5)
                        except Exception:
                            pass

                    if stop_event and stop_event.wait(timeout=2.0):
                        break
                    elif not stop_event:
                        time.sleep(2)
                    continue

                depletion_retries = 0
                item = items[0]
                region = resolve_region(item.region)
                puuid = item.puuid

                if not puuid and item.game_name and item.tag_line:
                    api = self.api_factory(region)
                    try:
                        puuid = api.get_puuid(item.game_name, item.tag_line)
                    except Exception:
                        puuid = None

                if not puuid:
                    if item.id:
                        self.queue_repo.mark_processed_by_id(item.id)
                    continue

                self.tracker.set_player(puuid[:12])
                sub_res = self.collect_player_matches(
                    puuid, region=region, count=batch_size, start_time=start_time
                )
                self.queue_repo.mark_processed(puuid)
                if item.id:
                    self.queue_repo.mark_processed_by_id(item.id)

                total_res.matches_saved += sub_res.matches_saved
                total_res.records_saved += sub_res.records_saved
                total_res.timelines_saved += sub_res.timelines_saved
                total_res.embeddings_computed += sub_res.embeddings_computed
                total_res.skipped += sub_res.skipped
                total_res.errors += sub_res.errors
                total_res.discovered_puuids.extend(sub_res.discovered_puuids)
                self.tracker.step(sub_res.matches_saved)

                # Snowballing: enqueue newly discovered participants
                if sub_res.discovered_puuids:
                    discovered_by_region: dict[str, list[str]] = {}
                    for p, r in sub_res.discovered_puuids:
                        discovered_by_region.setdefault(r, []).append(p)
                    for r, plist in discovered_by_region.items():
                        self.queue_repo.enqueue_puuids(plist, region=r, priority=0)

                batch_counter += 1
                if batch_counter % 10 == 0:
                    try:
                        with database.get_connection(self.db_path) as conn:
                            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    except Exception as e:
                        logger.warning("WAL checkpoint error: %s", e)

            self.tracker.set_completed()
        except Exception as e:
            self.tracker.set_error(str(e))
            raise
        return total_res
