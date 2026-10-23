"""Ingestion Pipeline & Quality Guard (MatchIngestor).

Responsible for:
1. Strict pre-match feature extraction (json_utils.extract_pre_match_features) ensuring zero post-game leakage.
2. Multi-head label extraction via feature_labels.extract_labels.
3. Quality filtering via MatchQualityFilter (gameMode, queueId, min_duration, valid winner).
4. Persisting raw matches to MatchRepository and training records to TrainingRepository.
5. Systematic timeline fetching and spatial embedding extraction via core.spatial_rnn.extract_and_store_timeline_embeddings.
6. Returning IngestionResult with counts and discovered participant PUUIDs for snowballing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from core import config, database, json_utils
from core.collection.models import IngestionResult, MatchQualityFilter
from core.feature_labels import extract_labels
from core.repository import MatchRepository, TrainingRepository

logger = logging.getLogger(__name__)


class MatchIngestor:
    """Validates, extracts features, and persists matches and training records."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        quality_filter: Optional[MatchQualityFilter] = None,
        match_repo: Optional[MatchRepository] = None,
        train_repo: Optional[TrainingRepository] = None,
    ):
        self.db_path = config.get_db_path(db_path, require_exists=False)
        if not Path(self.db_path).is_file():
            database.init_db(self.db_path)

        self.filter = quality_filter or MatchQualityFilter()
        self.match_repo = match_repo if match_repo is not None else MatchRepository(self.db_path)
        self.train_repo = train_repo if train_repo is not None else TrainingRepository(self.db_path)

    def build_training_record(
        self, match_id: str, details: dict, region: Optional[str] = None
    ) -> Optional[tuple]:
        """Convert raw Riot API match JSON into a leak-free training tuple.

        Returns (match_id, pre_match_feature_dict, winner_label, labels_dict)
        or None if validation fails.
        """
        try:
            if not isinstance(details, dict):
                return None

            info = details.get("info", details)
            if not isinstance(info, dict):
                return None

            game_mode = (info.get("gameMode") or "").upper()
            if self.filter.allowed_game_modes and game_mode not in self.filter.allowed_game_modes:
                return None

            duration = info.get("gameDuration", 0)
            if duration < self.filter.min_duration_seconds:
                return None

            queue_id = info.get("queueId")
            if self.filter.allowed_queue_ids and queue_id not in self.filter.allowed_queue_ids:
                return None

            teams = info.get("teams", [])
            if not isinstance(teams, list):
                return None

            team_a = next(
                (t for t in teams if isinstance(t, dict) and t.get("teamId") == 100),
                None,
            )
            team_b = next(
                (t for t in teams if isinstance(t, dict) and t.get("teamId") == 200),
                None,
            )
            if not team_a or not team_b:
                return None

            a_win = bool(team_a.get("win"))
            b_win = bool(team_b.get("win"))
            if a_win == b_win:
                return None  # Both won or both lost / draw

            winner = 1 if b_win else 0

            # Strictly pre-match features (leak-free)
            feature = json_utils.extract_pre_match_features(details, region=region)
            labels = extract_labels(details)
            if labels is None:
                return None

            return (match_id, feature, winner, labels)
        except Exception as e:
            logger.error("Error building training record for %s: %s", match_id, e)
            return None

    def ingest_match_ids(
        self,
        match_ids: list[str],
        api: Any,
        region: str,
        max_workers: int = 5,
        verbose: bool = False,
    ) -> IngestionResult:
        """Download, validate, extract features/timelines, and persist new matches."""
        result = IngestionResult()
        if not match_ids:
            return result

        # Deduplicate incoming match IDs while preserving order
        unique_mids = list(dict.fromkeys(match_ids))

        new_mids = [mid for mid in unique_mids if not self.match_repo.match_exists(mid)]
        already_existing_count = len(unique_mids) - len(new_mids)
        result.skipped += already_existing_count

        if not new_mids:
            return result

        try:
            if hasattr(api, "get_match_details_batch"):
                details_map = api.get_match_details_batch(
                    new_mids, max_workers=max_workers, verbose=verbose
                )
            elif hasattr(api, "get_match_details"):
                details_map = {mid: api.get_match_details(mid) for mid in new_mids}
            else:
                details_map = {}
        except Exception as e:
            logger.error("Error fetching match details: %s", e)
            result.errors += len(new_mids)
            return result

        if not details_map:
            result.errors += len(new_mids)
            return result

        matches_batch: list[tuple[str, dict]] = []
        training_batch: list[tuple] = []
        discovered_puuids_set: set[str] = set()

        for mid in new_mids:
            details = details_map.get(mid)
            if not details or not isinstance(details, dict):
                result.errors += 1
                continue

            info = details.get("info", details) if isinstance(details, dict) else {}
            if not isinstance(info, dict):
                result.errors += 1
                continue

            record = self.build_training_record(mid, details, region=region)
            if not record:
                result.skipped += 1
                continue

            matches_batch.append((mid, details))
            training_batch.append(record)

            # Discover participants for snowballing from info.participants and metadata.participants
            for p in info.get("participants", []):
                if isinstance(p, dict):
                    p_puuid = p.get("puuid")
                    if p_puuid and isinstance(p_puuid, str) and p_puuid.strip():
                        discovered_puuids_set.add(p_puuid.strip())

            metadata = details.get("metadata", {}) if isinstance(details, dict) else {}
            if isinstance(metadata, dict):
                for p_puuid in metadata.get("participants", []):
                    if p_puuid and isinstance(p_puuid, str) and p_puuid.strip():
                        discovered_puuids_set.add(p_puuid.strip())

        if matches_batch:
            self.match_repo.save_matches_batch(matches_batch)
            result.matches_saved = len(matches_batch)

        if training_batch:
            self.train_repo.save_training_records_batch(training_batch)
            result.records_saved = len(training_batch)

        # Systematic timeline collection
        timelines_to_fetch = [mid for mid, _ in matches_batch]
        if timelines_to_fetch:
            try:
                timelines = None
                if hasattr(api, "get_match_timelines_batch"):
                    timelines = api.get_match_timelines_batch(
                        timelines_to_fetch, max_workers=max_workers, verbose=verbose
                    )
                elif hasattr(api, "get_match_timeline"):
                    timelines = {}
                    for mid in timelines_to_fetch:
                        t = api.get_match_timeline(mid)
                        if t:
                            timelines[mid] = t

                if timelines:
                    self.match_repo.save_match_timelines_batch(list(timelines.items()))
                    result.timelines_saved = len(timelines)

                    try:
                        from core.spatial_rnn import extract_and_store_timeline_embeddings
                    except ImportError as err:
                        logger.warning("Could not import spatial_rnn: %s", err)
                    else:
                        for t_mid, t_data in timelines.items():
                            try:
                                m_details = details_map.get(t_mid, {})
                                info_dict = (
                                    m_details.get("info", {})
                                    if isinstance(m_details, dict)
                                    else {}
                                )
                                gtime = info_dict.get("gameCreation") or int(time.time() * 1000)
                                # 0 means skipped: the database holds embeddings from other encoder weights.
                                if extract_and_store_timeline_embeddings(
                                    t_mid, t_data, int(gtime), db_path=self.db_path
                                ):
                                    result.embeddings_computed += 1
                            except Exception as err:
                                logger.warning(
                                    "Spatial embedding failed for %s (%s): %s",
                                    t_mid,
                                    region,
                                    err,
                                )
            except Exception as e:
                logger.error("Error fetching timelines for batch: %s", e)

        result.discovered_puuids = [(p, region) for p in sorted(discovered_puuids_set)]
        return result
