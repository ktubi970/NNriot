"""Domain Feature Engineering package for League of Legends match modeling.

Provides domain-specific meta features, champion static properties,
matchup deltas, composition analytics, and a unified feature pipeline.
"""
from __future__ import annotations

from core.features.static_meta import (
    ChampionStaticProfile,
    get_champion_profile,
    get_champion_profile_by_id,
)
from core.features.matchup_matrix import (
    LaneMatchupTracker,
    get_lane_matchup_tracker,
)
from core.features.draft_composition import (
    compute_team_composition_traits,
    extract_draft_composition_features,
)
from core.features.pipeline import (
    DraftFeaturePipeline,
    get_feature_pipeline,
)

__all__ = [
    "ChampionStaticProfile",
    "get_champion_profile",
    "get_champion_profile_by_id",
    "LaneMatchupTracker",
    "get_lane_matchup_tracker",
    "compute_team_composition_traits",
    "extract_draft_composition_features",
    "DraftFeaturePipeline",
    "get_feature_pipeline",
]
