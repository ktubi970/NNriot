"""Enriched feature extraction combining draft compositions and point-in-time player priors.

Feature dimensions:
- Dense Neural (ENRICHED_DENSE_DIM = 203):
  170 champion composition difference indicator (+1 for Team A, -1 for Team B, 0 neither)
  + 33 point-in-time player performance priors (team aggregates, diffs, lane matchups)
- Tabular GBDT (ENRICHED_TABULAR_DIM = 213):
  10 canonical role-ordered champion IDs
  + 170 champion composition difference
  + 33 point-in-time player priors
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from core.champion_registry import NUM_CHAMPIONS, UNK_CHAMPION_ID
from core.draft_features import _order_team_participants
from core.player_priors import (
    PlayerPriorsTracker,
    TOTAL_PLAYER_PRIOR_DIM,
    extract_player_priors_vector,
)

ENRICHED_DENSE_DIM: int = NUM_CHAMPIONS + TOTAL_PLAYER_PRIOR_DIM  # 170 + 33 = 203
ENRICHED_TABULAR_DIM: int = 10 + NUM_CHAMPIONS + TOTAL_PLAYER_PRIOR_DIM  # 10 + 170 + 33 = 213


def _extract_composition_vector(
    team_a: Sequence[dict[str, Any]],
    team_b: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Extract length-NUM_CHAMPIONS vector: +1 for Team A champion, -1 for Team B champion."""
    vec = np.zeros(NUM_CHAMPIONS, dtype=np.float32)
    a_ids = _order_team_participants(team_a)
    b_ids = _order_team_participants(team_b)
    for cid in a_ids:
        if cid != UNK_CHAMPION_ID:
            vec[cid] += 1.0
    for cid in b_ids:
        if cid != UNK_CHAMPION_ID:
            vec[cid] -= 1.0
    return vec


def extract_enriched_features(
    feature_json: dict[str, Any] | None,
    mode: str = "dense",
    tracker: PlayerPriorsTracker | None = None,
) -> np.ndarray:
    """Extract an enriched feature vector for a match.

    Args:
        feature_json: Pre-match feature dict containing 'team_a' and 'team_b'.
        mode: 'dense' (203 features for neural net) or 'tabular' (213 features for GBDT).
        tracker: Optional PlayerPriorsTracker to compute point-in-time priors.
                 If None and no attached priors are present in participants,
                 uses neutral default priors.
    """
    if mode not in ("dense", "tabular"):
        raise ValueError(f"mode must be 'dense' or 'tabular', got {mode!r}")

    if not isinstance(feature_json, dict):
        team_a = []
        team_b = []
    else:
        team_a = feature_json.get("team_a") or []
        team_b = feature_json.get("team_b") or []

    # 1. Draft composition difference (NUM_CHAMPIONS, float32)
    comp_diff = _extract_composition_vector(team_a, team_b)

    # 2. Player priors (33, float32)
    priors_vec = extract_player_priors_vector(team_a, team_b, tracker=tracker)

    if mode == "dense":
        return np.concatenate([comp_diff, priors_vec]).astype(np.float32)

    # Tabular: 10 role-ordered champion IDs + comp_diff + priors
    t1_ids = _order_team_participants(team_a)
    t2_ids = _order_team_participants(team_b)
    role_ids = np.asarray(t1_ids + t2_ids, dtype=np.float32)

    return np.concatenate([role_ids, comp_diff, priors_vec]).astype(np.float32)


def batch_extract_enriched_features(
    features_list: Sequence[dict[str, Any] | None],
    mode: str = "dense",
    tracker: PlayerPriorsTracker | None = None,
) -> np.ndarray:
    """Batch extract enriched features for multiple pre-match records."""
    if not features_list:
        target_dim = ENRICHED_DENSE_DIM if mode == "dense" else ENRICHED_TABULAR_DIM
        return np.empty((0, target_dim), dtype=np.float32)

    extracted = [
        extract_enriched_features(fj, mode=mode, tracker=tracker)
        for fj in features_list
    ]
    return np.stack(extracted, axis=0)
