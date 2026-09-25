"""Domain-specific draft feature extraction for League of Legends matches.

Provides canonical, collision-free feature vectors for both tabular GBDT models
and neural networks.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from core.champion_registry import (
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    champion_to_id,
)

ROLE_NAMES: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_INDEX: dict[str, int] = {role: idx for idx, role in enumerate(ROLE_NAMES)}
_ROLE_ALIASES: dict[str, str] = {
    "MID": "MIDDLE",
    "BOT": "BOTTOM",
    "SUPPORT": "UTILITY",
    "SUP": "UTILITY",
    "ADC": "BOTTOM",
}

# Tabular: 10 role champion IDs (int) + NUM_CHAMPIONS composition diff indicators (+1/0/-1)
TABULAR_FEATURE_DIM: int = 10 + NUM_CHAMPIONS

# Dense Neural: 5 one-hot roles for Team A (5*N) + 5 one-hot roles for Team B (5*N) + comp diff (N)
DENSE_FEATURE_DIM: int = 11 * NUM_CHAMPIONS


def _order_team_participants(participants: list[dict[str, Any]]) -> list[int]:
    """Extract 5 champion IDs ordered strictly by [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY].

    If roles are unassigned or duplicated, maps greedily and fills remaining slots with UNK.
    """
    ordered_ids = [UNK_CHAMPION_ID] * 5
    assigned_slots: set[int] = set()
    unassigned_players: list[int] = []

    for p in participants:
        if not isinstance(p, dict):
            continue
        champ_name = p.get("champion") or p.get("championName") or p.get("champion_name")
        cid = champion_to_id(champ_name)
        raw_role = (p.get("role") or p.get("teamPosition") or p.get("individualPosition") or "").strip().upper()
        role = _ROLE_ALIASES.get(raw_role, raw_role)

        if role in ROLE_INDEX and ROLE_INDEX[role] not in assigned_slots:
            slot = ROLE_INDEX[role]
            ordered_ids[slot] = cid
            assigned_slots.add(slot)
        else:
            unassigned_players.append(cid)

    # Fill remaining unassigned slots
    unassigned_idx = 0
    for slot in range(5):
        if slot not in assigned_slots and unassigned_idx < len(unassigned_players):
            ordered_ids[slot] = unassigned_players[unassigned_idx]
            unassigned_idx += 1

    return ordered_ids


def extract_tabular_draft_features(features: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Extract a 1D vector of length TABULAR_FEATURE_DIM and corresponding feature names.

    Vector structure:
      - [0:5]: Team A champion IDs (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
      - [5:10]: Team B champion IDs (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
      - [10:10+NUM_CHAMPIONS]: Composition diff (+1.0 for Team B presence, -1.0 for Team A presence)
    """
    if not isinstance(features, dict):
        features = {}

    team_a = features.get("team_a") or []
    team_b = features.get("team_b") or []

    a_ids = _order_team_participants(team_a)
    b_ids = _order_team_participants(team_b)

    vec = np.zeros(TABULAR_FEATURE_DIM, dtype=np.float32)
    names: list[str] = []

    for role_name in ROLE_NAMES:
        names.append(f"team_a_{role_name}")
    for role_name in ROLE_NAMES:
        names.append(f"team_b_{role_name}")

    vec[0:5] = a_ids
    vec[5:10] = b_ids

    # Composition diff
    offset = 10
    for cid in range(NUM_CHAMPIONS):
        names.append(f"comp_diff_champ_{cid}")

    for cid in a_ids:
        if cid != UNK_CHAMPION_ID:
            vec[offset + cid] -= 1.0

    for cid in b_ids:
        if cid != UNK_CHAMPION_ID:
            vec[offset + cid] += 1.0

    return vec, names


def extract_dense_draft_features(features: dict[str, Any]) -> np.ndarray:
    """Extract a 1D float32 vector of length DENSE_FEATURE_DIM for neural networks.

    Vector structure:
      - 5 one-hot segments for Team A roles (5 * NUM_CHAMPIONS)
      - 5 one-hot segments for Team B roles (5 * NUM_CHAMPIONS)
      - 1 composition diff segment (NUM_CHAMPIONS)
    """
    if not isinstance(features, dict):
        features = {}

    team_a = features.get("team_a") or []
    team_b = features.get("team_b") or []

    a_ids = _order_team_participants(team_a)
    b_ids = _order_team_participants(team_b)

    dense = np.zeros(DENSE_FEATURE_DIM, dtype=np.float32)

    # Team A roles
    for slot, cid in enumerate(a_ids):
        if cid != UNK_CHAMPION_ID:
            dense[slot * NUM_CHAMPIONS + cid] = 1.0

    # Team B roles
    b_offset = 5 * NUM_CHAMPIONS
    for slot, cid in enumerate(b_ids):
        if cid != UNK_CHAMPION_ID:
            dense[b_offset + slot * NUM_CHAMPIONS + cid] = 1.0

    # Comp diff segment
    diff_offset = 10 * NUM_CHAMPIONS
    for cid in a_ids:
        if cid != UNK_CHAMPION_ID:
            dense[diff_offset + cid] -= 1.0
    for cid in b_ids:
        if cid != UNK_CHAMPION_ID:
            dense[diff_offset + cid] += 1.0

    return dense


def batch_extract_draft_features(
    features_list: list[dict[str, Any]], mode: str = "tabular"
) -> np.ndarray:
    """Batch feature extraction for lists of match feature dicts."""
    n = len(features_list)
    if mode == "tabular":
        out = np.zeros((n, TABULAR_FEATURE_DIM), dtype=np.float32)
        for i, feat in enumerate(features_list):
            out[i], _ = extract_tabular_draft_features(feat)
        return out
    elif mode == "dense":
        out = np.zeros((n, DENSE_FEATURE_DIM), dtype=np.float32)
        for i, feat in enumerate(features_list):
            out[i] = extract_dense_draft_features(feat)
        return out
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'tabular' or 'dense'.")
