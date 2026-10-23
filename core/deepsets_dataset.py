"""Structured tensor extraction for DeepSets pre-match modeling.

Extracts canonical (5, 8) tensors per team:
Each player is represented by:
[champion_id, role_id, smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp]
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from core.champion_registry import champion_to_id, UNK_CHAMPION_ID
from core.draft_features import ROLE_INDEX, _ROLE_ALIASES
from core.player_priors import PlayerPriorsTracker

PLAYER_TENSOR_DIM: int = 8  # champ_id (1) + role_id (1) + priors (6)


def _extract_single_team(
    team_participants: Any,
    tracker: PlayerPriorsTracker | None = None,
) -> np.ndarray:
    """Extract a (5, 8) float32 tensor for a single team ordered canonically by role."""
    tensor = np.zeros((5, PLAYER_TENSOR_DIM), dtype=np.float32)
    default_priors = PlayerPriorsTracker.get_default_priors()

    # Pre-fill defaults
    for i in range(5):
        tensor[i, 0] = UNK_CHAMPION_ID
        tensor[i, 1] = float(i)
        tensor[i, 2:] = default_priors

    if not isinstance(team_participants, (list, tuple)):
        return tensor

    assigned_slots: set[int] = set()
    unassigned_entries: list[tuple[float, float, list[float]]] = []

    for p in team_participants:
        if not isinstance(p, dict):
            continue
        champ = p.get("champion") or p.get("championName") or p.get("champion_name")
        cid = float(champion_to_id(champ))

        raw_role = (p.get("role") or p.get("teamPosition") or p.get("individualPosition") or "").strip().upper()
        role = _ROLE_ALIASES.get(raw_role, raw_role)

        if "priors" in p and isinstance(p["priors"], (list, tuple)) and len(p["priors"]) >= 6:
            priors = [float(x) for x in p["priors"][:6]]
        elif "smooth_wr" in p:
            priors = [
                float(p.get("smooth_wr", 0.50)),
                float(p.get("smooth_cwr", 0.50)),
                float(p.get("champ_exp", 0.0)),
                float(p.get("rec_wr", 0.50)),
                float(p.get("role_fam", 0.20)),
                float(p.get("tot_exp", 0.0)),
            ]
        elif tracker is not None:
            priors = tracker.get_player_priors(p.get("puuid"), champ, role)
        else:
            priors = default_priors

        if role in ROLE_INDEX and ROLE_INDEX[role] not in assigned_slots:
            slot = ROLE_INDEX[role]
            assigned_slots.add(slot)
            tensor[slot, 0] = cid
            tensor[slot, 1] = float(slot)
            tensor[slot, 2:] = priors
        else:
            unassigned_entries.append((cid, float(ROLE_INDEX.get(role, 0)), priors))

    # Place unassigned entries in free slots
    unassigned_idx = 0
    for slot in range(5):
        if slot not in assigned_slots and unassigned_idx < len(unassigned_entries):
            cid, _, priors = unassigned_entries[unassigned_idx]
            tensor[slot, 0] = cid
            tensor[slot, 1] = float(slot)
            tensor[slot, 2:] = priors
            unassigned_idx += 1

    return tensor


def extract_team_tensors(
    feature_json: dict[str, Any] | None,
    tracker: PlayerPriorsTracker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (team_a, team_b) tensors of shape (5, 8) each."""
    if not isinstance(feature_json, dict):
        team_a_parts = []
        team_b_parts = []
    else:
        team_a_parts = feature_json.get("team_a") or []
        team_b_parts = feature_json.get("team_b") or []

    t_a = _extract_single_team(team_a_parts, tracker=tracker)
    t_b = _extract_single_team(team_b_parts, tracker=tracker)
    return t_a, t_b


def batch_extract_team_tensors(
    features_list: Sequence[dict[str, Any] | None],
    tracker: PlayerPriorsTracker | None = None,
) -> dict[str, np.ndarray]:
    """Batch extract structured team tensors for multiple matches.
    
    Returns:
        {"team_a": np.ndarray of shape (N, 5, 8),
         "team_b": np.ndarray of shape (N, 5, 8)}
    """
    n = len(features_list)
    batch_a = np.empty((n, 5, PLAYER_TENSOR_DIM), dtype=np.float32)
    batch_b = np.empty((n, 5, PLAYER_TENSOR_DIM), dtype=np.float32)

    for i, fj in enumerate(features_list):
        ta, tb = extract_team_tensors(fj, tracker=tracker)
        batch_a[i] = ta
        batch_b[i] = tb

    return {"team_a": batch_a, "team_b": batch_b}
