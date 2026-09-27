"""Unified Draft Feature Pipeline for tabular GBDT and neural architectures.

Conforms to the pattern-design Strategy pattern:
- Produces canonical (5, 14) structured tensors per team for DeepSets / Cross-Attention
- Produces dense tabular feature vectors (232 features) for LightGBM / GBDT
- Guarantees exact mathematical anti-symmetry under team swap
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from core.champion_registry import (
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    champion_to_id,
)
from core.draft_features import (
    ROLE_INDEX,
    ROLE_NAMES,
    _ROLE_ALIASES,
    _order_team_participants,
)
from core.player_priors import (
    PlayerPriorsTracker,
    extract_player_priors_vector,
)
from core.features.static_meta import get_champion_profile_by_id
from core.features.matchup_matrix import LaneMatchupTracker, get_lane_matchup_tracker
from core.features.draft_composition import (
    extract_draft_composition_features,
    extract_macro_clash_vector,
    MACRO_CLASH_DIM,
)

# Tensor dimensions per player: 14 features:
# [champ_id, role_id, smooth_wr, smooth_cwr, champ_exp, rec_wr, role_fam, tot_exp,
#  smooth_kda, smooth_cspm, smooth_gpm, early_aggression, lane_matchup_delta, champ_base_wr]
ENRICHED_PLAYER_TENSOR_DIM: int = 14

# Tabular dimension: 10 role IDs + 170 comp diff + 33 player priors + 5 lane deltas + 7 comp diff + 7 comp sum = 232
ENRICHED_TABULAR_DIM: int = 10 + NUM_CHAMPIONS + 33 + 5 + 7 + 7


class DraftFeaturePipeline:
    """Strategy pipeline producing normalized, leak-free representations for pre-match modeling."""

    def __init__(
        self,
        matchup_tracker: LaneMatchupTracker | None = None,
        player_tracker: PlayerPriorsTracker | None = None,
    ) -> None:
        self.matchup_tracker = matchup_tracker or get_lane_matchup_tracker()
        self.player_tracker = player_tracker

    def extract_tensors(
        self,
        feature_dict: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (team_a, team_b) tensors of shape (5, 14) each."""
        t_a = np.zeros((5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
        t_b = np.zeros((5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)

        default_priors = PlayerPriorsTracker.get_default_priors(extended=True)

        if not isinstance(feature_dict, dict):
            feature_dict = {}

        team_a_parts = feature_dict.get("team_a") or []
        team_b_parts = feature_dict.get("team_b") or []

        # Order participants by canonical role [TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY]
        def _build_team_map(parts):
            role_map: dict[int, dict[str, Any]] = {}
            unassigned: list[dict[str, Any]] = []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                r_raw = (p.get("role") or p.get("teamPosition") or "").strip().upper()
                r = _ROLE_ALIASES.get(r_raw, r_raw)
                slot = ROLE_INDEX.get(r)
                if slot is not None and slot not in role_map:
                    role_map[slot] = p
                else:
                    unassigned.append(p)
            # Fill empty slots
            u_idx = 0
            for slot in range(5):
                if slot not in role_map:
                    if u_idx < len(unassigned):
                        role_map[slot] = unassigned[u_idx]
                        u_idx += 1
                    else:
                        role_map[slot] = {}
            return role_map

        map_a = _build_team_map(team_a_parts)
        map_b = _build_team_map(team_b_parts)

        # Populate tensors
        for slot in range(5):
            role_name = ROLE_NAMES[slot]
            pa = map_a[slot]
            pb = map_b[slot]

            cid_a = champion_to_id(pa.get("champion") or pa.get("championName"))
            cid_b = champion_to_id(pb.get("champion") or pb.get("championName"))

            # Player priors (10 features)
            def _get_priors(p, cid):
                if "priors" in p and isinstance(p["priors"], (list, tuple)):
                    p_list = [float(x) for x in p["priors"]]
                    if len(p_list) >= 10:
                        return p_list[:10]
                    return p_list[:6] + default_priors[6:10]
                if self.player_tracker is not None:
                    return self.player_tracker.get_player_priors(p.get("puuid"), cid, role_name, extended=True)
                return default_priors

            pri_a = _get_priors(pa, cid_a)
            pri_b = _get_priors(pb, cid_b)

            # Matchup delta: WR_role(cid_a | cid_b) - 0.5
            delta_a = self.matchup_tracker.get_matchup_delta(role_name, cid_a, cid_b)
            delta_b = -delta_a  # Exact anti-symmetry guarantee

            # Static domain profiles
            prof_a = get_champion_profile_by_id(cid_a)
            prof_b = get_champion_profile_by_id(cid_b)

            # Layout: [cid, slot, p0..p9 (10 priors), matchup_delta, base_wr] -> 14 features
            t_a[slot, 0] = float(cid_a)
            t_a[slot, 1] = float(slot)
            t_a[slot, 2:12] = pri_a
            t_a[slot, 12] = float(delta_a)
            t_a[slot, 13] = float(prof_a.base_winrate)

            t_b[slot, 0] = float(cid_b)
            t_b[slot, 1] = float(slot)
            t_b[slot, 2:12] = pri_b
            t_b[slot, 12] = float(delta_b)
            t_b[slot, 13] = float(prof_b.base_winrate)

        return t_a, t_b

    def extract_tabular(
        self,
        feature_dict: dict[str, Any] | None,
    ) -> np.ndarray:
        """Extract a 232-dimensional domain vector for GBDT / LightGBM."""
        if not isinstance(feature_dict, dict):
            feature_dict = {}

        team_a_parts = feature_dict.get("team_a") or []
        team_b_parts = feature_dict.get("team_b") or []

        a_ids = _order_team_participants(team_a_parts)
        b_ids = _order_team_participants(team_b_parts)

        # 1. Role champion IDs (10)
        role_ids = np.asarray(a_ids + b_ids, dtype=np.float32)

        # 2. Composition difference (+1 for Team B presence, -1 for Team A presence) (NUM_CHAMPIONS = 170)
        comp_diff = np.zeros(NUM_CHAMPIONS, dtype=np.float32)
        for cid in a_ids:
            if cid != UNK_CHAMPION_ID:
                comp_diff[cid] -= 1.0
        for cid in b_ids:
            if cid != UNK_CHAMPION_ID:
                comp_diff[cid] += 1.0

        # 3. Player performance priors (33)
        priors_vec = extract_player_priors_vector(team_a_parts, team_b_parts, tracker=self.player_tracker)

        # 4. Lane matchup deltas for 5 lanes (Team B - Team A) (5)
        lane_deltas = np.zeros(5, dtype=np.float32)
        for slot in range(5):
            role_name = ROLE_NAMES[slot]
            # Matchup advantage of Team B over Team A
            lane_deltas[slot] = self.matchup_tracker.get_matchup_delta(role_name, b_ids[slot], a_ids[slot])

        # 5. Composition trait differentials and sums (7 + 7 = 14)
        comp_trait_diff, comp_trait_sum = extract_draft_composition_features(a_ids, b_ids)

        return np.concatenate([
            role_ids,
            comp_diff,
            priors_vec,
            lane_deltas,
            comp_trait_diff,
            comp_trait_sum,
        ]).astype(np.float32)

    def batch_extract_tensors(
        self,
        features_list: Sequence[dict[str, Any] | None],
    ) -> dict[str, np.ndarray]:
        """Batch extract tensors for multiple matches."""
        n = len(features_list)
        batch_a = np.zeros((n, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)
        batch_b = np.zeros((n, 5, ENRICHED_PLAYER_TENSOR_DIM), dtype=np.float32)

        for i, fj in enumerate(features_list):
            ta, tb = self.extract_tensors(fj)
            batch_a[i] = ta
            batch_b[i] = tb

        return {"team_a": batch_a, "team_b": batch_b}

    def batch_extract_tabular(
        self,
        features_list: Sequence[dict[str, Any] | None],
    ) -> np.ndarray:
        """Batch extract tabular vectors for multiple matches."""
        n = len(features_list)
        batch_tab = np.zeros((n, ENRICHED_TABULAR_DIM), dtype=np.float32)

        for i, fj in enumerate(features_list):
            batch_tab[i] = self.extract_tabular(fj)

        return batch_tab

    def extract_macro_clash(
        self,
        feature_dict: dict[str, Any] | None,
    ) -> np.ndarray:
        """Extract 15-dimensional macro clash vector (Team B - Team A)."""
        if not isinstance(feature_dict, dict):
            feature_dict = {}
        team_a_parts = feature_dict.get("team_a") or []
        team_b_parts = feature_dict.get("team_b") or []
        a_ids = _order_team_participants(team_a_parts)
        b_ids = _order_team_participants(team_b_parts)
        return extract_macro_clash_vector(a_ids, b_ids)

    def batch_extract_macro_clash(
        self,
        features_list: Sequence[dict[str, Any] | None],
    ) -> np.ndarray:
        """Batch extract macro clash vectors for multiple matches."""
        n = len(features_list)
        batch_clash = np.zeros((n, MACRO_CLASH_DIM), dtype=np.float32)
        for i, fj in enumerate(features_list):
            batch_clash[i] = self.extract_macro_clash(fj)
        return batch_clash

    def batch_extract_v3(
        self,
        features_list: Sequence[dict[str, Any] | None],
    ) -> dict[str, np.ndarray]:
        """Extract both player tensors and macro clash vector for DeepSets v3."""
        tensors = self.batch_extract_tensors(features_list)
        clash = self.batch_extract_macro_clash(features_list)
        return {
            "team_a": tensors["team_a"],
            "team_b": tensors["team_b"],
            "macro_clash": clash,
        }


_GLOBAL_PIPELINE: DraftFeaturePipeline | None = None


def get_feature_pipeline() -> DraftFeaturePipeline:
    """Return shared global feature pipeline singleton."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = DraftFeaturePipeline()
    return _GLOBAL_PIPELINE
