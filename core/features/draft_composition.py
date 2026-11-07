"""Macro team composition traits, draft archetype synergies, and temporal power-spikes.

Computes composition-level attributes:
- Damage distribution: AD %, AP %, True Damage %
- Vulnerability indicators: Full AD flag (> 80% AD)
- Crowd Control (CC) / Engage score
- Scaling differential (early vs late curve)
- Ranged vs Melee composition balance
- 5 Draft Archetypes: Poke/Siege, Hard Engage/Teamfight, Pick/Catch, Split-Push, Hyper-Scaling
- Temporal Power-Spikes: 15 min, 25 min, 35 min projected advantage

All differentials guarantee exact mathematical anti-symmetry under team swap:
diff(Team B, Team A) == -diff(Team A, Team B).
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from core.champion_registry import UNK_CHAMPION_ID, champion_to_id
from core.features.static_meta import get_champion_profile_by_id, ChampionStaticProfile

COMPOSITION_TRAIT_COUNT: int = 7
DRAFT_ARCHETYPE_COUNT: int = 5
TEMPORAL_SPIKE_COUNT: int = 3
MACRO_CLASH_DIM: int = COMPOSITION_TRAIT_COUNT + DRAFT_ARCHETYPE_COUNT + TEMPORAL_SPIKE_COUNT  # 15 dimensions
MACRO_CLASH_FEATURE_DIM: int = MACRO_CLASH_DIM


def compute_team_composition_traits(
    champion_ids: Sequence[int],
) -> dict[str, float]:
    """Compute aggregate composition properties for a 5-player team."""
    valid_ids = [cid for cid in champion_ids if cid != UNK_CHAMPION_ID]
    if not valid_ids:
        return {
            "ad_ratio": 0.50,
            "ap_ratio": 0.45,
            "true_ratio": 0.05,
            "cc_score": 0.40,
            "scaling_score": 0.00,
            "ranged_ratio": 0.50,
            "is_full_ad": 0.0,
        }

    profiles = [get_champion_profile_by_id(cid) for cid in valid_ids]
    n = float(len(profiles))

    ad = sum(p.ad_ratio for p in profiles) / n
    ap = sum(p.ap_ratio for p in profiles) / n
    tr = sum(p.true_ratio for p in profiles) / n
    cc = sum(p.cc_score for p in profiles) / n
    sc = sum(p.scaling_score for p in profiles) / n
    rng = sum(p.is_ranged for p in profiles) / n
    full_ad = 1.0 if ad >= 0.80 else 0.0

    return {
        "ad_ratio": float(ad),
        "ap_ratio": float(ap),
        "true_ratio": float(tr),
        "cc_score": float(cc),
        "scaling_score": float(sc),
        "ranged_ratio": float(rng),
        "is_full_ad": full_ad,
    }


def compute_draft_archetypes(traits: dict[str, float]) -> dict[str, float]:
    """Compute normalized score [0, 1] for 5 canonical draft archetypes."""
    ad = traits["ad_ratio"]
    ap = traits["ap_ratio"]
    cc = traits["cc_score"]
    sc = traits["scaling_score"]
    rng = traits["ranged_ratio"]

    # 1. Poke / Siege: high range, high magic damage, safe distance
    poke = np.clip(rng * 0.45 + ap * 0.35 + (1.0 - cc) * 0.20, 0.0, 1.0)

    # 2. Hard Engage / 5v5 Teamfight: high CC, melee tankiness, initiation
    engage = np.clip(cc * 0.50 + (1.0 - rng) * 0.35 + ad * 0.15, 0.0, 1.0)

    # 3. Pick / Catch: reliable CC, burst execution, early priority
    pick = np.clip(cc * 0.40 + ap * 0.30 + (1.0 - sc) * 0.30, 0.0, 1.0)

    # 4. Split-Push (1-3-1): melee dueling, high physical DPS, side lane scaling
    split = np.clip((1.0 - rng) * 0.40 + sc * 0.35 + ad * 0.25, 0.0, 1.0)

    # 5. Hyper-Scaling Protect-the-Carry: massive late game potential
    hyper = np.clip(sc * 0.70 + rng * 0.30, 0.0, 1.0)

    return {
        "poke_siege": float(poke),
        "hard_engage": float(engage),
        "pick_catch": float(pick),
        "split_push": float(split),
        "hyper_scaling": float(hyper),
    }


def compute_temporal_power_spikes(traits: dict[str, float]) -> dict[str, float]:
    """Project relative team power across 3 game phases (15 min, 25 min, 35 min)."""
    sc = traits["scaling_score"]
    rng = traits["ranged_ratio"]
    cc = traits["cc_score"]
    full_ad = traits["is_full_ad"]

    # Phase 1: 15 min (Early lane dominance & skirmishes)
    p15 = np.clip((1.0 - sc) * 0.60 + rng * 0.40, 0.0, 1.0)

    # Phase 2: 25 min (Mid game objective fights & dragon stacks)
    p25 = np.clip(cc * 0.50 + 0.50 * (1.0 - abs(sc - 0.5) * 2.0), 0.0, 1.0)

    # Phase 3: 35 min (Late game Baron/Elder teamfights)
    p35 = np.clip(sc * 0.80 + 0.20 * (1.0 - full_ad), 0.0, 1.0)

    return {
        "spike_15m": float(p15),
        "spike_25m": float(p25),
        "spike_35m": float(p35),
    }


def extract_macro_clash_vector(
    team_a_ids: Sequence[int],
    team_b_ids: Sequence[int],
) -> np.ndarray:
    """Extract a 15-dimensional anti-symmetric macro clash vector (Team B - Team A).

    Guarantees:
        extract_macro_clash_vector(B, A) == -extract_macro_clash_vector(A, B)
    """
    t_a = compute_team_composition_traits(team_a_ids)
    t_b = compute_team_composition_traits(team_b_ids)

    # 1. Trait diffs (7)
    trait_keys = ["ad_ratio", "ap_ratio", "true_ratio", "cc_score", "scaling_score", "ranged_ratio", "is_full_ad"]
    v_a_traits = np.asarray([t_a[k] for k in trait_keys], dtype=np.float32)
    v_b_traits = np.asarray([t_b[k] for k in trait_keys], dtype=np.float32)
    diff_traits = v_b_traits - v_a_traits

    # 2. Archetype diffs (5)
    arch_a = compute_draft_archetypes(t_a)
    arch_b = compute_draft_archetypes(t_b)
    arch_keys = ["poke_siege", "hard_engage", "pick_catch", "split_push", "hyper_scaling"]
    v_a_arch = np.asarray([arch_a[k] for k in arch_keys], dtype=np.float32)
    v_b_arch = np.asarray([arch_b[k] for k in arch_keys], dtype=np.float32)
    diff_arch = v_b_arch - v_a_arch

    # 3. Temporal power-spike diffs (3)
    spikes_a = compute_temporal_power_spikes(t_a)
    spikes_b = compute_temporal_power_spikes(t_b)
    spike_keys = ["spike_15m", "spike_25m", "spike_35m"]
    v_a_spikes = np.asarray([spikes_a[k] for k in spike_keys], dtype=np.float32)
    v_b_spikes = np.asarray([spikes_b[k] for k in spike_keys], dtype=np.float32)
    diff_spikes = v_b_spikes - v_a_spikes

    return np.concatenate([diff_traits, diff_arch, diff_spikes]).astype(np.float32)


def extract_draft_composition_features(
    team_a_ids: Sequence[int],
    team_b_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract anti-symmetric difference and symmetric sum vectors of composition traits (backward compatible)."""
    t_a = compute_team_composition_traits(team_a_ids)
    t_b = compute_team_composition_traits(team_b_ids)

    keys = ["ad_ratio", "ap_ratio", "true_ratio", "cc_score", "scaling_score", "ranged_ratio", "is_full_ad"]

    v_a = np.asarray([t_a[k] for k in keys], dtype=np.float32)
    v_b = np.asarray([t_b[k] for k in keys], dtype=np.float32)

    diff_vec = v_b - v_a  # Anti-symmetric
    sum_vec = v_a + v_b   # Symmetric

    return diff_vec, sum_vec
