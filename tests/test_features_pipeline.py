"""Tests for domain feature engineering and unified draft feature pipeline."""
import pytest
import numpy as np

from core.features.static_meta import (
    get_champion_profile,
    get_champion_profile_by_id,
    ChampionStaticProfile,
)
from core.features.matchup_matrix import (
    LaneMatchupTracker,
)
from core.features.draft_composition import (
    compute_team_composition_traits,
    extract_draft_composition_features,
)
from core.features.pipeline import (
    DraftFeaturePipeline,
    ENRICHED_PLAYER_TENSOR_DIM,
    ENRICHED_TABULAR_DIM,
)
from core.champion_registry import champion_to_id


def test_static_champion_profiles():
    # Verify Zed is physical assassin with low CC
    zed = get_champion_profile("Zed")
    assert zed.ad_ratio > 0.80
    assert zed.ap_ratio < 0.10
    assert zed.cc_score < 0.30

    # Verify Leona is tank with high CC
    leona = get_champion_profile("Leona")
    assert leona.cc_score > 0.80
    assert leona.is_ranged == 0.0

    # Verify Kayle is late game scaling
    kayle = get_champion_profile("Kayle")
    assert kayle.scaling_score > 0.80


def test_lane_matchup_tracker_antisymmetry():
    tracker = LaneMatchupTracker(matchup_weight=5.0)
    cid_darius = champion_to_id("Darius")
    cid_sion = champion_to_id("Sion")

    # Team A has Darius, Team B has Sion. Team A wins 8 times out of 10
    for _ in range(8):
        tracker.update_match(
            [{"champion": "Darius", "role": "TOP"}],
            [{"champion": "Sion", "role": "TOP"}],
            winner=0,  # Team A wins
        )
    for _ in range(2):
        tracker.update_match(
            [{"champion": "Darius", "role": "TOP"}],
            [{"champion": "Sion", "role": "TOP"}],
            winner=1,  # Team B wins
        )

    delta_darius_vs_sion = tracker.get_matchup_delta("TOP", cid_darius, cid_sion)
    delta_sion_vs_darius = tracker.get_matchup_delta("TOP", cid_sion, cid_darius)

    assert delta_darius_vs_sion > 0.0
    assert delta_sion_vs_darius < 0.0
    # Strict anti-symmetry: delta(A, B) == -delta(B, A)
    np.testing.assert_allclose(delta_darius_vs_sion, -delta_sion_vs_darius, atol=1e-6)


def test_draft_composition_antisymmetry():
    t_a_ids = [champion_to_id("Renekton"), champion_to_id("Elise"), champion_to_id("Zed"),
               champion_to_id("Draven"), champion_to_id("Leona")]
    t_b_ids = [champion_to_id("Kayle"), champion_to_id("Sejuani"), champion_to_id("Kassadin"),
               champion_to_id("Vayne"), champion_to_id("Lulu")]

    diff_ab, sum_ab = extract_draft_composition_features(t_a_ids, t_b_ids)
    diff_ba, sum_ba = extract_draft_composition_features(t_b_ids, t_a_ids)

    # Differentials must be anti-symmetric
    np.testing.assert_allclose(diff_ba, -diff_ab, atol=1e-6)
    # Sums must be symmetric
    np.testing.assert_allclose(sum_ba, sum_ab, atol=1e-6)


def test_pipeline_tensor_extraction_and_team_swap():
    pipeline = DraftFeaturePipeline()

    match_dict = {
        "team_a": [
            {"champion": "Darius", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP"},
            {"champion": "Amumu", "role": "JUNGLE"},
            {"champion": "Syndra", "role": "MIDDLE"},
            {"champion": "Caitlyn", "role": "BOTTOM"},
            {"champion": "Leona", "role": "UTILITY"},
        ],
    }

    t_a, t_b = pipeline.extract_tensors(match_dict)
    assert t_a.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)
    assert t_b.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)

    # Swapped match
    swapped_dict = {"team_a": match_dict["team_b"], "team_b": match_dict["team_a"]}
    t_a_swapped, t_b_swapped = pipeline.extract_tensors(swapped_dict)

    # Swapping team A and team B should yield t_a_swapped == t_b and t_b_swapped == t_a
    np.testing.assert_allclose(t_a_swapped, t_b, atol=1e-6)
    np.testing.assert_allclose(t_b_swapped, t_a, atol=1e-6)


def test_pipeline_tabular_extraction_dimensions_and_antisymmetry():
    pipeline = DraftFeaturePipeline()

    match_dict = {
        "team_a": [
            {"champion": "Darius", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP"},
            {"champion": "Amumu", "role": "JUNGLE"},
            {"champion": "Syndra", "role": "MIDDLE"},
            {"champion": "Caitlyn", "role": "BOTTOM"},
            {"champion": "Leona", "role": "UTILITY"},
        ],
    }

    tab_ab = pipeline.extract_tabular(match_dict)
    assert tab_ab.shape == (ENRICHED_TABULAR_DIM,)
    assert not np.isnan(tab_ab).any()

    swapped_dict = {"team_a": match_dict["team_b"], "team_b": match_dict["team_a"]}
    tab_ba = pipeline.extract_tabular(swapped_dict)

    # Check composition diff indices (10 to 180) flip signs
    comp_diff_ab = tab_ab[10:180]
    comp_diff_ba = tab_ba[10:180]
    np.testing.assert_allclose(comp_diff_ba, -comp_diff_ab, atol=1e-6)


def test_pipeline_handles_malformed_and_empty():
    pipeline = DraftFeaturePipeline()

    t_a, t_b = pipeline.extract_tensors(None)
    assert t_a.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)
    assert t_b.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)
    assert not np.isnan(t_a).any()

    tab = pipeline.extract_tabular({})
    assert tab.shape == (ENRICHED_TABULAR_DIM,)
    assert not np.isnan(tab).any()


def test_macro_clash_anti_symmetry_and_v3_batch():
    from core.features.draft_composition import MACRO_CLASH_DIM
    pipeline = DraftFeaturePipeline()

    match_dict = {
        "team_a": [
            {"champion": "Darius", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP"},
            {"champion": "Amumu", "role": "JUNGLE"},
            {"champion": "Syndra", "role": "MIDDLE"},
            {"champion": "Caitlyn", "role": "BOTTOM"},
            {"champion": "Leona", "role": "UTILITY"},
        ],
    }

    clash_ab = pipeline.extract_macro_clash(match_dict)
    assert clash_ab.shape == (MACRO_CLASH_DIM,)
    assert not np.isnan(clash_ab).any()

    swapped_dict = {"team_a": match_dict["team_b"], "team_b": match_dict["team_a"]}
    clash_ba = pipeline.extract_macro_clash(swapped_dict)
    assert clash_ba.shape == (MACRO_CLASH_DIM,)

    # Mathematical domain invariant: clash(B, A) == -clash(A, B)
    np.testing.assert_allclose(clash_ba, -clash_ab, atol=1e-6)

    # Test batch v3
    v3_batch = pipeline.batch_extract_v3([match_dict, swapped_dict])
    assert "team_a" in v3_batch and v3_batch["team_a"].shape == (2, 5, ENRICHED_PLAYER_TENSOR_DIM)
    assert "team_b" in v3_batch and v3_batch["team_b"].shape == (2, 5, ENRICHED_PLAYER_TENSOR_DIM)
    assert "macro_clash" in v3_batch and v3_batch["macro_clash"].shape == (2, MACRO_CLASH_DIM)
    np.testing.assert_allclose(v3_batch["macro_clash"][1], -v3_batch["macro_clash"][0], atol=1e-6)
