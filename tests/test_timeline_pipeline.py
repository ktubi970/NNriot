"""Tests for timeline player priors integration into DraftFeaturePipeline."""
import numpy as np
import pytest

from core.champion_registry import champion_to_id
from core.features.pipeline import (
    DraftFeaturePipeline,
    ENRICHED_PLAYER_TENSOR_DIM,
    ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
    TIMELINE_PLAYER_FEATURE_COUNT,
)
from core.player_priors import PlayerPriorsTracker


def test_pipeline_timeline_constants():
    assert TIMELINE_PLAYER_FEATURE_COUNT == 4
    assert ENRICHED_PLAYER_TIMELINE_TENSOR_DIM == 18
    assert ENRICHED_PLAYER_TIMELINE_TENSOR_DIM == ENRICHED_PLAYER_TENSOR_DIM + TIMELINE_PLAYER_FEATURE_COUNT


def test_extract_tensors_shapes_default_vs_timeline():
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

    # Default: (5, 14)
    t_a, t_b = pipeline.extract_tensors(match_dict)
    assert t_a.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)
    assert t_b.shape == (5, ENRICHED_PLAYER_TENSOR_DIM)

    # Timeline enabled: (5, 18)
    t_a_tl, t_b_tl = pipeline.extract_tensors(match_dict, include_timeline=True)
    assert t_a_tl.shape == (5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM)
    assert t_b_tl.shape == (5, ENRICHED_PLAYER_TIMELINE_TENSOR_DIM)

    # Timeline + Spatial: (5, 50)
    t_a_tls, t_b_tls = pipeline.extract_tensors(match_dict, include_spatial=True, include_timeline=True)
    assert t_a_tls.shape == (5, 50)
    assert t_b_tls.shape == (5, 50)


def test_extract_tensors_antisymmetry_with_timeline():
    pipeline = DraftFeaturePipeline()
    match_dict = {
        "team_a": [
            {"champion": "Darius", "role": "TOP", "priors": [0.55] * 14},
            {"champion": "LeeSin", "role": "JUNGLE", "priors": [0.52] * 14},
            {"champion": "Ahri", "role": "MIDDLE", "priors": [0.58] * 14},
            {"champion": "Jinx", "role": "BOTTOM", "priors": [0.60] * 14},
            {"champion": "Thresh", "role": "UTILITY", "priors": [0.50] * 14},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP", "priors": [0.45] * 14},
            {"champion": "Amumu", "role": "JUNGLE", "priors": [0.48] * 14},
            {"champion": "Syndra", "role": "MIDDLE", "priors": [0.42] * 14},
            {"champion": "Caitlyn", "role": "BOTTOM", "priors": [0.40] * 14},
            {"champion": "Leona", "role": "UTILITY", "priors": [0.50] * 14},
        ],
    }

    t_a, t_b = pipeline.extract_tensors(match_dict, include_timeline=True)
    swapped_dict = {"team_a": match_dict["team_b"], "team_b": match_dict["team_a"]}
    t_a_swapped, t_b_swapped = pipeline.extract_tensors(swapped_dict, include_timeline=True)

    # Team swap exact invariance/anti-symmetry
    np.testing.assert_allclose(t_a_swapped, t_b, atol=1e-6)
    np.testing.assert_allclose(t_b_swapped, t_a, atol=1e-6)

    # Lane matchup delta at index 16 must be anti-symmetric
    np.testing.assert_allclose(t_a[:, 16], -t_b[:, 16], atol=1e-6)


def test_extract_tensors_timeline_layout():
    pipeline = DraftFeaturePipeline()
    priors_14 = [
        0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60,  # 10 base priors
        0.25, -0.30, 0.40, 0.80,                                      # 4 timeline priors
    ]
    match_dict = {
        "team_a": [
            {"champion": "Darius", "role": "TOP", "priors": priors_14},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP", "priors": priors_14},
        ],
    }

    t_a, t_b = pipeline.extract_tensors(match_dict, include_timeline=True)

    # Slot 0 layout check:
    # 0: cid
    assert t_a[0, 0] == float(champion_to_id("Darius"))
    assert t_b[0, 0] == float(champion_to_id("Malphite"))
    # 1: slot
    assert t_a[0, 1] == 0.0
    assert t_b[0, 1] == 0.0
    # 2..11: 10 base priors
    np.testing.assert_allclose(t_a[0, 2:12], priors_14[:10], atol=1e-6)
    # 12..15: 4 timeline priors
    np.testing.assert_allclose(t_a[0, 12:16], [0.25, -0.30, 0.40, 0.80], atol=1e-6)
    # 16: lane matchup delta
    assert isinstance(float(t_a[0, 16]), float)
    # 17: champ base wr
    assert 0.0 <= t_a[0, 17] <= 1.0


def test_extract_tensors_precomputed_priors_handling():
    pipeline = DraftFeaturePipeline()

    # Case 1: Participant has 10 priors and include_timeline=True -> appends [0.0, 0.0, 0.0, 0.10]
    priors_10 = [0.55, 0.52, 1.2, 0.50, 0.8, 2.0, 0.6, 0.7, 0.8, 0.4]
    match_10 = {
        "team_a": [{"champion": "Darius", "role": "TOP", "priors": priors_10}],
        "team_b": [{"champion": "Sion", "role": "TOP", "priors": priors_10}],
    }
    t_a_10, _ = pipeline.extract_tensors(match_10, include_timeline=True)
    np.testing.assert_allclose(t_a_10[0, 2:12], priors_10, atol=1e-6)
    np.testing.assert_allclose(t_a_10[0, 12:16], [0.0, 0.0, 0.0, 0.10], atol=1e-6)

    # Case 2: Participant has 14 priors and include_timeline=True -> uses them directly
    priors_14 = priors_10 + [0.45, -0.15, 0.35, 0.65]
    match_14 = {
        "team_a": [{"champion": "Darius", "role": "TOP", "priors": priors_14}],
        "team_b": [{"champion": "Sion", "role": "TOP", "priors": priors_14}],
    }
    t_a_14, _ = pipeline.extract_tensors(match_14, include_timeline=True)
    np.testing.assert_allclose(t_a_14[0, 2:12], priors_10, atol=1e-6)
    np.testing.assert_allclose(t_a_14[0, 12:16], [0.45, -0.15, 0.35, 0.65], atol=1e-6)

    # Case 3: Participant has 14 priors and include_timeline=False -> uses first 10, shape (5, 14)
    t_a_legacy, _ = pipeline.extract_tensors(match_14, include_timeline=False)
    assert t_a_legacy.shape == (5, 14)
    np.testing.assert_allclose(t_a_legacy[0, 2:12], priors_10, atol=1e-6)


def test_extract_tensors_with_player_tracker_timeline():
    tracker = PlayerPriorsTracker()
    puuid = "test_player_tl"
    tracker.update(
        puuid=puuid,
        champion="Darius",
        role="TOP",
        win=1,
        kills=6,
        deaths=1,
        assists=5,
        total_cs=180,
        gold_earned=12000,
        game_duration=1800,
    )
    tracker.record_timeline_metrics(
        puuid=puuid,
        gd15=1500.0,
        csd15=20.0,
        xpd15=1000.0,
        fbi10=1.0,
    )

    pipeline = DraftFeaturePipeline(player_tracker=tracker)
    match_dict = {
        "team_a": [{"puuid": puuid, "champion": "Darius", "role": "TOP"}],
        "team_b": [{"champion": "Malphite", "role": "TOP"}],
    }

    t_a, t_b = pipeline.extract_tensors(match_dict, include_timeline=True)
    expected_priors = tracker.get_player_priors(puuid, "Darius", "TOP", extended=True, include_timeline=True)

    np.testing.assert_allclose(t_a[0, 2:12], expected_priors[:10], atol=1e-6)
    np.testing.assert_allclose(t_a[0, 12:16], expected_priors[10:14], atol=1e-6)

    # Malphite in team B has no history -> default priors
    default_tl = PlayerPriorsTracker.get_default_priors(extended=True, include_timeline=True)
    np.testing.assert_allclose(t_b[0, 2:12], default_tl[:10], atol=1e-6)
    np.testing.assert_allclose(t_b[0, 12:16], default_tl[10:14], atol=1e-6)


def test_batch_extract_tensors_with_timeline():
    pipeline = DraftFeaturePipeline()
    match_dict = {
        "team_a": [{"champion": "Darius", "role": "TOP"}],
        "team_b": [{"champion": "Malphite", "role": "TOP"}],
    }

    res_default = pipeline.batch_extract_tensors([match_dict, match_dict])
    assert res_default["team_a"].shape == (2, 5, 14)
    assert res_default["team_b"].shape == (2, 5, 14)

    res_tl = pipeline.batch_extract_tensors([match_dict, match_dict], include_timeline=True)
    assert res_tl["team_a"].shape == (2, 5, 18)
    assert res_tl["team_b"].shape == (2, 5, 18)


def test_pipeline_handles_malformed_and_empty_with_timeline():
    pipeline = DraftFeaturePipeline()

    t_a, t_b = pipeline.extract_tensors(None, include_timeline=True)
    assert t_a.shape == (5, 18)
    assert t_b.shape == (5, 18)
    assert not np.isnan(t_a).any()
    assert not np.isnan(t_b).any()

    t_a_empty, t_b_empty = pipeline.extract_tensors({}, include_timeline=True)
    assert t_a_empty.shape == (5, 18)
    assert t_b_empty.shape == (5, 18)
    assert not np.isnan(t_a_empty).any()
    assert not np.isnan(t_b_empty).any()
