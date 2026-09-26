"""Tests for Selective Inference Wrapper and Decision Tiers."""
from __future__ import annotations

import math
import numpy as np
import pytest

from core.selective_infer import (
    DEFAULT_ACTION_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
    PRED_ABSTAIN,
    PRED_TEAM_A,
    PRED_TEAM_B,
    TIER_ACTIONABLE,
    TIER_COINFLIP,
    TIER_HIGH_CONFIDENCE,
    SelectiveInferenceResult,
    classify_batch_conviction,
    classify_match_conviction,
    format_batch_summary,
    predict_selective,
)


def test_classify_match_conviction_tiers():
    """Verify tier assignment and actionability based on conviction thresholds."""
    res_high = classify_match_conviction(0.62)
    assert res_high.prediction == "team_b"
    assert res_high.tier == "HIGH_CONFIDENCE"
    assert res_high.is_actionable is True
    assert pytest.approx(res_high.conviction, abs=1e-5) == 0.12
    assert pytest.approx(res_high.p_team_b, abs=1e-5) == 0.62
    assert pytest.approx(res_high.p_team_a, abs=1e-5) == 0.38

    res_actionable = classify_match_conviction(0.55)
    assert res_actionable.prediction == "team_b"
    assert res_actionable.tier == "ACTIONABLE"
    assert res_actionable.is_actionable is True
    assert pytest.approx(res_actionable.conviction, abs=1e-5) == 0.05

    res_coinflip = classify_match_conviction(0.51)
    assert res_coinflip.prediction == "abstain"
    assert res_coinflip.tier == "COINFLIP"
    assert res_coinflip.is_actionable is False
    assert pytest.approx(res_coinflip.conviction, abs=1e-5) == 0.01


def test_classify_match_conviction_team_a_direction():
    """When P(team_b) < 0.5, prediction should be team_a with identical conviction."""
    res_high_a = classify_match_conviction(0.38)
    assert res_high_a.prediction == "team_a"
    assert res_high_a.tier == "HIGH_CONFIDENCE"
    assert res_high_a.is_actionable is True
    assert pytest.approx(res_high_a.conviction, abs=1e-5) == 0.12
    assert pytest.approx(res_high_a.p_team_a, abs=1e-5) == 0.62
    assert pytest.approx(res_high_a.p_team_b, abs=1e-5) == 0.38

    res_actionable_a = classify_match_conviction(0.45)
    assert res_actionable_a.prediction == "team_a"
    assert res_actionable_a.tier == "ACTIONABLE"
    assert res_actionable_a.is_actionable is True
    assert pytest.approx(res_actionable_a.conviction, abs=1e-5) == 0.05

    res_coinflip_a = classify_match_conviction(0.49)
    assert res_coinflip_a.prediction == "abstain"
    assert res_coinflip_a.tier == "COINFLIP"
    assert res_coinflip_a.is_actionable is False
    assert pytest.approx(res_coinflip_a.conviction, abs=1e-5) == 0.01


def test_classify_match_conviction_strict_anti_symmetry():
    """Swapping P(team_b) -> 1 - P(team_b) must keep conviction and tier identical, flipping team prediction."""
    test_probs = [0.0, 0.10, 0.25, 0.38, 0.42, 0.46, 0.49, 0.50, 0.51, 0.54, 0.58, 0.62, 0.75, 0.90, 1.0]

    for p in test_probs:
        res_forward = classify_match_conviction(p)
        res_swapped = classify_match_conviction(1.0 - p)

        assert pytest.approx(res_forward.conviction, abs=1e-5) == res_swapped.conviction
        assert res_forward.tier == res_swapped.tier
        assert res_forward.is_actionable == res_swapped.is_actionable

        if res_forward.prediction == PRED_TEAM_B:
            assert res_swapped.prediction == PRED_TEAM_A
        elif res_forward.prediction == PRED_TEAM_A:
            assert res_swapped.prediction == PRED_TEAM_B
        else:
            assert res_forward.prediction == PRED_ABSTAIN
            assert res_swapped.prediction == PRED_ABSTAIN


def test_classify_match_conviction_boundary_values():
    """Verify behavior exactly at threshold boundaries."""
    # Actionable boundary: 0.04 conviction
    res_b_at_action = classify_match_conviction(0.54)
    assert res_b_at_action.tier == TIER_ACTIONABLE
    assert res_b_at_action.is_actionable is True
    assert res_b_at_action.prediction == PRED_TEAM_B

    res_a_at_action = classify_match_conviction(0.46)
    assert res_a_at_action.tier == TIER_ACTIONABLE
    assert res_a_at_action.is_actionable is True
    assert res_a_at_action.prediction == PRED_TEAM_A

    # High confidence boundary: 0.08 conviction
    res_b_at_high = classify_match_conviction(0.58)
    assert res_b_at_high.tier == TIER_HIGH_CONFIDENCE
    assert res_b_at_high.is_actionable is True
    assert res_b_at_high.prediction == PRED_TEAM_B

    res_a_at_high = classify_match_conviction(0.42)
    assert res_a_at_high.tier == TIER_HIGH_CONFIDENCE
    assert res_a_at_high.is_actionable is True
    assert res_a_at_high.prediction == PRED_TEAM_A

    # Exactly 0.50
    res_center = classify_match_conviction(0.50)
    assert res_center.conviction == 0.0
    assert res_center.tier == TIER_COINFLIP
    assert res_center.is_actionable is False
    assert res_center.prediction == PRED_ABSTAIN


def test_selective_inference_result_namedtuple_properties():
    """Verify NamedTuple unpacking, attribute access, and serialization."""
    res = classify_match_conviction(0.65)
    assert isinstance(res, tuple)
    assert res.p_team_a == pytest.approx(0.35, abs=1e-5)
    assert res.p_team_b == pytest.approx(0.65, abs=1e-5)
    assert res.conviction == pytest.approx(0.15, abs=1e-5)
    assert res.prediction == "team_b"
    assert res.tier == "HIGH_CONFIDENCE"
    assert res.is_actionable is True

    # NamedTuple unpacking
    p_a, p_b, conv, pred, tier, actionable = res
    assert p_a == pytest.approx(0.35, abs=1e-5)
    assert p_b == pytest.approx(0.65, abs=1e-5)

    # Dictionary serialization
    res_dict = res.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["p_team_a"] == pytest.approx(0.35, abs=1e-5)
    assert res_dict["p_team_b"] == pytest.approx(0.65, abs=1e-5)
    assert res_dict["conviction"] == pytest.approx(0.15, abs=1e-5)
    assert res_dict["prediction"] == "team_b"
    assert res_dict["tier"] == "HIGH_CONFIDENCE"
    assert res_dict["is_actionable"] is True


def test_classify_match_conviction_custom_thresholds():
    """Custom thresholds should properly alter decision boundaries."""
    # Stricter thresholds: action >= 0.06, high >= 0.12
    res_055 = classify_match_conviction(0.55, action_threshold=0.06, high_threshold=0.12)
    assert res_055.tier == TIER_COINFLIP
    assert res_055.is_actionable is False

    res_057 = classify_match_conviction(0.57, action_threshold=0.06, high_threshold=0.12)
    assert res_057.tier == TIER_ACTIONABLE
    assert res_057.is_actionable is True

    res_063 = classify_match_conviction(0.63, action_threshold=0.06, high_threshold=0.12)
    assert res_063.tier == TIER_HIGH_CONFIDENCE
    assert res_063.is_actionable is True


def test_classify_match_conviction_input_validation():
    """Verify ValueError is raised on invalid inputs."""
    with pytest.raises(ValueError, match="NaN"):
        classify_match_conviction(float("nan"))

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        classify_match_conviction(1.2)

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        classify_match_conviction(-0.1)

    with pytest.raises(ValueError, match="action_threshold cannot exceed high_threshold"):
        classify_match_conviction(0.6, action_threshold=0.10, high_threshold=0.05)

    with pytest.raises(ValueError, match="Thresholds must be non-negative"):
        classify_match_conviction(0.6, action_threshold=-0.01)


def test_batch_inference_formatting():
    """Verify batch classification and summary statistics."""
    probs = [0.62, 0.55, 0.51, 0.38, 0.45, 0.49]
    results = classify_batch_conviction(probs)

    assert len(results) == 6
    assert [r.prediction for r in results] == [
        "team_b",
        "team_b",
        "abstain",
        "team_a",
        "team_a",
        "abstain",
    ]
    assert [r.tier for r in results] == [
        "HIGH_CONFIDENCE",
        "ACTIONABLE",
        "COINFLIP",
        "HIGH_CONFIDENCE",
        "ACTIONABLE",
        "COINFLIP",
    ]
    assert [r.is_actionable for r in results] == [True, True, False, True, True, False]

    summary = format_batch_summary(results)
    assert summary["total_matches"] == 6
    assert summary["actionable_count"] == 4
    assert pytest.approx(summary["actionable_coverage"], abs=1e-5) == 4 / 6
    assert summary["high_confidence_count"] == 2
    assert summary["actionable_tier_count"] == 2
    assert summary["coinflip_count"] == 2
    assert summary["team_a_picks"] == 2
    assert summary["team_b_picks"] == 2
    assert summary["abstain_picks"] == 2


def test_predict_selective_with_callable_and_match_dict():
    """Verify predict_selective end-to-end with match input dict."""
    sample_match = {
        "team_a": [
            {"champion": "Aatrox", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP"},
            {"champion": "Sejuani", "role": "JUNGLE"},
            {"champion": "Orianna", "role": "MIDDLE"},
            {"champion": "Ezreal", "role": "BOTTOM"},
            {"champion": "Nami", "role": "UTILITY"},
        ],
    }

    # Callable mock predictor returning winner output
    def mock_model(inputs):
        # Predict 62% for team B
        return {"winner": np.array([[0.38, 0.62]])}

    result = predict_selective(mock_model, sample_match)
    assert isinstance(result, SelectiveInferenceResult)
    assert result.prediction == "team_b"
    assert result.tier == "HIGH_CONFIDENCE"
    assert result.is_actionable is True
    assert pytest.approx(result.conviction, abs=1e-5) == 0.12

    # Swapped match: swap team_a and team_b in match dict
    swapped_match = {
        "team_a": sample_match["team_b"],
        "team_b": sample_match["team_a"],
    }

    def mock_swapped_model(inputs):
        # Predict 38% for team B (which is team A of original match)
        return {"winner": np.array([[0.62, 0.38]])}

    swapped_result = predict_selective(mock_swapped_model, swapped_match)
    assert swapped_result.prediction == "team_a"
    assert swapped_result.tier == "HIGH_CONFIDENCE"
    assert swapped_result.is_actionable is True
    assert pytest.approx(swapped_result.conviction, abs=1e-5) == 0.12


def test_selective_infer_wrapper():
    """Verify SelectiveInferWrapper class with predict, predict_batch, and summarize_batch."""
    from core.selective_infer import SelectiveInferWrapper

    wrapper = SelectiveInferWrapper(
        model_or_checkpoint=lambda m: {"winner": np.array([[0.45, 0.55]])},
        action_threshold=0.04,
        high_threshold=0.08,
    )
    match_dict = {
        "team_a": [{"champion": "Ahri", "role": "MIDDLE"}],
        "team_b": [{"champion": "Zed", "role": "MIDDLE"}],
    }
    res = wrapper.predict(match_dict)
    assert res.prediction == "team_b"
    assert res.tier == "ACTIONABLE"
    assert res.is_actionable is True

    batch_res = wrapper.predict_batch([match_dict, match_dict])
    assert len(batch_res) == 2
    assert batch_res[0].prediction == "team_b"

    summary = wrapper.summarize_batch([match_dict, match_dict])
    assert summary["total_matches"] == 2
    assert summary["actionable_count"] == 2
    assert summary["actionable_coverage"] == 1.0


def test_predict_selective_with_real_equivariant_deepsets_model():
    """Verify end-to-end selective inference with a real Equivariant DeepSets model."""
    from core.equivariant_model import build_equivariant_deepsets_model
    from core.selective_infer import SelectiveInferWrapper

    model = build_equivariant_deepsets_model(player_feature_dim=8)
    wrapper = SelectiveInferWrapper(model)

    match_dict = {
        "team_a": [
            {"champion": "Aatrox", "role": "TOP"},
            {"champion": "LeeSin", "role": "JUNGLE"},
            {"champion": "Ahri", "role": "MIDDLE"},
            {"champion": "Jinx", "role": "BOTTOM"},
            {"champion": "Thresh", "role": "UTILITY"},
        ],
        "team_b": [
            {"champion": "Malphite", "role": "TOP"},
            {"champion": "Sejuani", "role": "JUNGLE"},
            {"champion": "Orianna", "role": "MIDDLE"},
            {"champion": "Ezreal", "role": "BOTTOM"},
            {"champion": "Nami", "role": "UTILITY"},
        ],
    }
    swapped_match = {
        "team_a": match_dict["team_b"],
        "team_b": match_dict["team_a"],
    }

    res = wrapper.predict(match_dict)
    res_swapped = wrapper.predict(swapped_match)

    assert pytest.approx(res.conviction, abs=1e-5) == res_swapped.conviction
    assert res.tier == res_swapped.tier
    assert res.is_actionable == res_swapped.is_actionable

    # Equivariant model strictly guarantees anti-symmetry:
    # P(winner=1 | B, A) == 1 - P(winner=1 | A, B)
    assert pytest.approx(res.p_team_b, abs=1e-5) == res_swapped.p_team_a
    assert pytest.approx(res.p_team_a, abs=1e-5) == res_swapped.p_team_b

    if res.prediction == "team_b":
        assert res_swapped.prediction == "team_a"
    elif res.prediction == "team_a":
        assert res_swapped.prediction == "team_b"
    else:
        assert res.prediction == "abstain"
        assert res_swapped.prediction == "abstain"

