"""Counterfactual credit contracts, including their team-relative sign convention."""
import json

import numpy as np
import pytest
import tensorflow as tf

from core.equivariant_model import build_equivariant_deepsets_model
from core.shapley_credit import compute_counterfactual_shapley


class LinearWinner:
    """Analytical oracle: each strength point adds .01 to its team's win chance."""

    def __call__(self, inputs, training=None):
        assert training is False, "Counterfactuals must disable dropout."
        a, b = inputs["team_a"], inputs["team_b"]
        # Role terms cancel only if masking keeps each player's role.
        p_b = 0.5 + 0.01 * tf.reduce_sum(b[:, :, 2] - a[:, :, 2], axis=1)
        p_b += 0.005 * tf.reduce_sum(b[:, :, 1] - a[:, :, 1], axis=1)
        return {"winner": tf.stack([1.0 - p_b, p_b], axis=1)}


def _linear_teams():
    a = np.zeros((1, 5, 14), dtype=np.float32)
    b = np.zeros_like(a)
    a[0, :, 1] = b[0, :, 1] = np.arange(5)
    a[0, :, 2] = np.arange(1, 6)
    b[0, :, 2] = np.arange(2, 7)
    return a, b


def test_counterfactual_values_and_inputs_are_not_mutated():
    a, b = _linear_teams()
    before_a, before_b = a.copy(), b.copy()
    result = compute_counterfactual_shapley(LinearWinner(), a, b)
    assert set(result) == {"draft_delta", "team_a_credits", "team_b_credits"}
    assert result["draft_delta"] == pytest.approx(0.05, abs=1e-7)
    # Plan convention: negative credits mean a player helps their own team.
    assert result["team_a_credits"] == pytest.approx([-.01, -.02, -.03, -.04, -.05], abs=1e-7)
    assert result["team_b_credits"] == pytest.approx([-.02, -.03, -.04, -.05, -.06], abs=1e-7)
    assert isinstance(result["draft_delta"], float)
    assert all(isinstance(x, float) for key in ("team_a_credits", "team_b_credits") for x in result[key])
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


def test_neutral_player_has_zero_credit_and_small_effects_are_preserved():
    a, b = _linear_teams()
    a[0, 2, 2:] = 0.0
    a[0, 3, 2] = 1e-4
    result = compute_counterfactual_shapley(LinearWinner(), a, b)
    assert result["team_a_credits"][2] == 0.0
    assert result["team_a_credits"][3] == pytest.approx(-1e-6, abs=1e-7)


@pytest.fixture(params=[(True, False), (False, False), (True, True)], scope="module")
def neural_case(request):
    evidential, macro = request.param
    tf.keras.utils.set_random_seed(604)
    model = build_equivariant_deepsets_model(
        player_feature_dim=14, embed_dim=8, latent_dim=16,
        use_evidential=evidential, include_macro_clash=macro, dropout_rate=0.3,
    )
    rng = np.random.default_rng(602)
    a = rng.uniform(.1, .9, (1, 5, 14)).astype(np.float32)
    b = rng.uniform(.1, .9, (1, 5, 14)).astype(np.float32)
    a[0, :, 0], b[0, :, 0] = np.arange(1, 6), np.arange(6, 11)
    a[0, :, 1] = b[0, :, 1] = np.arange(5)
    return model, tf.constant(a), tf.constant(b)


def test_real_model_credit_swap_and_player_permutation(neural_case):
    model, a, b = neural_case
    result = compute_counterfactual_shapley(model, a, b)
    swapped = compute_counterfactual_shapley(model, b, a)
    assert result["draft_delta"] == pytest.approx(-swapped["draft_delta"], abs=2e-6)
    assert result["team_a_credits"] == pytest.approx(swapped["team_b_credits"], abs=2e-6)
    assert result["team_b_credits"] == pytest.approx(swapped["team_a_credits"], abs=2e-6)
    assert np.any(np.abs(result["team_a_credits"]) > 1e-6)
    assert np.any(np.abs(result["team_b_credits"]) > 1e-6)

    perm_a, perm_b = [3, 0, 4, 1, 2], [2, 4, 1, 0, 3]
    permuted = compute_counterfactual_shapley(
        model, tf.gather(a, perm_a, axis=1), tf.gather(b, perm_b, axis=1),
    )
    assert permuted["draft_delta"] == pytest.approx(result["draft_delta"], abs=2e-6)
    assert permuted["team_a_credits"] == pytest.approx(np.array(result["team_a_credits"])[perm_a], abs=2e-6)
    assert permuted["team_b_credits"] == pytest.approx(np.array(result["team_b_credits"])[perm_b], abs=2e-6)

    repeated = compute_counterfactual_shapley(model, a, b)
    for key in result:
        np.testing.assert_allclose(repeated[key], result[key], atol=1e-7)


@pytest.mark.parametrize("shape_a,shape_b", [
    ((2, 5, 14), (2, 5, 14)),  # Scalar API must not silently drop matches.
    ((0, 5, 14), (0, 5, 14)),
    ((5, 14), (5, 14)),
    ((1, 4, 14), (1, 5, 14)),
    ((1, 5, 14), (1, 5, 8)),
    ((1, 5, 1), (1, 5, 1)),
])
def test_rejects_invalid_team_shapes(shape_a, shape_b):
    with pytest.raises(ValueError, match="shape"):
        compute_counterfactual_shapley(LinearWinner(), np.zeros(shape_a), np.zeros(shape_b))


def _service_match():
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    return {
        "team_a": [{"champion": champion, "role": role} for champion, role in zip(
            ["Aatrox", "LeeSin", "Ahri", "Jinx", "Leona"], roles,
        )],
        "team_b": [{"champion": champion, "role": role} for champion, role in zip(
            ["Ornn", "Viego", "Orianna", "Caitlyn", "Nautilus"], roles,
        )],
    }


def test_prediction_service_exposes_raw_diagnostics_and_keeps_calibration(neural_case):
    from core.features.pipeline import DraftFeaturePipeline
    from core.prediction_service import PredictionService

    model, _, _ = neural_case
    pipeline = DraftFeaturePipeline()
    service = PredictionService(feature_pipeline=pipeline)
    service._model = model
    service._temperature = 2.0
    service._is_calibrated = True
    match = _service_match()
    inputs = pipeline.batch_extract_tensors([match])
    if any(t.name == "macro_clash" for t in model.inputs):
        inputs["macro_clash"] = pipeline.batch_extract_macro_clash([match])
    raw = model(inputs, training=False)
    p_b = float(raw["winner"][0, 1])
    expected_calibrated = np.sqrt(p_b) / (np.sqrt(p_b) + np.sqrt(1.0 - p_b))

    result = service.predict_match(match, action_threshold=0.06)
    assert result["success"] is True
    assert result["winner"]["team_b"] == round(expected_calibrated, 4)
    assert result["selective_decision"]["threshold_applied"] == 0.06
    assert result["shapley_credits"]["draft_delta"] == pytest.approx(p_b - 0.5, abs=2e-6)
    assert len(result["shapley_credits"]["team_a_credits"]) == 5
    assert len(result["shapley_credits"]["team_b_credits"]) == 5
    if "winner_uncertainty" in raw:
        assert result["epistemic_uncertainty"] == pytest.approx(float(raw["winner_uncertainty"][0, 0]))
    else:
        assert result["epistemic_uncertainty"] is None
    json.dumps(result, allow_nan=False)

    swapped = service.predict_match({"team_a": match["team_b"], "team_b": match["team_a"]})
    assert result["winner"]["team_b"] == pytest.approx(1.0 - swapped["winner"]["team_b"], abs=1e-4)
    assert result["shapley_credits"]["team_a_credits"] == pytest.approx(
        swapped["shapley_credits"]["team_b_credits"], abs=2e-6,
    )
    assert result["epistemic_uncertainty"] == pytest.approx(swapped["epistemic_uncertainty"])


def test_prediction_service_loads_evidential_checkpoint(tmp_path):
    from core.prediction_service import PredictionService

    model = build_equivariant_deepsets_model(embed_dim=8, latent_dim=16)
    checkpoint = tmp_path / "evidential.keras"
    model.save(checkpoint)
    service = PredictionService(checkpoint_path=checkpoint)
    assert service.is_ready()
    result = service.predict_match(_service_match())
    assert result["success"] is True
    assert 0.0 < result["epistemic_uncertainty"] <= 1.0
    assert len(result["shapley_credits"]["team_b_credits"]) == 5
