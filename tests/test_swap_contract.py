"""Fixed-side composition swaps and within-team permutations are distinct contracts."""
import numpy as np
import pytest
import tensorflow as tf

from core.equivariant_model import build_equivariant_deepsets_model
from core.selective_classification import compute_conviction
from core.selective_infer import classify_match_conviction


TOLERANCE = 1e-5


@pytest.fixture(params=[14, 18, 46], ids=["base14", "timeline18", "spatial46"])
def teams(request):
    dim = request.param
    rng = np.random.default_rng(20261018 + dim)
    a = rng.uniform(0.1, 0.9, size=(3, 5, dim)).astype(np.float32)
    b = rng.uniform(0.1, 0.9, size=(3, 5, dim)).astype(np.float32)
    a[:, :, 0], b[:, :, 0] = np.arange(1, 6), np.arange(6, 11)
    # Unique roles make sorting a canonicalization of the same five players.
    a[:, :, 1] = b[:, :, 1] = np.arange(5)
    return a, b


@pytest.mark.parametrize("enable_side_bias", [False, True], ids=["neutral", "fixed_side_bias"])
def test_fresh_model_composition_swap_and_permutation_contract(teams, enable_side_bias):
    a, b = teams
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(20261018 + a.shape[-1])
    model = build_equivariant_deepsets_model(
        player_feature_dim=a.shape[-1], embed_dim=8, latent_dim=16,
        enable_side_bias=enable_side_bias,
    )
    if enable_side_bias:
        # Use a known nonzero bias; the numerical initialization is no invariant.
        model.get_layer("winner_side_bias").bias.assign([-0.25])

    def predict(x, y):
        return {k: v.numpy() for k, v in model({"team_a": x, "team_b": y}, training=False).items()}

    original = predict(a, b)
    permuted = predict(a[:, [3, 0, 4, 1, 2]], b[:, [2, 4, 1, 0, 3]])
    swapped = predict(b, a)
    mirror = predict(a, a)
    for head in ("draft_advantage", "winner"):
        np.testing.assert_allclose(original[head], permuted[head], atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(original["draft_advantage"], swapped["draft_advantage"][:, ::-1], atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(mirror["draft_advantage"], 0.5, atol=TOLERANCE, rtol=0)

    if not enable_side_bias:
        np.testing.assert_allclose(original["winner"], swapped["winner"][:, ::-1], atol=TOLERANCE, rtol=0)
        np.testing.assert_allclose(mirror["winner"], 0.5, atol=TOLERANCE, rtol=0)
        return

    bias_layer = model.get_layer("winner_side_bias")
    probe = tf.keras.Model(model.inputs, [model.get_layer("winner_draft_logit").output, bias_layer.output])
    draft_ab, match_ab = [v.numpy() for v in probe({"team_a": a, "team_b": b}, training=False)]
    draft_ba, match_ba = [v.numpy() for v in probe({"team_a": b, "team_b": a}, training=False)]
    np.testing.assert_allclose(draft_ab, -draft_ba, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(match_ab - draft_ab, -0.25, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(match_ba - draft_ba, -0.25, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(match_ab + match_ba, -0.5, atol=TOLERANCE, rtol=0)
    # Requiring complementarity here would erase the deliberately fixed side effect.
    assert np.max(np.abs(original["winner"] - swapped["winner"][:, ::-1])) > TOLERANCE
    assert np.all(mirror["winner"][:, 0] > 0.5)


@pytest.mark.parametrize("probability", [0.1, 0.42, 0.46, 0.5, 0.54, 0.58, 0.9])
def test_same_match_perspective_change_preserves_conviction_and_decision_tier(probability):
    opposite = 1.0 - probability
    np.testing.assert_allclose(compute_conviction(probability), compute_conviction(opposite), atol=1e-12, rtol=0)
    original = classify_match_conviction(probability)
    reversed_perspective = classify_match_conviction(opposite)
    assert original.conviction == reversed_perspective.conviction
    assert original.tier == reversed_perspective.tier
    assert original.is_actionable == reversed_perspective.is_actionable
    assert reversed_perspective.prediction == {
        "team_a": "team_b", "team_b": "team_a", "abstain": "abstain",
    }[original.prediction]
