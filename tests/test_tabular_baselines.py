import numpy as np
import pytest

from core.champion_registry import NUM_CHAMPIONS
from core.tabular_baselines import (
    fit_lightgbm, fit_logistic, load_baseline, predict_baseline, save_baseline, tabular_features,
)


def synthetic(n, dim, seed=0, signal=1.5):
    """Canonical role-ordered teams whose Red win depends on smooth_wr and one champion."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.3, 0.7, (n, 5, dim)).astype(np.float32)
    b = rng.uniform(0.3, 0.7, (n, 5, dim)).astype(np.float32)
    for team in (a, b):
        team[..., 0] = rng.integers(1, 20, (n, 5))
        team[..., 1] = np.arange(5)
    macro = rng.normal(0, 1, (n, 15)).astype(np.float32)
    logit = signal * 10 * (b[..., 2].mean(1) - a[..., 2].mean(1)) + 0.8 * ((b[..., 0] == 7).sum(1) - (a[..., 0] == 7).sum(1))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return {'team_a': a, 'team_b': b, 'macro_clash': macro}, y


@pytest.mark.parametrize('dim', [14, 18])
def test_features_are_signed_role_differences_levels_macro_and_champions(dim):
    x, _ = synthetic(6, dim)
    features = tabular_features(x)
    numeric = dim - 2
    assert features.shape == (6, 5 * numeric + 2 * numeric + 15 + NUM_CHAMPIONS)
    np.testing.assert_allclose(features[:, :numeric], x['team_b'][:, 0, 2:] - x['team_a'][:, 0, 2:])
    champions = features[:, -NUM_CHAMPIONS:]
    assert champions.sum(axis=1).tolist() == [0.0] * 6
    for i in range(6):
        expected = np.zeros(NUM_CHAMPIONS)
        np.add.at(expected, x['team_b'][i, :, 0].astype(int), 1)
        np.add.at(expected, x['team_a'][i, :, 0].astype(int), -1)
        np.testing.assert_array_equal(champions[i], expected)


def test_composition_swap_negates_signed_features_and_exchanges_levels():
    x, _ = synthetic(8, 18)
    swapped = {'team_a': x['team_b'], 'team_b': x['team_a'], 'macro_clash': -x['macro_clash']}
    f, g = tabular_features(x), tabular_features(swapped)
    numeric = 16
    signed = np.r_[0:5 * numeric, 7 * numeric:f.shape[1]]
    np.testing.assert_allclose(g[:, signed], -f[:, signed], atol=1e-6)
    levels = slice(5 * numeric, 7 * numeric)
    np.testing.assert_allclose(g[:, levels], np.roll(f[:, levels], numeric, axis=1))


def test_noncanonical_roles_and_unknown_champion_ids_are_rejected():
    x, _ = synthetic(4, 14)
    shuffled = {**x, 'team_a': x['team_a'][:, [1, 0, 2, 3, 4]]}
    with pytest.raises(ValueError, match='role order'):
        tabular_features(shuffled)
    x['team_b'][0, 0, 0] = NUM_CHAMPIONS
    with pytest.raises(ValueError, match='champion'):
        tabular_features(x)


@pytest.mark.parametrize('fit', [fit_logistic, fit_lightgbm], ids=['logistic', 'lightgbm'])
def test_baseline_learns_signal_selects_on_validation_and_round_trips(tmp_path, fit):
    train, y_train = synthetic(3000, 14, seed=1)
    validation, y_validation = synthetic(1000, 14, seed=2)
    model = fit(train, y_train, validation, y_validation, seed=42)
    assert model['selection']['partition'] == 'validation'
    assert model['calibration']['partition'] == 'validation'
    p = predict_baseline(model, validation)
    constant = np.full(len(y_validation), y_train.mean())
    def log_loss(prob):
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        return -np.mean(y_validation * np.log(prob) + (1 - y_validation) * np.log(1 - prob))
    assert log_loss(p) < log_loss(constant) - 0.02
    path = tmp_path / 'baseline.json'
    save_baseline(model, path)
    np.testing.assert_array_equal(predict_baseline(load_baseline(path), validation), p)


def test_predict_rejects_other_feature_dimension(tmp_path):
    train, y_train = synthetic(400, 14, seed=3)
    validation, y_validation = synthetic(200, 14, seed=4)
    model = fit_logistic(train, y_train, validation, y_validation, seed=0)
    other, _ = synthetic(10, 18)
    with pytest.raises(ValueError, match='dimension'):
        predict_baseline(model, other)
