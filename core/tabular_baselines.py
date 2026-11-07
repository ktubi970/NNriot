"""Logistic-regression and LightGBM baselines on the aligned 14D/18D tensors.

They read exactly the arrays the DeepSets trainers read, so comparing them with
DeepSets isolates the architecture. Hyperparameters and temperatures are chosen
on validation only; no function here accepts test data for fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from core.calibration import apply_temperature_to_probabilities, fit_temperature
from core.champion_registry import NUM_CHAMPIONS

KIND = 'nnriot.tabular_baseline.v1'
LOGISTIC_C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LIGHTGBM_PARAMS = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 31,
                   'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                   'bagging_freq': 1, 'lambda_l2': 1.0, 'verbose': -1, 'deterministic': True,
                   'force_row_wise': True, 'num_threads': 4}
LIGHTGBM_MAX_ROUNDS = 3000
LIGHTGBM_EARLY_STOPPING = 100


def tabular_features(inputs) -> np.ndarray:
    """Red-minus-Blue role differences, both team means, macro clash and a signed champion bag.

    Columns 0-1 of each player (champion and role IDs) are categorical; the rest are numeric.
    """
    a, b = (np.asarray(inputs[key], dtype=np.float32) for key in ('team_a', 'team_b'))
    macro = np.asarray(inputs['macro_clash'], dtype=np.float32)
    if a.ndim != 3 or a.shape != b.shape or a.shape[1] != 5 or macro.shape != (len(a), 15):
        raise ValueError('invalid aligned tensor shapes')
    for team in (a, b):
        if not np.array_equal(team[..., 1], np.broadcast_to(np.arange(5), team.shape[:2])):
            raise ValueError('teams must use the canonical role order')
        ids = team[..., 0]
        if not np.array_equal(ids, np.round(ids)) or ids.min() < 0 or ids.max() >= NUM_CHAMPIONS:
            raise ValueError('champion IDs outside the registry')
    numeric_a, numeric_b = a[..., 2:], b[..., 2:]
    champions = np.zeros((len(a), NUM_CHAMPIONS), dtype=np.float32)
    rows = np.repeat(np.arange(len(a)), 5)
    np.add.at(champions, (rows, b[..., 0].astype(int).ravel()), 1.0)
    np.add.at(champions, (rows, a[..., 0].astype(int).ravel()), -1.0)
    return np.concatenate([(numeric_b - numeric_a).reshape(len(a), -1), numeric_a.mean(axis=1),
                           numeric_b.mean(axis=1), macro, champions], axis=1)


def _log_loss(labels, probabilities) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _predict_raw(model, features) -> np.ndarray:
    if features.shape[1] != model['feature_count']:
        raise ValueError('tabular feature count differs from the fitted baseline')
    if model['family'] == 'logistic':
        z = (features.astype(np.float64) - np.asarray(model['mean'])) / np.asarray(model['scale'])
        return expit(z @ np.asarray(model['coef']) + model['intercept'])
    import lightgbm as lgb
    return lgb.Booster(model_str=model['booster']).predict(features)


def _finish(model, validation, y_validation) -> dict:
    raw = _predict_raw(model, tabular_features(validation))
    model['validation_log_loss_raw'] = _log_loss(y_validation, raw)
    model['calibration'] = {**fit_temperature(raw, np.asarray(y_validation)), 'method': 'temperature'}
    return model


def _header(family, train, y_train, features, seed) -> dict:
    return {'kind': KIND, 'family': family, 'input_dim': int(np.shape(train['team_a'])[-1]),
            'feature_count': int(features.shape[1]), 'seed': seed,
            'train_count': len(y_train), 'train_positive_rate': float(np.mean(y_train))}


def fit_logistic(train, y_train, validation, y_validation, *, seed) -> dict:
    """L2 logistic regression on standardized features; C chosen by validation log loss."""
    from sklearn.linear_model import LogisticRegression
    xt, xv = tabular_features(train), tabular_features(validation)
    mean, scale = xt.mean(axis=0), xt.std(axis=0)
    scale[scale == 0] = 1.0
    zt, zv = (xt - mean) / scale, (xv - mean) / scale
    trials = []
    for c in LOGISTIC_C_GRID:
        clf = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit(zt, y_train)
        trials.append({'C': c, 'validation_log_loss': _log_loss(y_validation, clf.predict_proba(zv)[:, 1]),
                       'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0])})
    best = min(trials, key=lambda t: t['validation_log_loss'])
    model = {**_header('logistic', train, y_train, xt, seed), 'mean': mean.tolist(), 'scale': scale.tolist(),
             'coef': best['coef'], 'intercept': best['intercept'],
             'selection': {'partition': 'validation', 'metric': 'log_loss', 'chosen_C': best['C'],
                           'boundary_hit': best['C'] in (LOGISTIC_C_GRID[0], LOGISTIC_C_GRID[-1]),
                           'trials': [{k: t[k] for k in ('C', 'validation_log_loss')} for t in trials]}}
    return _finish(model, validation, y_validation)


def fit_lightgbm(train, y_train, validation, y_validation, *, seed) -> dict:
    """Gradient-boosted trees with fixed parameters; rounds chosen by validation early stopping."""
    import lightgbm as lgb
    xt, xv = tabular_features(train), tabular_features(validation)
    booster = lgb.train({**LIGHTGBM_PARAMS, 'seed': seed}, lgb.Dataset(xt, y_train),
                        num_boost_round=LIGHTGBM_MAX_ROUNDS,
                        valid_sets=[lgb.Dataset(xv, y_validation)],
                        callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING, verbose=False)])
    model = {**_header('lightgbm', train, y_train, xt, seed),
             'booster': booster.model_to_string(num_iteration=booster.best_iteration),
             'selection': {'partition': 'validation', 'metric': 'binary_logloss',
                           'best_iteration': int(booster.best_iteration),
                           'max_rounds': LIGHTGBM_MAX_ROUNDS, 'early_stopping': LIGHTGBM_EARLY_STOPPING,
                           'params': LIGHTGBM_PARAMS}}
    return _finish(model, validation, y_validation)


def predict_baseline(model, inputs, *, calibrated=True) -> np.ndarray:
    """P(team B / Red wins) from a fitted baseline, temperature-scaled unless calibrated=False."""
    if np.shape(inputs['team_a'])[-1] != model['input_dim']:
        raise ValueError(f"input dimension {np.shape(inputs['team_a'])[-1]} differs from baseline {model['input_dim']}")
    raw = _predict_raw(model, tabular_features(inputs))
    return apply_temperature_to_probabilities(raw, model['calibration']['temperature']) if calibrated else raw


def save_baseline(model, path) -> None:
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(model, handle, allow_nan=False)


def load_baseline(path) -> dict:
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    if model.get('kind') != KIND or model.get('family') not in ('logistic', 'lightgbm'):
        raise ValueError('unsupported tabular baseline artifact')
    return model
