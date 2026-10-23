"""Paired retrospective benchmark for extended player-prior consumption.

Use only trusted local pickle caches. Every run writes to a new directory.
Run once before a feature change and again after it with identical arguments.
No hyperparameter or threshold is selected using test outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.enriched_features import batch_extract_enriched_features
from core.experiment import read_manifest, restore_splits
from core.player_priors import PlayerPriorsTracker


def _metrics(y, p):
    return {
        'count': len(y), 'coverage': 1.0,
        'accuracy': float(accuracy_score(y, p >= .5)),
        'log_loss': float(log_loss(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
    }


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(args.splits_cache)
    raw = cache.read_bytes()
    splits = pickle.loads(raw)
    manifest = read_manifest(args.manifest)
    splits = restore_splits([r for rows in splits.values() for r in rows], manifest)
    print('Verified frozen splits:', {k: len(v) for k, v in splits.items()}, flush=True)
    # Manifest fingerprints omit enriched priors: hash the entire cache as well.
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'variant': args.variant, 'cache_sha256': hashlib.sha256(raw).hexdigest(),
        'split_sha256': manifest['split_sha256'],
        'source_sha256': hashlib.sha256((ROOT / 'core/player_priors.py').read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'feature_policy': '213 existing enriched columns; no target-encoded lane statistics',
        'selection_policy': 'fixed hyperparameters, fixed seeds, threshold 0.5, coverage 100%',
        'limitations': ['Previously inspected historical test; not prospective validation.',
                       'Player priors inherit the cache temporal-availability assumptions.',
                       'Candidate only; no serving checkpoint is replaced.'],
        'time_ranges': {k: [rows[0]['game_creation_time'], rows[-1]['game_creation_time']]
                        for k, rows in splits.items()},
        'seeds': {},
    }
    x, xs, y = {}, {}, {}
    snapshots = {}
    for name, rows in splits.items():
        features = [r['feature_json'] for r in rows]
        if args.variant == 'legacy_replay':
            # Exact old behavior for extended priors: replace with neutral six.
            # This replay is also checked against a real pre-patch run.
            features = [{**f, **{
                team: [{**p, 'priors': (PlayerPriorsTracker.get_default_priors()
                                      if len(p.get('priors', [])) != 6 else p['priors'])}
                       for p in f[team]] for team in ('team_a', 'team_b')
            }} for f in features]
        swapped = [{**f, 'team_a': f['team_b'], 'team_b': f['team_a']} for f in features]
        x[name] = batch_extract_enriched_features(features, mode='tabular')
        xs[name] = batch_extract_enriched_features(swapped, mode='tabular')
        y[name] = np.array([r['labels_json']['winner'] for r in rows], dtype=int)
        snapshots[f'{name}_y'] = y[name]
        snapshots[f'{name}_ids'] = np.array([r['match_id'] for r in rows])
        report.setdefault('feature_sha256', {})[name] = hashlib.sha256(x[name].tobytes()).hexdigest()
        print('Extracted', name, x[name].shape, flush=True)
    parameters = dict(categorical_features=list(range(10)), max_iter=200,
                      learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=30,
                      l2_regularization=1., early_stopping=False)
    report['parameters'] = parameters
    train_x = np.concatenate([x['train'], xs['train']])
    train_y = np.concatenate([y['train'], 1-y['train']])
    for seed in args.seeds:
        start = time.monotonic()
        with threadpool_limits(limits=4):
            model = HistGradientBoostingClassifier(**parameters, random_state=seed)
            model.fit(train_x, train_y)
            metrics = {}
            for name in splits:
                p = .5 * (model.predict_proba(x[name])[:, 1] +
                          1. - model.predict_proba(xs[name])[:, 1])
                ps = .5 * (model.predict_proba(xs[name])[:, 1] +
                           1. - model.predict_proba(x[name])[:, 1])
                np.testing.assert_allclose(p + ps, 1., rtol=0., atol=2e-15)
                metrics[name] = _metrics(y[name], p)
                snapshots[f'seed_{seed}_{name}_p'] = p
            metrics['elapsed_seconds'] = time.monotonic() - start
        report['seeds'][str(seed)] = metrics
        joblib.dump({'model': model, 'requires_team_swap_symmetrization': True,
                     'feature_policy': report['feature_policy'], 'seed': seed},
                    out / f'candidate_{seed}.joblib')
        print('Seed', seed, json.dumps(metrics), flush=True)
    np.savez_compressed(out / 'predictions.npz', **snapshots)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Saved', out.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-cache', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=('before', 'after', 'legacy_replay'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    run(parser.parse_args())
