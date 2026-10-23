"""Frozen integration regression against A3; no tuning or serving activation.

This research script never writes a serving checkpoint or modifies the database.
Use trusted local pickle caches only. Output directory must not already exist.
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
sys.path.insert(0, str(ROOT))

from core.champion_registry import NUM_CHAMPIONS
from core.experiment import (
    read_manifest, restore_splits, resolve_result_end_times,
    write_result_end_times_artifact, compute_enriched_dataset_sha256,
)
from core.features.ordered_matchup import extract_ordered_features
from core.features.matchup_matrix import LaneMatchupTracker
from core.features.pipeline import DraftFeaturePipeline


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference_features(reference, variant, feature_hashes):
    expected = reference['variants'][variant]['feature_sha256']
    if feature_hashes != expected:
        raise ValueError(f'A3 {variant} matrix SHA256 mismatch; scores must not be interpreted')


def compare_reference_predictions(current, reference, *, strict):
    current, reference = np.asarray(current), np.asarray(reference)
    if current.shape != reference.shape:
        raise ValueError('A3 prediction shape mismatch')
    result = {
        'max_absolute_difference': float(np.max(np.abs(current - reference))),
        'classification_differences': int(np.sum((current >= .5) != (reference >= .5))),
        'probability_atol': 1e-12,
        'strict_same_versions': strict,
    }
    if strict:
        np.testing.assert_array_equal(current >= .5, reference >= .5)
        # Double precision tree probabilities; this permits only roundoff, not model drift.
        np.testing.assert_allclose(current, reference, rtol=0., atol=1e-12)
    return result


def frozen_features(splits, greedy):
    tracker = LaneMatchupTracker()
    if greedy:
        for r in splits['train']:
            f = r['feature_json']
            tracker.update_match(f['team_a'], f['team_b'], r['labels_json']['winner'])
    pipeline = DraftFeaturePipeline(tracker)
    x, xs = {}, {}
    for name, rows in splits.items():
        features = [r['feature_json'] for r in rows]
        swapped = [{**f, 'team_a': f['team_b'], 'team_b': f['team_a']} for f in features]
        x[name] = pipeline.batch_extract_tabular(features)
        xs[name] = pipeline.batch_extract_tabular(swapped)
        print('Extracted', 'greedy' if greedy else 'neutral', name, x[name].shape, flush=True)
    return x, xs


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(args.splits_cache)
    splits = pickle.loads(cache.read_bytes())
    manifest = read_manifest(args.manifest)
    splits = restore_splits([r for rows in splits.values() for r in rows], manifest)
    print('Verified frozen splits:', {k: len(v) for k, v in splits.items()}, flush=True)
    ends, provenance = resolve_result_end_times(splits, db_path=args.database)
    end_diagnostics = provenance['diagnostics']
    end_sha = write_result_end_times_artifact(out / 'model.result_end_times.json', splits, ends, provenance)
    reference_dir = Path(getattr(args, 'reference_dir', ROOT / 'runs/accuracy_matchup_20260927'))
    reference = json.loads((reference_dir / 'metrics.json').read_text(encoding='utf-8'))
    with np.load(reference_dir / 'predictions.npz') as archive:
        reference_predictions = {k: archive[k] for k in archive.files}
    same_versions = reference['sklearn_version'] == sklearn.__version__ and reference['numpy_version'] == np.__version__
    if reference['cache_sha256'] != sha(cache) or reference['split_sha256'] != manifest['split_sha256']:
        raise ValueError('A3 cache or split SHA256 mismatch')
    parameters = dict(categorical_features=list(range(10)), max_iter=200,
                      learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=30,
                      l2_regularization=1., early_stopping=False, random_state=42)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'cache_sha256': sha(cache), 'split_sha256': manifest['split_sha256'],
        'end_times_sha256': end_sha,
        'enriched_dataset_sha256': compute_enriched_dataset_sha256(splits),
        'reference_dir': str(reference_dir.resolve()),
        'same_library_versions_as_A3': same_versions,
        'database_path': str(Path(args.database).resolve()),
        'end_time_diagnostics': end_diagnostics,
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'parameters': parameters, 'matchup_weight': 15.0, 'threads': 4,
        'selection_policy': 'Predefined three variants, fixed parameters/seed, no tuning, threshold 0.5, coverage 100%.',
        'feature_policy': 'Same 232 columns; variants change ONLY five lane target statistics.',
        'availability_policy': 'Only training results with explicit gameEndTimestamp < query game_creation_time.',
        'limitations': [
            'End timestamp is an optimistic proxy for result availability; ingestion times are unrecorded.',
            'Player priors are unchanged cache snapshots with their existing temporal assumptions.',
            'Previously inspected March-May 2026 cohort; no prospective or current-patch accuracy claim.',
            'No neural retraining or serving checkpoint replacement.',
            'Paired iid diagnostics do not account for repeated players, temporal dependence or prior test inspection.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            Path(__file__), ROOT / 'core/player_priors.py',
            ROOT / 'core/features/pipeline.py', ROOT / 'core/features/matchup_matrix.py',
            ROOT / 'core/features/ordered_matchup.py',
            ROOT / 'core/point_in_time_dataset.py', ROOT / 'core/experiment.py',
        ]},
        'time_ranges': {k: [min(r['game_creation_time'] for r in rows),
                            max(r['game_creation_time'] for r in rows)]
                        for k, rows in splits.items()},
        'variants': {},
    }
    dump(out / 'protocol.json', report)
    y = {k: np.asarray([r['labels_json']['winner'] for r in rows], dtype=int)
         for k, rows in splits.items()}
    snapshots = {f'{k}_y': v for k, v in y.items()}
    snapshots.update({f'{k}_ids': np.asarray([r['match_id'] for r in rows])
                      for k, rows in splits.items()})
    for name in splits:
        np.testing.assert_array_equal(snapshots[f'{name}_ids'], reference_predictions[f'{name}_ids'])
        np.testing.assert_array_equal(y[name], reference_predictions[f'{name}_y'])
    invariant_hashes = {}
    lane_start = 10 + NUM_CHAMPIONS + 33
    lane_cols = slice(lane_start, lane_start + 5)
    for variant in ('greedy', 'ordered', 'neutral'):
        start = time.monotonic()
        if variant == 'ordered':
            x, xs, diagnostics = extract_ordered_features(splits, ends)
        else:
            x, xs = frozen_features(splits, greedy=variant == 'greedy')
            diagnostics = {}
        entry = {'diagnostics': diagnostics, 'feature_sha256': {}, 'swapped_feature_sha256': {}, 'metrics': {}, 'reference_comparison': {}}
        for k in splits:
            assert x[k].shape == xs[k].shape == (len(y[k]), lane_start + 19)
            assert np.isfinite(x[k]).all() and np.isfinite(xs[k]).all()
            np.testing.assert_array_equal(x[k][:, lane_cols], -xs[k][:, lane_cols])
            for label, matrix in (('original', x[k]), ('swapped', xs[k])):
                unchanged = np.concatenate([matrix[:, :lane_start], matrix[:, lane_start + 5:]], axis=1)
                digest = hashlib.sha256(unchanged.tobytes()).hexdigest()
                key = (k, label)
                if variant == 'greedy':
                    invariant_hashes[key] = digest
                else:
                    assert invariant_hashes[key] == digest, 'A variant changed non-matchup features'
            entry['feature_sha256'][k] = hashlib.sha256(x[k].tobytes()).hexdigest()
            entry['swapped_feature_sha256'][k] = hashlib.sha256(xs[k].tobytes()).hexdigest()
        verify_reference_features(reference, variant, entry['feature_sha256'])
        entry['reference_matrix_hashes_match'] = True
        with threadpool_limits(limits=4):
            model = HistGradientBoostingClassifier(**parameters)
            model.fit(np.concatenate([x['train'], xs['train']]),
                      np.concatenate([y['train'], 1 - y['train']]))
            for name in splits:
                raw = model.predict_proba(x[name])[:, 1]
                swapped_raw = model.predict_proba(xs[name])[:, 1]
                p = .5 * (raw + 1. - swapped_raw)
                ps = .5 * (swapped_raw + 1. - raw)
                np.testing.assert_allclose(p + ps, 1., rtol=0., atol=2e-15)
                entry['metrics'][name] = {
                    'count': len(y[name]), 'coverage': 1.0,
                    'accuracy': float(accuracy_score(y[name], p >= .5)),
                    'log_loss': float(log_loss(y[name], p)),
                    'roc_auc': float(roc_auc_score(y[name], p)),
                    'brier': float(brier_score_loss(y[name], p)),
                }
                snapshots[f'{variant}_{name}_p'] = p
                entry['reference_comparison'][name] = compare_reference_predictions(
                    p, reference_predictions[f'{variant}_{name}_p'], strict=same_versions)
        entry['elapsed_seconds'] = time.monotonic() - start
        report['variants'][variant] = entry
        joblib.dump({'model': model, 'variant': variant,
                     'requires_team_swap_symmetrization': True,
                     'experimental_only': True}, out / f'{variant}_candidate.joblib')
        dump(out / 'metrics.json', report)
        np.savez_compressed(out / 'predictions.npz', **snapshots)
        print('Completed', variant, json.dumps(entry['metrics']), flush=True)
        del x, xs, model
    report['completed'] = True
    dump(out / 'metrics.json', report)
    print('Saved', out.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-cache', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-dir', default=str(ROOT / 'runs/accuracy_matchup_20260927'))
    run(parser.parse_args())
