"""Paired diagnostics for the frozen accuracy experiments, without model tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.enriched_features import batch_extract_enriched_features
from core.player_priors import PlayerPriorsTracker


def compare(y, old, new):
    old_ok, new_ok = (old >= .5) == y, (new >= .5) == y
    gained = int(np.sum(new_ok & ~old_ok))
    lost = int(np.sum(old_ok & ~new_ok))
    delta = new_ok.astype(int) - old_ok.astype(int)
    rng = np.random.default_rng(20260927)
    counts = np.array([np.sum(delta == -1), np.sum(delta == 0), np.sum(delta == 1)])
    draws = rng.multinomial(len(y), counts/len(y), size=10000)
    bootstrap_delta = (draws[:, 2]-draws[:, 0])/len(y)
    return {
        'count': len(y), 'old_accuracy': float(old_ok.mean()),
        'new_accuracy': float(new_ok.mean()), 'delta': float(delta.mean()),
        'corrected_errors': gained, 'new_errors': lost,
        'mcnemar_exact_iid_p': float(binomtest(gained, gained+lost, .5).pvalue) if gained+lost else 1.,
        'paired_iid_bootstrap_95pct': np.quantile(bootstrap_delta, [.025, .975]).tolist(),
        'iid_warning': 'Repeated players and temporal dependence violate unverified iid assumptions.',
        'chronological_six_blocks': [
            {'count': len(ix), 'old_accuracy': float(old_ok[ix].mean()),
             'new_accuracy': float(new_ok[ix].mean()), 'delta': float(delta[ix].mean())}
            for ix in np.array_split(np.arange(len(y)), 6)
        ],
    }


def run(args):
    base = Path(args.experiment)
    before = np.load(base / 'before/predictions.npz')
    after = np.load(base / 'after/predictions.npz')
    ensemble = np.load(base / 'ensemble_legacy_config/predictions.npz')
    result = {'comparisons': {}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ('validation', 'test'):
        np.testing.assert_array_equal(before[f'{name}_ids'], after[f'{name}_ids'])
        np.testing.assert_array_equal(before[f'{name}_y'], after[f'{name}_y'])
        np.testing.assert_array_equal(after[f'{name}_ids'], ensemble[f'{name}_ids'])
        np.testing.assert_array_equal(after[f'{name}_y'], ensemble[f'{name}_y'])
        y = after[f'{name}_y']
        result['comparisons'][name] = {
            'restored_priors': compare(y, before[f'seed_42_{name}_p'], after[f'seed_42_{name}_p']),
            'ensemble_vs_neural': compare(y, ensemble[f'{name}_neural_p'], ensemble[f'{name}_ensemble_p']),
            'always_team_b_accuracy': float(y.mean()),
        }
    # Prove the after-code can exactly replay the real pre-patch feature matrices.
    splits = pickle.loads(Path(args.splits_cache).read_bytes())
    baseline_report = json.loads((base / 'before/metrics.json').read_text())
    assert baseline_report['cache_sha256'] == hashlib.sha256(Path(args.splits_cache).read_bytes()).hexdigest()
    result['legacy_replay_feature_sha256'] = {}
    for name, rows in splits.items():
        features = [{**r['feature_json'], **{
            team: [{**p, 'priors': PlayerPriorsTracker.get_default_priors()}
                   for p in r['feature_json'][team]] for team in ('team_a', 'team_b')
        }} for r in rows]
        # Cache contract was audited: every participant has ten attached priors.
        assert all(len(p['priors']) == 10 for r in rows for team in ('team_a', 'team_b')
                   for p in r['feature_json'][team])
        x = batch_extract_enriched_features(features, mode='tabular')
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        assert digest == baseline_report['feature_sha256'][name]
        result['legacy_replay_feature_sha256'][name] = digest
    output = Path(args.output)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--splits-cache', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
