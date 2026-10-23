"""Bounded two-model experiment; weights selected on validation before test."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.deepsets_dataset import batch_extract_team_tensors
from core.deepsets_layers import SiameseTeamEncoder, AntiSymmetricSoftmax
from core.equivariant_model import ThreeClassEquivariantSoftmax
from core.experiment import read_manifest, restore_splits
from scripts.benchmark_accuracy_priors import _metrics


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    raw = Path(args.splits_cache).read_bytes()
    splits = pickle.loads(raw)
    manifest = read_manifest(args.manifest)
    splits = restore_splits([r for rows in splits.values() for r in rows], manifest)
    tree_dir = Path(args.tree_run)
    tree_report = json.loads((tree_dir / 'metrics.json').read_text())
    assert tree_report['cache_sha256'] == hashlib.sha256(raw).hexdigest()
    assert tree_report['split_sha256'] == manifest['split_sha256']
    tree = np.load(tree_dir / 'predictions.npz')
    # Older checkpoints predate the three newly enabled encoder defaults.
    # Reconstruct their explicit old configuration in memory; never edit weights.
    with zipfile.ZipFile(args.neural_model) as archive:
        config = json.loads(archive.read('config.json'))
    legacy_defaults = []
    for layer in config['config']['layers']:
        if layer['class_name'] == 'SiameseTeamEncoder':
            for flag in ('use_set_transformer', 'use_sinkhorn_attention', 'use_periodic_embeddings'):
                if flag not in layer['config']:
                    layer['config'][flag] = False
                    legacy_defaults.append(flag)
    config['compile_config'] = None
    model = tf.keras.models.model_from_json(json.dumps(config), custom_objects={
        'SiameseTeamEncoder': SiameseTeamEncoder, 'AntiSymmetricSoftmax': AntiSymmetricSoftmax,
        'ThreeClassEquivariantSoftmax': ThreeClassEquivariantSoftmax,
    })
    model.load_weights(args.neural_model)
    print('Restored historical encoder defaults:', legacy_defaults, flush=True)
    predictions = {}
    metrics = {}
    selection = None
    for name in ('validation', 'test'):
        rows = splits[name]
        ids = np.array([r['match_id'] for r in rows])
        y = np.array([r['labels_json']['winner'] for r in rows])
        np.testing.assert_array_equal(ids, tree[f'{name}_ids'])
        np.testing.assert_array_equal(y, tree[f'{name}_y'])
        x = batch_extract_team_tensors([r['feature_json'] for r in rows])
        swapped = {'team_a': x['team_b'], 'team_b': x['team_a']}
        p0 = np.asarray(model.predict(x, batch_size=256, verbose=0)['winner'])[:, 1].astype(float)
        p1 = np.asarray(model.predict(swapped, batch_size=256, verbose=0)['winner'])[:, 1].astype(float)
        assert np.max(np.abs(p0 + p1 - 1.)) < 1e-5
        p = .5 * (p0 + 1. - p1)
        pt = tree[f'seed_42_{name}_p']
        if name == 'validation':
            grid = []
            for weight in np.linspace(0., 1., 21):
                value = _metrics(y, weight*p + (1.-weight)*pt)
                grid.append({'neural_weight': float(weight), **value})
            selection = min(grid, key=lambda r: (-r['accuracy'], r['log_loss']))
            (out / 'selection.json').write_text(json.dumps({
                'chosen': selection, 'validation_grid': grid,
                'policy': 'max validation accuracy; tie-break min log-loss; test not used',
                'cache_sha256': hashlib.sha256(raw).hexdigest(),
                'neural_sha256': hashlib.sha256(Path(args.neural_model).read_bytes()).hexdigest(),
                'tree_sha256': hashlib.sha256((tree_dir / 'candidate_42.joblib').read_bytes()).hexdigest(),
            }, indent=2) + '\n', encoding='utf-8')
            print('Frozen validation selection:', selection, flush=True)
        weight = selection['neural_weight']
        blend = weight*p + (1.-weight)*pt
        reverse = weight*(1.-p) + (1.-weight)*(1.-pt)
        np.testing.assert_allclose(blend+reverse, 1., rtol=0., atol=2e-15)
        metrics[name] = {'neural': _metrics(y,p), 'tree': _metrics(y,pt),
                         'ensemble': _metrics(y,blend)}
        for label, arr in [('ids', ids), ('y', y), ('neural_p', p), ('tree_p', pt), ('ensemble_p', blend)]:
            predictions[f'{name}_{label}'] = arr
        print(name, json.dumps(metrics[name]), flush=True)
    report = {'neural_weight': selection['neural_weight'], 'metrics': metrics,
              'historical_encoder_flags_set_false': legacy_defaults,
              'coverage': 1., 'calibrated': False, 'deployment_status': 'experimental_only',
              'cache_sha256': hashlib.sha256(raw).hexdigest(),
              'split_sha256': manifest['split_sha256'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': tree_report['limitations']}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(out / 'predictions.npz', **predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-cache', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--tree-run', required=True)
    parser.add_argument('--neural-model', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
