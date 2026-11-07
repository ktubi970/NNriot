"""Lock every candidate, then score the sealed test once with the six DeepSets checkpoints.

Run under the WSL TensorFlow environment after training and baselines finish. The lock binds
checkpoints, calibrations, baseline artifacts and sources before any test byte is read;
evaluate.py (Windows, LightGBM) refuses to run against any other lock.
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASET = ROOT / 'runs/aligned_dataset_20261023_prospective'
DIMS, SEEDS = (14, 18), (42, 43, 44)
FAMILIES = ('logistic', 'lightgbm')
TEST_COUNT = 10000
EPOCHS = 5
EVALUATION_SOURCES = ('core/training_metrics.py', 'core/common_cohort_evaluation.py', 'core/selective_classification.py',
                      'core/calibration.py', 'core/tabular_baselines.py', 'core/experiment.py',
                      'runs/prospective_20261023/infer_deepsets.py', 'runs/prospective_20261023/evaluate.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')


def publish_exclusive(path, writer):
    """Publish complete bytes atomically, refusing to replace any prior output."""
    fd, temporary = tempfile.mkstemp(prefix='.' + Path(path).name + '.', dir=Path(path).parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def checkpoint(dim, seed):
    return OUT / f'deepsets{dim}_s{seed}' / 'model.keras'


def bind():
    """Hash every non-test artifact the comparison depends on. Never opens sealed files."""
    from core.aligned_artifacts import checked_path, dataset_runtime_sources
    from core.calibration import load_calibration_artifact
    protocol, prep = read_json(OUT / 'protocol.json'), read_json(OUT / 'preparation_result.json')
    plan, status = read_json(OUT / 'training_plan.json'), read_json(OUT / 'training_status.json')
    baselines = read_json(OUT / 'baselines_status.json')
    manifest = read_json(DATASET / 'dataset.json')
    require(protocol['status'] == 'preregistered' and prep['status'] == 'verified', 'Unverified preparation')
    require(prep['test_metrics_computed'] is False, 'Preparation scored the test')
    require(status['status'] == 'completed' and status['test_status'] == 'reserved', 'Training incomplete')
    require(set(status['completed']) == {f'{d}_{s}' for d in DIMS for s in SEEDS}, 'Missing training run')
    require(baselines['status'] == 'completed' and baselines['test_status'] == 'reserved', 'Baselines incomplete')
    require(set(baselines['completed']) == {f'{f}{d}' for f in FAMILIES for d in DIMS}, 'Missing baseline')
    require(manifest['test_status'] == 'reserved' and manifest['selection']['protocol_sha256'] == digest(OUT / 'protocol.json'),
            'Dataset does not belong to this protocol')
    require(manifest['runtime_sources'] == dataset_runtime_sources(), 'Feature runtime changed')
    require(prep['dataset_manifest_sha256'] == digest(DATASET / 'dataset.json')
            == plan['dataset_manifest_sha256'] == baselines['dataset_manifest_sha256'], 'Dataset hash mismatch')
    for name, expected in plan['source_hashes'].items():
        require(digest(ROOT / name) == expected, 'Training source changed: ' + name)
    for name, expected in baselines['sources'].items():
        require(digest(ROOT / name) == expected, 'Baseline source changed: ' + name)
    files = {relative(p): digest(p) for p in (OUT / 'protocol.json', OUT / 'preparation_result.json',
             OUT / 'training_plan.json', OUT / 'training_status.json', OUT / 'baselines_status.json',
             DATASET / 'dataset.json')}
    for name in manifest['files']:
        if not name.startswith('sealed_test'):
            files[relative(checked_path(DATASET, manifest, name))] = manifest['files'][name]['sha256']
    candidates = {}
    for dim in DIMS:
        for seed in SEEDS:
            path = checkpoint(dim, seed)
            metrics = read_json(path.with_suffix('.metrics.json'))
            require(metrics['test_status'] == 'reserved' and metrics['player_feature_dim'] == dim, f'{dim}/{seed} metrics')
            require(metrics['checkpoint_sha256'] == digest(path) == status['completed'][f'{dim}_{seed}']['checkpoint_sha256'],
                    f'{dim}/{seed} checkpoint changed')
            require([r['epoch'] for r in metrics['history']] == list(range(1, EPOCHS + 1)), f'{dim}/{seed} epochs')
            calibration = load_calibration_artifact(path, input_dim=dim)
            require(calibration is not None, f'{dim}/{seed} calibration failed strict verification')
            for p in (path, path.with_suffix('.metrics.json'), path.with_suffix('.calibration.json'),
                      path.with_suffix('.split.json')):
                files[relative(p)] = digest(p)
            candidates[f'deepsets{dim}_s{seed}'] = dict(checkpoint=relative(path), temperature=calibration['temperature'],
                best_epoch=metrics['best_epoch'], best_val_log_loss=metrics['best_val_log_loss'])
    for key, info in baselines['completed'].items():
        require(digest(ROOT / info['path']) == info['sha256'], 'Baseline artifact changed: ' + key)
        files[info['path']] = info['sha256']
        candidates[key] = dict(path=info['path'], temperature=info['temperature'])
    sources = {**plan['source_hashes'], **baselines['sources'], **{n: digest(ROOT / n) for n in EVALUATION_SOURCES}}
    return dict(schema_version=1, kind='nnriot.locked_prospective_comparison.v1', test_count=TEST_COUNT,
                files=files, sources=sources, candidates=candidates, protocol=protocol, no_test_tuning=True,
                production_promotion=False)


def recheck(bound):
    for name, expected in {**bound['files'], **bound['sources']}.items():
        require(digest(ROOT / name) == expected, 'Locked artifact changed: ' + name)


def acquire_lock(bound):
    path = OUT / 'lock.json'
    if path.exists():
        require(read_json(path)['bound'] == bound, 'Existing lock differs; resume refused')
    else:
        publish_exclusive(path, lambda h: h.write(canonical({'created_utc': datetime.now(timezone.utc).isoformat(),
                                                             'bound': bound})))
    return digest(path)


def load_model(path, dim):
    import tensorflow as tf
    from core.equivariant_model import NNRIOT_CUSTOM_OBJECTS
    from core.serving_contract import FeatureContract
    model = tf.keras.models.load_model(path, custom_objects=NNRIOT_CUSTOM_OBJECTS, compile=False)
    FeatureContract(f'nnriot.players{dim}.v1').validate_model(model)
    require({'winner', 'draft_advantage'}.issubset(model.output_names), 'Required heads missing')
    return model


def validation_invariants(bound):
    """Permutation, composition swap, mirror and side-bias checks on a 64-row validation panel."""
    import numpy as np
    import tensorflow as tf
    from core.aligned_artifacts import read_train_validation
    result = {}
    for dim in DIMS:
        x, records, _ = read_train_validation(DATASET, f'nnriot.players{dim}.v1')
        count = len(records['validation'])
        rows = np.unique(np.linspace(0, count - 1, 64, dtype=int))
        panel = {k: v[rows] for k, v in x['validation'].items()}
        del x, records
        rng = np.random.default_rng(20261023)
        perm_a = np.array([rng.permutation(5) for _ in rows])
        perm_b = np.array([rng.permutation(5) for _ in rows])
        index = np.arange(len(rows))[:, None]
        for seed in SEEDS:
            model = load_model(checkpoint(dim, seed), dim)
            def predict(inputs):
                return {k: np.asarray(v) for k, v in model(inputs, training=False).items()}
            original = predict(panel)
            swap = predict({'team_a': panel['team_b'], 'team_b': panel['team_a'], 'macro_clash': -panel['macro_clash']})
            permuted = predict({'team_a': panel['team_a'][index, perm_a], 'team_b': panel['team_b'][index, perm_b],
                                'macro_clash': panel['macro_clash']})
            mirror = predict({'team_a': panel['team_a'], 'team_b': panel['team_a'],
                              'macro_clash': np.zeros_like(panel['macro_clash'])})
            bias = float(model.get_layer('winner_side_bias').bias.numpy()[0])
            alphas = np.logaddexp(0.0, np.array([-bias, bias])) + 1.0
            errors = dict(draft_swap=np.abs(original['draft_advantage'] - swap['draft_advantage'][:, ::-1]).max(),
                          draft_permutation=np.abs(original['draft_advantage'] - permuted['draft_advantage']).max(),
                          winner_permutation=np.abs(original['winner'] - permuted['winner']).max(),
                          draft_mirror=np.abs(mirror['draft_advantage'] - .5).max(),
                          winner_mirror_side_bias=np.abs(mirror['winner'] - alphas / alphas.sum()).max())
            errors = {k: float(v) for k, v in errors.items()}
            require(all(v <= 1e-5 for v in errors.values()), f'{dim}/{seed} invariant failure: {errors}')
            result[f'deepsets{dim}_s{seed}'] = dict(max_absolute_errors=errors, winner_side_bias=bias,
                                                    mirror_red_probability=float(alphas[1] / alphas.sum()))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    recheck(bound)
    return result


def main():
    import numpy as np
    import tensorflow as tf
    from core.aligned_artifacts import checked_path, validate_tensor_inputs
    from core.calibration import apply_temperature_to_probabilities
    from core.serving_contract import validate_probabilities
    bound = bind()
    lock_sha = acquire_lock(bound)
    target = OUT / 'predictions_deepsets.npz'
    require(not target.exists(), 'DeepSets predictions already exist; inference is never repeated')
    invariants = validation_invariants(bound)
    manifest = read_json(DATASET / 'dataset.json')
    split = read_json(checked_path(DATASET, manifest, 'model.split.json'))
    # First access to test content.
    records = read_json(checked_path(DATASET, manifest, 'sealed_test.json'))
    require([{'match_id': r['match_id'], 'game_creation_time': r['game_creation_time']} for r in records]
            == split['splits']['test'] and len(records) == TEST_COUNT, 'Test membership/order mismatch')
    saved = dict(match_ids=np.asarray([r['match_id'] for r in records]),
                 game_creation_time=np.asarray([r['game_creation_time'] for r in records], dtype=np.int64),
                 labels=np.asarray([r['labels_json']['winner'] for r in records], dtype=np.int8),
                 lock_sha256=np.asarray(lock_sha), invariants_json=np.asarray(json.dumps(invariants)))
    for dim in DIMS:
        with np.load(checked_path(DATASET, manifest, f'sealed_test{dim}.npz'), allow_pickle=False) as archive:
            x = {k: archive[k].copy() for k in archive.files}
        validate_tensor_inputs(x, TEST_COUNT, dim, 'test')
        for seed in SEEDS:
            key = f'deepsets{dim}_s{seed}'
            print(f'Locked test inference: {key}', flush=True)
            model = load_model(checkpoint(dim, seed), dim)
            parts = []
            for start in range(0, TEST_COUNT, 256):
                batch = {k: v[start:start + 256] for k, v in x.items()}
                parts.append(validate_probabilities(np.asarray(model(batch, training=False)['winner']), 'winner',
                                                    rows=len(batch['team_a']))[:, 1])
            saved[key + '_raw'] = np.concatenate(parts)
            saved[key + '_calibrated'] = apply_temperature_to_probabilities(
                saved[key + '_raw'], bound['candidates'][key]['temperature'])
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    recheck(bound)
    publish_exclusive(target, lambda handle: np.savez(handle, **saved))
    print('COMPLETE: predictions_deepsets.npz published; run evaluate.py on Windows next', flush=True)


if __name__ == '__main__':
    main()
