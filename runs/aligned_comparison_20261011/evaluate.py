"""Evaluate the locked 14D/18D pilot once; resume metrics from saved predictions.

Run under the verified WSL TensorFlow environment after both trainings finish.
No fitting, checkpoint selection, test-row filtering, or promotion happens here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TEST_COUNT = 1344
SEED = 20261011
BLOCK_SIZE = 100
REPEATS = 2000
PREDICTION_KEYS = ('14d_raw', '14d_calibrated', '18d_raw', '18d_calibrated')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def recorded_dataset_path(value):
    """Compare the preparation's Windows path with this checkout under WSL."""
    text = str(value).replace('\\', '/')
    if PureWindowsPath(text).drive:
        parts = PureWindowsPath(text).parts
        require('runs' in parts, 'Recorded dataset path has no runs component')
        return Path(*parts[parts.index('runs'):]).as_posix()
    path = Path(text)
    return relative(path) if path.is_absolute() else path.as_posix()


@contextmanager
def single_evaluator():
    # The actual execution target is WSL. Advisory locking releases on crashes.
    import fcntl
    with (OUT / 'evaluation.execution.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluation process is running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publish_exclusive(path, writer):
    """Publish complete bytes atomically, refusing to replace any prior output."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic exclusive publication on this filesystem.
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_json(path, value):
    publish_exclusive(path, lambda handle: handle.write(canonical(value)))


def environment():
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'tensorflow', 'keras'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': sys.platform, 'packages': packages}


def preflight(dataset, checkpoints):
    """Read and bind non-test artifacts only. Never hash/open sealed files here."""
    from core.aligned_artifacts import checked_path, dataset_runtime_sources
    from core.calibration import load_calibration_artifact
    from core.experiment import read_manifest
    from core.selective_classification import DEFAULT_CONVICTION_THRESHOLDS

    prep_path, protocol_path = OUT / 'preparation_result.json', OUT / 'protocol.json'
    prep, protocol = read_json(prep_path), read_json(protocol_path)
    training_plan_path, training_status_path = OUT / 'training_plan.json', OUT / 'training_status.json'
    training_plan, training_status = read_json(training_plan_path), read_json(training_status_path)
    require(training_status.get('status') == 'completed', 'Both training jobs must be completed')
    require(training_status.get('test_status') == 'reserved', 'Training status did not reserve the test')
    require(set(training_status.get('completed', {})) == {'14', '18'}, 'Missing completed training job')
    require(prep.get('status') == 'verified', 'Preparation is not verified')
    for key in ('hashes_verified', 'temporal_purges_verified', 'previous_tests_excluded'):
        require(prep.get(key) is True, 'Preparation verification missing: ' + key)
    require(prep.get('test_metrics_computed') is False, 'Preparation already scored test')
    require(protocol.get('status') == 'preregistered', 'Protocol is not preregistered')
    for recorded in (prep.get('dataset_dir'), protocol.get('dataset_dir')):
        require(recorded_dataset_path(recorded) == relative(dataset), 'Dataset path mismatch')
    require(protocol.get('seed') == 42 and protocol.get('epochs') == 5
            and protocol.get('early_stopping') is False, 'Unexpected training protocol')
    require(protocol.get('batch_size') == 64 and protocol.get('learning_rate') == .0008,
            'Unexpected optimization protocol')
    require(protocol.get('calibration_partition') == 'validation'
            and protocol.get('production_promotion') is False, 'Unexpected evaluation policy')
    require(protocol.get('test_minimum') == TEST_COUNT, 'Unexpected prospective test size')
    require(prep['cohort']['splits']['test']['count'] == TEST_COUNT, 'Preparation test size mismatch')
    require(protocol.get('preparation_script_sha256') == digest(OUT / 'prepare.py'),
            'Preparation script changed')
    manifest_path = dataset / 'dataset.json'
    manifest = read_json(manifest_path)
    manifest_sha = digest(manifest_path)
    require(prep.get('dataset_manifest_sha256') == manifest_sha, 'Prepared dataset hash changed')
    require(training_plan.get('dataset_manifest_sha256') == manifest_sha, 'Training plan dataset mismatch')
    require(training_plan.get('protocol_sha256') == digest(protocol_path), 'Training plan protocol mismatch')
    training_hashes = training_plan.get('source_hashes')
    require(isinstance(training_hashes, dict) and training_hashes, 'Training source hashes missing')
    required_training_sources = {'core/equivariant_model.py', 'core/deepsets_layers.py',
        'core/generate_graph.py', 'core/pcgrad.py', 'core/training_utils.py', 'core/aligned_training.py',
        'scripts/train_deepsets_v3.py', 'scripts/train_deepsets_v4_timeline.py',
        'scripts/run_gpu_training.py', 'scripts/run_with_cuda.sh'}
    require(required_training_sources.issubset(training_hashes), 'Training plan omits architecture or execution sources')
    require(set(manifest['runtime_sources']).issubset(training_hashes), 'Training plan omits dataset runtime sources')
    for name, expected in training_hashes.items():
        source_path = (ROOT / name).resolve()
        require(not Path(name).is_absolute() and source_path.is_relative_to(ROOT),
                'Training source path outside repository: ' + name)
        require(digest(source_path) == expected, 'Source changed since training plan: ' + name)
        if name in manifest['runtime_sources']:
            require(expected == manifest['runtime_sources'][name], 'Training/dataset source mismatch: ' + name)
    require(manifest.get('kind') == 'nnriot.serving_aligned_dataset.v1'
            and manifest.get('schema_version') == 1, 'Unsupported dataset')
    require(manifest.get('test_status') == 'reserved', 'Dataset test is not reserved')
    require(manifest['selection'].get('protocol_sha256') == digest(protocol_path),
            'Dataset does not correspond to this protocol')
    require(manifest['runtime_sources'] == dataset_runtime_sources(), 'Feature runtime changed')
    split_path = checked_path(dataset, manifest, 'model.split.json')
    split = read_manifest(split_path)
    for key in ('dataset_sha256', 'split_sha256'):
        require(split[key] == manifest[key], 'Dataset/split mismatch: ' + key)
    require(len(split['splits']['test']) == TEST_COUNT, 'Reserved test count changed')
    files = {relative(p): digest(p) for p in (prep_path, protocol_path, manifest_path, split_path,
                                             training_plan_path, training_status_path)}
    # Bind expected test hashes without reading their bytes until lock publication.
    for name in ('sealed_test.json', 'sealed_test14.npz', 'sealed_test18.npz'):
        require(name in manifest['files'], 'Missing reserved artifact entry: ' + name)
    artifacts = {}
    for dim, checkpoint in checkpoints.items():
        completed = training_status['completed'][str(dim)]
        require(completed.get('status') == 'completed', f'{dim}D training was not completed')
        command = training_plan.get('commands', {}).get(str(dim))
        require(isinstance(command, list) and all(isinstance(part, str) for part in command),
                f'{dim}D training command missing')
        expected_script = 'scripts/train_deepsets_v3.py' if dim == 14 else 'scripts/train_deepsets_v4_timeline.py'
        require(expected_script in command and 'scripts/run_gpu_training.py' in command
                and 'scripts/run_with_cuda.sh' in command, f'{dim}D training wrapper/script mismatch')
        options = {'--dataset-dir': relative(dataset), '--output-dir': relative(checkpoint.parent),
                   '--epochs': '5', '--batch-size': '64', '--learning-rate': '0.0008', '--seed': '42'}
        if dim == 18:
            options['--patience'] = '0'
        for flag, expected in options.items():
            require(command.count(flag) == 1, f'{dim}D training command lacks unique {flag}')
            index = command.index(flag)
            require(index + 1 < len(command) and command[index + 1] == expected,
                    f'{dim}D training command differs from protocol: {flag}')
        contract = f'nnriot.players{dim}.v1'
        require(manifest['availability'][contract]['status'] == 'ready', contract + ' unavailable')
        metrics_path = checkpoint.with_suffix('.metrics.json')
        metrics = read_json(metrics_path)
        require(metrics.get('test_status') == 'reserved', f'{dim}D test already evaluated')
        require(metrics.get('player_feature_dim') == dim, f'{dim}D input dimension mismatch')
        require(metrics.get('calibrated') is True and metrics.get('calibration_status') == 'fitted',
                f'{dim}D calibration is not fitted')
        require(metrics.get('dataset_manifest_sha256') == manifest_sha, f'{dim}D dataset mismatch')
        require(metrics.get('split_sha256') == manifest['split_sha256'], f'{dim}D split mismatch')
        require(metrics.get('checkpoint_sha256') == digest(checkpoint), f'{dim}D checkpoint changed')
        require(completed.get('checkpoint_sha256') == digest(checkpoint), f'{dim}D training status checkpoint mismatch')
        require(completed.get('metrics_sha256') == digest(metrics_path), f'{dim}D training status metrics mismatch')
        require([r['epoch'] for r in metrics.get('history', [])] == list(range(1, 6)),
                f'{dim}D did not complete the five preregistered epochs')
        model_split_path = checkpoint.with_suffix('.split.json')
        require(digest(model_split_path) == digest(split_path), f'{dim}D split file differs')
        calibration = load_calibration_artifact(checkpoint, input_dim=dim)
        require(calibration is not None, f'{dim}D strict calibration verification failed')
        calibration_path = checkpoint.with_suffix('.calibration.json')
        for path in (checkpoint, metrics_path, model_split_path, calibration_path):
            files[relative(path)] = digest(path)
        artifacts[str(dim)] = {'checkpoint': relative(checkpoint),
            'checkpoint_sha256': digest(checkpoint), 'calibration_sha256': digest(calibration_path),
            'temperature': calibration['temperature'], 'best_epoch': metrics['best_epoch'],
            'calibration_fit': calibration['fit']}
    for name in ('train.json', 'validation.json', 'validation14.npz', 'validation18.npz'):
        path = checked_path(dataset, manifest, name)
        files[relative(path)] = digest(path)
    code = ('core/common_cohort_evaluation.py', 'core/training_metrics.py', 'core/calibration.py',
            'core/selective_classification.py', 'core/experiment.py', 'core/aligned_training.py',
            'core/pcgrad.py', 'core/generate_graph.py', 'core/periodic_embeddings.py',
            'scripts/train_deepsets_v3.py', 'scripts/train_deepsets_v4_timeline.py')
    runtime = {**manifest['runtime_sources'], **{name: digest(ROOT / name) for name in code},
               **training_hashes,
               relative(__file__): digest(__file__)}
    bound = {'schema_version': 1, 'kind': 'nnriot.locked_aligned_pair.v1',
        'dataset_dir': relative(dataset), 'files': files, 'runtime_sources': runtime,
        'training_source_hashes': training_hashes,
        'dataset_files': manifest['files'], 'candidates': artifacts, 'environment': environment(),
        'test_count': TEST_COUNT, 'primary_metric': 'calibrated log_loss: 18D minus 14D',
        'secondary_metric': 'raw log_loss: 18D minus 14D',
        'bootstrap': {'seed': SEED, 'block_size': BLOCK_SIZE, 'repeats': REPEATS},
        'invariant_panel': {'partition': 'validation', 'maximum_rows': 64, 'atol': 1e-5},
        'selective_thresholds': list(DEFAULT_CONVICTION_THRESHOLDS),
        'selective_policy': 'fixed default thresholds; no test threshold selection or optimization',
        'baseline_policy': 'constant P(team B) from training labels only',
        'no_test_tuning': True, 'production_promotion': False}
    return bound, manifest, split


def acquire_lock(bound):
    path = OUT / 'lock.json'
    if path.exists():
        lock = read_json(path)
        require(lock.get('bound') == bound, 'Existing evaluation lock differs; resume refused')
    else:
        lock = {'created_utc': datetime.now(timezone.utc).isoformat(), 'bound': bound}
        publish_json(path, lock)
    return digest(path)


def recheck_bound(bound, lock_sha):
    require(digest(OUT / 'lock.json') == lock_sha, 'Evaluation lock changed')
    for name, expected in {**bound['files'], **bound['runtime_sources']}.items():
        require(digest(ROOT / name) == expected, 'Locked artifact changed: ' + name)


def records_for(dataset, manifest, split, partition):
    from core.aligned_artifacts import checked_path
    name = 'sealed_test' if partition == 'test' else partition
    records = read_json(checked_path(dataset, manifest, name + '.json'))
    membership = [{'match_id': r['match_id'], 'game_creation_time': r['game_creation_time']}
                  for r in records]
    require(membership == split['splits'][partition], partition + ' membership/order mismatch')
    keys = [(r['game_creation_time'], r['match_id']) for r in records]
    require(len({r['match_id'] for r in records}) == len(records), partition + ' duplicate ID')
    require(keys == sorted(keys), partition + ' is not chronological')
    require(all(type(r['labels_json']['winner']) is int and r['labels_json']['winner'] in (0, 1)
                for r in records), partition + ' invalid binary labels')
    return records


def arrays_for(dataset, manifest, partition, dim, count):
    import numpy as np
    from core.aligned_artifacts import checked_path, validate_tensor_inputs
    prefix = 'sealed_test' if partition == 'test' else partition
    with np.load(checked_path(dataset, manifest, f'{prefix}{dim}.npz'), allow_pickle=False) as archive:
        inputs = {key: archive[key].copy() for key in archive.files}
    validate_tensor_inputs(inputs, count, dim, partition)
    return inputs


def common_columns(arrays):
    import numpy as np
    for team in ('team_a', 'team_b'):
        np.testing.assert_array_equal(arrays[14][team], arrays[18][team][..., [*range(12), 16, 17]])
    np.testing.assert_array_equal(arrays[14]['macro_clash'], arrays[18]['macro_clash'])


def load_model(checkpoint, dim):
    import tensorflow as tf
    from core.equivariant_model import NNRIOT_CUSTOM_OBJECTS
    from core.serving_contract import FeatureContract
    model = tf.keras.models.load_model(checkpoint, custom_objects=NNRIOT_CUSTOM_OBJECTS, compile=False)
    FeatureContract(f'nnriot.players{dim}.v1').validate_model(model)
    require({'winner', 'draft_advantage'}.issubset(model.output_names), 'Required model heads missing')
    return model


def validation_invariants(dataset, manifest, split, checkpoints):
    import numpy as np
    import tensorflow as tf
    from core.serving_contract import validate_probabilities
    records = records_for(dataset, manifest, split, 'validation')
    count = len(records)
    indices = np.unique(np.linspace(0, count - 1, min(64, count), dtype=int))
    arrays = {dim: arrays_for(dataset, manifest, 'validation', dim, count) for dim in (14, 18)}
    common_columns(arrays)
    panel = {dim: {k: v[indices] for k, v in arrays[dim].items()} for dim in arrays}
    del arrays
    result = {'status': 'verified', 'partition': 'validation', 'atol': 1e-5,
              'match_ids': [records[i]['match_id'] for i in indices], 'models': {}}
    for dim, checkpoint in checkpoints.items():
        model = load_model(checkpoint, dim)
        x = panel[dim]
        def predict(inputs):
            values = {k: np.asarray(v) for k, v in model(inputs, training=False).items()}
            require(all(np.isfinite(v).all() for v in values.values()), 'Non-finite invariant output')
            for head in ('winner', 'draft_advantage'):
                validate_probabilities(values[head], head, rows=len(indices))
            return values
        original = predict(x)
        swap = predict({'team_a': x['team_b'], 'team_b': x['team_a'], 'macro_clash': -x['macro_clash']})
        rng = np.random.default_rng(SEED)
        permutations_a = np.array([rng.permutation(5) for _ in indices])
        permutations_b = np.array([rng.permutation(5) for _ in indices])
        permuted = predict({'team_a': x['team_a'][np.arange(len(indices))[:, None], permutations_a],
            'team_b': x['team_b'][np.arange(len(indices))[:, None], permutations_b],
            'macro_clash': x['macro_clash']})
        mirror = predict({'team_a': x['team_a'], 'team_b': x['team_a'],
                          'macro_clash': np.zeros_like(x['macro_clash'])})
        errors = {'draft_swap': float(np.max(np.abs(original['draft_advantage'] - swap['draft_advantage'][:, ::-1]))),
            'draft_permutation': float(np.max(np.abs(original['draft_advantage'] - permuted['draft_advantage']))),
            'winner_permutation': float(np.max(np.abs(original['winner'] - permuted['winner']))),
            'draft_mirror': float(np.max(np.abs(mirror['draft_advantage'] - .5)))}
        bias = float(model.get_layer('winner_side_bias').bias.numpy()[0])
        if any(layer.name == 'winner_evidential' for layer in model.layers):
            alphas = np.logaddexp(0.0, np.array([-bias, bias])) + 1.0
            expected = alphas / alphas.sum()
            mapping = 'evidential Dirichlet softplus evidence plus one'
        else:
            blue = 1.0 / (1.0 + np.exp(2.0 * bias))
            expected, mapping = np.array([blue, 1.0 - blue]), 'antisymmetric softmax'
        errors['winner_mirror_side_bias'] = float(np.max(np.abs(mirror['winner'] - expected)))
        require(all(value <= 1e-5 for value in errors.values()), f'{dim}D invariant failure: {errors}')
        result['models'][str(dim)] = {'status': 'verified', 'max_absolute_errors': errors,
            'winner_side_bias': bias, 'winner_mirror_expected': expected.tolist(), 'winner_mapping': mapping,
            'winner_swap_complementarity_required': False, 'training': False}
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return result


def read_test(dataset, manifest, split):
    records = records_for(dataset, manifest, split, 'test')
    require(len(records) == TEST_COUNT, 'Test must contain all 1344 preregistered matches')
    arrays = {dim: arrays_for(dataset, manifest, 'test', dim, len(records)) for dim in (14, 18)}
    common_columns(arrays)
    return records, arrays


def validate_saved_predictions(saved, lock_sha, records, candidates):
    import numpy as np
    from core.calibration import apply_temperature_to_probabilities
    fields = set(PREDICTION_KEYS) | {'match_ids', 'game_creation_time', 'labels', 'lock_sha256', 'invariants_json'}
    require(set(saved) == fields, 'Saved prediction archive has unexpected fields')
    require(saved['lock_sha256'].shape == () and str(saved['lock_sha256'].item()) == lock_sha,
            'Predictions are not bound to this lock')
    np.testing.assert_array_equal(saved['match_ids'], [r['match_id'] for r in records])
    np.testing.assert_array_equal(saved['game_creation_time'], [r['game_creation_time'] for r in records])
    np.testing.assert_array_equal(saved['labels'], [r['labels_json']['winner'] for r in records])
    for key in PREDICTION_KEYS:
        values = saved[key]
        require(values.shape == (TEST_COUNT,) and np.isfinite(values).all()
                and np.all((values >= 0) & (values <= 1)), 'Invalid saved probabilities: ' + key)
    for dim in (14, 18):
        expected = apply_temperature_to_probabilities(saved[f'{dim}d_raw'], candidates[str(dim)]['temperature'])
        np.testing.assert_array_equal(saved[f'{dim}d_calibrated'], expected)
    invariant = json.loads(str(saved['invariants_json'].item()))
    require(invariant.get('status') == 'verified' and invariant.get('partition') == 'validation',
            'Saved predictions lack verified validation invariants')
    require(0 < len(invariant['match_ids']) <= 64, 'Invalid invariant panel')
    for dim in ('14', '18'):
        require(invariant['models'][dim]['status'] == 'verified', dim + 'D invariant check missing')
    return invariant


def make_predictions(dataset, manifest, split, checkpoints, bound, lock_sha):
    import numpy as np
    import tensorflow as tf
    from core.calibration import apply_temperature_to_probabilities
    from core.serving_contract import validate_probabilities
    invariants = validation_invariants(dataset, manifest, split, checkpoints)
    recheck_bound(bound, lock_sha)
    # This is the first test-content access in a fresh evaluation.
    records, arrays = read_test(dataset, manifest, split)
    saved = {'match_ids': np.asarray([r['match_id'] for r in records]),
        'game_creation_time': np.asarray([r['game_creation_time'] for r in records], dtype=np.int64),
        'labels': np.asarray([r['labels_json']['winner'] for r in records], dtype=np.int8),
        'lock_sha256': np.asarray(lock_sha), 'invariants_json': np.asarray(canonical(invariants).decode('utf-8'))}
    for dim, checkpoint in checkpoints.items():
        print(f'Locked test inference: {dim}D, {len(records)} matches', flush=True)
        model = load_model(checkpoint, dim)
        parts = []
        for start in range(0, len(records), 128):
            batch = {k: v[start:start + 128] for k, v in arrays[dim].items()}
            output = model(batch, training=False)
            parts.append(validate_probabilities(np.asarray(output['winner']), 'winner', rows=len(batch['team_a']))[:, 1])
        saved[f'{dim}d_raw'] = np.concatenate(parts)
        saved[f'{dim}d_calibrated'] = apply_temperature_to_probabilities(
            saved[f'{dim}d_raw'], bound['candidates'][str(dim)]['temperature'])
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    validate_saved_predictions(saved, lock_sha, records, bound['candidates'])
    recheck_bound(bound, lock_sha)
    from core.aligned_artifacts import checked_path
    for name in ('sealed_test.json', 'sealed_test14.npz', 'sealed_test18.npz'):
        checked_path(dataset, manifest, name)
    publish_exclusive(OUT / 'predictions.npz', lambda handle: np.savez(handle, **saved))
    return saved, records


def summarize(saved, train, validation, lock_sha, bound):
    import numpy as np
    from core.training_metrics import evaluate_winner
    from core.common_cohort_evaluation import paired_block_bootstrap
    from core.selective_classification import evaluate_selective_coverage
    y = saved['labels']
    train_y = np.asarray([r['labels_json']['winner'] for r in train], dtype=int)
    validation_y = np.asarray([r['labels_json']['winner'] for r in validation], dtype=int)
    train_rate = float(train_y.mean())
    def classes(labels):
        counts = np.bincount(labels, minlength=2)
        return {'count': len(labels), 'team_a_wins': int(counts[0]), 'team_b_wins': int(counts[1]),
                'team_a_rate': float(counts[0] / len(labels)), 'team_b_rate': float(counts[1] / len(labels))}
    metrics = {}
    for name in PREDICTION_KEYS:
        score = evaluate_winner(y, saved[name], train_rate, diagnostics=True)
        selective = []
        for threshold in bound['selective_thresholds']:
            row = evaluate_selective_coverage(y, saved[name], threshold=threshold, confidence_level=.95)
            row['status'] = 'estimated' if row['count'] else 'no_retained_matches'
            row['risk'] = 1.0 - row['accuracy'] if row['count'] else None
            row['risk_wilson95'] = ([1.0 - row['wilson_upper'], 1.0 - row['wilson_lower']]
                                    if row['count'] else None)
            if not row['count']:
                # The shared utility uses zeros for empty masks; make missing
                # statistical estimates explicit in this final report.
                row['accuracy'] = row['wilson_lower'] = row['wilson_upper'] = None
            selective.append(row)
        score['selective_fixed_thresholds'] = selective
        metrics[name] = score
    paired = {kind: paired_block_bootstrap(y, saved[f'18d_{kind}'], saved[f'14d_{kind}'],
                repeats=REPEATS, seed=SEED, block_size=BLOCK_SIZE) for kind in ('calibrated', 'raw')}
    return {'schema_version': 1, 'status': 'completed', 'classification': 'bounded comparative pilot',
        'lock_sha256': lock_sha, 'predictions_sha256': digest(OUT / 'predictions.npz'),
        'test_count': TEST_COUNT, 'inference_count_each_model': TEST_COUNT, 'dropped_test_rows': 0,
        'class_ratios': {'train': classes(train_y), 'validation': classes(validation_y), 'test': classes(y)},
        'baseline': {'source': 'training labels only', 'probability_team_b': train_rate},
        'metrics': metrics, 'paired_18d_minus_14d': paired,
        'selective_policy': bound['selective_policy'],
        'selective_thresholds': bound['selective_thresholds'],
        'primary': paired['calibrated']['deltas']['log_loss_a_minus_b'],
        'delta_interpretation': 'Negative log-loss difference favors 18D; no automatic promotion',
        'invariants': json.loads(str(saved['invariants_json'].item())),
        'candidates': bound['candidates'], 'production_promotion': False,
        'bootstrap_context': {'chronological_match_block_size': BLOCK_SIZE,
            'equivalent_nonoverlapping_block_count': TEST_COUNT / BLOCK_SIZE,
            'sampled_blocks_per_replicate': int(np.ceil(TEST_COUNT / BLOCK_SIZE)),
            'repeats': REPEATS, 'seed': SEED},
        'limitations': ['Only 1344 prospective test matches versus the original target of 3000.',
            'Blocks contain 100 chronologically ordered matches, not a fixed duration.',
            'Only 13.44 blocks of 100 matches fit in the test; percentile intervals are unstable with few blocks.',
            'Shared players and temporal dependencies can span blocks; intervals are exploratory, not confirmatory significance.',
            'One training seed per architecture; the intervals do not include training randomness.',
            'Results are assumed observable at match end; actual historical ingestion times are unknown.',
            'Current static champion profiles lack historical patch provenance.',
            'Validation selected checkpoints and fitted temperatures; validation scores are not an independent test.',
            'Descriptive calibration and fixed-threshold risk/coverage do not establish a serving risk guarantee.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', default='runs/aligned_dataset_20261011_pilot')
    parser.add_argument('--model14', default='runs/aligned_comparison_20261011/14d/model.keras')
    parser.add_argument('--model18', default='runs/aligned_comparison_20261011/18d/model.keras')
    args = parser.parse_args()
    def root_path(value):
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    dataset = root_path(args.dataset_dir)
    checkpoints = {14: root_path(args.model14), 18: root_path(args.model18)}
    with single_evaluator():
        bound, manifest, split = preflight(dataset, checkpoints)
        lock_sha = acquire_lock(bound)
        train = records_for(dataset, manifest, split, 'train')
        validation = records_for(dataset, manifest, split, 'validation')
        require(train and validation, 'Empty train/validation partition')
        prediction_path = OUT / 'predictions.npz'
        comparison_path = OUT / 'comparison.json'
        if prediction_path.exists():
            # This route intentionally never imports TensorFlow or loads weights.
            import numpy as np
            records, arrays = read_test(dataset, manifest, split)
            del arrays
            with np.load(prediction_path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            validate_saved_predictions(saved, lock_sha, records, bound['candidates'])
            print('Reusing locked predictions; no inference repeated', flush=True)
        else:
            require(not comparison_path.exists(), 'Comparison exists without prediction archive')
            saved, records = make_predictions(dataset, manifest, split, checkpoints, bound, lock_sha)
        recheck_bound(bound, lock_sha)
        if comparison_path.exists():
            previous = read_json(comparison_path)
            require(previous.get('status') == 'completed' and previous.get('lock_sha256') == lock_sha
                    and previous.get('predictions_sha256') == digest(prediction_path)
                    and previous.get('test_count') == TEST_COUNT, 'Existing comparison differs')
            print('Evaluation already complete; all locked hashes verified', flush=True)
            return
        result = summarize(saved, train, validation, lock_sha, bound)
        recheck_bound(bound, lock_sha)
        publish_json(comparison_path, result)
        print('COMPLETE: comparison.json saved; production configuration unchanged', flush=True)


if __name__ == '__main__':
    main()
