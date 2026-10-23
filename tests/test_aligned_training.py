import json
from pathlib import Path
import pytest
from tests.test_aligned_artifacts import tiny_bundle

@pytest.mark.parametrize('name', ['train_deepsets_v3', 'train_deepsets_v4_timeline'])
def test_aligned_training_does_not_read_legacy_or_test(tmp_path, monkeypatch, name):
    import importlib
    trainer = importlib.import_module('scripts.' + name)
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name.startswith('sealed_test') or path.name == 'source_rows.json.gz':
            raise AssertionError('training attempted to open reserved outcomes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    def legacy_forbidden(*args, **kwargs):
        raise AssertionError('legacy data path used')
    monkeypatch.setattr(trainer, 'load_enriched_records', legacy_forbidden)
    if name == 'train_deepsets_v3':
        monkeypatch.setattr(trainer, 'prepare_ordered_training', legacy_forbidden)
    class FirstBatchReached(Exception):
        pass
    def stop(*args, **kwargs):
        raise FirstBatchReached
    monkeypatch.setattr(trainer, 'pcgrad_step', stop)
    with pytest.raises(FirstBatchReached):
        getattr(trainer, name)('unused.db', str(tmp_path / name),
                              dataset_dir=str(directory), epochs=1)

def test_aligned_and_legacy_cache_are_mutually_exclusive(tmp_path):
    from scripts.train_deepsets_v3 import train_deepsets_v3
    with pytest.raises(ValueError, match='exclusive'):
        train_deepsets_v3('unused.db', str(tmp_path / 'out'),
            dataset_dir='frozen', splits_cache='old.pkl')
    assert not (tmp_path / 'out').exists()


def test_v3_compiled_batch_trains_pruned_model_and_reloads_checkpoint(tmp_path):
    """The compiled path must ignore losses for heads absent from the model."""
    import tensorflow as tf
    from scripts.train_deepsets_v3 import train_deepsets_v3
    from core.aligned_artifacts import read_train_validation
    from core.equivariant_model import NNRIOT_CUSTOM_OBJECTS

    directory, _ = tiny_bundle(tmp_path, timeline=True)
    output = tmp_path / 'compiled_v3'
    result = train_deepsets_v3(
        'unused.db', str(output), dataset_dir=str(directory),
        epochs=1, batch_size=64,
    )

    assert result['best_epoch'] == 1
    assert len(result['history']) == 1
    assert np.isfinite(result['history'][0]['train_loss'])
    assert np.isfinite(result['best_val_log_loss'])
    assert result['test_status'] == 'reserved'
    loaded = tf.keras.models.load_model(
        output / 'model.keras', custom_objects=NNRIOT_CUSTOM_OBJECTS,
    )
    assert int(loaded.optimizer.iterations.numpy()) == 1
    x, _, _ = read_train_validation(directory, 'nnriot.players14.v1')
    probabilities = loaded(x['validation'], training=False)['winner'].numpy()
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

import numpy as np
from core.aligned_training import finish_aligned_pilot, prepare_aligned_training
from core.calibration import load_calibration_artifact
from core.aligned_artifacts import write_dataset
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

@pytest.mark.parametrize('labels', [np.array([0,1,0,1]), np.zeros(20, dtype=int)])
def test_insufficient_calibration_has_no_fallback_or_test(tmp_path, labels):
    directory, meta = tiny_bundle(tmp_path)
    output = tmp_path / 'pilot'; output.mkdir()
    (output / 'model.keras').write_bytes(b'fixture-checkpoint')
    result = finish_aligned_pilot(output_dir=output, dataset_dir=directory,
        dataset_manifest=meta, input_dim=14, validation_probabilities=np.full(len(labels), .5),
        validation_labels=labels, history=[], best_epoch=1, best_val_loss=.7)
    assert result['calibration_status'] == 'insufficient_validation'
    assert result['calibrated'] is False
    assert result['test_status'] == 'reserved'
    assert 'test_raw' not in result and 'test_calibrated' not in result
    assert not (output / 'model.calibration.json').exists()

def test_completed_pilot_calibration_binds_dataset_manifest(tmp_path, monkeypatch):
    rows = [source_row(f'NA1_{i:03d}', 10*i+10, 10*i+19, i % 2) for i in range(70)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, purged = partition_targets(targets, after_ms=400,
                                      validation_count=20, test_count=20)
    directory = tmp_path / 'dataset'
    meta = write_dataset(directory, rows, splits, source_db='fixture',
                         selection={}, exclusions=purged)
    output = tmp_path / 'pilot'
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name.startswith('sealed_test') or path.name == 'source_rows.json.gz':
            raise AssertionError('pilot accessed reserved outcomes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    _, records, loaded = prepare_aligned_training(directory, output, 'nnriot.players14.v1')
    (output / 'model.keras').write_bytes(b'fixture-checkpoint')
    labels = np.array([r['labels_json']['winner'] for r in records['validation']])
    result = finish_aligned_pilot(output_dir=output, dataset_dir=directory,
        dataset_manifest=loaded, input_dim=14, validation_probabilities=np.full(len(labels), .55),
        validation_labels=labels, history=[], best_epoch=1, best_val_loss=.7)
    assert result['calibrated'] is True
    assert load_calibration_artifact(output / 'model.keras', input_dim=14) is not None
