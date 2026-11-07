import json
import numpy as np
import pytest
from pathlib import Path
from core.aligned_artifacts import write_dataset, read_train_validation
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

def tiny_bundle(tmp_path, *, timeline=False):
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19, i % 2,
                       timeline=timeline) for i in range(10)]
    targets, excluded = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, purged = partition_targets(targets, after_ms=60,
                                       validation_count=2, test_count=2)
    directory = tmp_path / 'dataset'
    manifest = write_dataset(directory, rows, splits, source_db='fixture.db',
        selection={'history_window_limit': 10000}, exclusions=excluded + purged)
    return directory, manifest

def test_missing_timelines_are_reported_and_18d_refused(tmp_path):
    directory, meta = tiny_bundle(tmp_path)
    x, records, loaded = read_train_validation(directory, 'nnriot.players14.v1')
    assert set(x) == set(records) == {'train', 'validation'}
    assert meta['availability']['nnriot.players18.v1']['status'] == 'blocked'
    with pytest.raises(ValueError, match='unavailable'):
        read_train_validation(directory, 'nnriot.players18.v1')

def test_exact_shared_membership_and_changed_tensor_rejected(tmp_path):
    directory, meta = tiny_bundle(tmp_path, timeline=True)
    for version in ('nnriot.players14.v1', 'nnriot.players18.v1'):
        x, records, _ = read_train_validation(directory, version)
        assert len(x['train']['team_a']) == len(records['train'])
    tensor = directory / 'train14.npz'
    tensor.write_bytes(tensor.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        read_train_validation(directory, 'nnriot.players14.v1')

def test_manifest_path_cannot_escape_dataset(tmp_path):
    directory, meta = tiny_bundle(tmp_path)
    meta['files']['train14.npz']['path'] = '../foreign.npz'
    (directory / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='outside dataset'):
        read_train_validation(directory, 'nnriot.players14.v1')

def test_occupied_output_is_preserved(tmp_path):
    directory, _ = tiny_bundle(tmp_path)
    sentinel = directory / 'keep.txt'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        write_dataset(directory, [], {}, source_db='unused', selection={}, exclusions=[])
    assert sentinel.read_text() == 'keep'

def test_target_cutoff_must_match_row_creation(tmp_path):
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19) for i in range(10)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, _ = partition_targets(targets, after_ms=60, validation_count=2, test_count=2)
    splits['train'][0]['feature_json']['prediction_time_ms'] += 1
    with pytest.raises(ValueError, match='prediction cutoff'):
        write_dataset(tmp_path / 'bad', rows, splits, source_db='fixture', selection={}, exclusions=[])

@pytest.mark.parametrize('rows', [[], [{'match_id': 'NA1_1',
    'game_creation_time': True, 'partition': 'test'}]])
def test_invalid_previous_cohort_fails_before_database_read(tmp_path, rows):
    from scripts.prepare_aligned_dataset import prepare
    cohort = tmp_path / 'cohort.jsonl'
    cohort.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='previous cohort'):
        prepare(tmp_path / 'absent.db', tmp_path / 'output', cohort)
    assert not (tmp_path / 'output').exists()


def test_verify_cli_has_no_prediction_metrics(tmp_path, capsys):
    from scripts.prepare_aligned_dataset import main
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    assert main(['verify', '--dataset-dir', str(directory)]) == 0
    text = capsys.readouterr().out
    assert 'verified' in text
    assert 'test_metrics_computed' in text
    assert 'accuracy' not in text and 'log_loss' not in text
