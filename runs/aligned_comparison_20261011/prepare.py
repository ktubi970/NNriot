"""Freeze the predeclared pilot from audited source facts, with no SQL writes."""
from collections import Counter
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / 'runs/timeline18_audit_20261011'
DATASET = ROOT / 'runs/aligned_dataset_20261011_pilot'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def main():
    import numpy as np
    from core.aligned_artifacts import write_dataset, checked_path, validate_tensor_inputs, dataset_runtime_sources
    from core.aligned_features import extract_aligned_inputs
    from core.aligned_partitions import select_targets, partition_targets
    from core.frozen_history import FrozenHistoryProvider
    from core.features.pipeline import DraftFeaturePipeline
    from core.features.matchup_matrix import LaneMatchupTracker

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    started = time.time()
    assert not DATASET.exists(), 'Output directory must be new'
    audit = json.loads((AUDIT / 'result.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'completed' and audit['runtime_unchanged']
    for name, expected in audit['runtime_sources'].items():
        assert digest(ROOT / name) == expected, f'Audited runtime changed: {name}'
    runtime = dataset_runtime_sources()
    source = AUDIT / 'source_rows.json.gz'
    assert digest(source) == audit['source_snapshot_sha256']
    previous = json.loads((ROOT / 'runs/aligned_dataset_20261009_reviewed/model.split.json').read_text(encoding='utf-8'))
    protected = set(previous['split_metadata']['excluded_test_ids'])
    protected.update(r['match_id'] for r in previous['splits']['test'])
    cutoff = max(r['game_creation_time'] for r in previous['splits']['test'])
    protocol = dict(status='preregistered', created_utc=datetime.now(timezone.utc).isoformat(),
        kind='bounded_pilot', dataset_dir=str(DATASET), source_limit=61000, warmup_rows=10000,
        history_window_limit=10000, validation_minimum=3000, test_minimum=1344,
        test_after_utc=utc(cutoff), test_size_exception='1344 available new targets; initial prospective protocol required 3000',
        prior_test_ids_protected=len(protected), seed=42, epochs=5, batch_size=64,
        learning_rate=0.0008, early_stopping=False, checkpoint_selection='minimum validation winner log-loss',
        calibration_partition='validation', primary_metric='paired calibrated test log-loss: 18D minus 14D',
        test_policy='lock both checkpoints and calibrations, then evaluate both once; no tuning after test',
        collection_restarted=False, production_promotion=False, source_audit_sha256=digest(AUDIT / 'result.json'),
        audited_source_sha256=digest(source), preparation_script_sha256=digest(__file__))
    save('protocol.json', protocol)
    logging.info('Loading compact audited source snapshot (no SQLite payload reads)')
    with gzip.open(source, 'rt', encoding='utf-8') as handle:
        all_rows = json.load(handle)
    assert len(all_rows) == audit['source_rows']
    assert all((a['game_creation_time'], a['match_id']) < (b['game_creation_time'], b['match_id'])
               for a, b in zip(all_rows, all_rows[1:]))
    rows = all_rows[-61000:]
    del all_rows
    gc.collect()
    logging.info('Retained %d source rows from %s to %s', len(rows), utc(rows[0]['game_creation_time']), utc(rows[-1]['game_creation_time']))
    eligibility = {}
    with gzip.open(AUDIT / 'targets.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            eligibility[record['match_id']] = record['status']
    targets, exclusions = select_targets(rows, excluded_ids=protected, warmup_rows=10000)
    retained = []
    for target in targets:
        status = eligibility[target['match_id']]
        if status == 'eligible':
            retained.append(target)
        else:
            exclusions.append(dict(match_id=target['match_id'], reason='timeline_unavailable_in_audit', detail=status))
    splits, purged = partition_targets(retained, after_ms=cutoff, validation_count=3000, test_count=1344)
    exclusions.extend(purged)
    assert len(splits['test']) == 1344
    assert all(r['game_creation_time'] > cutoff for r in splits['test'])
    assert not protected.intersection(r['match_id'] for group in splits.values() for r in group)
    validation_start = min(r['game_creation_time'] for r in splits['validation'])
    test_start = min(r['game_creation_time'] for r in splits['test'])
    assert all(r['result_end_ms'] < validation_start for r in splits['train'])
    assert all(r['result_end_ms'] < test_start for r in splits['validation'])
    summary = dict(source_rows=len(rows), source_first_utc=utc(rows[0]['game_creation_time']),
        source_last_utc=utc(rows[-1]['game_creation_time']),
        splits={k: dict(count=len(v), first_utc=utc(v[0]['game_creation_time']), last_utc=utc(v[-1]['game_creation_time'])) for k, v in splits.items()},
        exclusions=dict(Counter(r['reason'] for r in exclusions)), temporal_purge_count=len(purged))
    save('cohort.json', summary)
    logging.info('Locked cohort: %s', summary)
    selection = dict(target_queues=[420, 440], mode='CLASSIC', source_limit=61000,
        warmup_rows=10000, history_window_limit=10000, after_ms=cutoff,
        validation_minimum=3000, test_minimum=1344, excluded_test_ids=sorted(protected),
        history_policy='result_end_strictly_before_start.v1',
        test_scope='1344 new targets reserved for this bounded pilot; prior research limitations declared',
        common_eligibility='18D timeline availability assessed before fitting, applied to both dimensions',
        protocol_sha256=digest(OUT / 'protocol.json'), audit_result_sha256=digest(AUDIT / 'result.json'),
        audited_source_snapshot_sha256=audit['source_snapshot_sha256'])
    pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
    original_extract = extract_aligned_inputs

    def shared_extract(match, version):
        return original_extract(match, version, pipeline=pipeline)

    # The public extractor explicitly accepts an immutable neutral pipeline.
    # Scope this optimization to the experiment; production files are unchanged.
    with patch('core.aligned_artifacts.extract_aligned_inputs', shared_extract):
        manifest = write_dataset(DATASET, rows, splits, source_db=str(ROOT / 'training_data.db'),
                                 selection=selection, exclusions=exclusions)
    assert all(a['status'] == 'ready' for a in manifest['availability'].values())
    assert dataset_runtime_sources() == runtime
    logging.info('Materialization completed; verifying common columns and independent replay panel')
    provider = FrozenHistoryProvider(rows)
    panel, comparisons, total = [], 0, 0
    for partition, records in splits.items():
        prefix = 'sealed_test' if partition == 'test' else partition
        arrays = {}
        for dim in (14, 18):
            with np.load(checked_path(DATASET, manifest, f'{prefix}{dim}.npz'), allow_pickle=False) as archive:
                arrays[dim] = {k: archive[k].copy() for k in archive.files}
            validate_tensor_inputs(arrays[dim], len(records), dim, partition)
        for key in ('team_a', 'team_b'):
            np.testing.assert_array_equal(arrays[14][key], arrays[18][key][..., [*range(12), 16, 17]])
        np.testing.assert_array_equal(arrays[14]['macro_clash'], arrays[18]['macro_clash'])
        total += len(records)
        for index in sorted(set(np.linspace(0, len(records) - 1, min(64, len(records)), dtype=int).tolist())):
            record = records[index]
            for dim in (14, 18):
                enriched = provider.enrich(record['feature_json'], include_timeline=dim == 18)
                actual = original_extract(enriched, f'nnriot.players{dim}.v1')
                for key in ('team_a', 'team_b', 'macro_clash'):
                    np.testing.assert_array_equal(actual[key][0], arrays[dim][key][index])
                    comparisons += 1
            panel.append(dict(partition=partition, match_id=record['match_id']))
    for name in manifest['files']:
        checked_path(DATASET, manifest, name)
    assert dataset_runtime_sources() == runtime
    save('preparation_result.json', dict(status='verified', dataset_dir=str(DATASET),
        cohort=summary, availability=manifest['availability'],
        common_columns_verified_targets=total, independent_replay_panel=panel,
        panel_tensor_comparisons=comparisons, replay_validation='deterministic sample, not exhaustive',
        hashes_verified=True, temporal_purges_verified=True, previous_tests_excluded=True,
        test_metrics_computed=False, dataset_manifest_sha256=digest(DATASET / 'dataset.json'),
        elapsed_seconds=round(time.time() - started, 2)))
    logging.info('COMPLETE: both dimensions ready, %d targets, %d replay tensor comparisons', total, comparisons)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('preparation_failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
