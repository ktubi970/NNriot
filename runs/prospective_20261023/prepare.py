"""Preregister the prospective comparison and freeze its dataset, with no SQL writes.

Test: the latest 10,000 eligible ranked targets, all created after every previously
evaluated test. Validation: the 5,000 completed targets just before. Train: the rest.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
DATASET = ROOT / 'runs/aligned_dataset_20261023_prospective'
ORIGINAL_DB = ROOT / 'training_data.db'
# Copy of ORIGINAL_DB (+ WAL, checkpointed into the copy) on the SSD, taken with no writer
# running: the HDD original is too fragmented for random payload reads.
# source_state() asserts both hold the same logical rows.
DB = Path(r'C:\Users\ktubi\AppData\Local\Temp\claude\D--projet-NNriot\7b480b51-cff3-4235-aa95-5cffdb5c449f'
          r'\scratchpad\snapshot\training_data.db')
SOURCE_LIMIT, WARMUP, VALIDATION, TEST = 150000, 10000, 5000, 10000
SEEDS = (42, 43, 44)
SMOKE = False  # set by a smoke harness only; never in the preregistered run
EVALUATED_SPLITS = ('runs/aligned_dataset_20261011_pilot/model.split.json',
                    'runs/aligned_dataset_20261009_reviewed/model.split.json')
EVALUATED_COHORT = 'runs/served_model_comparison_20261009_complete/cohort.jsonl'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def evaluated_tests():
    """Every match ID any earlier comparison scored, and the end of the latest such test."""
    ids, times = set(), []
    for name in EVALUATED_SPLITS:
        split = json.loads((ROOT / name).read_text(encoding='utf-8'))
        ids.update(split['split_metadata'].get('excluded_test_ids', []))
        ids.update(r['match_id'] for r in split['splits']['test'])
        times += [r['game_creation_time'] for r in split['splits']['test']]
    for line in (ROOT / EVALUATED_COHORT).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['partition'] in ('validation', 'test'):
            ids.add(row['match_id'])
            times.append(row['game_creation_time'])
    return ids, max(times)


def source_state(db):
    """Logical facts that change with any match/label write (data_version is per connection)."""
    # Index-only queries: the fragmented HDD original cannot afford table-row reads here.
    queries = dict(classic_matches="SELECT COUNT(*) FROM matches WHERE game_mode='CLASSIC'",
                   last_creation_ms='SELECT MAX(game_creation_time) FROM matches',
                   max_rowid='SELECT MAX(rowid) FROM matches', training_rows='SELECT COUNT(*) FROM training_dataset',
                   training_max_id='SELECT MAX(id) FROM training_dataset',
                   timelines='SELECT COUNT(*) FROM match_timelines')
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        return {name: conn.execute(sql).fetchone()[0] for name, sql in queries.items()}


def main():
    import numpy as np
    from core.aligned_artifacts import checked_path, dataset_runtime_sources, validate_tensor_inputs, write_dataset
    from core.aligned_features import extract_aligned_inputs
    from core.aligned_partitions import partition_targets, select_targets
    from core.features.matchup_matrix import LaneMatchupTracker
    from core.features.pipeline import DraftFeaturePipeline
    from core.frozen_history import FrozenHistoryProvider
    from core.pre_match_history import read_recent_rows
    from core.serving_contract import ContractError

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    started = time.time()
    assert not DATASET.exists(), 'Output directory must be new'
    runtime = dataset_runtime_sources()
    evaluated, after_ms = evaluated_tests()
    state_before = source_state(DB)
    assert source_state(ORIGINAL_DB) == state_before, 'Snapshot differs from the original database'
    assert SMOKE or state_before['classic_matches'] <= SOURCE_LIMIT, 'Source limit would drop old history'
    protocol = dict(status='preregistered', created_utc=datetime.now(timezone.utc).isoformat(),
        kind='prospective_architecture_comparison', dataset_dir=str(DATASET),
        source_db=str(ORIGINAL_DB), source_snapshot=str(DB),
        source_snapshot_method='file copy of database and WAL with no writer running; WAL checkpointed into the copy',
        source_state=state_before, source_limit=SOURCE_LIMIT, warmup_rows=WARMUP,
        history_window_limit=10000, validation_count=VALIDATION, test_count=TEST,
        test_after_utc=utc(after_ms), previously_evaluated_ids=len(evaluated),
        previously_evaluated_policy='never in validation or test; allowed in train',
        common_eligibility='18D extraction succeeds; applied to both dimensions before partitioning',
        deepsets=dict(scripts={'14': 'scripts/train_deepsets_v3.py', '18': 'scripts/train_deepsets_v4_timeline.py'},
                      seeds=list(SEEDS), epochs=5, batch_size=64, learning_rate=0.0008, early_stopping=False,
                      side_prior_init='training Blue rate', checkpoint_selection='minimum validation winner log-loss',
                      ensemble='mean of the three calibrated seed probabilities; no fitting'),
        baselines=dict(module='core/tabular_baselines.py', families=['logistic', 'lightgbm'], seed=42,
                       selection='validation only (C grid / early stopping)'),
        calibration_partition='validation', constant_baseline='training P(Red)',
        primary_metrics=['calibrated test log-loss: DeepSets 18D ensemble minus LightGBM 18D',
                         'calibrated test log-loss: DeepSets 18D ensemble minus logistic 18D'],
        secondary_metrics=['each calibrated model minus constant', '18D minus 14D within each family',
                           'per-seed DeepSets spread', 'AUC, Brier, accuracy, fixed-threshold selective risk'],
        bootstrap=dict(method='paired moving chronological block bootstrap', block_size=100,
                       repeats=2000, seed=20261023),
        decision_rule=('DeepSets shows added value only if both primary 95% intervals lie entirely below 0; '
                       'otherwise the simpler baseline is preferred'),
        test_policy='lock every checkpoint, calibration and baseline, then score the test once; no tuning after',
        collection_restarted=False, production_promotion=False,
        preparation_script_sha256=digest(__file__))
    save('protocol.json', protocol)

    logging.info('Reading %s read-only (all CLASSIC rows, timelines compacted)', DB)
    rows = read_recent_rows(DB, SOURCE_LIMIT, timelines=True, include_labels=True)
    assert source_state(DB) == state_before, 'Source changed during the read'
    logging.info('Read %d rows from %s to %s', len(rows), utc(rows[0]['game_creation_time']),
                 utc(rows[-1]['game_creation_time']))
    targets, exclusions = select_targets(rows, excluded_ids=set(), warmup_rows=WARMUP)
    pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
    provider = FrozenHistoryProvider(rows)
    eligible = []
    for n, target in enumerate(targets, 1):
        try:
            extract_aligned_inputs(provider.enrich(target['feature_json'], include_timeline=True),
                                   'nnriot.players18.v1', pipeline=pipeline)
            eligible.append(target)
        except ContractError as exc:
            exclusions.append(dict(match_id=target['match_id'], reason='timeline18_unavailable',
                                   detail=type(exc).__name__ + ': ' + str(exc)))
        if n % 10000 == 0:
            logging.info('18D eligibility %d/%d, eligible=%d', n, len(targets), len(eligible))
    del provider
    gc.collect()
    splits, purged = partition_targets(eligible, after_ms=after_ms, validation_count=VALIDATION, test_count=TEST)
    exclusions.extend(purged)
    assert len(splits['test']) == TEST and len(splits['validation']) == VALIDATION
    assert all(r['game_creation_time'] > after_ms for p in ('validation', 'test') for r in splits[p])
    assert not evaluated.intersection(r['match_id'] for p in ('validation', 'test') for r in splits[p])
    validation_start = min(r['game_creation_time'] for r in splits['validation'])
    test_start = min(r['game_creation_time'] for r in splits['test'])
    assert all(r['result_end_ms'] < validation_start for r in splits['train'])
    assert all(r['result_end_ms'] < test_start for r in splits['validation'])
    summary = dict(source_rows=len(rows), source_first_utc=utc(rows[0]['game_creation_time']),
        source_last_utc=utc(rows[-1]['game_creation_time']),
        splits={k: dict(count=len(v), first_utc=utc(v[0]['game_creation_time']),
                        last_utc=utc(v[-1]['game_creation_time']),
                        red_win_rate=sum(r['labels_json']['winner'] for r in v) / len(v)) for k, v in splits.items()},
        previously_evaluated_in_train=sum(r['match_id'] in evaluated for r in splits['train']),
        exclusions=dict(Counter(r['reason'] for r in exclusions)), temporal_purge_count=len(purged))
    save('cohort.json', summary)
    logging.info('Locked cohort: %s', json.dumps(summary))
    selection = dict(target_queues=[420, 440], mode='CLASSIC', source_limit=SOURCE_LIMIT,
        warmup_rows=WARMUP, history_window_limit=10000, after_ms=after_ms,
        validation_minimum=VALIDATION, test_minimum=TEST, excluded_test_ids=[],
        previously_evaluated_ids_sha256=hashlib.sha256('\n'.join(sorted(evaluated)).encode()).hexdigest(),
        history_policy='result_end_strictly_before_start.v1',
        test_scope='latest 10000 eligible targets after every previously evaluated test',
        common_eligibility=protocol['common_eligibility'], protocol_sha256=digest(OUT / 'protocol.json'))

    def shared_extract(match, version):
        return extract_aligned_inputs(match, version, pipeline=pipeline)

    # Same scoped optimization as the 11 October pilot: one immutable neutral pipeline.
    with patch('core.aligned_artifacts.extract_aligned_inputs', shared_extract):
        manifest = write_dataset(DATASET, rows, splits, source_db=str(ORIGINAL_DB), selection=selection,
                                 exclusions=exclusions)
    assert all(a['status'] == 'ready' for a in manifest['availability'].values()), manifest['availability']
    assert dataset_runtime_sources() == runtime
    logging.info('Materialization completed; verifying common columns and a replay panel')
    provider = FrozenHistoryProvider(rows)
    panel, comparisons = [], 0
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
        for index in sorted(set(np.linspace(0, len(records) - 1, 64, dtype=int).tolist())):
            record = records[index]
            for dim in (14, 18):
                actual = extract_aligned_inputs(provider.enrich(record['feature_json'], include_timeline=dim == 18),
                                                f'nnriot.players{dim}.v1')
                for key in ('team_a', 'team_b', 'macro_clash'):
                    np.testing.assert_array_equal(actual[key][0], arrays[dim][key][index])
                    comparisons += 1
            panel.append(dict(partition=partition, match_id=record['match_id']))
    assert source_state(DB) == source_state(ORIGINAL_DB) == state_before, 'Source changed during preparation'
    save('preparation_result.json', dict(status='verified', dataset_dir=str(DATASET), cohort=summary,
        availability={k: {f: v[f] for f in ('status', 'eligible', 'total', 'coverage')}
                      for k, v in manifest['availability'].items()},
        independent_replay_panel=panel, panel_tensor_comparisons=comparisons,
        replay_validation='deterministic sample, not exhaustive', hashes_verified=True,
        temporal_purges_verified=True, previous_tests_excluded=True, test_metrics_computed=False,
        dataset_manifest_sha256=digest(DATASET / 'dataset.json'), elapsed_seconds=round(time.time() - started, 2)))
    logging.info('COMPLETE: %d targets, %d replay tensor comparisons', sum(map(len, splits.values())), comparisons)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (OUT / 'preparation_failure.json').write_text(json.dumps(dict(type=type(exc).__name__, message=str(exc))),
                                                      encoding='utf-8')
        raise
