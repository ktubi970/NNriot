# scripts/prepare_aligned_dataset.py : imports et inventaire
import argparse
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
import sys

def inspect_sources(paths):
    results = []
    for path in paths:
        path = Path(path).resolve()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.execute('PRAGMA query_only=ON')
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            count, first, last = conn.execute(
                'SELECT COUNT(*), MIN(game_creation_time), MAX(game_creation_time) FROM matches').fetchone()
            timelines = conn.execute('SELECT COUNT(*) FROM match_timelines').fetchone()[0] if 'match_timelines' in tables else 0
            results.append(dict(path=str(path), bytes=path.stat().st_size,
                matches=count, first_creation_ms=first, last_creation_ms=last,
                timeline_rows=timelines))
    return results

def main(argv=None):
    from core.serving_contract import ContractError
    parser = argparse.ArgumentParser(description='Prepare and audit serving-aligned datasets')
    commands = parser.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory')
    inventory.add_argument('--db', type=Path, action='append', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--db', type=Path, required=True)
    freeze.add_argument('--after-cohort', type=Path, required=True)
    freeze.add_argument('--output-dir', type=Path, required=True)
    freeze.add_argument('--source-limit', type=int, default=61000)
    replay = commands.add_parser('verify')
    replay.add_argument('--dataset-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inventory':
            result = inspect_sources(args.db)
        elif args.command == 'freeze':
            result = prepare(args.db, args.output_dir, args.after_cohort,
                             source_limit=args.source_limit)
        else:
            result = verify(args.dataset_dir)
        report = result if args.command in ('inventory', 'verify') else {
            'dataset_dir': str(args.output_dir.resolve()), 'status': 'ready_14d',
            'availability': {v: {k: info[k] for k in ('status', 'eligible', 'total', 'coverage')}
                             for v, info in result['availability'].items()}}
        if args.command == 'verify':
            report = {**result, 'availability': {v: {k: info[k] for k in ('status', 'eligible', 'total', 'coverage')}
                      for v, info in result['availability'].items()}}
        print(json.dumps(report, allow_nan=False, ensure_ascii=False))
        return 0
    except (ValueError, FileExistsError, FileNotFoundError, ContractError, sqlite3.Error) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

def verify(dataset_dir):
    import gzip
    import json
    import numpy as np
    from core.aligned_artifacts import checked_path, dataset_runtime_sources, validate_tensor_inputs
    from core.aligned_features import extract_aligned_inputs
    from core.frozen_history import FrozenHistoryProvider
    root = Path(dataset_dir).resolve()
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    if meta['runtime_sources'] != dataset_runtime_sources():
        raise ValueError('feature runtime SHA-256 mismatch')
    for name in meta['files']:
        checked_path(root, meta, name)
    rows = json.loads(gzip.decompress(checked_path(root, meta, 'source_rows.json.gz').read_bytes()))
    provider = FrozenHistoryProvider(rows)
    comparisons = 0
    for version, availability in meta['availability'].items():
        if availability['status'] != 'ready':
            continue
        dim = {'nnriot.players14.v1': 14, 'nnriot.players18.v1': 18}[version]
        for prefix in ('train', 'validation', 'sealed_test'):
            records = json.loads(checked_path(root, meta, prefix + '.json').read_text(encoding='utf-8'))
            with np.load(checked_path(root, meta, f'{prefix}{dim}.npz'), allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
                validate_tensor_inputs(arrays, len(records), dim, prefix)
                for index, record in enumerate(records):
                    enriched = provider.enrich(record['feature_json'], include_timeline=dim == 18)
                    x = extract_aligned_inputs(enriched, version)
                    for key in ('team_a', 'team_b', 'macro_clash'):
                        if not np.array_equal(x[key][0], arrays[key][index]):
                            raise ValueError(f'tensor mismatch: {record["match_id"]}/{key}/{dim}')
                        comparisons += 1
    return {'status': 'verified', 'tensor_comparisons': comparisons,
            'test_metrics_computed': False, 'availability': meta['availability']}

def prepare(db, output, after_cohort, *, source_limit=61000):
    import json
    from core.pre_match_history import read_recent_rows
    from core.aligned_partitions import select_targets, partition_targets
    from core.aligned_artifacts import write_dataset
    from core.calibration import sha256_file
    if Path(output).exists():
        raise FileExistsError(f'occupied output directory: {output}')
    cohort = [json.loads(line) for line in Path(after_cohort).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if not cohort or any(type(r.get('game_creation_time')) is not int
            or r['game_creation_time'] <= 0
            or r.get('partition') not in ('history', 'validation', 'test')
            or not isinstance(r.get('match_id'), str) or not r['match_id'] for r in cohort):
        raise ValueError('invalid previous cohort')
    if len({r['match_id'] for r in cohort}) != len(cohort):
        raise ValueError('duplicate match_id in previous cohort')
    if not any(r['partition'] == 'test' for r in cohort):
        raise ValueError('previous cohort has no test partition')
    after_ms = max(row['game_creation_time'] for row in cohort)
    excluded_ids = {row['match_id'] for row in cohort if row['partition'] == 'test'}
    rows = read_recent_rows(db, source_limit, timelines=True, include_labels=True)
    targets, exclusions = select_targets(rows, excluded_ids=excluded_ids)
    splits, purged = partition_targets(targets, after_ms=after_ms)
    selection = dict(target_queues=[420, 440], mode='CLASSIC', source_limit=source_limit,
        warmup_rows=10000, history_window_limit=10000, after_ms=after_ms,
        validation_minimum=3000, test_minimum=3000,
        after_cohort_sha256=sha256_file(after_cohort), excluded_test_ids=sorted(excluded_ids),
        history_policy='result_end_strictly_before_start.v1',
        test_scope='reserved for this future experiment; prior research limitations declared')
    return write_dataset(output, rows, splits, source_db=str(db),
                         selection=selection, exclusions=exclusions + purged)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    raise SystemExit(main())
