"""Read-only 18D coverage audit; no training, fetching, or model changes.

Run from repository root with .venv312/Scripts/python.exe -u
runs/timeline18_audit_20261011/audit.py. Source facts are captured by the
production readers with a persistent SQLite data_version observer rejecting
any concurrent source commit. Replay uses production code.
"""
from __future__ import annotations

import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
DB = ROOT / 'training_data.db'


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding='utf-8')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def timeline_reason(timeline, participants):
    """Diagnostic only; acceptance always comes from the production extractor."""
    import math
    info = timeline.get('info', timeline) if isinstance(timeline, dict) else None
    if not isinstance(info, dict):
        return 'invalid_info'
    frames = info.get('frames')
    if not isinstance(frames, list) or not frames:
        return 'missing_frames'
    frame = frames[min(15, len(frames) - 1)]
    if not isinstance(frame, dict):
        return 'invalid_target_frame'
    if any(not isinstance(f, dict) or not isinstance(f.get('events'), list)
           for f in frames):
        return 'missing_event_lists'
    pframes = frame.get('participantFrames')
    if not isinstance(pframes, dict):
        return 'missing_participant_frames'
    for i in range(1, 11):
        p = pframes.get(str(i))
        if not isinstance(p, dict):
            return 'missing_participant_frames'
        for key in ('totalGold', 'minionsKilled', 'jungleMinionsKilled', 'xp'):
            v = p.get(key)
            if type(v) not in (int, float) or not math.isfinite(v):
                return 'missing_or_invalid_' + key
    return 'other_strict_extractor_rejection'


def read_chunk(args):
    snapshot, limit, before_ms, signature = args
    cache = OUT / 'chunks' / f'{before_ms}-{limit}.json.gz'
    if cache.exists():
        with gzip.open(cache, 'rt', encoding='utf-8') as f:
            saved = json.load(f)
        if saved['signature'] == signature:
            return saved['rows'], Counter(saved['observed'])
    from core import pre_match_history as history_module
    from core import point_in_time_dataset, pre_match_dataset
    observed = Counter()
    parser = history_module._parse_timeline_json
    extractor = history_module.extract_timeline_lane_metrics
    original_json = (point_in_time_dataset.json, pre_match_dataset.json)
    try:
        from pydantic_core import from_json
    except ImportError:
        from_json = None

    class CheckedJson:
        def __getattr__(self, name):
            return getattr(json, name)

        def loads(self, data, *args, **kwargs):
            if from_json is None or args or kwargs:
                return json.loads(data, *args, **kwargs)
            try:
                result = from_json(data)
            except ValueError:
                observed['fast_json_fallback'] += 1
                return json.loads(data)
            observed['fast_json_calls'] += 1
            if observed['fast_json_calls'] % 100 == 1:
                assert result == json.loads(data), 'JSON parser parity failure'
                observed['json_parser_parity_checks'] += 1
            return result

    point_in_time_dataset.json = pre_match_dataset.json = CheckedJson()

    def audit_parse(raw):
        timeline = parser(raw)
        observed['timeline_read_calls'] += 1
        if not raw:
            observed['timeline_absent_or_empty'] += 1
        elif not timeline:
            observed['timeline_unparseable_or_empty_object'] += 1
        else:
            observed['timeline_parsed_nonempty'] += 1
        return timeline

    def audit_extract(timeline, participants=None, **kwargs):
        try:
            result = extractor(timeline, participants=participants, **kwargs)
        except (TypeError, ValueError, KeyError, AttributeError):
            observed['extractor_exception'] += 1
            raise
        if result:
            observed['strict_timeline_accepted'] += 1
        else:
            observed['strict_timeline_rejected'] += 1
            observed['reject_' + timeline_reason(timeline, participants)] += 1
        return result

    history_module._parse_timeline_json = audit_parse
    history_module.extract_timeline_lane_metrics = audit_extract
    try:
        rows = history_module.read_recent_rows(snapshot, limit, before_ms=before_ms,
                                               timelines=True, include_labels=True)
    finally:
        history_module._parse_timeline_json = parser
        history_module.extract_timeline_lane_metrics = extractor
        point_in_time_dataset.json, pre_match_dataset.json = original_json
    with gzip.open(cache.with_suffix('.tmp'), 'wt', encoding='utf-8') as f:
        json.dump({'signature': signature, 'rows': rows, 'observed': dict(observed)},
                  f, separators=(',', ':'), allow_nan=False)
    cache.with_suffix('.tmp').replace(cache)
    return rows, observed


def main():
    started = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from core import pre_match_history as history_module
    from core.aligned_partitions import select_targets, partition_targets
    from core.frozen_history import FrozenHistoryProvider
    from core.aligned_features import extract_aligned_inputs
    from core.features.pipeline import DraftFeaturePipeline
    from core.features.matchup_matrix import LaneMatchupTracker
    from core.serving_contract import ContractError, runtime_feature_sources

    runtime = runtime_feature_sources()
    for name in ('core/frozen_history.py', 'core/aligned_partitions.py'):
        runtime[name] = digest(ROOT / name)
    write_json('plan.json', {
        'source': str(DB), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All positive-integer-timestamp CLASSIC source rows; 10000-row warmup; queues 420/440 targets',
        'acceptance': 'Production FrozenHistoryProvider plus extract_aligned_inputs players18.v1 for every selected target',
        'snapshot': 'Parallel read-only production readers over disjoint complete creation groups; persistent data_version guard rejects any concurrent commit',
        'writes': 'Audit artifacts only; no source SQL writes, collection, training, prediction metrics or promotion',
        'runtime_sources': runtime,
    })
    source_read_start = datetime.now(timezone.utc).isoformat()
    (OUT / 'chunks').mkdir(exist_ok=True)
    observer = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True)
    observer.execute('PRAGMA query_only=ON')
    initial_data_version = observer.execute('PRAGMA data_version').fetchone()[0]
    logging.info('Guarded read-only capture, data_version=%d', initial_data_version)
    # Both scans use covering indexes: creation is after the huge raw_json
    # column in table storage, so a naive mode-filtered query reads gigabytes.
    time_by_rowid = dict(observer.execute(
        "SELECT rowid,game_creation_time FROM matches INDEXED BY idx_matches_game_creation "
        "WHERE +rowid IN (SELECT rowid FROM matches WHERE game_mode='CLASSIC') "
        "AND game_creation_time>0 AND typeof(game_creation_time)='integer'"))
    headers = sorted(((mid, time_by_rowid[rid]) for rid, mid in observer.execute(
        'SELECT rowid,match_id FROM matches INDEXED BY sqlite_autoindex_matches_1')
        if rid in time_by_rowid), key=lambda r: (r[1], r[0]))
    db_counts = {t: observer.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                 for t in ('matches', 'training_dataset', 'match_timelines')}
    signature = {'runtime': runtime, 'db_counts': db_counts,
                 'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                           for p in (DB, Path(str(DB) + '-wal')) if p.exists()},
                 'headers_sha256': hashlib.sha256(json.dumps(headers).encode()).hexdigest()}
    tasks = []
    start = 0
    while start < len(headers):
        stop = min(start + 1500, len(headers))
        while stop < len(headers) and headers[stop][1] == headers[stop-1][1]:
            stop += 1
        tasks.append((str(DB), stop-start, headers[stop-1][1]+1, signature))
        start = stop
    rows, observed = [], Counter()
    logging.info('Inventory complete: %d source rows, %d chunks, 3 workers', len(headers), len(tasks))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(read_chunk, task) for task in tasks]):
            chunk, counts = future.result()
            rows.extend(chunk)
            observed.update(counts)
            logging.info('Captured %d/%d rows; strict timelines accepted=%d, rejected=%d',
                         len(rows), len(headers), observed['strict_timeline_accepted'],
                         observed['strict_timeline_rejected'])
            assert observer.execute('PRAGMA data_version').fetchone()[0] == initial_data_version, 'Source changed during capture'
    final_data_version = observer.execute('PRAGMA data_version').fetchone()[0]
    observer.close()
    assert final_data_version == initial_data_version, 'Source changed during capture'
    rows.sort(key=lambda r: (r['game_creation_time'], r['match_id']))
    assert [(r['match_id'], r['game_creation_time']) for r in rows] == headers
    source_read_end = datetime.now(timezone.utc).isoformat()
    logging.info('Source captured: %d rows, timeline checks %s', len(rows), dict(observed))
    with gzip.open(OUT / 'source_rows.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(rows, f, separators=(',', ':'), allow_nan=False)

    # Exclusions for any future experiment preserve all known test memberships.
    old = json.loads((ROOT / 'runs/aligned_dataset_20261009_reviewed/model.split.json').read_text())
    protected = set(old['split_metadata']['excluded_test_ids'])
    protected.update(r['match_id'] for r in old['splits']['test'])
    previous_end = max(r['game_creation_time'] for r in old['splits']['test'])
    targets, rejected = select_targets(rows, excluded_ids=set(), warmup_rows=10000)
    provider = FrozenHistoryProvider(rows)
    pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
    row_by_id = {r['match_id']: r for r in rows}
    groups = defaultdict(Counter)
    failures, player_histogram, history_counts = Counter(), Counter(), Counter()
    missing_sources, uncovered_players = Counter(), Counter()
    eligible_ids = set()
    samples = {}
    checks = []
    source_metrics = Counter('valid' if r.get('timeline_metrics') else 'missing_or_rejected' for r in rows)

    with gzip.open(OUT / 'targets.jsonl.gz', 'wt', encoding='utf-8') as f:
        for n, target in enumerate(targets):
            match = target['feature_json']
            as_of = target['game_creation_time']
            identities = {p.get('puuid') for k in ('team_a', 'team_b')
                          for p in match[k] if p.get('puuid')}
            stop = bisect.bisect_left(provider.times, as_of)
            start = max(0, stop - provider.history_limit)
            indexes = set()
            for puuid in identities:
                positions = provider.positions.get(puuid, [])
                indexes.update(positions[bisect.bisect_left(positions, start):
                                         bisect.bisect_left(positions, stop)])
            player_counts = Counter()
            completed = 0
            backfill = []
            for i in sorted(indexes):
                rec = provider.facts[i]
                parts = [p for p in rec['participants'] if p.get('puuid') in identities]
                if any(type(p.get('win')) is not bool for p in parts):
                    continue
                end = rec['result_end_ms']
                if type(end) is not int or end >= as_of:
                    continue
                completed += 1
                valid = {p for p, m in (rec.get('timeline_metrics') or {}).items()
                         if p in identities and isinstance(m, dict)
                         and all(k in m and isinstance(m[k], (int, float))
                                 and __import__('math').isfinite(m[k])
                                 for k in ('gd15', 'csd15', 'xpd15', 'fbi10'))}
                player_counts.update(valid)
                if not valid:
                    backfill.append(rec['match_id'])
            enriched = provider.enrich(match, include_timeline=True)
            provenance = enriched['priors_provenance']
            assert completed == provenance['history_matches']
            assert bool(player_counts) == provenance['timeline_source_available']
            try:
                inputs = extract_aligned_inputs(enriched, 'nnriot.players18.v1', pipeline=pipeline)
                assert inputs['team_a'].shape == inputs['team_b'].shape == (1, 5, 18)
                assert inputs['macro_clash'].shape == (1, 15)
                status = 'eligible'
                eligible_ids.add(target['match_id'])
            except ContractError as exc:
                status = type(exc).__name__ + ': ' + str(exc)
                failures[status] += 1
            if not player_counts:
                reason = ('no_prior_player_match_in_window' if not indexes else
                          'no_completed_valid_player_result' if not completed else
                          'completed_history_without_usable_timeline')
                history_counts[reason] += 1
                missing_sources.update(backfill)
                uncovered_players.update(identities)
            player_histogram[len(player_counts)] += 1
            keys = ('all', 'date:' + utc(as_of)[:10], 'region:' + str(target['region']),
                    'patch:' + str(target['patch']),
                    'recent_30d' if as_of >= provider.times[-1] - 30 * 86400000 else 'older_than_30d')
            for key in keys:
                groups[key]['targets'] += 1
                groups[key]['eligible'] += int(status == 'eligible')
                groups[key]['player_slots'] += 10
                groups[key]['players_with_timeline'] += len(player_counts)
            record = dict(match_id=target['match_id'], game_creation_time=as_of,
                          status=status, player_timeline_count=len(player_counts),
                          minimum_player_history=min(player_counts.get(p.get('puuid'), 0)
                                                     for k in ('team_a', 'team_b') for p in match[k]),
                          timeline_history_matches=provenance['timeline_matches'],
                          history_matches=provenance['history_matches'],
                          own_timeline_usable=bool(row_by_id[target['match_id']].get('timeline_metrics')),
                          protected_prior_test=target['match_id'] in protected)
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
            sample_key = status if status != 'eligible' else 'eligible_players_' + str(len(player_counts))
            if sample_key not in samples:
                samples[sample_key] = record
            if n % 2500 == 0 or n == len(targets) - 1:
                logging.info('Audit targets %d/%d, eligible=%d', n + 1, len(targets), len(eligible_ids))
            # Independent direct history replay checks diagnostic counts on a fixed panel.
            if n % 2000 == 0:
                hist = history_module.build_target_history(
                    [provider.facts[i] for i in sorted(indexes)], match, as_of_ms=as_of)
                real_counts = {p: t['count'] for p, t in hist.tracker.player_timeline_totals.items()}
                assert dict(player_counts) == real_counts
                checks.append(target['match_id'])

    candidate_targets = [t for t in targets if t['match_id'] not in protected]
    partition_reports = {}
    for name, population in (
        ('all_admissible_unprotected', candidate_targets),
        ('timeline_eligible_unprotected', [t for t in candidate_targets if t['match_id'] in eligible_ids])):
        try:
            splits, purged = partition_targets(population, after_ms=previous_end)
            partition_reports[name] = {
                'status': 'partitionable', 'purged': len(purged),
                'splits': {k: {'count': len(v),
                               'eligible18': sum(t['match_id'] in eligible_ids for t in v),
                               'first_utc': utc(v[0]['game_creation_time']),
                               'last_utc': utc(v[-1]['game_creation_time'])} for k, v in splits.items()},
            }
            write_json('prospective_' + name + '_split.json', {
                k: [dict(match_id=t['match_id'], game_creation_time=t['game_creation_time']) for t in v]
                for k, v in splits.items()})
        except ValueError as exc:
            partition_reports[name] = {'status': 'blocked', 'reason': str(exc)}

    current_hashes = {p: digest(ROOT / p) for p in runtime}
    assert current_hashes == runtime, 'Production sources changed during audit'
    result = {
        'status': 'completed', 'database': str(DB), 'database_counts_at_snapshot': db_counts,
        'snapshot_method': 'Read-only source with persistent data_version guard, disjoint complete-creation chunks using production read_recent_rows; full header equality asserted',
        'source_data_version_start': initial_data_version, 'source_data_version_end': final_data_version,
        'snapshot_read_started_utc': source_read_start, 'snapshot_read_finished_utc': source_read_end,
        'source_snapshot_sha256': digest(OUT / 'source_rows.json.gz'),
        'source_rows': len(rows), 'source_first_utc': utc(provider.times[0]),
        'source_last_utc': utc(provider.times[-1]),
        'source_timeline_diagnostics': dict(observed), 'source_metric_status': dict(source_metrics),
        'invalid_source_facts': len(provider.invalid), 'warmup_rows': 10000,
        'target_selection_rejections': dict(Counter(r['reason'] for r in rejected)),
        'target_count': len(targets), 'eligible18': len(eligible_ids),
        'coverage18': len(eligible_ids) / len(targets) if targets else None,
        'contract_failures': dict(failures), 'history_unavailability_reasons': dict(history_counts),
        'players_with_history_per_target': dict(sorted(player_histogram.items())),
        'groups': {k: dict(v) for k, v in sorted(groups.items())},
        'prospective_partitioning': partition_reports,
        'protected_prior_test_ids': len(protected), 'future_test_after_utc': utc(previous_end),
        'backfill_existing_history_candidates': len(missing_sources),
        'blocked_target_unique_players': len(uncovered_players),
        'top_backfill_existing_history': missing_sources.most_common(25),
        'examples': samples, 'diagnostic_replay_panel_checks': len(checks),
        'runtime_sources': runtime, 'runtime_unchanged': True,
        'python_version': sys.version,
        'capture_optimization': 'Installed pydantic_core JSON parser, checked against stdlib every 100 parses; production extraction and history semantics unchanged; checkpointed chunks bound to source metadata and headers',
        'elapsed_seconds': round(time.time() - started, 2),
        'limitations': [
            'recent_30d means 30 days ending at the newest source match creation, not the wall-clock audit date.',
            'Coverage is data and feature availability, not predictive performance.',
            'Any timeline-admissible subset is a changed population; exclusions must remain explicit.',
            'No new dataset frozen, model trained, or reserved test predictive metric computed.',
            'Historical results assumed observable at result end; ingestion timestamps unknown.',
        ],
    }
    write_json('result.json', result)
    write_json('selection_rejections.json', rejected)
    write_json('backfill_candidates.json', [{'match_id': mid, 'blocked_targets_using_history': n}
                                           for mid, n in missing_sources.most_common()])
    logging.info('COMPLETE %d/%d eligible (%.3f%%); result.json saved',
                 len(eligible_ids), len(targets), 100 * result['coverage18'])


if __name__ == '__main__':
    main()
