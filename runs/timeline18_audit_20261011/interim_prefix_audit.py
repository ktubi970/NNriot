"""Interim audit from a proven complete chronological prefix of cached rows."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DB = ROOT / 'training_data.db'
sys.path.insert(0, str(ROOT))
from core.aligned_partitions import select_targets
from core.frozen_history import FrozenHistoryProvider
from core.aligned_features import extract_aligned_inputs
from core.features.pipeline import DraftFeaturePipeline
from core.features.matchup_matrix import LaneMatchupTracker
from core.serving_contract import ContractError


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    started = time.time()
    paths = sorted((OUT / 'chunks').glob('*.json.gz'), key=lambda p: int(p.name.split('-')[0]))[:12]
    assert len(paths) == 12
    signature = None
    rows = []
    chunks = []
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            data = json.load(handle)
        if signature is None:
            signature = data['signature']
        assert data['signature'] == signature, 'Chunk source signature differs'
        before_ms, count = [int(v) for v in path.name.removesuffix('.json.gz').split('-')]
        assert count == len(data['rows']) == 1500
        ordered = sorted(data['rows'], key=lambda row: (row['game_creation_time'], row['match_id']))
        assert ordered == data['rows']
        assert before_ms == ordered[-1]['game_creation_time'] + 1
        if rows:
            assert rows[-1]['game_creation_time'] < ordered[0]['game_creation_time']
        rows.extend(ordered)
        chunks.append(dict(path=path.name, sha256=digest(path), rows=count))
    assert len(rows) == 18000
    assert len({row['match_id'] for row in rows}) == len(rows)
    runtime = signature['runtime']
    assert {name: digest(ROOT / name) for name in runtime} == runtime
    current_files = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                     for p in (DB, Path(str(DB) + '-wal')) if p.exists()}
    assert current_files == signature['files'], 'Source file metadata differs from cached capture'
    with closing(sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True)) as observer:
        observer.execute('PRAGMA query_only=ON')
        version_before = observer.execute('PRAGMA data_version').fetchone()[0]
        time_by_rowid = dict(observer.execute(
            "SELECT rowid,game_creation_time FROM matches INDEXED BY idx_matches_game_creation "
            "WHERE +rowid IN (SELECT rowid FROM matches WHERE game_mode='CLASSIC') "
            "AND game_creation_time>0 AND typeof(game_creation_time)='integer'"))
        headers = sorted(((mid, time_by_rowid[rid]) for rid, mid in observer.execute(
            'SELECT rowid,match_id FROM matches INDEXED BY sqlite_autoindex_matches_1')
            if rid in time_by_rowid), key=lambda r: (r[1], r[0]))
        version_after = observer.execute('PRAGMA data_version').fetchone()[0]
    assert version_before == version_after
    assert hashlib.sha256(json.dumps(headers).encode()).hexdigest() == signature['headers_sha256']
    assert [(r['match_id'], r['game_creation_time']) for r in rows] == headers[:18000]
    assert len(headers) == 18000 or rows[-1]['game_creation_time'] < headers[18000][1]
    print(json.dumps({'prefix_verified': len(rows), 'source_first_utc': utc(rows[0]['game_creation_time']),
                      'source_last_utc': utc(rows[-1]['game_creation_time'])}), flush=True)
    targets, rejected = select_targets(rows, excluded_ids=set(), warmup_rows=10000)
    provider = FrozenHistoryProvider(rows)
    pipeline = DraftFeaturePipeline(matchup_tracker=LaneMatchupTracker())
    eligible = 0
    failures = Counter()
    source_timeline_histogram = Counter()
    cold_players = Counter()
    samples = []
    for index, target in enumerate(targets):
        enriched = provider.enrich(target['feature_json'], include_timeline=True)
        provenance = enriched['priors_provenance']
        source_timeline_histogram[bool(provenance['timeline_source_available'])] += 1
        cold_players[provenance['cold_start_players']] += 1
        try:
            inputs = extract_aligned_inputs(enriched, 'nnriot.players18.v1', pipeline=pipeline)
            assert inputs['team_a'].shape == inputs['team_b'].shape == (1, 5, 18)
            assert inputs['macro_clash'].shape == (1, 15)
            eligible += 1
        except ContractError as exc:
            reason = type(exc).__name__ + ': ' + str(exc)
            failures[reason] += 1
            if len(samples) < 3:
                samples.append(dict(match_id=target['match_id'], reason=reason,
                                    history_matches=provenance['history_matches'],
                                    timeline_matches=provenance['timeline_matches']))
        if (index + 1) % 1000 == 0:
            print(json.dumps({'processed': index + 1, 'targets': len(targets), 'eligible18': eligible}), flush=True)
    assert {name: digest(ROOT / name) for name in runtime} == runtime
    result = dict(status='interim_completed', measured_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, source='training_data.db', source_payload_reads=0,
                  source_header_reads_only=True, prefix_complete_and_exact_against_source_headers=True,
                  header_data_version_before=version_before, header_data_version_after=version_after,
                  cached_signatures_identical=True, source_metadata_matches_cached_signature=True,
                  source_header_sha256=signature['headers_sha256'], chunks=chunks,
                  source_rows=len(rows), source_first_utc=utc(rows[0]['game_creation_time']),
                  source_last_utc=utc(rows[-1]['game_creation_time']), warmup_rows=10000,
                  target_count=len(targets), target_first_utc=utc(targets[0]['game_creation_time']) if targets else None,
                  target_last_utc=utc(targets[-1]['game_creation_time']) if targets else None,
                  target_selection_rejections=dict(Counter(r['reason'] for r in rejected)),
                  eligible18=eligible, failed18=len(targets)-eligible,
                  coverage18=eligible/len(targets) if targets else None, contract_failures=dict(failures),
                  timeline_source_available_targets=source_timeline_histogram[True],
                  timeline_source_unavailable_targets=source_timeline_histogram[False],
                  cold_start_players_per_target=dict(sorted(cold_players.items())), failure_examples=samples,
                  runtime_sources=runtime, runtime_unchanged=True, elapsed_seconds=round(time.time()-started,2),
                  limitations=['Only the oldest complete 18000-row source prefix and targets after its 10000-row warmup.',
                               'Not representative of the full database or newly collected matches.',
                               'Prior protected test IDs are included for this availability audit; no experiment partitions built.',
                               'Availability only; no predictive metrics, training, dataset freeze, or model promotion.'])
    (OUT / 'interim_prefix_coverage.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime_sources','chunks','failure_examples')}), flush=True)


if __name__ == '__main__':
    main()
