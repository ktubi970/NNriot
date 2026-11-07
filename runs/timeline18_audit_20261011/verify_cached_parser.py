"""Independent Python 3.12 parity panel against cached capture rows."""
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
from core.pre_match_dataset import _decode
from core.point_in_time_dataset import _parse_timeline_json
from core.features.timeline_metrics import extract_timeline_lane_metrics


def main():
    paths = sorted((OUT / 'chunks').glob('*.json.gz'))[:12]
    panel = []
    for index, path in enumerate(paths):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            saved = json.load(handle)
        positions = [0, len(saved['rows']) - 1]
        if index < 6:
            positions.append(len(saved['rows']) // 2)
        for position in positions:
            panel.append((path.name, position, saved['rows'][position]))
    ids = [row['match_id'] for _, _, row in panel]
    assert len(ids) == len(set(ids)) == 30
    findings = []
    with closing(sqlite3.connect((ROOT / 'training_data.db').as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        version_before = conn.execute('PRAGMA data_version').fetchone()[0]
        conn.execute('BEGIN')
        conn.row_factory = sqlite3.Row
        placeholders = ','.join('?' for _ in ids)
        sql = f'''SELECT m.match_id,m.game_creation_time,m.raw_json,
            t.winner_label,t.labels_json,
            (SELECT timeline_json FROM match_timelines mt WHERE mt.match_id=m.match_id) AS timeline_json
            FROM matches m LEFT JOIN training_dataset t ON t.match_id=m.match_id
            WHERE m.match_id IN ({placeholders})'''
        fetched = {row['match_id']: dict(row) for row in conn.execute(sql, ids)}
        assert set(fetched) == set(ids)
        conn.commit()
        version_after = conn.execute('PRAGMA data_version').fetchone()[0]
    assert version_before == version_after, 'Source committed during verification panel'
    part_keys = ('puuid','participantId','teamId','championName','teamPosition','win','kills','deaths','assists',
                 'totalMinionsKilled','neutralMinionsKilled','goldEarned')
    info_keys = ('gameStartTimestamp','gameEndTimestamp','gameDuration','gameVersion','gameMode','queueId')
    for chunk, position, cached in panel:
        raw = fetched[cached['match_id']]
        decoded = _decode(raw['raw_json'])
        info = decoded['info']
        parts = [{k: p[k] for k in part_keys if k in p} for p in info.get('participants', [])]
        small = {k: info[k] for k in info_keys if k in info}
        small['gameCreation'] = raw['game_creation_time']
        small['participants'] = parts
        timeline = _parse_timeline_json(raw['timeline_json'])
        metrics = extract_timeline_lane_metrics(timeline, participants=parts, require_observations=True) if timeline else None
        checks = {
            'compact_raw_facts_equal': {'info': small} == _decode(cached['raw_json']),
            'timeline_metrics_equal': metrics == cached.get('timeline_metrics'),
            'creation_equal': raw['game_creation_time'] == cached['game_creation_time'],
            'winner_equal': raw['winner_label'] == cached['winner_label'],
            'labels_equal': _decode(raw['labels_json']) == _decode(cached['labels_json']),
            'no_unexpected_source_error': not cached.get('source_error'),
        }
        entry = dict(match_id=cached['match_id'], chunk=chunk, row_position=position,
                     checks=checks, timeline_present=bool(raw['timeline_json']),
                     original_timeline_parsed=bool(timeline), original_metrics_present=bool(metrics),
                     cached_metrics_present=bool(cached.get('timeline_metrics')),
                     source_timeline_type=type(raw['timeline_json']).__name__,
                     source_timeline_bytes=len(raw['timeline_json'] or ''),
                     raw_payload_sha256=hashlib.sha256(raw['raw_json'].encode()).hexdigest())
        findings.append(entry)
        if not all(checks.values()):
            print(json.dumps({'MISMATCH': entry}), flush=True)
    result = dict(status='passed' if all(all(r['checks'].values()) for r in findings) else 'failed',
                  verified_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                  sqlite_version=sqlite3.sqlite_version, parser='original production stdlib json in .venv312',
                  method='Read 30 exact IDs only; first/last rows in oldest 12 completed chunks, plus midpoint of oldest 6',
                  source_read_only=True, source_data_version_before=version_before,
                  source_data_version_after=version_after, checked_rows=len(findings),
                  checked_chunks=len(paths), failed_rows=sum(not all(r['checks'].values()) for r in findings),
                  limitations=['Deterministic panel, not exhaustive equality of all cached rows.'], rows=findings)
    (OUT / 'parser_parity_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
