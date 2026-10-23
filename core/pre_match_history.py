"""Read-only result-end ordered priors, shared by serving and evaluation.

Retrospective availability assumes historical Riot results were observable at
result end. Acquisition timestamps are absent in this schema; that assumption
is reported, not presented as a verified historical ingestion schedule.
"""
from __future__ import annotations

import copy
import heapq
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from core.player_priors import PlayerPriorsTracker
from core.pre_match_dataset import _decode, to_pre_match_features
from core.point_in_time_dataset import _parse_timeline_json
from core.features.timeline_metrics import extract_timeline_lane_metrics
from core.serving_contract import HISTORY_POLICY, ContractError
from core.collection.regions import TAG_TO_REGION


def to_serving_features(match: dict) -> dict:
    """Allowlist draft context and player identity, never current match statistics."""
    clean = to_pre_match_features(match)
    info = match.get('info', match)
    if 'participants' in info:
        for key, tid in (('team_a', 100), ('team_b', 200)):
            parts = [p for p in info['participants'] if p.get('teamId') == tid]
            for out, p in zip(clean[key], parts):
                out['puuid'] = p.get('puuid')
    else:
        for key in ('team_a', 'team_b'):
            for out, p in zip(clean[key], match.get(key, [])):
                out['puuid'] = p.get('puuid')
    clean['prediction_time_ms'] = match.get('prediction_time_ms', info.get('gameCreation'))
    return clean


def decode_history_row(row) -> dict:
    raw = _decode(row['raw_json'])
    if not isinstance(raw, dict) or not isinstance(raw.get('info'), dict):
        raise ContractError('Invalid Riot match payload')
    info = raw['info']
    creation = row['game_creation_time']
    start, end = info.get('gameStartTimestamp'), info.get('gameEndTimestamp')
    valid_end = (type(start) is int and type(end) is int and
                 creation <= start < end and end - creation <= 24 * 3600 * 1000)
    if not isinstance(info.get('participants'), list) or not all(isinstance(p, dict) for p in info['participants']):
        raise ContractError('Invalid Riot participant list')
    if isinstance(row, dict) and row.get('source_error'):
        raise ContractError(row['source_error'])
    clean = to_serving_features(raw)
    # Region is known before play from the Riot platform prefix. Preserve the
    # original tree encoder's NA/EUW context rather than discarding it to UNK.
    if not clean.get('region'):
        region = TAG_TO_REGION.get(row['match_id'].split('_', 1)[0].upper())
        if region is not None:
            clean['region'] = region
    return {'match_id': row['match_id'], 'game_creation_time': creation,
            'result_end_ms': end if valid_end else None,
            'participants': info.get('participants', []),
            'timeline': row['timeline_json'] if 'timeline_json' in row.keys() else None,
            'timeline_metrics': row.get('timeline_metrics') if isinstance(row, dict) else None,
            'feature_json': clean,
            'patch': info.get('gameVersion', 'UNKNOWN'), 'raw_info': info}


def read_recent_rows(db_path, limit: int, *, before_ms: int | None = None, timelines: bool = True, puuids=None, include_labels: bool = False, history_only: bool = False) -> list[dict]:
    if type(limit) is not int or limit < 1:
        raise ValueError('History/cohort limit must be positive')
    if history_only and include_labels:
        raise ValueError('History-only reads cannot include training labels')
    uri = Path(db_path).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('PRAGMA cache_size=-65536')
        conn.execute('BEGIN')
        has_timeline = timelines and conn.execute("SELECT 1 FROM sqlite_master WHERE name='match_timelines'").fetchone()
        training_columns = {r[1] for r in conn.execute('PRAGMA table_info(training_dataset)')}
        labels_expr = 't.labels_json' if include_labels and 'labels_json' in training_columns else 'NULL'
        winner_expr = 'NULL' if history_only else 't.winner_label'
        training_join = '' if history_only else ' LEFT JOIN training_dataset t ON m.match_id=t.match_id'
        timeline_expr = '(SELECT timeline_json FROM match_timelines mt WHERE mt.match_id=m.match_id)' if has_timeline else 'NULL'
        if puuids is not None:
            puuids = sorted(set(p for p in puuids if p))
            if not puuids:
                return []
        # Creation and mode indexes cover rowids, avoiding metadata after the
        # huge raw_json column. Resolve match-ID ordering only at the boundary
        # tie; fetch the final selected payloads in rowid order. All reads share
        # the explicit transaction, including boundary resolution.
        table = conn.execute("SELECT type,sql FROM sqlite_master WHERE name='matches'").fetchone()
        prefix = ''
        if table and table['type'] == 'table' and 'WITHOUT ROWID' not in table['sql'].upper():
            mode_filter = "+rowid IN (SELECT rowid FROM matches WHERE game_mode='CLASSIC')"
            window = conn.execute(f'''SELECT rowid,game_creation_time FROM matches
                WHERE {mode_filter} AND game_creation_time>0 AND typeof(game_creation_time)='integer'
                AND game_creation_time < ? ORDER BY game_creation_time DESC LIMIT ?''',
                (before_ms or 2**63-1, limit)).fetchall()
            rowids = [r[0] for r in window]
            if len(window) == limit:
                boundary = window[-1][1]
                rowids = [r[0] for r in window if r[1] > boundary]
                tied = conn.execute(f'''SELECT rowid FROM matches WHERE {mode_filter}
                    AND game_creation_time=? ORDER BY match_id DESC LIMIT ?''',
                    (boundary, limit - len(rowids))).fetchall()
                rowids.extend(r[0] for r in tied)
            tail = 'FROM matches m' + training_join + ' WHERE m.rowid IN (SELECT value FROM json_each(?))'
            params = (json_dumps(rowids),)
        else:
            # General schemas lack usable rowids; sort headers before payloads.
            prefix = '''WITH recent AS MATERIALIZED (
                SELECT match_id,game_creation_time FROM matches
                WHERE +game_mode='CLASSIC' AND game_creation_time>0 AND typeof(game_creation_time)='integer'
                AND game_creation_time < ? ORDER BY game_creation_time DESC,match_id DESC LIMIT ?)
            '''
            tail = 'FROM recent r JOIN matches m ON m.match_id=r.match_id' + training_join + ' WHERE 1=1'
            params = (before_ms or 2**63-1, limit)
        if puuids is not None:
            # Select the SAME global creation window first, then fetch only
            # player-relevant raw payloads via the indexed participant table.
            # Older rows can have raw payloads without the backfilled index.
            # Fetch incomplete indexes too, then filter identities from raw;
            # otherwise live features would differ from frozen evaluation.
            extra = (' AND (EXISTS (SELECT 1 FROM match_participants mp WHERE mp.match_id=m.match_id AND mp.puuid IN ('
                     + ','.join('?' for _ in puuids) + ')) OR '
                     '(SELECT COUNT(DISTINCT mp.puuid) FROM match_participants mp WHERE mp.match_id=m.match_id) != 10)')
            tail += extra
            params = (*params, *puuids)
        query = prefix + f'''SELECT m.match_id,m.game_creation_time,m.raw_json,{winner_expr} AS winner_label,
                             {timeline_expr} AS timeline_json,{labels_expr} AS labels_json ''' + tail
        cursor = conn.execute(query,params)
        rows = []
        # Riot payloads contain hundreds of post-match fields. Retain only the
        # source facts needed for historical replay, not gigabytes of unrelated
        # combat/item/perk payloads. One read transaction freezes these facts.
        for row in cursor:
            compact = dict(row)
            raw = _decode(row['raw_json'])
            if isinstance(raw, dict) and isinstance(raw.get('info'), dict):
                info = raw['info']
                part_keys = ('puuid','participantId','teamId','championName','teamPosition','win','kills','deaths','assists',
                             'totalMinionsKilled','neutralMinionsKilled','goldEarned')
                source_parts = info.get('participants', [])
                if not isinstance(source_parts, list) or not all(isinstance(p, dict) for p in source_parts):
                    compact['source_error'] = 'Invalid Riot participant list'
                    compact['raw_json'] = json_dumps({'info':{}})
                    compact['timeline_json'] = None
                    rows.append(compact)
                    continue
                parts = [{k:p[k] for k in part_keys if k in p} for p in source_parts]
                small = {k:info[k] for k in ('gameStartTimestamp','gameEndTimestamp','gameDuration','gameVersion','gameMode','queueId') if k in info}
                small['gameCreation'] = row['game_creation_time']
                small['participants'] = parts
                compact['raw_json'] = json_dumps({'info':small})
                timeline = _parse_timeline_json(row['timeline_json'])
                try:
                    compact['timeline_metrics'] = extract_timeline_lane_metrics(timeline, participants=parts, require_observations=True) if timeline else None
                except (TypeError, ValueError, KeyError, AttributeError) as exc:
                    compact['source_error'] = f'Invalid timeline payload: {exc}'
                    compact['timeline_metrics'] = None
                compact['timeline_json'] = None
            else:
                compact['source_error'] = 'Invalid Riot match payload'
                compact['raw_json'] = json_dumps({'info':{}})
                compact['timeline_json'] = None
            rows.append(compact)
            if len(rows) % 1000 == 0:
                import logging
                logging.getLogger(__name__).info('Read and compacted %d historical source rows', len(rows))
    return sorted(rows, key=lambda r: (r['game_creation_time'], r['match_id']))


class ResultOrderedHistory:
    def __init__(self):
        self.tracker = PlayerPriorsTracker()
        self.pending = []
        self.latest_end = None
        self.count = 0
        self.missing_end = 0
        self.timeline_count = 0
        self._serial = 0
        self._as_of = -1

    def schedule(self, record: dict) -> None:
        end = record.get('result_end_ms')
        if type(end) is not int:
            self.missing_end += 1
            return
        self._serial += 1
        heapq.heappush(self.pending, (end, self._serial, record))

    def advance(self, as_of_ms: int) -> None:
        if as_of_ms < self._as_of:
            raise ContractError('History replay cannot move backwards in time')
        self._as_of = as_of_ms
        while self.pending and self.pending[0][0] < as_of_ms:
            end, _, rec = heapq.heappop(self.pending)
            parts = rec['participants']
            for p in parts:
                self.tracker.update(puuid=p.get('puuid'), champion=p.get('championName'),
                                    role=p.get('teamPosition'), win=p.get('win', False),
                                    kills=p.get('kills', 0), deaths=p.get('deaths', 0), assists=p.get('assists', 0),
                                    total_cs=p.get('totalMinionsKilled', 0) + p.get('neutralMinionsKilled', 0),
                                    gold_earned=p.get('goldEarned', 0), game_duration=rec.get('raw_info', {}).get('gameDuration', 0))
            timeline = _parse_timeline_json(rec.get('timeline'))
            metrics = rec.get('timeline_metrics')
            if metrics is None and timeline:
                metrics = extract_timeline_lane_metrics(timeline, participants=parts, require_observations=True)
            if metrics:
                for puuid, m in metrics.items():
                    self.tracker.record_timeline_metrics(puuid, gd15=m['gd15'], csd15=m['csd15'], xpd15=m['xpd15'], fbi10=m['fbi10'])
                self.timeline_count += 1
            self.latest_end = end
            self.count += 1

    def enrich(self, match: dict, *, include_timeline: bool, as_of_ms: int) -> dict:
        self.advance(as_of_ms)
        enriched = copy.deepcopy(match)
        cold = 0
        for key in ('team_a', 'team_b'):
            for p in enriched.get(key, []):
                puuid = p.get('puuid')
                if not puuid or puuid not in self.tracker.player_totals:
                    cold += 1
                p['priors'] = self.tracker.get_player_priors(puuid, p.get('champion') or p.get('championName'),
                                                           p.get('role') or p.get('teamPosition'), extended=True, include_timeline=include_timeline)
        enriched['priors_provenance'] = {'policy': HISTORY_POLICY, 'as_of_ms': as_of_ms,
                                       'latest_result_end_ms': self.latest_end, 'history_matches': self.count,
                                       'timeline_matches': self.timeline_count, 'cold_start_players': cold,
                                       'timeline_source_available': self.timeline_count > 0,
                                       'excluded_missing_result_end': self.missing_end,
                                       'availability_assumption': 'Riot results observable at result end; ingestion timestamps unknown'}
        return enriched


class DatabaseHistoryProvider:
    def __init__(self, db_path, *, history_limit: int = 10000):
        self.db_path = db_path
        self.history_limit = history_limit

    def enrich(self, match: dict, *, include_timeline: bool) -> dict:
        as_of = match.get('prediction_time_ms') or int(time.time()*1000)
        if type(as_of) is not int:
            raise ContractError('prediction_time_ms must be integer UTC milliseconds')
        identities = {p.get('puuid') for k in ('team_a','team_b') for p in match.get(k, []) if p.get('puuid')}
        records = []
        for row in read_recent_rows(self.db_path, self.history_limit, before_ms=as_of, timelines=include_timeline, puuids=identities, history_only=True):
            try:
                rec = decode_history_row(row)
            except ContractError:
                records.append(None)
                continue
            records.append(rec)
        history = build_target_history(records, match, as_of_ms=as_of)
        result = history.enrich(match, include_timeline=include_timeline, as_of_ms=as_of)
        result['priors_provenance']['history_window_limit'] = self.history_limit
        return result


def build_target_history(records, match, *, as_of_ms: int) -> ResultOrderedHistory:
    """Shared replay of a preselected creation window; target labels are irrelevant.

    Player features are separable, so updating only requested identities gives
    exactly the same priors as updating every player, with much less work.
    """
    identities = {p.get('puuid') for k in ('team_a','team_b') for p in match.get(k, []) if p.get('puuid')}
    history = ResultOrderedHistory()
    for rec in records:
        if rec is None:
            history.missing_end += 1
            continue
        if rec['game_creation_time'] >= as_of_ms:
            continue
        parts = [p for p in rec['participants'] if p.get('puuid') in identities]
        if not parts:
            continue
        # A Riot result can enter history without an ML training label. Its
        # observed participant win field must still be explicit and coherent.
        if any(type(p.get('win')) is not bool for p in parts):
            history.missing_end += 1
            continue
        target = dict(rec);target['participants'] = parts
        metrics = rec.get('timeline_metrics')
        if metrics is None:
            timeline = _parse_timeline_json(rec.get('timeline'))
            metrics = extract_timeline_lane_metrics(timeline, participants=rec['participants'], require_observations=True) if timeline else None
        valid_metrics = {}
        for puuid, m in (metrics or {}).items():
            if puuid in identities and isinstance(m, dict) and all(k in m and isinstance(m[k], (int,float)) and math.isfinite(m[k]) for k in ('gd15','csd15','xpd15','fbi10')):
                valid_metrics[puuid] = m
        target['timeline'] = None
        target['timeline_metrics'] = valid_metrics
        history.schedule(target)
    history.advance(as_of_ms)
    return history


def json_dumps(value):
    import json
    return json.dumps(value, separators=(',', ':'))
