# core/frozen_history.py
from bisect import bisect_left
import copy
from core.pre_match_history import decode_history_row, build_target_history
from core.serving_contract import ContractError

class FrozenHistoryProvider:
    def __init__(self, rows, *, history_limit=10000):
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('history_limit must be a positive integer')
        self.rows = sorted(copy.deepcopy(rows), key=lambda r: (
            r['game_creation_time'], r['match_id']))
        ids = [r['match_id'] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate match_id')
        self.times = [r['game_creation_time'] for r in self.rows]
        self.history_limit = history_limit
        self.facts, self.positions, self.invalid = [], {}, []
        for index, row in enumerate(self.rows):
            try:
                fact = decode_history_row(row)
            except ContractError:
                fact = None
                self.invalid.append(index)
            self.facts.append(fact)
            if fact is not None:
                for puuid in {p.get('puuid') for p in fact['participants']} - {None}:
                    self.positions.setdefault(puuid, []).append(index)

    def enrich(self, match, *, include_timeline):
        as_of = match.get('prediction_time_ms')
        if type(as_of) is not int or as_of <= 0:
            raise ContractError('Frozen replay requires positive prediction_time_ms')
        stop = bisect_left(self.times, as_of)
        start = max(0, stop - self.history_limit)
        identities = {p.get('puuid') for k in ('team_a', 'team_b')
                      for p in match.get(k, []) if p.get('puuid')}
        indexes = set()
        for puuid in identities:
            positions = self.positions.get(puuid, [])
            indexes.update(positions[bisect_left(positions, start):
                                     bisect_left(positions, stop)])
        indexes.update(self.invalid[bisect_left(self.invalid, start):
                                    bisect_left(self.invalid, stop)])
        facts = [self.facts[i] for i in sorted(indexes)]
        history = build_target_history(facts, match, as_of_ms=as_of)
        enriched = history.enrich(match, include_timeline=include_timeline,
                                  as_of_ms=as_of)
        enriched['priors_provenance']['history_window_limit'] = self.history_limit
        return enriched
