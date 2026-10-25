"""Live history windows are reused per player set without sharing a mutated tracker."""
from types import SimpleNamespace

import core.serving_history as serving_history
from core.serving_contract import HISTORY_POLICY
from core.serving_history import CachedHistoryProvider

ROLES = ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')


def _match(champion='Ahri', **extra):
    match = {key: [{'puuid': f'{key}_{i}', 'champion': champion, 'role': role}
                   for i, role in enumerate(ROLES)]
             for key in ('team_a', 'team_b')}
    match.update(extra)
    return match


def _record():
    # One completed match for team_a_0, decoded as decode_history_row returns it.
    return {'match_id': 'EUW1_1', 'game_creation_time': 1_000, 'result_end_ms': 2_000,
            'participants': [{'puuid': 'team_a_0', 'championName': 'Ahri', 'teamPosition': 'TOP',
                              'win': True, 'kills': 5, 'deaths': 1, 'assists': 7,
                              'totalMinionsKilled': 150, 'neutralMinionsKilled': 10, 'goldEarned': 11_000}],
            'timeline': None, 'timeline_metrics': None, 'raw_info': {'gameDuration': 1_800}}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _provider(monkeypatch, clock=None):
    reads = []
    def read_recent_rows(db_path, limit, **kwargs):
        reads.append(kwargs)
        return [{'row': 1}]
    monkeypatch.setattr(serving_history, 'read_recent_rows', read_recent_rows)
    monkeypatch.setattr(serving_history, 'decode_history_row', lambda row: _record())
    exact = []
    inner = SimpleNamespace(db_path='history.db', history_limit=10_000,
                            enrich=lambda match, include_timeline: exact.append(match) or {'exact': True})
    return CachedHistoryProvider(inner, clock=clock or Clock()), reads, exact


def test_same_players_reuse_one_window_even_when_champions_change(monkeypatch):
    provider, reads, _ = _provider(monkeypatch)
    first = provider.enrich(_match('Ahri'), include_timeline=False)
    second = provider.enrich(_match('Zed'), include_timeline=False)
    assert len(reads) == 1
    for result in (first, second):
        provenance = result['priors_provenance']
        assert provenance['policy'] == HISTORY_POLICY
        assert provenance['history_window_limit'] == 10_000
        assert provenance['history_matches'] == 1
        # The tracker is rebuilt per request, so lookups never turn cold players warm.
        assert provenance['cold_start_players'] == 9
    assert first['priors_provenance']['as_of_ms'] == second['priors_provenance']['as_of_ms']
    assert second['team_a'][1]['champion'] == 'Zed'


def test_window_expires_after_the_ttl(monkeypatch):
    clock = Clock()
    provider, reads, _ = _provider(monkeypatch, clock)
    provider.enrich(_match(), include_timeline=False)
    clock.now = serving_history.HISTORY_CACHE_TTL_SECONDS + 1
    provider.enrich(_match(), include_timeline=False)
    assert len(reads) == 2


def test_different_players_or_timeline_mode_do_not_share_a_window(monkeypatch):
    provider, reads, _ = _provider(monkeypatch)
    provider.enrich(_match(), include_timeline=False)
    provider.enrich(_match(), include_timeline=True)
    other = _match()
    other['team_b'][0]['puuid'] = 'someone_else'
    provider.enrich(other, include_timeline=False)
    assert len(reads) == 3


def test_explicit_prediction_time_is_replayed_exactly(monkeypatch):
    provider, reads, exact = _provider(monkeypatch)
    assert provider.enrich(_match(prediction_time_ms=5_000), include_timeline=False) == {'exact': True}
    assert len(exact) == 1 and not reads


def test_cache_is_bounded(monkeypatch):
    provider, _, _ = _provider(monkeypatch)
    provider.max_entries = 2
    for n in range(3):
        match = _match()
        match['team_a'][0]['puuid'] = f'player_{n}'
        provider.enrich(match, include_timeline=False)
    assert len(provider._entries) == 2
    assert provider.history_limit == 10_000
