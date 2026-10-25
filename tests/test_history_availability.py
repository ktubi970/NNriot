"""Opt-in availability filter: history only from rows ingested before the target starts."""
import sqlite3

from core import database
from core.availability_history import (AVAILABILITY_ASSUMPTION, IngestedDatabaseHistoryProvider,
                                       IngestedFrozenHistoryProvider, build_available_history,
                                       read_ingestion_times)
from core.pre_match_history import DatabaseHistoryProvider, build_target_history, read_recent_rows


def record(mid, start, end, win, puuid='p'):
    return {'match_id': mid, 'game_creation_time': start, 'result_end_ms': end, 'timeline': None,
            'participants': [{'puuid': puuid, 'championName': 'Ahri', 'teamPosition': 'MIDDLE', 'win': win}]}


MATCH = {'team_a': [{'puuid': 'p', 'champion': 'Ahri', 'role': 'MIDDLE'}],
         'team_b': [{'puuid': 'q', 'champion': 'Ahri', 'role': 'MIDDLE'}]}
RECORDS = [record('early', 10, 20, True), record('late', 11, 21, False), record('legacy', 12, 22, True),
           record('at_start', 13, 23, False), record('q_early', 10, 20, False, puuid='q'),
           record('q_late', 11, 21, True, puuid='q'), record('other', 10, 20, True, puuid='r')]
INGESTED = {'early': 30, 'late': 95, 'legacy': None, 'at_start': 90, 'q_early': 30, 'q_late': 95, 'other': 95}


def games(history, puuid='p'):
    return history.tracker.player_totals.get(puuid, {}).get('games', 0)


def test_rows_ingested_at_or_after_start_or_never_recorded_are_excluded():
    history, excluded = build_available_history(RECORDS, MATCH, as_of_ms=90, ingested_at=INGESTED)
    assert (games(history), history.tracker.player_totals['p']['wins']) == (1, 1)
    assert games(history, 'q') == 1
    # 'other' involves no target player: it is not counted.
    assert excluded == 4
    assert games(build_target_history(RECORDS, MATCH, as_of_ms=90)) == 4


def test_fully_available_history_reproduces_the_default_replay():
    early = dict.fromkeys(INGESTED, 0)
    history, excluded = build_available_history(RECORDS, MATCH, as_of_ms=90, ingested_at=early)
    kwargs = dict(include_timeline=False, as_of_ms=90)
    assert excluded == 0
    assert history.enrich(MATCH, **kwargs)['team_a'] == build_target_history(RECORDS, MATCH, as_of_ms=90).enrich(MATCH, **kwargs)['team_a']


def test_availability_filter_commutes_with_team_swap():
    swapped = {'team_a': MATCH['team_b'], 'team_b': MATCH['team_a']}
    kwargs = dict(include_timeline=False, as_of_ms=90)
    original = build_available_history(RECORDS, MATCH, as_of_ms=90, ingested_at=INGESTED)[0].enrich(MATCH, **kwargs)
    mirrored = build_available_history(RECORDS, swapped, as_of_ms=90, ingested_at=INGESTED)[0].enrich(swapped, **kwargs)
    assert mirrored['team_a'][0]['priors'] == original['team_b'][0]['priors']
    assert mirrored['team_b'][0]['priors'] == original['team_a'][0]['priors']


def test_ingestion_times_are_unknown_before_schema_v16(tmp_path):
    db = tmp_path / 'v15.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER)')
        conn.execute("INSERT INTO matches VALUES ('EUW1_1', 10)")
    assert read_ingestion_times(db, ['EUW1_1', 'EUW1_1', None]) == {'EUW1_1': None}


def _payload(mid, creation):
    parts = [{'puuid': f'p{i}', 'championName': 'Ahri', 'teamPosition': 'MIDDLE',
              'teamId': 100 if i < 5 else 200, 'win': i < 5} for i in range(10)]
    return {'metadata': {'matchId': mid}, 'info': {
        'gameMode': 'CLASSIC', 'queueId': 420, 'gameCreation': creation, 'gameVersion': '16.18.1',
        'gameStartTimestamp': creation + 1, 'gameEndTimestamp': creation + 1000,
        'gameDuration': 1, 'participants': parts}}


def test_database_and_frozen_providers_apply_the_same_filter(tmp_path):
    db = str(tmp_path / 'v16.db')
    database.init_db(db)
    for mid, creation, ingested in (('EUW1_EARLY', 1000, 2000), ('EUW1_LATE', 1100, 20_000),
                                    ('EUW1_LEGACY', 1200, 3000)):
        database.save_matches_batch([(mid, _payload(mid, creation))], db_path=db, ingested_at=ingested)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM match_facts WHERE match_id = 'EUW1_LEGACY'")  # stored before v16
    match = {'prediction_time_ms': 10_000,
             'team_a': [{'puuid': 'p0', 'champion': 'Ahri', 'role': 'MIDDLE'}],
             'team_b': [{'puuid': 'p5', 'champion': 'Ahri', 'role': 'MIDDLE'}]}

    default = DatabaseHistoryProvider(db).enrich(match, include_timeline=False)
    strict = IngestedDatabaseHistoryProvider(db).enrich(match, include_timeline=False)
    assert default['priors_provenance']['history_matches'] == 3
    assert 'availability_policy' not in default['priors_provenance']
    assert strict['priors_provenance']['history_matches'] == 1
    assert strict['priors_provenance']['excluded_not_yet_ingested'] == 2
    assert strict['priors_provenance']['availability_assumption'] == AVAILABILITY_ASSUMPTION
    assert strict['priors_provenance']['history_window_limit'] == 10000

    rows = read_recent_rows(db, 10000, timelines=False, history_only=True)
    times = read_ingestion_times(db, [r['match_id'] for r in rows])
    frozen = IngestedFrozenHistoryProvider(rows, ingested_at=times).enrich(match, include_timeline=False)
    assert frozen['team_a'] == strict['team_a'] and frozen['team_b'] == strict['team_b']
    assert frozen['priors_provenance'] == strict['priors_provenance']
