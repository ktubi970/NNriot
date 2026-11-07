from core.pre_match_history import ResultOrderedHistory


def record(mid, start, end, win):
    return {'match_id': mid, 'game_creation_time': start, 'result_end_ms': end,
            'participants': [{'puuid': 'p', 'championName': 'Ahri', 'teamPosition': 'MIDDLE', 'win': win}],
            'timeline': None}


def test_overlapping_results_never_update_before_end():
    history = ResultOrderedHistory()
    history.schedule(record('long', 10, 100, True))
    history.schedule(record('short', 20, 30, False))
    history.advance(30)
    assert 'p' not in history.tracker.player_totals
    history.advance(31)
    assert history.tracker.player_totals['p']['games'] == 1
    assert history.tracker.player_totals['p']['wins'] == 0
    history.advance(100)
    assert history.tracker.player_totals['p']['games'] == 1
    history.advance(101)
    assert history.tracker.player_totals['p']['games'] == 2


def test_unknown_result_end_is_excluded():
    history = ResultOrderedHistory()
    history.schedule(record('unknown', 10, None, True))
    history.advance(1000)
    assert 'p' not in history.tracker.player_totals


def test_empty_timeline_metrics_do_not_establish_availability():
    history=ResultOrderedHistory()
    rec=record('empty',10,20,True);rec['timeline_metrics']={}
    history.schedule(rec);history.advance(30)
    assert history.timeline_count==0


def test_shared_history_uses_creation_cutoff_and_riot_results_without_ml_label():
    from core.pre_match_history import build_target_history
    m={'team_a':[{'puuid':'p','champion':'Ahri','role':'MIDDLE'}],'team_b':[]}
    # Current match was created at 90 and starts at 100. Prediction cutoff is
    # creation, so result ending at 95 is not yet observable to either path.
    h=build_target_history([record('past',10,95,True)],m,as_of_ms=90)
    assert 'p' not in h.tracker.player_totals
    old=record('label_missing',10,80,True);old['winner_label']=None
    h=build_target_history([old],m,as_of_ms=90)
    assert h.tracker.player_totals['p']['wins']==1


def test_database_provider_replays_raw_history_without_participant_index(tmp_path):
    import json
    import sqlite3
    from core.pre_match_history import DatabaseHistoryProvider, build_target_history, decode_history_row, read_recent_rows
    db = tmp_path / 'history.db'
    parts = [{'puuid': f'p{i}', 'championName': 'Ahri', 'teamPosition': 'MIDDLE',
              'teamId': 100 if i < 5 else 200, 'win': i < 5} for i in range(10)]
    payload = json.dumps({'info': {'gameCreation': 10, 'gameStartTimestamp': 11,
                                  'gameEndTimestamp': 80, 'gameDuration': 1, 'participants': parts}})
    with sqlite3.connect(db) as conn:
        conn.executescript('''CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER, game_mode TEXT, raw_json TEXT);
                             CREATE TABLE training_dataset(match_id TEXT, winner_label INTEGER);
                             CREATE TABLE match_participants(match_id TEXT, puuid TEXT);''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)', ('past', 10, 'CLASSIC', payload))
    match = {'prediction_time_ms': 90, 'team_a': [{'puuid': 'p0', 'champion': 'Ahri', 'role': 'MIDDLE'}], 'team_b': []}
    rows = read_recent_rows(db, 10000, before_ms=90, timelines=False)
    history = build_target_history([decode_history_row(r) for r in rows], match, as_of_ms=90)
    expected = history.enrich(match, include_timeline=False, as_of_ms=90)['team_a'][0]['priors']
    provider = DatabaseHistoryProvider(db)
    assert provider.enrich(match, include_timeline=False)['team_a'][0]['priors'] == expected
    # Completing the index may optimize selection, never change the features.
    with sqlite3.connect(db) as conn:
        conn.executemany('INSERT INTO match_participants VALUES(?,?)', [('past', p['puuid']) for p in parts])
    assert provider.enrich(match, include_timeline=False)['team_a'][0]['priors'] == expected


def test_historical_context_preserves_original_tree_region_categories():
    import json
    from core.pre_match_history import decode_history_row
    row = {'match_id': 'NA1_1', 'game_creation_time': 10, 'timeline_json': None,
           'raw_json': json.dumps({'info': {'participants': [], 'gameVersion': '16.18.817.5716',
                                           'gameStartTimestamp': 11, 'gameEndTimestamp': 20}})}
    feature = decode_history_row(row)['feature_json']
    assert feature['region'] == 'NA'
    assert feature['game_version'] == '16.18.817.5716'
    assert feature['season_id'] == 16
    row['match_id'] = 'UNRECOGNIZED_1'
    assert 'region' not in decode_history_row(row)['feature_json']
