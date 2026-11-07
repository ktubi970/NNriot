import pytest
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

def test_bad_current_label_rejected_but_not_removed_from_history():
    row = source_row('NA1_1', 10, 20)
    row['winner_label'] = 1
    selected, rejected = select_targets([row], excluded_ids=set(), warmup_rows=0)
    assert selected == []
    assert rejected[0]['reason'] == 'winner_disagrees_with_riot'
    assert row['raw_json']

def test_ties_and_result_purge_at_boundaries():
    rows = [source_row(f'NA1_{i}', t, e) for i, (t, e) in enumerate(
        [(10, 19), (20, 31), (30, 39), (30, 39), (40, 51), (50, 59), (60, 69)])]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=0)
    parts, purged = partition_targets(targets, after_ms=40,
                                     validation_count=2, test_count=2)
    assert [r['game_creation_time'] for r in parts['train']] == [10]
    assert [r['game_creation_time'] for r in parts['validation']] == [30, 30]
    assert [r['game_creation_time'] for r in parts['test']] == [50, 60]
    assert {r['game_creation_time'] for r in purged} == {20, 40}

def test_insufficient_new_test_is_an_error():
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19) for i in range(4)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=0)
    with pytest.raises(ValueError, match='Insufficient'):
        partition_targets(targets, after_ms=35, validation_count=1, test_count=2)

def test_previous_test_id_excluded_and_unknown_queue_rejected():
    import json
    known = source_row('NA1_seen', 10, 20)
    unknown = source_row('NA1_unknown', 30, 40)
    raw = json.loads(unknown['raw_json']); raw['info'].pop('queueId')
    unknown['raw_json'] = json.dumps(raw)
    selected, errors = select_targets([known, unknown],
        excluded_ids={'NA1_seen'}, warmup_rows=0)
    assert selected == []
    assert {e['reason'] for e in errors} == {'previous_test', 'unsupported_queue'}

def test_snapshot_labels_are_opt_in_and_queue_survives(tmp_path):
    import json
    import sqlite3
    from core.pre_match_history import read_recent_rows
    db = tmp_path / 'snapshot.db'
    row = source_row('NA1_1', 10, 20)
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER,
                                 game_mode TEXT, raw_json TEXT);
            CREATE TABLE training_dataset(match_id TEXT, winner_label INTEGER);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);
        ''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     (row['match_id'], 10, 'CLASSIC', row['raw_json']))
        conn.execute('INSERT INTO training_dataset VALUES(?,?)', (row['match_id'], 0))
    assert read_recent_rows(db, 10, include_labels=True)[0]['labels_json'] is None
    with sqlite3.connect(db) as conn:
        conn.execute('ALTER TABLE training_dataset ADD COLUMN labels_json TEXT')
        conn.execute('UPDATE training_dataset SET labels_json=?', (row['labels_json'],))
    assert read_recent_rows(db, 10)[0]['labels_json'] is None
    stored = read_recent_rows(db, 10, include_labels=True)[0]
    assert stored['labels_json'] == row['labels_json']
    assert json.loads(stored['raw_json'])['info']['queueId'] == 420


def test_warmup_ties_do_not_use_an_incomplete_global_history_window():
    rows = [source_row(f'NA1_{i}', t, t + 9)
            for i, t in enumerate((10, 20, 20, 30))]
    selected, errors = select_targets(rows, excluded_ids=set(), warmup_rows=2)
    assert [r['game_creation_time'] for r in selected] == [30]
    assert errors[0]['reason'] == 'history_context_incomplete'
def test_source_window_limits_payload_reads_before_sorting(tmp_path, monkeypatch):
    import sqlite3
    from core.pre_match_history import read_recent_rows
    from tests.aligned_dataset_fixture import source_row
    db = tmp_path / 'bounded.db'
    original_connect = sqlite3.connect
    with original_connect(db) as conn:
        conn.executescript('''
            CREATE TABLE source_matches(match_id TEXT PRIMARY KEY, game_creation_time INTEGER,
                                        game_mode TEXT, raw_json TEXT);
            CREATE VIEW matches AS SELECT match_id,game_creation_time,game_mode,
                                          payload_read(raw_json) AS raw_json FROM source_matches;
            CREATE TABLE training_dataset(match_id TEXT PRIMARY KEY, winner_label INTEGER);
        ''')
        for number in range(20):
            row = source_row(f'm{number:02}', 10 + number, 50 + number)
            conn.execute('INSERT INTO source_matches VALUES(?,?,?,?)',
                         (row['match_id'], row['game_creation_time'], 'CLASSIC', row['raw_json']))
    reads = []
    def payload_read(value):
        reads.append(value)
        return value
    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.create_function('payload_read', 1, payload_read)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    rows = read_recent_rows(db, 3, before_ms=29, timelines=False)
    assert [r['match_id'] for r in rows] == ['m16', 'm17', 'm18']
    assert len(reads) == 3


@pytest.mark.parametrize('without_rowid', [False, True])
@pytest.mark.parametrize('complete_index', [False, True])
def test_global_boundary_ties_precede_identity_filter(tmp_path, without_rowid, complete_index):
    import json
    import sqlite3
    from core.pre_match_history import DatabaseHistoryProvider, read_recent_rows
    db = tmp_path / 'ties.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE matches(match_id TEXT PRIMARY KEY, raw_json TEXT, '
                     'game_mode TEXT, game_creation_time INTEGER)' +
                     (' WITHOUT ROWID' if without_rowid else ''))
        conn.executescript('''CREATE INDEX idx_matches_game_creation ON matches(game_creation_time);
            CREATE INDEX idx_matches_game_mode ON matches(game_mode);
            CREATE TABLE training_dataset(match_id TEXT PRIMARY KEY, winner_label INTEGER);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);''')
        for mid in ('NA1_y', 'NA1_z'):
            row = source_row(mid, 30, 35)
            raw = json.loads(row['raw_json'])
            if mid == 'NA1_z':
                raw['info']['participants'][0]['puuid'] = 'other'
            conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                         (mid, json.dumps(raw), 'CLASSIC', 30))
            if complete_index:
                conn.executemany('INSERT INTO match_participants VALUES(?,?)',
                    [(mid, p['puuid']) for p in raw['info']['participants']])
    rows = read_recent_rows(db, 1, before_ms=40, timelines=False)
    assert [r['match_id'] for r in rows] == ['NA1_z']
    query = {'prediction_time_ms': 40, 'team_a': [
        {'puuid': 'p100_0', 'champion': 'Ahri', 'role': 'TOP'}], 'team_b': []}
    enriched = DatabaseHistoryProvider(db, history_limit=1).enrich(query, include_timeline=False)
    assert enriched['priors_provenance']['history_matches'] == 0


def test_live_history_never_reads_multitask_training_data(tmp_path, monkeypatch):
    import sqlite3
    from core.pre_match_history import DatabaseHistoryProvider
    from tests.aligned_dataset_fixture import target
    original_connect = sqlite3.connect
    db = tmp_path / 'raw_only.db'
    row = source_row('NA1_old', 10, 20)
    with original_connect(db) as conn:
        conn.executescript('''CREATE TABLE matches(match_id TEXT PRIMARY KEY, raw_json TEXT,
            game_mode TEXT, game_creation_time INTEGER);
            CREATE TABLE training_dataset(match_id TEXT PRIMARY KEY, winner_label INTEGER);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     (row['match_id'], row['raw_json'], 'CLASSIC', 10))
    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_authorizer(lambda action, table, *_: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and table == 'training_dataset' else sqlite3.SQLITE_OK)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    enriched = DatabaseHistoryProvider(db).enrich(
        target(source_row('NA1_current', 30, 40)), include_timeline=False)
    assert enriched['priors_provenance']['history_matches'] == 1
    assert enriched['team_a'][0]['priors'][0] == pytest.approx(6/11)
