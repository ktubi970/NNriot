import copy
import numpy as np
import pytest
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from core.serving_contract import FeatureUnavailableError
from scripts.prepare_aligned_dataset import verify
from tests.test_aligned_artifacts import tiny_bundle
from tests.aligned_dataset_fixture import source_row, target


def incomplete_timeline(problem):
    frame = {'participantFrames': {
        str(i): dict(totalGold=500, minionsKilled=0,
                     jungleMinionsKilled=0, xp=0) for i in range(1, 11)},
        'events': []}
    if problem == 'empty_frame':
        frame = {}
    elif problem == 'missing_player':
        del frame['participantFrames']['1']
    elif problem == 'missing_measure':
        del frame['participantFrames']['1']['xp']
    elif problem == 'invalid_measure':
        frame['participantFrames']['1']['xp'] = 'unobserved'
    elif problem == 'missing_events':
        del frame['events']
    return {'info': {'frames': [frame]}}


@pytest.mark.parametrize('problem', ['empty_frame', 'missing_player',
                                   'missing_measure', 'invalid_measure',
                                   'missing_events'])
def test_incomplete_timeline_cannot_rescue_18d_or_remove_riot_priors(problem):
    import json
    old = source_row('NA1_old', 10, 20)
    old['timeline_json'] = json.dumps(incomplete_timeline(problem))
    query = target(source_row('NA1_current', 30, 40))
    provider = FrozenHistoryProvider([old])
    enriched = provider.enrich(query, include_timeline=True)
    assert enriched['team_a'][0]['priors'][0] == 6 / 11
    assert enriched['priors_provenance']['timeline_matches'] == 0
    assert enriched['priors_provenance']['timeline_source_available'] is False
    fourteen = provider.enrich(query, include_timeline=False)
    assert fourteen['team_a'][0]['priors'][0] == 6 / 11
    assert extract_aligned_inputs(fourteen, 'nnriot.players14.v1')['team_a'].shape == (1, 5, 14)
    with pytest.raises(FeatureUnavailableError, match='timeline history source unavailable'):
        extract_aligned_inputs(enriched, 'nnriot.players18.v1')


def test_writer_keeps_14d_ready_and_blocks_unobserved_18d(tmp_path):
    import json
    from core.aligned_artifacts import write_dataset
    from core.aligned_partitions import select_targets, partition_targets
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19, i % 2)
            for i in range(10)]
    for row in rows:
        row['timeline_json'] = json.dumps(incomplete_timeline('empty_frame'))
    targets, excluded = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, purged = partition_targets(targets, after_ms=60,
                                       validation_count=2, test_count=2)
    directory = tmp_path / 'incomplete'
    meta = write_dataset(directory, rows, splits, source_db='fixture.db',
                         selection={}, exclusions=excluded + purged)
    assert meta['availability']['nnriot.players14.v1']['status'] == 'ready'
    availability = meta['availability']['nnriot.players18.v1']
    assert availability['status'] == 'blocked'
    assert availability['eligible'] == 0
    assert availability['coverage'] == 0
    assert all('timeline history source unavailable' in r['reason']
               for r in availability['errors'])
    assert not list(directory.glob('*18.npz'))


def test_sqlite_partial_timeline_keeps_result_and_blocks_18d(tmp_path):
    import json
    import sqlite3
    from core.pre_match_history import read_recent_rows, DatabaseHistoryProvider
    db = tmp_path / 'partial.db'
    old = source_row('NA1_old', 10, 20)
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER,
                                 game_mode TEXT, raw_json TEXT);
            CREATE TABLE match_timelines(match_id TEXT, timeline_json TEXT);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);
        ''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     (old['match_id'], 10, 'CLASSIC', old['raw_json']))
        conn.execute('INSERT INTO match_timelines VALUES(?,?)',
                     (old['match_id'], json.dumps(incomplete_timeline('missing_measure'))))
    rows = read_recent_rows(db, 10000, timelines=True, history_only=True)
    assert not rows[0].get('source_error')
    query = target(source_row('NA1_current', 30, 40))
    a = FrozenHistoryProvider(rows).enrich(query, include_timeline=True)
    b = DatabaseHistoryProvider(db).enrich(query, include_timeline=True)
    for enriched in (a, b):
        assert enriched['team_a'][0]['priors'][0] == 6 / 11
        assert enriched['priors_provenance']['timeline_matches'] == 0
        with pytest.raises(FeatureUnavailableError):
            extract_aligned_inputs(enriched, 'nnriot.players18.v1')
    a14 = FrozenHistoryProvider(rows).enrich(query, include_timeline=False)
    b14 = DatabaseHistoryProvider(db).enrich(query, include_timeline=False)
    for key, value in extract_aligned_inputs(a14, 'nnriot.players14.v1').items():
        assert np.array_equal(value, extract_aligned_inputs(b14, 'nnriot.players14.v1')[key])

def test_frozen_replay_is_exact(tmp_path):
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    report = verify(directory)
    assert report['status'] == 'verified'
    assert report['tensor_comparisons'] > 0

def test_permutation_swap_and_target_outcomes_do_not_change_features():
    history = [source_row('NA1_1', 10, 20, timeline=True)]
    match = target(source_row('NA1_2', 30, 130))
    provider = FrozenHistoryProvider(history)
    for dim in (14, 18):
        version = f'nnriot.players{dim}.v1'
        original = provider.enrich(match, include_timeline=dim == 18)
        expected = extract_aligned_inputs(original, version)
        permuted = copy.deepcopy(original)
        for team in ('team_a', 'team_b'):
            permuted[team].reverse()
            for p in permuted[team]:
                p.update(kills=999999, win=False, goldEarned=999999)
        got = extract_aligned_inputs(permuted, version)
        for key in expected:
            assert np.array_equal(expected[key], got[key])
        swapped = dict(original, team_a=original['team_b'], team_b=original['team_a'])
        reverse = extract_aligned_inputs(swapped, version)
        assert np.array_equal(expected['team_a'], reverse['team_b'])
        assert np.array_equal(expected['team_b'], reverse['team_a'])
        assert np.array_equal(expected['macro_clash'], -reverse['macro_clash'])

def test_no_timeline_outside_global_window_can_rescue_18d():
    old = source_row('NA1_1', 10, 20, timeline=True)
    recent = source_row('NA1_2', 30, 40, timeline=False)
    match = target(source_row('NA1_3', 50, 150))
    enriched = FrozenHistoryProvider([old, recent], history_limit=1).enrich(
        match, include_timeline=True)
    with pytest.raises(FeatureUnavailableError):
        extract_aligned_inputs(enriched, 'nnriot.players18.v1')

def test_replay_refuses_changed_feature_runtime(tmp_path, monkeypatch):
    import core.aligned_artifacts as artifacts
    directory, _ = tiny_bundle(tmp_path)
    monkeypatch.setattr(artifacts, 'dataset_runtime_sources', lambda: {'changed': '0'*64})
    with pytest.raises(ValueError, match='runtime SHA-256'):
        verify(directory)


def test_replay_rejects_float64_even_with_matching_values(tmp_path):
    import json
    from core.calibration import sha256_file
    directory, meta = tiny_bundle(tmp_path)
    path = directory / 'sealed_test14.npz'
    with np.load(path, allow_pickle=False) as data:
        x = {key: data[key].astype(np.float64) for key in data.files}
    np.savez(path, **x)
    meta['files'][path.name]['sha256'] = sha256_file(path)
    (directory / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='tensor contract'):
        verify(directory)


def test_replay_loads_each_archive_array_once(tmp_path, monkeypatch):
    directory, _ = tiny_bundle(tmp_path)
    original = np.load
    accesses = []

    class Archive:
        def __init__(self, path, **kwargs):
            self.data = original(path, **kwargs)
            self.files = self.data.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.data.close()

        def __getitem__(self, key):
            accesses.append(key)
            return self.data[key]

    monkeypatch.setattr(np, 'load', Archive)
    verify(directory)
    assert len(accesses) == 9  # Three inputs for each of three 14D partitions.

@pytest.mark.parametrize('complete_index', [False, True])
@pytest.mark.parametrize('dim', [14, 18])
def test_sqlite_and_frozen_priors_and_tensors_are_equal(tmp_path, complete_index, dim):
    import json
    import sqlite3
    from core.pre_match_history import DatabaseHistoryProvider, read_recent_rows
    from tests.test_point_in_time_timeline import _create_sample_timeline
    db = tmp_path / 'parity.db'
    old = source_row('NA1_1', 1000, 1001000)
    raw = json.loads(old['raw_json'])
    raw['info']['gameDuration'] = 1000
    participants = raw['info']['participants']
    timeline = _create_sample_timeline()
    timeline['info']['participants'] = [
        {'participantId': p['participantId'], 'puuid': p['puuid'],
         'teamPosition': p['teamPosition']} for p in participants]
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER,
                                 game_mode TEXT, raw_json TEXT);
            CREATE TABLE training_dataset(match_id TEXT, winner_label INTEGER,
                                          labels_json TEXT);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);
            CREATE TABLE match_timelines(match_id TEXT, timeline_json TEXT);
        ''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     ('NA1_1', 1000, 'CLASSIC', json.dumps(raw)))
        # History is usable even with no ML label.
        conn.execute('INSERT INTO training_dataset VALUES(?,?,?)', ('NA1_1', None, None))
        conn.execute('INSERT INTO match_timelines VALUES(?,?)', ('NA1_1', json.dumps(timeline)))
        if complete_index:
            conn.executemany('INSERT INTO match_participants VALUES(?,?)',
                             [('NA1_1', p['puuid']) for p in participants])
    rows = read_recent_rows(db, 10000, timelines=True)
    query = target(source_row('NA1_query', 2000000, 2100000))
    a = FrozenHistoryProvider(rows).enrich(query, include_timeline=dim == 18)
    b = DatabaseHistoryProvider(db).enrich(query, include_timeline=dim == 18)
    assert [p['priors'] for team in ('team_a', 'team_b') for p in a[team]] == [
        p['priors'] for team in ('team_a', 'team_b') for p in b[team]]
    xa = extract_aligned_inputs(a, f'nnriot.players{dim}.v1')
    xb = extract_aligned_inputs(b, f'nnriot.players{dim}.v1')
    for key in xa:
        assert np.array_equal(xa[key], xb[key])
    for enriched in (a, b):
        provenance = enriched['priors_provenance']
        assert provenance['as_of_ms'] == 2000000
        assert provenance['history_window_limit'] == 10000
        assert provenance['latest_result_end_ms'] == 1001000
