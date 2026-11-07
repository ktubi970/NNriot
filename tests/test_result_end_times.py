"""Availability provenance checks for the isolated matchup benchmark."""
import base64
import gzip
import json
import sqlite3

import pytest
from core.point_in_time_dataset import load_result_end_times


def _database(tmp_path, entries):
    path = tmp_path / 'source.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation_time INTEGER, raw_json TEXT)')
        conn.executemany('INSERT INTO matches VALUES (?,?,?)', entries)
    return path


@pytest.mark.parametrize('compressed', [False, True])
def test_uses_explicit_end_instead_of_creation_plus_duration(tmp_path, compressed):
    raw = json.dumps({'info': {'gameStartTimestamp': 110, 'gameEndTimestamp': 999, 'gameDuration': 1}})
    if compressed:
        raw = base64.b64encode(gzip.compress(raw.encode())).decode()
    path = _database(tmp_path, [('one', 100, raw)])
    before = path.read_bytes()
    ends, diagnostics = load_result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])
    assert ends == {'one': 999}
    assert diagnostics == {'requested': 1, 'found': 1, 'invalid': 0, 'valid': 1, 'missing': 0}
    assert path.read_bytes() == before


def test_missing_or_invalid_raw_metadata_does_not_invent_availability(tmp_path):
    path = _database(tmp_path, [
        ('no_end', 100, json.dumps({'info': {'gameStartTimestamp': 110, 'gameDuration': 1}})),
        ('backwards', 100, json.dumps({'info': {'gameStartTimestamp': 110, 'gameEndTimestamp': 109}})),
        ('bad_json', 100, '{'),
    ])
    rows = [{'match_id': mid, 'game_creation_time': 100}
            for mid in ('no_end', 'backwards', 'bad_json', 'absent')]
    ends, diagnostics = load_result_end_times(path, rows)
    assert ends == {}
    assert diagnostics == {'requested': 4, 'found': 3, 'invalid': 3, 'valid': 0, 'missing': 1}


def test_database_cache_creation_disagreement_fails_loudly(tmp_path):
    path = _database(tmp_path, [('one', 101, '{}')])
    with pytest.raises(ValueError, match='timestamp mismatch'):
        load_result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])


def test_truncated_gzip_is_classified_invalid(tmp_path):
    payload = gzip.compress(b'{"info":{"gameStartTimestamp":110,"gameEndTimestamp":150}}')
    raw = base64.b64encode(payload[:-5]).decode()
    path = _database(tmp_path, [('one', 100, raw)])
    ends, diagnostics = load_result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])
    assert ends == {}
    assert diagnostics == {'requested': 1, 'found': 1, 'invalid': 1, 'valid': 0, 'missing': 0}


def test_missing_raw_json_schema_is_an_error(tmp_path):
    path = tmp_path / 'source.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation_time INTEGER)')
    with pytest.raises(sqlite3.OperationalError, match='raw_json'):
        load_result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])


def test_missing_database_is_an_error(tmp_path):
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        load_result_end_times(tmp_path / 'missing.db', [{'match_id': 'one', 'game_creation_time': 100}])


def test_corrupt_deflate_body_is_invalid_but_does_not_hide_valid_results(tmp_path):
    payload = bytearray(gzip.compress(b''))
    payload[10] = 255
    bad = base64.b64encode(payload).decode()
    good = json.dumps({'info': {'gameStartTimestamp': 110, 'gameEndTimestamp': 150}})
    db = _database(tmp_path, [('bad', 100, bad), ('good', 100, good)])
    ends, diagnostics = load_result_end_times(db, [
        {'match_id': mid, 'game_creation_time': 100} for mid in ('bad', 'good')
    ])
    assert ends == {'good': 150}
    assert diagnostics == {'requested': 2, 'found': 2, 'invalid': 1, 'valid': 1, 'missing': 0}
