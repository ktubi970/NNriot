"""Availability provenance checks for the isolated matchup benchmark."""
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest


RESEARCH = Path(__file__).resolve().parents[1] / 'docs/research-improvement'
SPEC = importlib.util.spec_from_file_location('accuracy_matchup_experiment', RESEARCH / 'accuracy_matchup_experiment.py')
MODULE = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(RESEARCH), *sys.path]):
    SPEC.loader.exec_module(MODULE)


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
    ends, diagnostics = MODULE.result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])
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
    ends, diagnostics = MODULE.result_end_times(path, rows)
    assert ends == {}
    assert diagnostics == {'requested': 4, 'found': 3, 'invalid': 3, 'valid': 0, 'missing': 1}


def test_database_cache_creation_disagreement_fails_loudly(tmp_path):
    path = _database(tmp_path, [('one', 101, '{}')])
    with pytest.raises(ValueError, match='timestamp mismatch'):
        MODULE.result_end_times(path, [{'match_id': 'one', 'game_creation_time': 100}])
