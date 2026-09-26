import sqlite3
import json

import pytest

from analytics.extract import ExtractionError, extract_match
from analytics.source import (
    connect_source,
    decode_raw_json,
    iter_source_matches,
    source_fingerprint,
    source_snapshot,
)


def test_extracts_one_match_two_teams_ten_players(classic_match):
    classic_match["info"]["participants"][0]["visionScore"] = 15.5
    result = extract_match("EUW1_1", classic_match)

    assert result.match["region"] == "EUW"
    assert result.match["patch"] == "16.14"
    assert len(result.teams) == 2
    assert len(result.players) == 10
    assert sum(row["win"] for row in result.teams) == 1
    assert result.players[0]["vision_score"] == 15.5
    assert result.teams[0]["gold"] == sum(
        player["gold"] for player in result.players if player["team_id"] == 100
    )


def test_rejects_non_classic_without_partial_rows(classic_match):
    classic_match["info"]["gameMode"] = "ARAM"

    with pytest.raises(ExtractionError, match="UNSUPPORTED_GAME_MODE") as raised:
        extract_match("EUW1_2", classic_match)

    assert raised.value.code == "UNSUPPORTED_GAME_MODE"
    assert raised.value.field == "info.gameMode"


def test_rejects_duplicate_puuid(classic_match):
    classic_match["info"]["participants"][1]["puuid"] = classic_match["info"][
        "participants"
    ][0]["puuid"]

    with pytest.raises(ExtractionError) as raised:
        extract_match("EUW1_3", classic_match)

    assert raised.value.code == "DUPLICATE_PARTICIPANT"


def test_source_access_is_query_only_and_decodes_matches(source_db):
    rows = list(iter_source_matches(str(source_db)))
    assert len(rows) == 1
    assert rows[0][0] == "EUW1_1"
    assert rows[0][1]["info"]["gameMode"] == "CLASSIC"

    with connect_source(str(source_db)) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM matches")


def test_source_tokens_are_deterministic_and_snapshot_is_cheap(source_db):
    first = source_fingerprint(str(source_db), "extractor-v1")
    second = source_fingerprint(str(source_db), "extractor-v1")

    assert first == second
    assert first["source_schema_version"] == "11"
    assert first["source_match_count"] == "1"
    assert first["source_max_creation_ms"] == "1"
    assert source_snapshot(str(source_db)) == {
        key: first[key]
        for key in (
            "source_schema_version",
            "source_match_count",
            "source_max_creation_ms",
        )
    }


def test_decoder_accepts_legacy_plain_json():
    payload = {"info": {"gameMode": "CLASSIC"}}
    assert decode_raw_json(json.dumps(payload)) == payload


def test_source_iterator_defaults_to_ordered_but_can_scan_payloads_sequentially(tmp_path):
    from analytics.source import iter_source_rows
    from tests.analytics.factories import classic_match, write_source_db

    source = write_source_db(tmp_path / "source.db", [
        classic_match(match_id=f"EUW1_{i}", created_ms=i) for i in (9, 1, 7, 2)
    ])
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE INDEX mode_match ON matches(game_mode,match_id)")
    before = source_fingerprint(str(source), "extractor-v1")
    assert [mid for mid, _ in iter_source_rows(str(source))] == ["EUW1_1", "EUW1_2", "EUW1_7", "EUW1_9"]
    assert [mid for mid, _ in iter_source_rows(str(source), ordered=False)] == ["EUW1_9", "EUW1_1", "EUW1_7", "EUW1_2"]
    assert [mid for mid, _ in iter_source_rows(str(source), "EUW1_1")] == ["EUW1_2", "EUW1_7", "EUW1_9"]
    assert [mid for mid, _ in iter_source_rows(str(source), "EUW1_1", ordered=False)] == ["EUW1_9", "EUW1_7", "EUW1_2"]
    assert source_fingerprint(str(source), "extractor-v1") == before


def test_fingerprint_scans_sequentially_but_keeps_exact_ordered_digest(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import hashlib
    import analytics.source as source_module
    from tests.analytics.factories import classic_match, write_source_db

    source = write_source_db(tmp_path / "source.db", [
        classic_match(match_id=f"EUW1_{i}", created_ms=i * 100) for i in (9, 1, 7, 2)
    ])
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE INDEX mode_only ON matches(game_mode)")
    original_connect = source_module.connect_source
    with original_connect(str(source)) as conn:
        ordered = conn.execute("SELECT match_id,game_creation_time FROM matches WHERE game_mode='CLASSIC' ORDER BY match_id").fetchall()
    expected = hashlib.sha256(b"extractor-v1")
    for match_id, created in ordered:
        expected.update(f"{match_id}:{int(created or 0)}\n".encode("utf-8"))
    statements = []

    @contextmanager
    def traced(path):
        with original_connect(path) as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(source_module, "connect_source", traced)
    actual = source_module.source_fingerprint(str(source), "extractor-v1")
    assert actual["dataset_version"] == expected.hexdigest()
    sql = next(statement for statement in statements if "game_creation_time" in statement)
    with original_connect(str(source)) as conn:
        plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
    assert "SCAN matches" in plan, plan
    assert any("TEMP B-TREE FOR ORDER BY" in step for step in plan), plan


@pytest.fixture
def snapshot_db(tmp_path):
    from contextlib import closing
    from core.database import _MIGRATIONS

    path = tmp_path / "snapshot.db"
    with closing(sqlite3.connect(path)) as conn:
        for _, statements in _MIGRATIONS:
            for sql in statements:
                conn.execute(sql)
        conn.commit()
    return path


@pytest.mark.parametrize("samples", [
    [],
    [("ARAM", 999)],
    [("CLASSIC", None)],
    [("CLASSIC", -5), ("CLASSIC", -1)],
    [("CLASSIC", 10), ("CLASSIC", None), ("CLASSIC", 20)],
    [("CLASSIC", 10), ("ARAM", 999), ("CLASSIC", None)],
    [("CLASSIC", 10), (None, 999)],
])
def test_snapshot_matches_original_aggregate_exactly(snapshot_db, samples):
    from contextlib import closing

    with closing(sqlite3.connect(snapshot_db)) as conn:
        conn.executemany(
            "INSERT INTO matches(match_id,game_mode,game_creation_time) VALUES(?,?,?)",
            [(str(i), mode, created) for i, (mode, created) in enumerate(samples)],
        )
        count, maximum = conn.execute(
            "SELECT COUNT(*),COALESCE(MAX(game_creation_time),0) "
            "FROM matches WHERE game_mode='CLASSIC'"
        ).fetchone()
        conn.commit()
    assert source_snapshot(str(snapshot_db)) == {
        "source_schema_version": "12",
        "source_match_count": str(count),
        "source_max_creation_ms": str(maximum),
    }


def test_all_classic_snapshot_uses_covering_fast_path_in_one_statement(snapshot_db, monkeypatch):
    from contextlib import closing, contextmanager
    import analytics.source as source_module

    with closing(sqlite3.connect(snapshot_db)) as conn:
        conn.executemany(
            "INSERT INTO matches(match_id,game_mode,game_creation_time) VALUES(?,'CLASSIC',?)",
            [(str(i), i) for i in range(500)],
        )
        conn.commit()
    statements = []
    original_connect = source_module.connect_source

    @contextmanager
    def traced(path):
        with original_connect(path) as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(source_module, "connect_source", traced)
    assert source_module.source_snapshot(str(snapshot_db))["source_max_creation_ms"] == "499"
    snapshot_statements = [sql for sql in statements if "COUNT(*)" in sql]
    assert len(snapshot_statements) == 1
    sql = snapshot_statements[0]
    with original_connect(str(snapshot_db)) as conn:
        plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
        assert any("COVERING INDEX idx_matches_game_creation" in step for step in plan), plan

        def vm_steps(query):
            steps = 0

            def progress():
                nonlocal steps
                steps += 1
                return 0

            conn.set_progress_handler(progress, 1)
            try:
                conn.execute(query).fetchone()
            finally:
                conn.set_progress_handler(None, 0)
            return steps

        baseline = "SELECT COUNT(*),COALESCE(MAX(game_creation_time),0) FROM matches WHERE game_mode='CLASSIC'"
        assert vm_steps(sql) < vm_steps(baseline)
