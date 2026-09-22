import hashlib
import sqlite3

import analytics.build as build_module
import pytest

pytestmark = pytest.mark.integration

from analytics.build import build_analytics
from analytics.cli import main
from analytics.db import EvidenceRepository, all_meta, connect_analytics
from tests.analytics.factories import write_source_db


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_build_is_idempotent_and_does_not_write_source(source_db, tmp_path):
    target = tmp_path / "analytics.db"
    before = _sha(source_db)

    first = build_analytics(str(source_db), str(target))
    second = build_analytics(str(source_db), str(target))

    assert _sha(source_db) == before
    assert first.inserted_matches == second.inserted_matches == 1
    assert first.quality_gate_passed is True
    with connect_analytics(str(target), readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM match_facts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM team_facts").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM player_facts").fetchone()[0] == 10


def test_incremental_adds_only_new_match(source_db_factory, tmp_path):
    source = source_db_factory(match_count=1)
    target = tmp_path / "analytics.db"
    build_analytics(str(source), str(target))
    source_db_factory(path=source, match_count=2)

    report = build_analytics(str(source), str(target), incremental=True)

    assert report.eligible_matches == 2
    assert report.inserted_matches == 1
    with connect_analytics(str(target), readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM match_facts").fetchone()[0] == 2
    metadata = all_meta(str(target))
    assert metadata["dataset_version"] == report.dataset_version
    assert metadata["source_match_count"] == "2"


def test_incremental_invalidates_evidence_from_previous_dataset(
    source_db_factory, tmp_path
):
    source = source_db_factory(match_count=1)
    target = tmp_path / "analytics.db"
    first = build_analytics(str(source), str(target))
    repository = EvidenceRepository(str(target))
    repository.put("ev-1", first.dataset_version, "request", "1.0.0", {"ok": True})
    source_db_factory(path=source, match_count=2)

    build_analytics(str(source), str(target), incremental=True)

    assert repository.get_by_id("ev-1") is None


def test_expected_extraction_error_is_audited_without_partial_rows(
    classic_match, tmp_path
):
    classic_match["info"]["participants"][1]["puuid"] = classic_match["info"][
        "participants"
    ][0]["puuid"]
    source = write_source_db(tmp_path / "source.db", [classic_match])
    target = tmp_path / "analytics.db"

    report = build_analytics(str(source), str(target))

    assert report.issue_counts == {"DUPLICATE_PARTICIPANT": 1}
    assert report.unexplained_failures == 0
    with connect_analytics(str(target), readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM match_facts").fetchone()[0] == 0
        issue = conn.execute(
            "SELECT issue_code,field FROM extraction_issues"
        ).fetchone()
    assert tuple(issue) == (
        "DUPLICATE_PARTICIPANT",
        "info.participants[1].puuid",
    )


def test_null_raw_json_is_an_audited_exclusion(source_db, tmp_path):
    with sqlite3.connect(source_db) as conn:
        conn.execute("DROP TABLE matches")
        conn.execute(
            "CREATE TABLE matches("
            "match_id TEXT PRIMARY KEY,raw_json TEXT,game_mode TEXT,"
            "game_creation_time INTEGER,game_duration INTEGER,season_id INTEGER)"
        )
        conn.execute(
            "INSERT INTO matches VALUES ('EUW1_1',NULL,'CLASSIC',1,1800,16)"
        )
    target = tmp_path / "analytics.db"

    report = build_analytics(str(source_db), str(target))

    assert report.eligible_matches == 1
    assert report.inserted_matches == 0
    assert report.issue_counts == {"INVALID_RAW_JSON": 1}
    assert report.unexplained_failures == 0
    assert report.quality_gate_passed is True
    with connect_analytics(str(target), readonly=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM match_facts").fetchone()[0] == 0
        issue = conn.execute(
            "SELECT issue_code,field FROM extraction_issues"
        ).fetchone()
    assert tuple(issue) == ("INVALID_RAW_JSON", "raw_json")


def test_failed_full_build_keeps_last_published_database(
    source_db, tmp_path, monkeypatch
):
    target = tmp_path / "analytics.db"
    build_analytics(str(source_db), str(target))
    before = _sha(target)

    def fail_unexpectedly(match_id, payload):
        raise RuntimeError("synthetic extractor failure")

    monkeypatch.setattr(build_module, "extract_match", fail_unexpectedly)
    report = build_analytics(str(source_db), str(target))

    assert report.quality_gate_passed is False
    assert report.unexplained_failures == 1
    assert _sha(target) == before


def test_full_build_rechecks_source_after_index_creation(
    source_db, tmp_path, monkeypatch
):
    target = tmp_path / "analytics.db"
    build_analytics(str(source_db), str(target))
    before = _sha(target)
    original = build_module.create_secondary_indexes

    def create_indexes_then_mutate_source(conn):
        original(conn)
        with sqlite3.connect(source_db) as source_conn:
            source_conn.execute(
                "UPDATE matches SET game_creation_time=game_creation_time+1"
            )

    monkeypatch.setattr(
        build_module,
        "create_secondary_indexes",
        create_indexes_then_mutate_source,
    )

    with pytest.raises(RuntimeError, match="source changed"):
        build_analytics(str(source_db), str(target))

    assert _sha(target) == before


def test_cli_health_prints_machine_readable_metadata(source_db, tmp_path, capsys):
    target = tmp_path / "analytics.db"
    build_analytics(str(source_db), str(target))

    exit_code = main(
        ["--source", str(source_db), "--analytics", str(target), "health"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"dataset_version"' in output
    assert '"quality_gate_passed": "true"' in output
