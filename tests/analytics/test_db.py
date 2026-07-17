import sqlite3

import pytest

from analytics.db import EvidenceRepository, connect_analytics, initialize_schema


def test_schema_has_required_fact_tables(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_schema(str(path))

    with connect_analytics(str(path), readonly=True) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert {
        "analytics_meta",
        "match_facts",
        "team_facts",
        "player_facts",
        "extraction_issues",
        "evidence_cache",
    } <= names


def test_readonly_connection_refuses_writes(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_schema(str(path))

    with connect_analytics(str(path), readonly=True) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO analytics_meta(key,value) VALUES ('unsafe','write')"
            )


def test_evidence_cache_is_scoped_by_dataset_and_method(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_schema(str(path))
    repo = EvidenceRepository(str(path))

    repo.put("ev-1", "dataset-a", "request-a", "1.0.0", {"value": 1})

    assert repo.get("dataset-a", "request-a", "1.0.0") == {"value": 1}
    assert repo.get("dataset-b", "request-a", "1.0.0") is None
    assert repo.get("dataset-a", "request-a", "2.0.0") is None
    assert repo.get_by_id("ev-1") == {"value": 1}
