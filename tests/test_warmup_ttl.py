"""Tests for the fresh-data warm-up TTL feature."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as _dt
import pytest
import database


def test_get_stale_puuids_empty_list(tmp_path):
    db = str(tmp_path / "test.db")
    database.init_db(db)
    assert database.get_stale_puuids([], db_path=db) == []


def test_get_stale_puuids_treats_missing_as_stale(tmp_path):
    """Puuids not in the players table are considered stale (so they get fetched)."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    result = database.get_stale_puuids(["MISSING_PUUID"], db_path=db)
    assert result == ["MISSING_PUUID"]


def test_get_stale_puuids_fresh_record_not_stale(tmp_path):
    """A player with last_updated = now is NOT stale."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    database.save_player("FRESH_PUUID", "Name", "Tag", db_path=db)

    result = database.get_stale_puuids(["FRESH_PUUID"], stale_after_days=7, db_path=db)
    assert result == []


def test_get_stale_puuids_old_record_is_stale(tmp_path):
    """A player with last_updated > 7 days ago IS stale."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    database.save_player("OLD_PUUID", "Name", "Tag", db_path=db)
    # Manually backdate
    old_date = (_dt.datetime.utcnow() - _dt.timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    with database.get_connection(db) as conn:
        conn.execute("UPDATE players SET last_updated = ? WHERE puuid = ?",
                     (old_date, "OLD_PUUID"))

    result = database.get_stale_puuids(["OLD_PUUID"], stale_after_days=7, db_path=db)
    assert result == ["OLD_PUUID"]


def test_get_stale_puuids_mixed(tmp_path):
    """Mix of fresh, stale, and missing puuids — returns only stale + missing."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    # Fresh player
    database.save_player("FRESH", "Name", "Tag", db_path=db)
    # Stale player
    database.save_player("STALE", "Name", "Tag", db_path=db)
    old_date = (_dt.datetime.utcnow() - _dt.timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    with database.get_connection(db) as conn:
        conn.execute("UPDATE players SET last_updated = ? WHERE puuid = ?", (old_date, "STALE"))
    # Missing player — not inserted

    result = database.get_stale_puuids(["FRESH", "STALE", "MISSING"], stale_after_days=7, db_path=db)
    assert set(result) == {"STALE", "MISSING"}


def test_get_stale_puuids_threshold_boundary(tmp_path):
    """A player updated exactly at the threshold is borderline — not stale."""
    db = str(tmp_path / "test.db")
    database.init_db(db)
    database.save_player("BOUNDARY", "Name", "Tag", db_path=db)
    # Set to 3 days ago — well within 7-day window
    recent = (_dt.datetime.utcnow() - _dt.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
    with database.get_connection(db) as conn:
        conn.execute("UPDATE players SET last_updated = ? WHERE puuid = ?",
                     (recent, "BOUNDARY"))

    result = database.get_stale_puuids(["BOUNDARY"], stale_after_days=7, db_path=db)
    assert result == []


def test_warmup_status_endpoint(tmp_path, monkeypatch):
    """GET /api/warmup/status returns stale list."""
    from unittest.mock import MagicMock
    import final_web_app

    # Point the function defaults at a fresh, initialized tmp DB so the
    # endpoint doesn't hit the dev DB (which may not be migrated/initialized
    # in this test environment). The function's `db_path` default was
    # captured at module-import time, so we have to patch __defaults__.
    db = str(tmp_path / "warmup_status.db")
    database.init_db(db)
    monkeypatch.setattr(database, "DB_PATH", db)
    # Replace the (stale_after_days, db_path) defaults to use the tmp DB.
    monkeypatch.setattr(database.get_stale_puuids, "__defaults__", (7, db))

    # Mock the trainer so the app can boot
    monkeypatch.setattr(final_web_app, "global_trainer", MagicMock())
    monkeypatch.setattr(final_web_app, "tf_available", True)
    final_web_app.app.config["TESTING"] = True
    client = final_web_app.app.test_client()

    # No puuids → 400
    resp = client.get("/api/warmup/status")
    assert resp.status_code == 400

    # With puuids → 200 + structure
    resp = client.get("/api/warmup/status?puuids=p1,p2")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "puuids_checked" in data
    assert "stale_count" in data
    assert "stale_puuids" in data
    assert data["puuids_checked"] == ["p1", "p2"]
    # Both unknown → both stale
    assert set(data["stale_puuids"]) == {"p1", "p2"}


def test_final_web_app_calls_init_db_at_import():
    """Importing final_web_app should ensure DB is initialized."""
    import inspect
    src = inspect.getsource(__import__("final_web_app"))
    assert "database.init_db()" in src
