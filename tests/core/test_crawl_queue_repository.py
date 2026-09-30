"""Tests for SQLite Migration v14 and CrawlQueueRepository."""

import sqlite3
import pytest
from core import database
from core.collection import CrawlQueueRepository, CrawlItem


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary initialized SQLite database."""
    db_file = tmp_path / "test_crawl_queue.db"
    database.init_db(str(db_file))
    return str(db_file)


def test_crawl_queue_repository_exported():
    """Verify CrawlQueueRepository is exported from core.collection."""
    import core.collection
    assert hasattr(core.collection, "CrawlQueueRepository")
    assert core.collection.CrawlQueueRepository is CrawlQueueRepository


def test_migration_14_schema_and_data_preservation(tmp_path):
    """Test upgrading from schema v13 to v14 preserves existing queue data and creates indices."""
    db_file = str(tmp_path / "migration_test.db")

    # Manually create schema at v13
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (13)")
        conn.execute(
            """
            CREATE TABLE crawl_queue (
                game_name TEXT,
                tag_line  TEXT,
                priority  INTEGER DEFAULT 0,
                is_processed INTEGER DEFAULT 0,
                added_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(game_name, tag_line)
            )
            """
        )
        conn.execute("CREATE INDEX idx_crawl_processed ON crawl_queue(is_processed)")
        conn.execute(
            "INSERT INTO crawl_queue (game_name, tag_line, priority, is_processed) VALUES ('Faker', 'KR1', 5, 0)"
        )
        conn.execute(
            "INSERT INTO crawl_queue (game_name, tag_line, priority, is_processed) VALUES ('Caps', 'EUW', 10, 1)"
        )

    # Run migration
    database.migrate_db(db_file)

    # Verify schema version
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert ver == 14

        # Verify table columns
        cols = {row["name"]: row["type"] for row in conn.execute("PRAGMA table_info(crawl_queue)").fetchall()}
        assert "id" in cols
        assert "puuid" in cols
        assert "region" in cols
        assert "game_name" in cols
        assert "tag_line" in cols
        assert "priority" in cols
        assert "is_processed" in cols
        assert "created_at" in cols
        assert "processed_at" in cols

        # Verify data preservation with default region EUW
        rows = conn.execute("SELECT * FROM crawl_queue ORDER BY priority DESC").fetchall()
        assert len(rows) == 2
        caps = rows[0]
        assert caps["game_name"] == "Caps"
        assert caps["tag_line"] == "EUW"
        assert caps["priority"] == 10
        assert caps["is_processed"] == 1
        assert caps["region"] == "EUW"

        faker = rows[1]
        assert faker["game_name"] == "Faker"
        assert faker["tag_line"] == "KR1"
        assert faker["priority"] == 5
        assert faker["is_processed"] == 0
        assert faker["region"] == "EUW"

        # Verify indices exist
        indices = [row["name"] for row in conn.execute("PRAGMA index_list(crawl_queue)").fetchall()]
        assert any("puuid" in idx for idx in indices)


def test_enqueue_puuids_and_conflict_resolution(test_db):
    """Test batch enqueueing PUUIDs and duplicate conflict resolution."""
    repo = CrawlQueueRepository(test_db)

    # Empty list
    assert repo.enqueue_puuids([], region="EUW") == 0

    # Insert batch of 3
    inserted = repo.enqueue_puuids(["p1", "p2", "p3"], region="EUW", priority=5)
    assert inserted == 3

    # Re-insert with overlapping puuids
    # "p2" and "p3" should be ignored; "p4" inserted
    inserted2 = repo.enqueue_puuids(["p2", "p3", "p4"], region="EUW", priority=10)
    assert inserted2 == 1

    stats = repo.get_stats()
    assert stats["total"] == 4
    assert stats["pending"] == 4


def test_enqueue_riot_id(test_db):
    """Test enqueueing Riot IDs with and without PUUID."""
    repo = CrawlQueueRepository(test_db)

    # Without PUUID
    ok = repo.enqueue_riot_id(game_name="Agurin", tag_line="EUW", region="EUW", priority=8)
    assert ok is True

    items = repo.pop_next(limit=10)
    assert len(items) == 1
    assert items[0].game_name == "Agurin"
    assert items[0].tag_line == "EUW"
    assert items[0].region == "EUW"
    assert items[0].priority == 8
    assert items[0].puuid is None

    # With PUUID - initial insert
    ok2 = repo.enqueue_riot_id(
        game_name="Faker", tag_line="KR1", region="KR", priority=5, puuid="puuid_faker"
    )
    assert ok2 is True

    # With PUUID - upsert (update priority and name)
    ok3 = repo.enqueue_riot_id(
        game_name="Hide on bush", tag_line="KR1", region="KR", priority=15, puuid="puuid_faker"
    )
    assert ok3 is True

    items = repo.pop_next(limit=10)
    # Should have 2 items total (Agurin + Faker updated)
    assert len(items) == 2
    faker_item = next(it for it in items if it.puuid == "puuid_faker")
    assert faker_item.game_name == "Hide on bush"
    assert faker_item.priority == 15


def test_pop_next_priority_and_ordering(test_db):
    """Test pop_next respects priority DESC and id ASC (FIFO for same priority)."""
    repo = CrawlQueueRepository(test_db)

    repo.enqueue_puuids(["low"], region="EUW", priority=1)
    repo.enqueue_puuids(["high_1"], region="EUW", priority=10)
    repo.enqueue_puuids(["mid"], region="EUW", priority=5)
    repo.enqueue_puuids(["high_2"], region="EUW", priority=10)

    items = repo.pop_next(limit=10)
    assert len(items) == 4
    assert isinstance(items[0], CrawlItem)
    assert [item.puuid for item in items] == ["high_1", "high_2", "mid", "low"]


def test_mark_processed(test_db):
    """Test marking items as processed by PUUID and by ID."""
    repo = CrawlQueueRepository(test_db)
    repo.enqueue_puuids(["p_alpha", "p_beta"], region="EUW", priority=5)
    repo.enqueue_riot_id("RiotPlayer", "NA1", region="NA", priority=3)

    items = repo.pop_next(limit=10)
    assert len(items) == 3

    # Mark p_alpha processed by PUUID
    repo.mark_processed(puuid="p_alpha")

    # Mark RiotPlayer processed by ID
    riot_item = next(it for it in items if it.game_name == "RiotPlayer")
    assert riot_item.id is not None
    repo.mark_processed_by_id(riot_item.id)

    # Now pop_next should only yield p_beta
    pending = repo.pop_next(limit=10)
    assert len(pending) == 1
    assert pending[0].puuid == "p_beta"


def test_get_stats(test_db):
    """Test get_stats calculates total, processed, and pending counts accurately."""
    repo = CrawlQueueRepository(test_db)
    assert repo.get_stats() == {"total": 0, "processed": 0, "pending": 0}

    repo.enqueue_puuids(["p1", "p2", "p3"], region="EUW", priority=1)
    assert repo.get_stats() == {"total": 3, "processed": 0, "pending": 3}

    repo.mark_processed(puuid="p1")
    assert repo.get_stats() == {"total": 3, "processed": 1, "pending": 2}
