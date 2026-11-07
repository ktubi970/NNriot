import sqlite3
from analytics.build import build_analytics
from analytics.db import initialize_schema, connect_analytics
from tests.analytics.factories import classic_match as make_classic_match
from tests.analytics.factories import write_source_db

def test_analytics_db_persists_wal_mode(tmp_path):
    mart_path = str(tmp_path / "test_mart.db")
    initialize_schema(mart_path)

    # Reopen with standard sqlite3 connection to inspect persistent PRAGMA
    conn = sqlite3.connect(mart_path)
    cur = conn.cursor()
    mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    assert mode.lower() == "wal"


def test_analytics_full_build_persists_wal_mode(tmp_path):
    source_db = tmp_path / "source.db"
    matches = [make_classic_match(match_id="EUW1_1", created_ms=1)]
    write_source_db(source_db, matches)
    target = tmp_path / "analytics.db"
    build_analytics(str(source_db), str(target))

    conn = sqlite3.connect(str(target))
    cur = conn.cursor()
    mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    assert mode.lower() == "wal"
