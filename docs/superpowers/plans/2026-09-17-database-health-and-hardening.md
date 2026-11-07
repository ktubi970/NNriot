# Database Health, Integrity & Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore full database health across `training_data.db` and `analytics.db` by fixing broken tests, backfilling `match_participants` to unblock player history, purging polluted non-CLASSIC training data, harmonizing configuration paths, and enforcing WAL mode.

**Architecture:** 
1. Fix the regression test suite so baseline tests pass at 100%.
2. Harmonize analytics configuration between `core/config.py`, `analytics/config.py`, and `app/__init__.py`.
3. Add WAL mode persistence to `analytics/db.py` and `analytics/build.py`.
4. Provide safe, verified data sanitation in `core/database.py` to purge `TEST_MATCH` and non-CLASSIC game modes from `training_dataset`.
5. Optimize and execute `match_participants` backfill so that player match history queries operate in $O(1)$ without recency truncation.
6. Clean repository root artifacts and document operational procedures.

**Tech Stack:** Python 3.12, SQLite 3 (WAL mode), Flask, Pytest, NumPy.

## Global Constraints

- Never break existing migration order; schema version is 12.
- All database operations must respect transaction safety (commit on success, rollback on exception, close unconditionally).
- Tests must run under `.venv312` with 0 failures.
- No destructive deletions without explicit match_id or game_mode filtering.

---

### Task 1: Fix Regression Test & Assertions (Green Test Suite)

**Files:**
- Modify: `tests/test_database_v11_migration.py:59-61`
- Test: `tests/test_database_v11_migration.py`

**Interfaces:**
- Consumes: `core.database.SCHEMA_VERSION` (currently 12)
- Produces: Passing test suite asserting current schema version 12 and migration contract.

- [ ] **Step 1: Write the updated test asserting schema version 12**

Modify `tests/test_database_v11_migration.py`:
```python
def test_schema_version_is_at_least_12():
    assert database.SCHEMA_VERSION >= 12
    assert database.SCHEMA_VERSION == 12
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_database_v11_migration.py -v`
Expected: PASS with 4 passed tests.

- [ ] **Step 3: Run full test suite to verify 100% green status**

Run: `& .venv312\Scripts\pytest.exe -q`
Expected: 284 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_database_v11_migration.py
git commit -m "fix(tests): update schema version assertion to v12"
```

---

### Task 2: Harmonize Database Path Configurations Across Layers

**Files:**
- Modify: `core/config.py:16-18`
- Modify: `analytics/config.py:14-18`
- Modify: `app/__init__.py:39-41`
- Create: `tests/test_db_config_harmony.py`

**Interfaces:**
- Consumes: Environment variables `NNRIOT_ANALYTICS_DB_PATH`, `ANALYTICS_DB_PATH`, `NNRIOT_DB_PATH`
- Produces: Consistent resolution order and default path (`analytics.db`) across all layers.

- [ ] **Step 1: Write the failing configuration test**

Create `tests/test_db_config_harmony.py`:
```python
import os
from unittest.mock import patch

def test_analytics_config_prefers_nnriot_prefix_and_falls_back():
    from analytics.config import AnalyticsConfig
    
    with patch.dict(os.environ, {"NNRIOT_ANALYTICS_DB_PATH": "custom_nn.db", "ANALYTICS_DB_PATH": "custom_plain.db"}):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == "custom_nn.db"

    with patch.dict(os.environ, {"ANALYTICS_DB_PATH": "custom_plain.db"}, clear=True):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == "custom_plain.db"

    with patch.dict(os.environ, {}, clear=True):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == "analytics.db"
```

- [ ] **Step 2: Run test to verify failure**

Run: `& .venv312\Scripts\pytest.exe tests/test_db_config_harmony.py -v`
Expected: FAIL (Attribute or value mismatch with fallback).

- [ ] **Step 3: Implement unified configuration logic**

In `core/config.py`:
```python
# Database Configuration
DB_PATH = os.environ.get("NNRIOT_DB_PATH", "training_data.db")
ANALYTICS_DB_PATH = os.environ.get("NNRIOT_ANALYTICS_DB_PATH") or os.environ.get("ANALYTICS_DB_PATH", "analytics.db")
MAX_DB_SIZE = int(os.environ.get("NNRIOT_MAX_DB_SIZE_GB", "50")) * 1024 * 1024 * 1024
```

In `analytics/config.py`:
```python
    analytics_path: str = field(
        default_factory=lambda: os.getenv("NNRIOT_ANALYTICS_DB_PATH")
        or os.getenv("ANALYTICS_DB_PATH", "analytics.db")
    )
```

In `app/__init__.py`:
```python
    app.config["ANALYTICS_DB_PATH"] = (
        os.getenv("NNRIOT_ANALYTICS_DB_PATH")
        or os.getenv("ANALYTICS_DB_PATH", "analytics.db")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_db_config_harmony.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/config.py analytics/config.py app/__init__.py tests/test_db_config_harmony.py
git commit -m "fix(config): harmonize analytics DB path resolution across core, analytics, and app"
```

---

### Task 3: Enforce WAL Journal Mode in Analytics Mart

**Files:**
- Modify: `analytics/db.py:137-147`
- Modify: `analytics/build.py:219-225`
- Create: `tests/test_analytics_wal_mode.py`

**Interfaces:**
- Consumes: `connect_analytics`, `build_analytics`
- Produces: Analytics mart file persisted with `PRAGMA journal_mode=WAL` to prevent table locking under concurrent web requests.

- [ ] **Step 1: Write failing test for WAL enforcement**

Create `tests/test_analytics_wal_mode.py`:
```python
import sqlite3
from analytics.db import initialize_schema, connect_analytics

def test_analytics_db_persists_wal_mode(tmp_path):
    mart_path = str(tmp_path / "test_mart.db")
    initialize_schema(mart_path)

    # Reopen with standard sqlite3 connection to inspect persistent PRAGMA
    conn = sqlite3.connect(mart_path)
    cur = conn.cursor()
    mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    assert mode.lower() == "wal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv312\Scripts\pytest.exe tests/test_analytics_wal_mode.py -v`
Expected: FAIL (mode is `delete`).

- [ ] **Step 3: Implement WAL enforcement in analytics**

In `analytics/db.py`:
```python
def initialize_schema(path: str) -> None:
    """Create or validate the version-one analytical schema."""
    with connect_analytics(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA_SQL)
        set_meta(conn, "schema_version", str(ANALYTICS_SCHEMA_VERSION))
```

In `analytics/build.py`:
Add WAL persistence after index creation in `_full_build`:
```python
            create_secondary_indexes(conn)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _write_metadata(conn, fingerprint, report, config, "full")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_analytics_wal_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analytics/db.py analytics/build.py tests/test_analytics_wal_mode.py
git commit -m "fix(analytics): enforce persistent WAL journal mode on analytics mart"
```

---

### Task 4: Purge Contaminated Training Data & Fixtures

**Files:**
- Modify: `core/database.py` (add `purge_polluted_training_data`)
- Create: `tests/test_db_purge_polluted.py`

**Interfaces:**
- Consumes: `matches`, `training_dataset`
- Produces: Cleaned dataset containing only legitimate CLASSIC matches.

- [ ] **Step 1: Write failing test for data purge**

Create `tests/test_db_purge_polluted.py`:
```python
import sqlite3
from core import database

def test_purge_polluted_training_data(tmp_path):
    db_path = str(tmp_path / "test_purge.db")
    database.init_db(db_path)

    with database.get_connection(db_path) as conn:
        # 1 valid match
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('VALID_1', 'CLASSIC')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('VALID_1', '{}', 1)")
        # 1 ARAM match
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('ARAM_1', 'ARAM')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('ARAM_1', '{}', 0)")
        # 1 TEST_MATCH
        conn.execute("INSERT INTO matches (match_id, game_mode) VALUES ('TEST_MATCH', 'CLASSIC')")
        conn.execute("INSERT INTO training_dataset (match_id, feature_json, winner_label) VALUES ('TEST_MATCH', '{}', 1)")

    summary = database.purge_polluted_training_data(db_path=db_path)

    assert summary["deleted_test_matches"] == 1
    assert summary["deleted_polluted_training_records"] == 2  # ARAM + TEST_MATCH
    
    with database.get_connection(db_path) as conn:
        rem_td = conn.execute("SELECT match_id FROM training_dataset").fetchall()
        assert len(rem_td) == 1
        assert rem_td[0]["match_id"] == "VALID_1"
```

- [ ] **Step 2: Run test to verify failure**

Run: `& .venv312\Scripts\pytest.exe tests/test_db_purge_polluted.py -v`
Expected: FAIL (`AttributeError: module 'core.database' has no attribute 'purge_polluted_training_data'`).

- [ ] **Step 3: Implement `purge_polluted_training_data` in `core/database.py`**

Add to `core/database.py`:
```python
def purge_polluted_training_data(db_path: str = DB_PATH) -> dict:
    """
    Remove non-CLASSIC matches and dummy test fixtures from training_dataset
    and delete TEST_MATCH from matches.
    """
    with get_connection(db_path) as conn:
        # Delete training records for TEST_MATCH or non-CLASSIC game modes
        cur = conn.execute(
            """
            DELETE FROM training_dataset
            WHERE match_id = 'TEST_MATCH'
               OR match_id IN (
                   SELECT match_id FROM matches WHERE game_mode != 'CLASSIC' OR game_mode IS NULL OR game_mode = ''
               )
            """
        )
        deleted_training = cur.rowcount

        # Delete TEST_MATCH from matches table
        cur2 = conn.execute("DELETE FROM matches WHERE match_id = 'TEST_MATCH'")
        deleted_test = cur2.rowcount

    logger.info(
        "Purged %d polluted training records and %d test matches.",
        deleted_training,
        deleted_test,
    )
    return {
        "deleted_polluted_training_records": deleted_training,
        "deleted_test_matches": deleted_test,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_db_purge_polluted.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/database.py tests/test_db_purge_polluted.py
git commit -m "feat(db): add purge_polluted_training_data to clean ARAM/URF and test fixtures"
```

---

### Task 5: High-Throughput Batch Backfill for `match_participants`

**Files:**
- Modify: `core/database.py:1099-1175` (optimize batching query)
- Create: `scripts/backfill_participants.py`
- Create: `tests/test_backfill_participants_script.py`

**Interfaces:**
- Consumes: `matches.raw_json`
- Produces: Fully populated `match_participants` table and reliable `/api/player/history` for all players.

- [ ] **Step 1: Write test for the backfill CLI script**

Create `tests/test_backfill_participants_script.py`:
```python
import pytest
from core import database
from scripts.backfill_participants import run_backfill

def test_backfill_participants_script(tmp_path):
    db_path = str(tmp_path / "test_backfill.db")
    database.init_db(db_path)

    # Insert sample match
    match_data = {
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1700000000000,
            "gameDuration": 1500,
            "participants": [
                {
                    "puuid": "p-123",
                    "championName": "Jinx",
                    "teamPosition": "BOTTOM",
                    "win": True,
                    "kills": 10, "deaths": 1, "assists": 5,
                    "totalMinionsKilled": 180, "goldEarned": 12000,
                }
            ]
        }
    }
    database.save_match("MATCH_TEST_1", match_data, db_path=db_path)

    # Clear participants table to simulate pre-migration state
    with database.get_connection(db_path) as conn:
        conn.execute("DELETE FROM match_participants")

    total = run_backfill(batch_size=50, max_matches=100, db_path=db_path)
    assert total == 1

    history = database.get_player_match_history("p-123", limit=5, db_path=db_path)
    assert len(history) == 1
    assert history[0]["champion"] == "Jinx"
```

- [ ] **Step 2: Run test to verify failure**

Run: `& .venv312\Scripts\pytest.exe tests/test_backfill_participants_script.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.backfill_participants'`).

- [ ] **Step 3: Implement `scripts/backfill_participants.py`**

Create `scripts/backfill_participants.py`:
```python
#!/usr/bin/env python3
"""
backfill_participants.py - High-throughput backfill of match_participants from matches.raw_json.
"""
import os
import sys
import time
import argparse
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_participants")

def run_backfill(batch_size: int = 500, max_matches: int = None, db_path: str = database.DB_PATH) -> int:
    start_time = time.time()
    total_backfilled = 0

    logger.info("Starting match_participants backfill on '%s' (batch_size=%d)...", db_path, batch_size)
    while True:
        limit = batch_size
        if max_matches is not None:
            limit = min(batch_size, max_matches - total_backfilled)
            if limit <= 0:
                break

        count = database.backfill_match_participants(limit=limit, db_path=db_path)
        if count == 0:
            break
        total_backfilled += count
        elapsed = time.time() - start_time
        rate = total_backfilled / max(1e-3, elapsed)
        logger.info("Backfilled %d matches so far (%.1f matches/s)...", total_backfilled, rate)

    total_time = time.time() - start_time
    logger.info("Finished backfill: %d total matches processed in %.2f seconds.", total_backfilled, total_time)
    return total_backfilled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill match_participants table")
    parser.add_argument("--batch-size", type=int, default=500, help="Matches per transaction batch")
    parser.add_argument("--max", type=int, default=None, help="Max matches to backfill")
    parser.add_argument("--db", type=str, default=database.DB_PATH, help="Path to database")
    args = parser.parse_args()

    run_backfill(batch_size=args.batch_size, max_matches=args.max, db_path=args.db)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_backfill_participants_script.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_participants.py tests/test_backfill_participants_script.py
git commit -m "feat(db): add backfill_participants utility script with tests"
```

---

### Task 6: Clean Up Repository Root & Workspace Hygiene

**Files:**
- Remove: Orphaned temp files in workspace root (`temp_body.html`, `test.db`)

- [ ] **Step 1: Check git status and remove stray root test/debug artifacts**

Remove transient artifacts that are already in `.gitignore` or are obsolete build artifacts:
`temp_body.html`, `test.db`.

- [ ] **Step 2: Verify git status is clean**

Run: `git status`
Expected: Clean working tree.

- [ ] **Step 3: Run complete test suite**

Run: `& .venv312\Scripts\pytest.exe -q`
Expected: 288 passed, 0 failed.

---

## Verification Plan

### Automated Tests
- `& .venv312\Scripts\pytest.exe tests/test_database_v11_migration.py -v`
- `& .venv312\Scripts\pytest.exe tests/test_db_config_harmony.py -v`
- `& .venv312\Scripts\pytest.exe tests/test_analytics_wal_mode.py -v`
- `& .venv312\Scripts\pytest.exe tests/test_db_purge_polluted.py -v`
- `& .venv312\Scripts\pytest.exe tests/test_backfill_participants_script.py -v`
- `& .venv312\Scripts\pytest.exe -q` (Entire suite passing without failure)

### Operational Execution (Post-implementation)
- Execute `python -m scripts.backfill_participants` to populate `match_participants` in `training_data.db`.
- Execute `python -c "from core.database import purge_polluted_training_data; purge_polluted_training_data()"` on production DB.
- Run `python -m analytics update` to synchronize `analytics.db`.
