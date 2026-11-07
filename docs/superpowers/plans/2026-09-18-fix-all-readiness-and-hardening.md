# Fix All: Database Readiness, Test Isolation & Data Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all blockers preventing database filling: isolate test database operations from production `training_data.db`, lock down data collectors to CLASSIC game mode to prevent repollution, optimize and complete the `match_participants` backfill (17.4k remaining matches), synchronize `analytics.db` to 117k matches with persistent WAL mode, and clean orphan root artifacts.

**Architecture:** 
1. Isolate `tests/test_components.py` to run against a temporary SQLite database using `tempfile.TemporaryDirectory`, preventing any injection of dummy `TEST_MATCH` rows into `training_data.db` and preventing lock contention.
2. Harden `scripts/data_collector.py` and `scripts/background_collector.py` by requiring `game_mode == "CLASSIC"` in `_build_training_record` and filtering for queue 420 (Ranked Solo/Duo).
3. Optimize `core.database.backfill_match_participants` using `WHERE NOT EXISTS` and protect against corrupted/empty match records so the backfill cannot enter an infinite loop.
4. Execute `scripts/backfill_participants.py` on `training_data.db` until 100% of matches are processed into `match_participants`.
5. Run `python -m analytics update` to synchronize `analytics.db` with all matches in `training_data.db` and enforce persistent WAL journal mode.
6. Clean up root directory database artifacts (`test.db`, `test_old.db`, `test_wal_check2.db`) and run the full test suite to guarantee 100% green status.

**Tech Stack:** Python 3.12, SQLite 3 (WAL mode), Pytest, Repository Pattern.

## Global Constraints

- Never break existing migration order; schema version is 12.
- Production `training_data.db` must never be touched by automated test fixtures.
- All database write operations must adhere to transaction safety (commit on success, rollback on exception, close unconditionally).
- Tests must pass with 0 failures under `.venv312`.
- Zero non-CLASSIC records in `training_dataset`.

---

### Task 1: Isolate `tests/test_components.py` from Production Database

**Files:**
- Modify: `tests/test_components.py:40-110`
- Test: `tests/test_components.py`

**Interfaces:**
- Consumes: `core.database.init_db(db_path)`, `core.repository.MatchRepository(db_path)`, `core.repository.TrainingRepository(db_path)`, `core.repository.SystemRepository(db_path)`
- Produces: Isolated database test that executes in an ephemeral directory, verifies Repository DAO functionality, and leaves `training_data.db` untouched.

- [ ] **Step 1: Write the updated test with temp directory isolation**

Modify `test_database()` in `tests/test_components.py`:
```python
def test_database():
    """
    Test database functionality using Repository Pattern in an isolated temp database.
    """
    import os
    import tempfile
    try:
        from core import database
        from core.repository import MatchRepository, TrainingRepository, SystemRepository
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = os.path.join(tmpdir, "test_components.db")
            database.init_db(test_db)

            match_repo = MatchRepository(test_db)
            train_repo = TrainingRepository(test_db)
            sys_repo = SystemRepository(test_db)

            # Satisfaction of Foreign Key constraint: insert a dummy match first
            with match_repo.get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO matches (match_id, game_mode) VALUES (?, ?)",
                    ("TEST_MATCH", "CLASSIC"),
                )
                conn.commit()

            # perform a quick insert and fetch
            dummy_feature = {"foo": "bar"}
            train_repo.save_training_records_batch([
                (
                    "TEST_MATCH",
                    dummy_feature,
                    0,  # winner_label
                    {"test": True},  # labels_json
                )
            ])

            stats = sys_repo.get_db_stats()
            required_keys = [
                "players",
                "matches",
                "trained_records",
                "untrained_records",
                "trained_ratio",
            ]
            missing = [k for k in required_keys if k not in stats]

            if missing:
                print(f"[FAIL] Database stats missing keys: {missing}")
                return False

            with train_repo.get_connection() as conn:
                row = conn.execute(
                    "SELECT match_id, feature_json FROM training_dataset WHERE match_id = ?",
                    ("TEST_MATCH",),
                ).fetchone()

            if not row:
                print("[FAIL] Test record not found")
                return False

            print(
                f"[OK] Database stats consistent: {stats['matches']} matches, {stats['trained_records']} trained"
            )
            print("[OK] Database training record save/fetch working")
            return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Database failed: {e}")
        return False
```

- [ ] **Step 2: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_components.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 3: Verify production `training_data.db` contains 0 `TEST_MATCH` entries**

Run: `& .venv312\Scripts\python.exe -c "import sqlite3; conn=sqlite3.connect('file:training_data.db?mode=ro', uri=True); cur=conn.cursor(); print('TEST_MATCH count:', cur.execute(\"SELECT count(*) FROM matches WHERE match_id='TEST_MATCH'\").fetchone()[0]); conn.close()"`
Expected: `TEST_MATCH count: 0`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_components.py
git commit -m "fix(tests): isolate test_components to temporary database"
```

---

### Task 2: Lock Down Data Collectors to CLASSIC Game Mode

**Files:**
- Modify: `scripts/data_collector.py:50-53, 90-95, 149-152`
- Modify: `scripts/background_collector.py:88-91, 109-112`
- Create: `tests/test_collector_game_mode_guard.py`

**Interfaces:**
- Consumes: Riot match detail payload dictionary
- Produces: `_build_training_record` returns `None` for any match whose `gameMode` is not `"CLASSIC"`.

- [ ] **Step 1: Write failing test for collector game mode guard**

Create `tests/test_collector_game_mode_guard.py`:
```python
from scripts.data_collector import _build_training_record

def test_build_training_record_rejects_all_non_classic():
    # Valid CLASSIC payload stub
    valid_payload = {
        "info": {
            "gameMode": "CLASSIC",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [{"teamId": 100, "championId": 1, "win": True} for _ in range(10)],
            "gameDuration": 1200,
        }
    }
    
    # ARAM mode
    aram_payload = {
        "info": {
            "gameMode": "ARAM",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [{"teamId": 100, "championId": 1, "win": True} for _ in range(10)],
            "gameDuration": 1200,
        }
    }
    
    # URF mode
    urf_payload = {
        "info": {
            "gameMode": "URF",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [{"teamId": 100, "championId": 1, "win": True} for _ in range(10)],
            "gameDuration": 1200,
        }
    }

    assert _build_training_record("ARAM_MATCH", aram_payload) is None
    assert _build_training_record("URF_MATCH", urf_payload) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv312\Scripts\pytest.exe tests/test_collector_game_mode_guard.py -v`
Expected: FAIL (ARAM/URF not rejected by `_build_training_record`).

- [ ] **Step 3: Update `scripts/data_collector.py` and `scripts/background_collector.py`**

In `scripts/data_collector.py`:
```python
def _build_training_record(
    match_id: str, details: dict, region: str = None
) -> tuple | None:
    try:
        info = details.get("info", details)
        game_mode = (info.get("gameMode") or "").upper()
        if game_mode != "CLASSIC":
            return None
```
And in `collect_training_data`:
```python
        match_ids = api.get_match_ids(
            puuid, count=matches_per_player, start_time=start_time, queue=420, type="ranked"
        )
```

In `scripts/background_collector.py`:
```python
            # Fetch last 20 ranked solo matches
            match_ids = api.get_match_ids(puuid, count=20, queue=420, type="ranked")
```
And in match processing loop:
```python
                info = details.get("info", {})
                if (info.get("gameMode") or "").upper() != "CLASSIC":
                    continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_collector_game_mode_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_collector.py scripts/background_collector.py tests/test_collector_game_mode_guard.py
git commit -m "fix(collector): enforce strict CLASSIC mode and queue 420 in collectors"
```

---

### Task 3: Optimize `backfill_match_participants` and Protect Against Infinite Loops

**Files:**
- Modify: `core/database.py:1099-1165`
- Create: `tests/test_backfill_loop_guard.py`

**Interfaces:**
- Consumes: `matches.raw_json`
- Produces: `backfill_match_participants(limit, db_path, order_by_recency)` returning count of processed matches, inserting tombstone record for empty/corrupt matches to guarantee termination.

- [ ] **Step 1: Write test for loop guard and performance query**

Create `tests/test_backfill_loop_guard.py`:
```python
import sqlite3
from core import database

def test_backfill_handles_corrupt_or_empty_match_without_loop(tmp_path):
    db_path = str(tmp_path / "test_corrupt.db")
    database.init_db(db_path)

    # Insert a corrupt match with invalid JSON
    with database.get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("CORRUPT_1", "INVALID_BASE64_OR_NOT_JSON", "CLASSIC"),
        )
        conn.execute(
            "INSERT INTO matches (match_id, raw_json, game_mode) VALUES (?, ?, ?)",
            ("EMPTY_1", "", "CLASSIC"),
        )

    # Run backfill: should process 2 matches and not loop
    count1 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count1 == 2

    # Second run must find 0 unbackfilled matches
    count2 = database.backfill_match_participants(limit=10, db_path=db_path)
    assert count2 == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `& .venv312\Scripts\pytest.exe tests/test_backfill_loop_guard.py -v`
Expected: FAIL (second run returns 2 instead of 0 because corrupt matches remain unbackfilled).

- [ ] **Step 3: Implement optimized query and tombstone handling in `core/database.py`**

In `core/database.py`:
```python
def backfill_match_participants(limit: int = 500, db_path: str = DB_PATH, order_by_recency: bool = False) -> int:
    """
    Backfill match_participants from compressed raw_json in matches table.
    Uses WHERE NOT EXISTS for optimal query plan.
    Handles corrupt/empty records gracefully to avoid infinite loops.
    """
    order_clause = "ORDER BY m.game_creation_time DESC" if order_by_recency else ""
    query = f"""
        SELECT m.match_id, m.raw_json, m.game_mode, m.game_creation_time, m.game_duration
        FROM matches m
        WHERE NOT EXISTS (
            SELECT 1 FROM match_participants mp WHERE mp.match_id = m.match_id
        )
        {order_clause}
        LIMIT ?
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        if not rows:
            return 0

        p_rows = []
        for r in rows:
            raw = _decode_compressed_json(r["raw_json"])
            participants = raw.get("info", {}).get("participants", []) if raw else []
            
            if not participants:
                # Insert a sentinel tombstone row so this match is marked as processed and never re-queried
                p_rows.append(
                    (
                        r["match_id"],
                        "__TOMBSTONE__",
                        "Unknown",
                        "UNKNOWN",
                        0, 0, 0, 0, 0, 0,
                        r["game_creation_time"] or 0,
                        r["game_duration"] or 0,
                        r["game_mode"] or "UNKNOWN",
                        "[]",
                    )
                )
                continue

            info = raw.get("info", {})
            for p in participants:
                p_puuid = p.get("puuid") or f"__UNKNOWN_{r['match_id']}_{p.get('participantId', 0)}"
                items = [p.get(f"item{i}", 0) for i in range(7)]
                p_rows.append(
                    (
                        r["match_id"],
                        p_puuid,
                        p.get("championName", "Unknown"),
                        p.get("teamPosition") or p.get("individualPosition", "UNKNOWN"),
                        1 if p.get("win") else 0,
                        p.get("kills", 0),
                        p.get("deaths", 0),
                        p.get("assists", 0),
                        p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                        p.get("goldEarned", 0),
                        r["game_creation_time"] or info.get("gameCreation"),
                        r["game_duration"] or info.get("gameDuration"),
                        r["game_mode"] or info.get("gameMode", "CLASSIC"),
                        json.dumps(items),
                    )
                )

        if p_rows:
            try:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO match_participants
                        (match_id, puuid, champion_name, team_position, win, kills, deaths, assists,
                         total_cs, gold_earned, game_creation_time, game_duration, game_mode, items_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    p_rows,
                )
            except sqlite3.OperationalError:
                pass
        return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv312\Scripts\pytest.exe tests/test_backfill_loop_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/database.py tests/test_backfill_loop_guard.py
git commit -m "perf(db): optimize backfill query with WHERE NOT EXISTS and add tombstone guard"
```

---

### Task 4: Complete the Remaining `match_participants` Backfill on Production

**Files:**
- Run: `scripts/backfill_participants.py`
- Verify: `training_data.db`

**Interfaces:**
- Consumes: `training_data.db`
- Produces: 100% of matches in `matches` table covered in `match_participants`.

- [ ] **Step 1: Execute backfill on production database**

Run: `& .venv312\Scripts\python.exe scripts/backfill_participants.py --batch-size 2000`
Expected: Logs show ~17,400 matches backfilled until loop terminates with 0 remaining.

- [ ] **Step 2: Verify match coverage in `training_data.db`**

Run:
```powershell
& .venv312\Scripts\python.exe -c "import sqlite3; conn=sqlite3.connect('file:training_data.db?mode=ro', uri=True); cur=conn.cursor(); m=cur.execute('SELECT count(*) FROM matches').fetchone()[0]; mp=cur.execute('SELECT count(DISTINCT match_id) FROM match_participants').fetchone()[0]; print(f'Matches: {m:,}, Backfilled: {mp:,}, Remaining: {m-mp}'); conn.close()"
```
Expected: `Remaining: 0`.

- [ ] **Step 3: Commit log or operational record if required**

---

### Task 5: Synchronize `analytics.db` with Production Matches

**Files:**
- Run: `analytics` CLI (`python -m analytics update`)
- Verify: `analytics.db`

**Interfaces:**
- Consumes: `training_data.db` (source matches), `analytics.db` (destination mart)
- Produces: Updated `analytics.db` containing ~117k matches with `journal_mode=wal`.

- [ ] **Step 1: Execute analytics update**

Run: `& .venv312\Scripts\python.exe -m analytics update`
Expected: JSON report with `"quality_gate_passed": true` and updated match count matching `training_data.db`.

- [ ] **Step 2: Verify analytics journal mode is WAL**

Run:
```powershell
& .venv312\Scripts\python.exe -c "import sqlite3; conn=sqlite3.connect('analytics.db'); cur=conn.cursor(); mode=cur.execute('PRAGMA journal_mode;').fetchone()[0]; print('Analytics Journal Mode:', mode); conn.close()"
```
Expected: `Analytics Journal Mode: wal`.

---

### Task 6: Repository Hygiene & Complete Regression Suite

**Files:**
- Remove: Stray test database files in root (`test.db`, `test_old.db`, `test_wal_check2.db`)
- Test: Full test suite (`pytest -v`)

- [ ] **Step 1: Remove stray root test files**

Run: `Remove-Item -Force test.db, test_old.db, test_wal_check2.db -ErrorAction SilentlyContinue`

- [ ] **Step 2: Run complete test suite**

Run: `& .venv312\Scripts\pytest.exe -q`
Expected: 290+ passed, 0 failed.

- [ ] **Step 3: Verify git status is clean**

Run: `git status`
Expected: Clean working tree.
