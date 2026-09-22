# API Production Hardening and Veracity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate fake data, prevent server-freezing authentication loops, bound background task concurrency, prevent dataset corruption, and unify database telemetry across the NNriot API.

**Architecture:** 
1. `core/riot_api.py`: Convert to `requests.Session` with fast-failing `RiotAuthError` on 401/403, structured logging, and modern Spectator endpoints.
2. `app/routes/history.py` + `core/database.py`: Replace synthetic/random match generator in `/api/player/history` with real SQLite match extraction and graceful empty states.
3. `core/task_runner.py`: Implement a bounded, thread-safe background job executor with unique `job_id` tracking to replace unmanaged `threading.Thread(daemon=True)`.
4. `app/routes/api.py`: Abort the fake training simulation that corrupts dataset status (`mark_as_trained`), secure file writing in `/api/stream/process_frame`, and unify database telemetry via `SystemService`.

**Tech Stack:** Python 3.12, Flask, SQLite3, requests.Session, concurrent.futures, pytest.

## Global Constraints

- Never return fabricated / randomized match statistics for players under the guise of real API data.
- Never block server worker threads indefinitely with `while True: time.sleep(...)` on expired API credentials.
- Never spawn unbounded raw threads (`threading.Thread(daemon=True)`) directly from HTTP route handlers.
- Never mark records as trained in SQLite unless an ML model has genuinely computed gradients on them.
- Preserve backward compatibility for existing UI routes and responses while upgrading backend resilience.

---

### Task 1: Riot API Client Hardening & Fail-Fast Authentication

**Files:**
- Modify: `core/riot_api.py:1-210`
- Test: `tests/test_riot_api_hardening.py`

**Interfaces:**
- Consumes: `os.getenv("RIOT_API_KEY")`, `requests.Session`
- Produces: `class RiotAuthError(Exception)`, `class RiotAPI` with connection pooling, non-blocking auth error handling, and structured logging.

- [ ] **Step 1: Write the failing test for RiotAuthError and non-blocking key expiration**

Create `tests/test_riot_api_hardening.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from core.riot_api import RiotAPI, RiotAuthError


def test_riot_api_raises_auth_error_on_401():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-expired", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="EUW")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {}

        with patch.object(api.session, "get", return_value=mock_resp):
            with patch.object(api, "reload_key", return_value=None):
                with pytest.raises(RiotAuthError) as exc_info:
                    api._make_request("https://europe.api.riotgames.com/test")
                assert "401" in str(exc_info.value) or "unauthorized" in str(exc_info.value).lower()


def test_riot_api_uses_persistent_session():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="KR")
        assert hasattr(api, "session")
        assert isinstance(api.session, object)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest tests/test_riot_api_hardening.py -v`
Expected: FAIL with `ImportError: cannot import name 'RiotAuthError' from 'core.riot_api'`

- [ ] **Step 3: Implement RiotAuthError, Session reuse, and remove infinite loop in `core/riot_api.py`**

In `core/riot_api.py`:
1. Define `RiotAuthError`:
```python
class RiotAuthError(Exception):
    """Raised when Riot API rejects the API key (HTTP 401 or 403)."""
    pass
```
2. Initialize `self.session = requests.Session()` in `RiotAPI.__init__`.
3. In `_make_request`, replace `requests.get` with `self.session.get`.
4. Replace lines 165-175:
```python
                elif response.status_code in (401, 403):
                    logger.warning(
                        "Riot API returned HTTP %s (expired or invalid key). Attempting key reload from .env...",
                        response.status_code
                    )
                    new_key = self.reload_key()
                    if new_key and new_key != headers.get("X-Riot-Token"):
                        logger.info("New RIOT_API_KEY reloaded from .env. Retrying request...")
                        continue
                    raise RiotAuthError(
                        f"Riot API token rejected with HTTP {response.status_code}. "
                        "Please update RIOT_API_KEY in your .env file."
                    )
```
5. Replace `print(...)` with `logger.info(...)` and `logger.warning(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest tests/test_riot_api_hardening.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/riot_api.py tests/test_riot_api_hardening.py
git commit -m "fix(riot_api): add RiotAuthError, persistent requests.Session, and eliminate infinite sleep loop on 401/403"
```

---

### Task 2: Real Player History & Elimination of Fake Data in `/api/player/history`

**Files:**
- Modify: `core/database.py:750-810`
- Modify: `app/routes/history.py:215-318`
- Test: `tests/test_player_history_real.py`

**Interfaces:**
- Consumes: `database.get_player_by_name_tag`, `database.get_player_match_history`, `core.riot_api.RiotAuthError`
- Produces: `GET /api/player/history` returning 100% genuine database matches and stats with zero random/fake mock generation.

- [ ] **Step 1: Write the failing test for real player history retrieval**

Create `tests/test_player_history_real.py`:
```python
import json
import pytest
from unittest.mock import patch, MagicMock
from app import app as flask_app
from core import database


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_player_history_returns_real_db_data(client, tmp_path):
    test_db = str(tmp_path / "test_history.db")
    database.init_db(test_db)
    
    # Insert a real player
    puuid = "test-puuid-123"
    database.save_player(puuid, "RealPlayer", "EUW", db_path=test_db)
    
    # Insert a real match with participant info
    match_data = {
        "metadata": {"matchId": "EUW1_9999999999"},
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1700000000000,
            "gameDuration": 1800,
            "participants": [
                {
                    "puuid": puuid,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "kills": 8,
                    "deaths": 2,
                    "assists": 11,
                    "totalMinionsKilled": 210,
                    "goldEarned": 14500,
                    "item0": 3089, "item1": 3020, "item2": 0, "item3": 0, "item4": 0, "item5": 0, "item6": 3364,
                }
            ]
        }
    }
    database.save_match("EUW1_9999999999", match_data, db_path=test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        resp = client.get("/api/player/history?name=RealPlayer&tag=EUW&count=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["player"]["name"] == "RealPlayer"
        assert len(data["matches"]) == 1
        m = data["matches"][0]
        assert m["champion"] == "Ahri"
        assert m["kills"] == 8
        assert m["deaths"] == 2
        assert m["assists"] == 11
        assert m["win"] is True
        assert m["match_id"] == "EUW1_9999999999"


def test_player_history_no_matches_returns_empty_not_fake(client, tmp_path):
    test_db = str(tmp_path / "test_empty.db")
    database.init_db(test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        resp = client.get("/api/player/history?name=UnknownGuy&tag=NA1&count=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["matches"] == []
        assert data["player"]["total_games"] == 0
        # Must not fabricate Challenger LP
        assert data["player"].get("is_pro") is False or data["player"].get("is_pro") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest tests/test_player_history_real.py -v`
Expected: FAIL (because current `/api/player/history` returns fake randomized matches with 1240 LP Challenger).

- [ ] **Step 3: Implement `get_player_match_history` in `core/database.py` and refactor `/api/player/history`**

In `core/database.py`:
```python
def get_player_match_history(puuid: str, limit: int = 10, db_path: str = DB_PATH) -> list[dict]:
    """Retrieve actual match records for a player by inspecting participant data in stored matches."""
    matches = []
    with get_connection(db_path) as conn:
        # Search for matches containing the puuid
        rows = conn.execute(
            """
            SELECT match_id, raw_json, game_mode, game_creation_time, game_duration
            FROM matches
            WHERE raw_json LIKE ?
            ORDER BY game_creation_time DESC
            LIMIT ?
            """,
            (f"%{puuid}%", limit * 2),
        ).fetchall()

    for r in rows:
        raw = _decode_compressed_json(r["raw_json"])
        if not raw:
            continue
        info = raw.get("info", {})
        participants = info.get("participants", [])
        p_data = next((p for p in participants if p.get("puuid") == puuid), None)
        if not p_data:
            continue

        dur_secs = r["game_duration"] or info.get("gameDuration", 0)
        dur_mins = max(1, dur_secs // 60)
        kills = p_data.get("kills", 0)
        deaths = p_data.get("deaths", 0)
        assists = p_data.get("assists", 0)
        kda = round((kills + assists) / max(1, deaths), 2)
        cs = p_data.get("totalMinionsKilled", 0) + p_data.get("neutralMinionsKilled", 0)
        items = [p_data.get(f"item{i}", 0) for i in range(7)]

        matches.append({
            "match_id": r["match_id"],
            "champion": p_data.get("championName", "Unknown"),
            "role": p_data.get("teamPosition") or p_data.get("individualPosition", "UNKNOWN"),
            "win": bool(p_data.get("win", False)),
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "kda": str(kda),
            "cs": cs,
            "cs_per_min": f"{(cs / dur_mins):.1f}",
            "gold": p_data.get("goldEarned", 0),
            "duration": f"{dur_mins}:{dur_secs % 60:02d}",
            "game_mode": r["game_mode"] or info.get("gameMode", "CLASSIC"),
            "items": items,
        })
        if len(matches) >= limit:
            break

    return matches
```

In `app/routes/history.py`:
Delete lines 253-318 (the `random.seed` fake match block).
Replace with genuine lookup:
```python
    # 2. Query real player and match history from DB
    db_path = current_app.config.get("NNRIOT_DB_PATH", database.DB_PATH)
    player_rows = database.search_players(name, limit=5, db_path=db_path)
    target_player = next((p for p in player_rows if p.get("tag_line", "").upper() == tag.upper()), None)
    
    puuid = target_player["puuid"] if target_player else None
    if not puuid:
        target_player = next((p for p in player_rows if p.get("game_name", "").lower() == name.lower()), None)
        puuid = target_player["puuid"] if target_player else None

    matches = []
    stats = None
    if puuid:
        matches = database.get_player_match_history(puuid, limit=count, db_path=db_path)
        stats = database.get_player_stats(puuid, db_path=db_path)

    wins_count = sum(1 for m in matches if m["win"])
    total_games = len(matches) if not stats else stats.get("total_matches", len(matches))
    winrate = round((wins_count / max(1, len(matches))) * 100, 1) if matches else 0.0
    avg_kda = str(round(stats.get("avg_kda", 0.0), 2)) if stats else ("0.0" if not matches else str(round(sum(float(m["kda"]) for m in matches) / len(matches), 2)))

    return jsonify({
        "success": True,
        "player": {
            "name": name,
            "tag": tag,
            "puuid": puuid,
            "winrate": winrate,
            "total_games": total_games,
            "avg_kda": avg_kda,
            "has_data": len(matches) > 0,
        },
        "matches": matches,
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest tests/test_player_history_real.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/database.py app/routes/history.py tests/test_player_history_real.py
git commit -m "fix(history): eliminate fake randomized player history and return authentic database matches"
```

---

### Task 3: Bounded Background Task Runner to Replace Daemon Threads

**Files:**
- Create: `core/task_runner.py`
- Modify: `app/routes/history.py:1-180`
- Modify: `app/routes/api.py:277-325, 405-416`
- Test: `tests/test_task_runner.py`

**Interfaces:**
- Consumes: `concurrent.futures.ThreadPoolExecutor`
- Produces: `task_runner.submit_task(task_type, fn, *args, **kwargs) -> str (job_id)`, `task_runner.get_status(job_id) -> dict`, `GET /api/jobs/<job_id>`.

- [ ] **Step 1: Write the failing test for task runner**

Create `tests/test_task_runner.py`:
```python
import time
import pytest
from core.task_runner import TaskRunner


def test_task_runner_executes_and_tracks_job():
    runner = TaskRunner(max_workers=2)

    def simple_job(x, y):
        time.sleep(0.05)
        return x + y

    job_id = runner.submit_task("test_add", simple_job, 10, 20)
    assert job_id is not None
    status = runner.get_status(job_id)
    assert status["status"] in ("queued", "running", "completed")

    time.sleep(0.1)
    status = runner.get_status(job_id)
    assert status["status"] == "completed"
    assert status["result"] == 30
    assert status["error"] is None
    runner.shutdown()


def test_task_runner_records_failure():
    runner = TaskRunner(max_workers=2)

    def failing_job():
        raise ValueError("Boom")

    job_id = runner.submit_task("fail_job", failing_job)
    time.sleep(0.1)
    status = runner.get_status(job_id)
    assert status["status"] == "failed"
    assert "Boom" in status["error"]
    runner.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest tests/test_task_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.task_runner'`

- [ ] **Step 3: Implement `core/task_runner.py` and register `/api/jobs/<job_id>`**

Create `core/task_runner.py`:
```python
import uuid
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskRunner:
    """Bounded, thread-safe background task executor with UUID tracking."""
    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="NNTask")
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit_task(self, task_type: str, fn: Callable, *args, **kwargs) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job_record = {
            "job_id": job_id,
            "type": task_type,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
        }
        with self.lock:
            self.jobs[job_id] = job_record

        def _wrapper():
            with self.lock:
                self.jobs[job_id]["status"] = "running"
                self.jobs[job_id]["started_at"] = time.time()
            try:
                result = fn(*args, **kwargs)
                with self.lock:
                    self.jobs[job_id]["status"] = "completed"
                    self.jobs[job_id]["completed_at"] = time.time()
                    self.jobs[job_id]["result"] = result
            except Exception as exc:
                logger.error("Task %s (%s) failed: %s", job_id, task_type, exc, exc_info=True)
                with self.lock:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["completed_at"] = time.time()
                    self.jobs[job_id]["error"] = str(exc)

        self.executor.submit(_wrapper)
        return job_id

    def get_status(self, job_id: str) -> Optional[dict]:
        with self.lock:
            rec = self.jobs.get(job_id)
            return dict(rec) if rec else None

    def get_latest_by_type(self, task_type: str) -> Optional[dict]:
        with self.lock:
            matching = [j for j in self.jobs.values() if j["type"] == task_type]
            if not matching:
                return None
            return dict(sorted(matching, key=lambda x: x["created_at"])[-1])

    def shutdown(self, wait: bool = False):
        self.executor.shutdown(wait=wait)


global_task_runner = TaskRunner(max_workers=3)
```

In `app/routes/history.py` and `app/routes/api.py`:
Replace `threading.Thread(target=..., daemon=True).start()` with `global_task_runner.submit_task(...)`.
Add endpoint in `app/routes/api.py`:
```python
@api_bp.route("/api/jobs/<job_id>", methods=["GET"])
def api_get_job_status(job_id):
    from core.task_runner import global_task_runner
    status = global_task_runner.get_status(job_id)
    if not status:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, "job": status}), 200
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest tests/test_task_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/task_runner.py app/routes/history.py app/routes/api.py tests/test_task_runner.py
git commit -m "feat(tasks): replace unbounded daemon threads with bounded TaskRunner and /api/jobs endpoint"
```

---

### Task 4: Fix Data Corruption in Manual Train & Secure Stream OCR

**Files:**
- Modify: `app/routes/api.py:150-275`
- Test: `tests/test_api_train_and_stream.py`

**Interfaces:**
- Consumes: `core.tf_available`, `core.global_trainer`
- Produces: Fail-fast HTTP 503 on unready trainer in `/api/train/manual` (no fake database updates), and thread-safe unique filenames in `/api/stream/process_frame`.

- [ ] **Step 1: Write the failing test for manual train safety and OCR race condition**

Create `tests/test_api_train_and_stream.py`:
```python
import os
import pytest
from unittest.mock import patch
from app import app as flask_app
from app import core as app_core
from core import database


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_manual_train_fails_fast_without_marking_records(client, tmp_path):
    test_db = str(tmp_path / "test_train.db")
    database.init_db(test_db)
    
    database.save_training_record("MATCH_1", {"f": 1}, 0, db_path=test_db)

    with patch.dict("os.environ", {"NNRIOT_DB_PATH": test_db}):
        with patch.object(app_core, "tf_available", False):
            with patch.object(app_core, "global_trainer", None):
                resp = client.post("/api/train/manual")
                assert resp.status_code == 503
                data = resp.get_json()
                assert "unavailable" in data["error"].lower() or "not loaded" in data["error"].lower()

                untrained = database.get_untrained_records(limit=10, db_path=test_db)
                assert len(untrained) == 1
                assert untrained[0]["match_id"] == "MATCH_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest tests/test_api_train_and_stream.py -v`
Expected: FAIL (returns 200 with simulation: "Processed 1 new records" and marks record as trained!).

- [ ] **Step 3: Remove destructive simulation from `/api/train/manual` and secure `/api/stream/process_frame`**

In `app/routes/api.py`:
1. Modify `api_train_manual` (lines 242-262):
```python
@api_bp.route("/api/train/manual", methods=["POST"])
def api_train_manual():
    """API endpoint to trigger a single training step manually."""
    try:
        if not core.tf_available or core.global_trainer is None:
            return jsonify({
                "success": False,
                "error": "TensorFlow model is not loaded or available. Real training step cannot run.",
            }), 503

        records_processed = core.global_trainer.run_training_step()

        return jsonify(
            {
                "success": True,
                "records_processed": records_processed,
                "message": f"Retraining step complete. Processed {records_processed} new records.",
            }
        )
    except Exception as e:
        logger.error(f"Manual train error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
```
2. In `api_stream_process_frame` (lines 175-186), use unique filenames:
```python
        import uuid
        frame_id = uuid.uuid4().hex[:8]
        filename = f"capture_{frame_id}.jpg"
        filepath = os.path.join(screenshot_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest tests/test_api_train_and_stream.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/routes/api.py tests/test_api_train_and_stream.py
git commit -m "fix(api): eliminate destructive fake training fallback and prevent OCR file collision"
```

---

### Task 5: Database Telemetry Unification & Standardized REST Responses

**Files:**
- Modify: `app/services/system_service.py`
- Modify: `app/routes/api.py:61-70, 615-703`
- Modify: `app/routes/history.py:187-214`
- Test: `tests/test_api_telemetry_and_errors.py`

**Interfaces:**
- Consumes: `SystemService.get_full_telemetry()`
- Produces: Unified telemetry across `/api/db-health`, `/api/db/stats`, and `/api/monitor/metrics` without redundant inline SQL.

- [ ] **Step 1: Write the failing test for telemetry consistency**

Create `tests/test_api_telemetry_and_errors.py`:
```python
import pytest
from app import app as flask_app
from core import database


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "telemetry.db")
    database.init_db(test_db)
    database.save_player("p1", "TestPlayer", "EUW", db_path=test_db)
    flask_app.config["TESTING"] = True
    flask_app.config["NNRIOT_DB_PATH"] = test_db
    return flask_app.test_client()


def test_telemetry_endpoints_are_consistent(client):
    r_health = client.get("/api/db-health")
    r_stats = client.get("/api/db/stats")
    r_metrics = client.get("/api/monitor/metrics")

    assert r_health.status_code == 200
    assert r_stats.status_code == 200
    assert r_metrics.status_code == 200

    d_health = r_health.get_json()
    d_stats = r_stats.get_json()
    d_metrics = r_metrics.get_json()

    assert d_health["players"] == 1
    assert d_stats["total_players"] == 1
    assert d_metrics["players"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\pytest tests/test_api_telemetry_and_errors.py -v`
Expected: May pass or fail depending on path routing, but demonstrates schema discrepancies.

- [ ] **Step 3: Enhance `SystemService` and wire up routes**

In `app/services/system_service.py`:
```python
    def get_full_telemetry(self, db_path: str = None) -> dict:
        target_path = db_path or getattr(self.repo, 'db_path', database.DB_PATH)
        stats = database.get_db_stats(target_path)
        crawl_stats = database.get_crawl_queue_stats(target_path)
        
        size_mb = 0.0
        if target_path and os.path.exists(target_path):
            size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)

        return {
            "players": stats.get("players", 0),
            "matches": stats.get("matches", 0),
            "training_records": stats.get("training_records", 0),
            "trained_records": stats.get("trained_records", 0),
            "untrained_records": stats.get("untrained_records", 0),
            "trained_ratio": stats.get("trained_ratio", 0.0),
            "crawl_queue_pending": crawl_stats.get("pending", 0),
            "db_size_mb": size_mb,
            "schema_version": stats.get("schema_version", 0),
            "success": True,
            "ok": True,
        }
```

Refactor `/api/db-health`, `/api/db/stats`, and `/api/monitor/metrics` to delegate to `_system_service.get_full_telemetry()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\pytest tests/test_api_telemetry_and_errors.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to guarantee zero regression**

Run: `.\.venv312\Scripts\pytest tests/test_final_web_app.py tests/test_player_history_real.py tests/test_riot_api_hardening.py tests/test_task_runner.py tests/test_api_train_and_stream.py tests/test_api_telemetry_and_errors.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/system_service.py app/routes/api.py app/routes/history.py tests/test_api_telemetry_and_errors.py
git commit -m "refactor(api): unify telemetry endpoints through SystemService and standardize responses"
```
