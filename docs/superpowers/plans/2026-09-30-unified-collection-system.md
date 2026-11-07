# Unified Collection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify and modernize the League of Legends match data collection system in NNriot into a clean, thread-safe service package in `core/collection/` that eliminates post-game data leakage, unifies SQLite crawl queuing on PUUID and region, systematically ingests timelines and 32D spatial embeddings, cleans up Flask routes, and maintains 100% backward compatibility with existing tests and scripts.

**Architecture:** A domain service package `core/collection/` structured with single-responsibility components: `regions.py` (region resolution), `models.py` (typed dataclasses), `queue.py` (`CrawlQueueRepository` over SQLite v14 migration), `tracker.py` (`CollectionTracker` thread-safe status), `ingestor.py` (`MatchIngestor` leak-free parsing, timeline ingestion, spatial embeddings, repo persistence), and `service.py` (`CollectionService` facade). Flask routes and CLI scripts delegate exclusively to `CollectionService`.

**Tech Stack:** Python 3.12, SQLite 3 (WAL mode), Requests, Scipy/Numpy, Pytest.

## Global Constraints

- **Python Version:** Python 3.12 (`.\.venv312\Scripts\python.exe`).
- **Domain Boundaries:** No Flask route (`app/routes/`) may import from `scripts/`. Domain collection logic belongs strictly in `core/collection/`.
- **Zero Post-Game Leakage:** Training records must be generated using `json_utils.extract_pre_match_features` and `feature_labels.extract_labels` (never `extract_match_features` with final KDA/gold).
- **Systematic Timelines:** Every collected match details batch must fetch timelines via `RiotAPI.get_match_timelines_batch` and extract 32D spatial embeddings via `spatial_rnn.extract_and_store_timeline_embeddings`, with graceful handling if a timeline is missing from Riot.
- **Backward Compatibility:** Existing functions in `scripts/data_collector.py` (`collect_training_data`, `collect_top_leagues_data`, `collect_by_puuid`, `collect_batch_with_smurfs`, `resolve_region`, `_build_training_record`, `batch_status`, `batch_lock`) must remain exported as shims to ensure `tests/test_collection_database_selection.py` continues to pass.

---

### Task 1: Region Resolution & Core Data Models

**Files:**
- Create: `core/collection/__init__.py`
- Create: `core/collection/regions.py`
- Create: `core/collection/models.py`
- Test: `tests/core/test_collection_regions_models.py`

**Interfaces:**
- Consumes: None (pure domain utilities and dataclasses).
- Produces:
  - `core.collection.regions.resolve_region(tag: str) -> str`
  - `core.collection.regions.TAG_TO_REGION: dict[str, str]`
  - `core.collection.models.CrawlItem`
  - `core.collection.models.IngestionResult`
  - `core.collection.models.MatchQualityFilter`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_collection_regions_models.py`:
```python
import pytest
from core.collection.regions import resolve_region, TAG_TO_REGION
from core.collection.models import CrawlItem, IngestionResult, MatchQualityFilter


def test_resolve_region():
    assert resolve_region("KR1") == "KR"
    assert resolve_region("kr") == "KR"
    assert resolve_region("EUW1") == "EUW"
    assert resolve_region("NA1") == "NA"
    assert resolve_region("UNKNOWN_TAG") == "EUW"
    assert "BR" in TAG_TO_REGION


def test_crawl_item_dataclass():
    item = CrawlItem(puuid="puuid-123", region="KR", priority=5)
    assert item.puuid == "puuid-123"
    assert item.region == "KR"
    assert item.priority == 5
    assert item.is_processed == 0


def test_ingestion_result_dataclass():
    res = IngestionResult(matches_saved=10, records_saved=10, timelines_saved=10)
    assert res.matches_saved == 10
    assert res.skipped == 0
    assert res.errors == 0


def test_match_quality_filter_defaults():
    qf = MatchQualityFilter()
    assert "CLASSIC" in qf.allowed_game_modes
    assert 420 in qf.allowed_queue_ids
    assert qf.min_duration_seconds == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_regions_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.collection'`

- [ ] **Step 3: Implement `regions.py`, `models.py`, and `__init__.py`**

Create `core/collection/regions.py`:
```python
from typing import Final

TAG_TO_REGION: Final[dict[str, str]] = {
    "BR1": "BR",
    "BR": "BR",
    "EUN1": "EUN",
    "EUN": "EUN",
    "EUW1": "EUW",
    "EUW": "EUW",
    "JP1": "JP",
    "JP": "JP",
    "KR1": "KR",
    "KR": "KR",
    "KR2": "KR",
    "LA1": "LA1",
    "LA2": "LA2",
    "NA1": "NA",
    "NA": "NA",
    "OC1": "OCE",
    "OCE": "OCE",
    "RU": "RU",
    "TR1": "TR",
    "TR": "TR",
}


def resolve_region(tag: str) -> str:
    """Convert a Riot tag or platform suffix to the canonical region code.

    Falls back to EUW if the tag is unrecognised.
    """
    if not tag:
        return "EUW"
    return TAG_TO_REGION.get(tag.upper(), "EUW")
```

Create `core/collection/models.py`:
```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrawlItem:
    region: str
    puuid: Optional[str] = None
    game_name: Optional[str] = None
    tag_line: Optional[str] = None
    priority: int = 0
    is_processed: int = 0
    id: Optional[int] = None


@dataclass
class IngestionResult:
    matches_saved: int = 0
    records_saved: int = 0
    timelines_saved: int = 0
    embeddings_computed: int = 0
    skipped: int = 0
    errors: int = 0
    discovered_puuids: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class MatchQualityFilter:
    allowed_game_modes: frozenset[str] = frozenset({"CLASSIC"})
    allowed_queue_ids: frozenset[int] = frozenset({420})
    min_duration_seconds: int = 300
```

Create `core/collection/__init__.py`:
```python
from core.collection.regions import resolve_region, TAG_TO_REGION
from core.collection.models import CrawlItem, IngestionResult, MatchQualityFilter

__all__ = [
    "resolve_region",
    "TAG_TO_REGION",
    "CrawlItem",
    "IngestionResult",
    "MatchQualityFilter",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_regions_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/collection/ regions.py models.py __init__.py tests/core/test_collection_regions_models.py
git commit -m "feat(collection): add region resolver and core data models"
```

---

### Task 2: SQLite Migration v14 & CrawlQueueRepository

**Files:**
- Modify: `core/database.py`
- Create: `core/collection/queue.py`
- Test: `tests/core/test_crawl_queue_repository.py`

**Interfaces:**
- Consumes: `core.database.get_connection`, `core.collection.models.CrawlItem`
- Produces:
  - `core.collection.queue.CrawlQueueRepository`:
    - `enqueue_puuids(puuids: list[str], region: str, priority: int = 0) -> int`
    - `enqueue_riot_id(game_name: str, tag_line: str, region: str, priority: int = 10, puuid: str | None = None) -> bool`
    - `pop_next(limit: int = 1) -> list[CrawlItem]`
    - `mark_processed(puuid: str, region: str | None = None) -> None`
    - `get_stats() -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_crawl_queue_repository.py`:
```python
import sqlite3
import pytest
from core import database
from core.collection.queue import CrawlQueueRepository
from core.collection.models import CrawlItem


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_queue.db")
    database.init_db(db_file)
    return CrawlQueueRepository(db_path=db_file)


def test_enqueue_and_pop_puuids(repo):
    added = repo.enqueue_puuids(["p1", "p2", "p3"], region="KR", priority=5)
    assert added == 3

    # Duplicate enqueue should not duplicate
    added_again = repo.enqueue_puuids(["p1"], region="KR", priority=1)
    assert added_again == 0

    items = repo.pop_next(limit=2)
    assert len(items) == 2
    assert items[0].puuid in ("p1", "p2", "p3")
    assert items[0].region == "KR"
    assert items[0].priority == 5


def test_mark_processed_and_stats(repo):
    repo.enqueue_puuids(["p_proc"], region="EUW", priority=10)
    stats_before = repo.get_stats()
    assert stats_before["total"] == 1
    assert stats_before["pending"] == 1
    assert stats_before["processed"] == 0

    repo.mark_processed("p_proc")
    stats_after = repo.get_stats()
    assert stats_after["total"] == 1
    assert stats_after["pending"] == 0
    assert stats_after["processed"] == 1

    # Pop next should now be empty
    assert repo.pop_next(limit=1) == []


def test_enqueue_riot_id(repo):
    repo.enqueue_riot_id("Faker", "KR1", region="KR", priority=10, puuid="p_faker")
    items = repo.pop_next(limit=1)
    assert len(items) == 1
    assert items[0].puuid == "p_faker"
    assert items[0].game_name == "Faker"
    assert items[0].tag_line == "KR1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_crawl_queue_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.collection.queue'`

- [ ] **Step 3: Implement Migration 14 in `core/database.py` and `CrawlQueueRepository` in `core/collection/queue.py`**

In `core/database.py`:
Update `SCHEMA_VERSION = 14`.
Add to `_MIGRATIONS`:
```python
    # v13 → v14 : migrate crawl_queue to support PUUID and region indexing
    (
        13,
        [
            """
            CREATE TABLE IF NOT EXISTS crawl_queue_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puuid TEXT UNIQUE,
                region TEXT NOT NULL,
                game_name TEXT,
                tag_line TEXT,
                priority INTEGER DEFAULT 0,
                is_processed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_v2_status_prio 
                ON crawl_queue_v2(is_processed, priority DESC, id ASC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_crawl_v2_puuid 
                ON crawl_queue_v2(puuid)
            """,
            """
            INSERT OR IGNORE INTO crawl_queue_v2 (game_name, tag_line, region, priority, is_processed, created_at)
            SELECT game_name, tag_line, 'EUW', priority, is_processed, created_at FROM crawl_queue
            """,
            "DROP TABLE IF EXISTS crawl_queue",
            "ALTER TABLE crawl_queue_v2 RENAME TO crawl_queue",
        ],
    ),
```

Create `core/collection/queue.py`:
```python
import logging
from typing import Optional
from core import database, config
from core.collection.models import CrawlItem

logger = logging.getLogger(__name__)


class CrawlQueueRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = config.get_db_path(db_path, require_exists=False)

    def enqueue_puuids(self, puuids: list[str], region: str, priority: int = 0) -> int:
        """Insert multiple PUUIDs into the crawl queue.

        Uses ON CONFLICT(puuid) DO NOTHING to prevent duplicate queue items.
        """
        valid_items = [(p.strip(), region.upper(), priority) for p in puuids if p and p.strip()]
        if not valid_items:
            return 0

        with database.get_connection(self.db_path) as conn:
            cursor = conn.executemany(
                """
                INSERT INTO crawl_queue (puuid, region, priority)
                VALUES (?, ?, ?)
                ON CONFLICT(puuid) DO NOTHING
                """,
                valid_items,
            )
            return cursor.rowcount

    def enqueue_riot_id(
        self,
        game_name: str,
        tag_line: str,
        region: str,
        priority: int = 10,
        puuid: Optional[str] = None,
    ) -> bool:
        """Insert or update a player identified by Riot ID and/or PUUID."""
        with database.get_connection(self.db_path) as conn:
            if puuid:
                conn.execute(
                    """
                    INSERT INTO crawl_queue (puuid, region, game_name, tag_line, priority)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(puuid) DO UPDATE SET
                        priority = MAX(crawl_queue.priority, excluded.priority),
                        game_name = COALESCE(excluded.game_name, crawl_queue.game_name),
                        tag_line = COALESCE(excluded.tag_line, crawl_queue.tag_line)
                    """,
                    (puuid, region.upper(), game_name, tag_line, priority),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO crawl_queue (region, game_name, tag_line, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    (region.upper(), game_name, tag_line, priority),
                )
            return True

    def pop_next(self, limit: int = 1) -> list[CrawlItem]:
        """Fetch the next pending players to crawl ordered by priority DESC, id ASC."""
        with database.get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, puuid, region, game_name, tag_line, priority, is_processed
                FROM crawl_queue
                WHERE is_processed = 0
                ORDER BY priority DESC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items = []
        for r in rows:
            items.append(
                CrawlItem(
                    id=r["id"],
                    puuid=r["puuid"],
                    region=r["region"],
                    game_name=r["game_name"],
                    tag_line=r["tag_line"],
                    priority=r["priority"],
                    is_processed=r["is_processed"],
                )
            )
        return items

    def mark_processed(self, puuid: str, region: Optional[str] = None) -> None:
        """Mark a crawled player as processed."""
        with database.get_connection(self.db_path) as conn:
            if puuid:
                conn.execute(
                    """
                    UPDATE crawl_queue
                    SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                    WHERE puuid = ?
                    """,
                    (puuid,),
                )
            elif region:
                pass

    def mark_processed_by_id(self, queue_id: int) -> None:
        """Mark a queue item as processed by its primary key ID."""
        with database.get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE crawl_queue
                SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (queue_id,),
            )

    def get_stats(self) -> dict[str, int]:
        """Return counts of total, pending, and processed players in queue."""
        with database.get_connection(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM crawl_queue").fetchone()[0]
            processed = conn.execute(
                "SELECT COUNT(*) FROM crawl_queue WHERE is_processed = 1"
            ).fetchone()[0]
        return {
            "total": total,
            "processed": processed,
            "pending": total - processed,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_crawl_queue_repository.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/database.py core/collection/queue.py tests/core/test_crawl_queue_repository.py
git commit -m "feat(collection): add schema migration 14 and CrawlQueueRepository"
```

---

### Task 3: Thread-Safe Status Tracker (`CollectionTracker`)

**Files:**
- Create: `core/collection/tracker.py`
- Test: `tests/core/test_collection_tracker.py`

**Interfaces:**
- Consumes: None (pure Python threading and structures).
- Produces:
  - `core.collection.tracker.CollectionTracker`:
    - `set_running(total: int, player: str = "")`
    - `set_completed()`
    - `set_error(msg: str)`
    - `set_player(player: str)`
    - `step(increment: int = 1)`
    - `add_log(msg: str)`
    - `to_dict() -> dict`
  - `core.collection.tracker.get_default_tracker() -> CollectionTracker`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_collection_tracker.py`:
```python
import pytest
from core.collection.tracker import CollectionTracker, get_default_tracker


def test_tracker_lifecycle_and_dict():
    tracker = CollectionTracker()
    assert tracker.to_dict()["status"] == "idle"

    tracker.set_running(total=10, player="Player1#KR1")
    state = tracker.to_dict()
    assert state["status"] == "running"
    assert state["total"] == 10
    assert state["current"] == 0
    assert state["player"] == "Player1#KR1"

    tracker.step(2)
    tracker.add_log("Fetched 2 matches")
    state = tracker.to_dict()
    assert state["current"] == 2
    assert len(state["logs"]) == 1
    assert "Fetched 2 matches" in state["logs"][0]

    tracker.set_completed()
    assert tracker.to_dict()["status"] == "completed"


def test_tracker_log_limit():
    tracker = CollectionTracker(max_logs=5)
    for i in range(10):
        tracker.add_log(f"log {i}")
    state = tracker.to_dict()
    assert len(state["logs"]) == 5
    assert state["logs"][-1] == "log 9"


def test_default_tracker_singleton():
    t1 = get_default_tracker()
    t2 = get_default_tracker()
    assert t1 is t2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.collection.tracker'`

- [ ] **Step 3: Implement `CollectionTracker` in `core/collection/tracker.py`**

Create `core/collection/tracker.py`:
```python
import threading
import time
from typing import Optional


class CollectionTracker:
    def __init__(self, max_logs: int = 150):
        self._lock = threading.Lock()
        self.max_logs = max_logs
        self.status = "idle"
        self.total = 0
        self.current = 0
        self.player = ""
        self.logs: list[str] = []
        self.start_time: Optional[float] = None

    def set_running(self, total: int = 0, player: str = "") -> None:
        with self._lock:
            self.status = "running"
            self.total = total
            self.current = 0
            self.player = player
            self.logs = []
            self.start_time = time.time()

    def set_completed(self) -> None:
        with self._lock:
            self.status = "completed"

    def set_error(self, err_msg: str) -> None:
        with self._lock:
            self.status = "error"
            self._append_log_unlocked(f"[ERROR] {err_msg}")

    def set_player(self, player: str) -> None:
        with self._lock:
            self.player = player

    def step(self, increment: int = 1) -> None:
        with self._lock:
            self.current += increment

    def add_log(self, msg: str) -> None:
        with self._lock:
            self._append_log_unlocked(msg)

    def _append_log_unlocked(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

    def to_dict(self) -> dict:
        with self._lock:
            rate = 0.0
            eta_min = 0.0
            if self.start_time and self.current > 0:
                elapsed = max(1.0, time.time() - self.start_time)
                rate = (self.current / elapsed) * 60.0
                remaining = max(0, self.total - self.current)
                eta_min = (remaining / rate) if rate > 0 else 0.0

            return {
                "status": self.status,
                "total": self.total,
                "current": self.current,
                "player": self.player,
                "logs": list(self.logs),
                "speed_matches_per_min": round(rate, 1),
                "eta_minutes": round(eta_min, 1),
            }


_GLOBAL_TRACKER = CollectionTracker()


def get_default_tracker() -> CollectionTracker:
    return _GLOBAL_TRACKER
```

Update `core/collection/__init__.py` to export `CollectionTracker` and `get_default_tracker`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_tracker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/collection/tracker.py core/collection/__init__.py tests/core/test_collection_tracker.py
git commit -m "feat(collection): implement thread-safe CollectionTracker"
```

---

### Task 4: Ingestion Pipeline & Quality Guard (`MatchIngestor`)

**Files:**
- Create: `core/collection/ingestor.py`
- Test: `tests/core/test_match_ingestor.py`

**Interfaces:**
- Consumes:
  - `core.repository.MatchRepository`, `core.repository.TrainingRepository`
  - `core.json_utils.extract_pre_match_features`
  - `core.feature_labels.extract_labels`
  - `core.riot_api.RiotAPI`
  - `core.spatial_rnn.extract_and_store_timeline_embeddings`
- Produces:
  - `core.collection.ingestor.MatchIngestor`:
    - `build_training_record(match_id: str, details: dict, region: str = None) -> tuple | None`
    - `ingest_match_ids(match_ids: list[str], api: RiotAPI, region: str, max_workers: int = 5) -> IngestionResult`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_match_ingestor.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from core import database
from core.collection.ingestor import MatchIngestor
from core.collection.models import IngestionResult


@pytest.fixture
def ingestor(tmp_path):
    db_file = str(tmp_path / "test_ingest.db")
    database.init_db(db_file)
    return MatchIngestor(db_path=db_file)


def _make_match_payload(game_mode="CLASSIC", queue_id=420, duration=600, winner_team=200):
    teams = [
        {"teamId": 100, "win": winner_team == 100},
        {"teamId": 200, "win": winner_team == 200},
    ]
    participants = []
    for i in range(10):
        t_id = 100 if i < 5 else 200
        participants.append({
            "puuid": f"puuid_{i}",
            "teamId": t_id,
            "teamPosition": "TOP" if i % 5 == 0 else "MIDDLE",
            "championName": "Ahri",
            "win": (t_id == winner_team),
            "kills": 3,
            "deaths": 1,
            "assists": 2,
            "goldEarned": 5000,
        })
    return {
        "info": {
            "gameMode": game_mode,
            "queueId": queue_id,
            "gameDuration": duration,
            "gameCreation": 1700000000000,
            "gameVersion": "14.1.1",
            "teams": teams,
            "participants": participants,
        }
    }


def test_build_training_record_leak_free(ingestor):
    details = _make_match_payload(winner_team=200)
    record = ingestor.build_training_record("EUW1_123", details, region="EUW")
    assert record is not None
    mid, feature, winner, labels = record
    assert mid == "EUW1_123"
    assert winner == 1
    # Verify zero post-game stats in features
    for p in feature["team_a"] + feature["team_b"]:
        assert "kda" not in p
        assert "gold" not in p


def test_build_training_record_rejects_non_classic_and_short(ingestor):
    # Non-classic
    bad_mode = _make_match_payload(game_mode="ARAM")
    assert ingestor.build_training_record("M1", bad_mode) is None

    # Remake duration < 300
    remake = _make_match_payload(duration=180)
    assert ingestor.build_training_record("M2", remake) is None


def test_ingest_match_ids_success(ingestor):
    mock_api = MagicMock()
    mock_api.get_match_details_batch.return_value = {
        "EUW1_1": _make_match_payload(),
        "EUW1_2": _make_match_payload(),
    }
    mock_api.get_match_timelines_batch.return_value = {
        "EUW1_1": {"info": {}},
        "EUW1_2": {"info": {}},
    }

    with patch("core.spatial_rnn.extract_and_store_timeline_embeddings") as mock_spatial:
        result = ingestor.ingest_match_ids(["EUW1_1", "EUW1_2"], api=mock_api, region="EUW")

    assert result.matches_saved == 2
    assert result.records_saved == 2
    assert result.timelines_saved == 2
    assert len(result.discovered_puuids) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_match_ingestor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.collection.ingestor'`

- [ ] **Step 3: Implement `MatchIngestor` in `core/collection/ingestor.py`**

Create `core/collection/ingestor.py`:
```python
import logging
import time
from typing import Optional
from core import config, json_utils
from core.feature_labels import extract_labels
from core.repository import MatchRepository, TrainingRepository
from core.collection.models import IngestionResult, MatchQualityFilter

logger = logging.getLogger(__name__)


class MatchIngestor:
    def __init__(
        self,
        db_path: Optional[str] = None,
        quality_filter: Optional[MatchQualityFilter] = None,
    ):
        self.db_path = config.get_db_path(db_path, require_exists=False)
        self.match_repo = MatchRepository(self.db_path)
        self.train_repo = TrainingRepository(self.db_path)
        self.filter = quality_filter or MatchQualityFilter()

    def build_training_record(
        self, match_id: str, details: dict, region: Optional[str] = None
    ) -> Optional[tuple]:
        """Convert raw Riot API match JSON into a leak-free training tuple.

        Returns (match_id, pre_match_feature_dict, winner_label, labels_dict)
        or None if validation fails.
        """
        try:
            info = details.get("info", details)
            game_mode = (info.get("gameMode") or "").upper()
            if game_mode not in self.filter.allowed_game_modes:
                return None

            duration = info.get("gameDuration", 0)
            if duration < self.filter.min_duration_seconds:
                return None

            teams = info.get("teams", [])
            winner = None
            for t in teams:
                if t.get("win") and t.get("teamId") == 200:
                    winner = 1
                    break
                elif t.get("win") and t.get("teamId") == 100:
                    winner = 0
                    break

            if winner is None:
                return None

            # Strictly pre-match features (leak-free)
            feature = json_utils.extract_pre_match_features(details, region=region)
            labels = extract_labels(details)
            if labels is None:
                return None

            return (match_id, feature, winner, labels)
        except Exception as e:
            logger.error("Error building training record for %s: %s", match_id, e)
            return None

    def ingest_match_ids(
        self,
        match_ids: list[str],
        api,
        region: str,
        max_workers: int = 5,
        verbose: bool = False,
    ) -> IngestionResult:
        """Download, validate, extract features/timelines, and persist new matches."""
        result = IngestionResult()
        new_mids = [mid for mid in match_ids if not self.match_repo.match_exists(mid)]
        if not new_mids:
            return result

        details_map = api.get_match_details_batch(new_mids, max_workers=max_workers, verbose=verbose)
        if not details_map:
            return result

        matches_batch = []
        training_batch = []
        discovered_puuids_set: set[str] = set()

        for mid, details in details_map.items():
            if not details or not isinstance(details, dict):
                result.skipped += 1
                continue

            info = details.get("info", {})
            # Extra queue check if present
            queue_id = info.get("queueId")
            if queue_id is not None and queue_id not in self.filter.allowed_queue_ids:
                result.skipped += 1
                continue

            record = self.build_training_record(mid, details, region=region)
            if not record:
                result.skipped += 1
                continue

            matches_batch.append((mid, details))
            training_batch.append(record)

            # Discover participants for snowballing
            for p in info.get("participants", []):
                p_puuid = p.get("puuid")
                if p_puuid:
                    discovered_puuids_set.add(p_puuid)

        if matches_batch:
            self.match_repo.save_matches_batch(matches_batch)
            result.matches_saved = len(matches_batch)

        if training_batch:
            self.train_repo.save_training_records_batch(training_batch)
            result.records_saved = len(training_batch)

        # Systematic timeline collection
        timelines_to_fetch = [mid for mid, _ in matches_batch]
        if timelines_to_fetch:
            try:
                timelines = api.get_match_timelines_batch(
                    timelines_to_fetch, max_workers=max_workers, verbose=verbose
                )
                if timelines:
                    self.match_repo.save_match_timelines_batch(list(timelines.items()))
                    result.timelines_saved = len(timelines)

                    try:
                        from core.spatial_rnn import extract_and_store_timeline_embeddings
                        for t_mid, t_data in timelines.items():
                            m_details = details_map.get(t_mid, {})
                            gtime = m_details.get("info", {}).get("gameCreation", int(time.time() * 1000))
                            extract_and_store_timeline_embeddings(
                                t_mid, t_data, gtime, db_path=self.db_path
                            )
                            result.embeddings_computed += 1
                    except Exception as err:
                        logger.warning("Spatial embeddings extraction warning for %s: %s", region, err)
            except Exception as e:
                logger.error("Error fetching timelines for batch: %s", e)

        result.discovered_puuids = [(p, region) for p in discovered_puuids_set]
        return result
```

Update `core/collection/__init__.py` to export `MatchIngestor`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_match_ingestor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/collection/ingestor.py core/collection/__init__.py tests/core/test_match_ingestor.py
git commit -m "feat(collection): implement leak-free MatchIngestor with timeline integration"
```

---

### Task 5: Orchestration Facade (`CollectionService`)

**Files:**
- Create: `core/collection/service.py`
- Test: `tests/core/test_collection_service.py`

**Interfaces:**
- Consumes:
  - `core.collection.regions.resolve_region`
  - `core.collection.queue.CrawlQueueRepository`
  - `core.collection.ingestor.MatchIngestor`
  - `core.collection.tracker.CollectionTracker`
  - `core.repository.PlayerRepository`
- Produces:
  - `core.collection.service.CollectionService`:
    - `collect_player_matches(...) -> IngestionResult`
    - `collect_seed_players(...) -> IngestionResult`
    - `collect_by_puuid_accounts(...) -> dict`
    - `collect_top_leagues(...) -> IngestionResult`
    - `collect_batch_with_smurfs(...) -> IngestionResult`
    - `run_crawler(...) -> IngestionResult`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_collection_service.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from core import database
from core.collection.service import CollectionService


@pytest.fixture
def service(tmp_path):
    db_file = str(tmp_path / "test_service.db")
    database.init_db(db_file)
    return CollectionService(db_path=db_file)


def test_collect_player_matches(service):
    with patch("core.riot_api.RiotAPI") as MockAPI:
        api_instance = MockAPI.return_value
        api_instance.get_match_ids.return_value = ["EUW1_100"]
        api_instance.get_match_details_batch.return_value = {
            "EUW1_100": {
                "info": {
                    "gameMode": "CLASSIC",
                    "queueId": 420,
                    "gameDuration": 1200,
                    "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
                    "participants": [{"puuid": "p0", "teamId": 100, "championName": "Ashe", "teamPosition": "BOTTOM"}],
                }
            }
        }
        api_instance.get_match_timelines_batch.return_value = {"EUW1_100": {"info": {}}}

        res = service.collect_player_matches(puuid="p0", region="EUW", count=1)
        assert res.matches_saved == 1
        assert res.records_saved == 1


def test_collect_seed_players(service):
    with patch("core.riot_api.RiotAPI") as MockAPI:
        api_instance = MockAPI.return_value
        api_instance.get_puuid.return_value = "p_faker"
        api_instance.get_match_ids.return_value = []

        res = service.collect_seed_players([("Faker", "KR1")], matches_per_player=5)
        assert res.errors == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.collection.service'`

- [ ] **Step 3: Implement `CollectionService` in `core/collection/service.py`**

Create `core/collection/service.py`:
```python
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional, Callable
from core import config, database, riot_api
from core.repository import MatchRepository, PlayerRepository, TrainingRepository
from core.collection.regions import resolve_region
from core.collection.models import IngestionResult, CrawlItem
from core.collection.queue import CrawlQueueRepository
from core.collection.ingestor import MatchIngestor
from core.collection.tracker import CollectionTracker, get_default_tracker

logger = logging.getLogger(__name__)

DEFAULT_REGIONS = ["KR", "NA", "EUW", "EUN", "BR", "LA1", "LA2", "OCE", "JP", "RU", "TR"]
DEFAULT_PLAYERS_PER_REGION = 500
DEFAULT_MATCHES_PER_PLAYER = 20
DEFAULT_LOOKBACK_DAYS = 60


class CollectionService:
    def __init__(
        self,
        db_path: Optional[str] = None,
        tracker: Optional[CollectionTracker] = None,
    ):
        self.db_path = config.get_db_path(db_path, require_exists=False)
        database.init_db(self.db_path)
        self.match_repo = MatchRepository(self.db_path)
        self.player_repo = PlayerRepository(self.db_path)
        self.train_repo = TrainingRepository(self.db_path)
        self.queue_repo = CrawlQueueRepository(self.db_path)
        self.ingestor = MatchIngestor(self.db_path)
        self.tracker = tracker or get_default_tracker()

    def collect_player_matches(
        self,
        puuid: str,
        region: str,
        count: int = 20,
        start_time: Optional[int] = None,
        queue: int = 420,
    ) -> IngestionResult:
        """Fetch and ingest recent matches for a single player by PUUID."""
        canon_region = resolve_region(region)
        api = riot_api.RiotAPI(canon_region)
        try:
            match_ids = api.get_match_ids(
                puuid, count=count, start_time=start_time, queue=queue, type="ranked"
            )
            if not match_ids:
                return IngestionResult()
            result = self.ingestor.ingest_match_ids(match_ids, api=api, region=canon_region)
            if result.records_saved > 0:
                self.player_repo.increment_player_matches(puuid, result.records_saved)
            return result
        except Exception as e:
            logger.error("Error collecting matches for %s (%s): %s", puuid[:12], canon_region, e)
            res = IngestionResult()
            res.errors = 1
            return res

    def collect_seed_players(
        self,
        seed_players: list[tuple[str, str]],
        matches_per_player: int = 10,
        days_back: Optional[int] = None,
    ) -> IngestionResult:
        """Collect match history for a list of (name, tag) seeds."""
        total_res = IngestionResult()
        start_time = int(time.time()) - (days_back * 86400) if days_back else None
        self.tracker.set_running(total=len(seed_players), player="Seeds Batch")

        for idx, (name, tag) in enumerate(seed_players):
            self.tracker.set_player(f"{name}#{tag}")
            region = resolve_region(tag)
            try:
                api = riot_api.RiotAPI(region)
                puuid = api.get_puuid(name, tag)
                if not puuid:
                    self.tracker.add_log(f"PUUID resolution failed for {name}#{tag}")
                    total_res.skipped += 1
                    self.tracker.step(1)
                    continue

                self.player_repo.save_player(puuid, name, tag)
                sub_res = self.collect_player_matches(
                    puuid, region=region, count=matches_per_player, start_time=start_time
                )
                total_res.matches_saved += sub_res.matches_saved
                total_res.records_saved += sub_res.records_saved
                total_res.timelines_saved += sub_res.timelines_saved
                total_res.errors += sub_res.errors

                self.tracker.add_log(
                    f"Saved {sub_res.matches_saved} matches for {name}#{tag}"
                )
            except Exception as e:
                logger.error("Error collecting seed player %s#%s: %s", name, tag, e)
                total_res.errors += 1
                self.tracker.add_log(f"[ERROR] {name}#{tag}: {e}")
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def collect_by_puuid_accounts(
        self,
        accounts: list[dict],
        matches_per_player: int = 20,
    ) -> dict:
        """Collect match history for accounts where PUUID is known (e.g. lolpros)."""
        stats = {"processed": 0, "saved": 0, "skipped": 0, "errors": 0}
        self.tracker.set_running(total=len(accounts), player="PUUID Accounts")

        for acc in accounts:
            puuid = acc.get("puuid", "").strip()
            server = acc.get("server", "").upper()
            name = acc.get("gamename") or acc.get("summoner_name") or "?"
            tag = acc.get("tagline") or server or "?"

            if not puuid:
                stats["skipped"] += 1
                self.tracker.step(1)
                continue

            region = resolve_region(server)
            self.tracker.set_player(f"{name}#{tag}")
            try:
                self.player_repo.save_player(puuid, name, tag)
                res = self.collect_player_matches(
                    puuid, region=region, count=matches_per_player
                )
                stats["saved"] += res.records_saved
                stats["processed"] += 1
                self.tracker.add_log(f"[{region}] Saved {res.matches_saved} matches for {name}#{tag}")
            except Exception as e:
                logger.error("Error for %s#%s: %s", name, tag, e)
                stats["errors"] += 1
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return stats

    def collect_top_leagues(
        self,
        regions: list[str] = ["KR"],
        total_players_per_region: int = 100,
        matches_per_player: int = 10,
        days_back: int = 30,
    ) -> IngestionResult:
        """Fetch top league players (Challenger/GM/Master) across regions and collect their matches."""
        total_res = IngestionResult()
        all_entries = []
        for region in regions:
            canon = resolve_region(region)
            api = riot_api.RiotAPI(canon)
            try:
                entries = api.get_top_players(total_count=total_players_per_region)
                for entry in entries:
                    entry["_region"] = canon
                all_entries.extend(entries)
            except Exception as e:
                logger.error("Error fetching top league entries for %s: %s", canon, e)

        self.tracker.set_running(total=len(all_entries), player="Top Leagues")
        start_time = int(time.time()) - (days_back * 86400)

        for entry in all_entries:
            puuid = entry.get("puuid")
            region = entry.get("_region", "EUW")
            if not puuid:
                self.tracker.step(1)
                continue

            api = riot_api.RiotAPI(region)
            account_info = api.get_account_by_puuid(puuid)
            game_name = account_info.get("game_name", "Unknown") if account_info else "Unknown"
            tag_line = account_info.get("tag_line", region) if account_info else region
            self.player_repo.save_player(puuid, game_name, tag_line)
            self.tracker.set_player(f"{game_name}#{tag_line} ({region})")

            sub_res = self.collect_player_matches(
                puuid, region=region, count=matches_per_player, start_time=start_time
            )
            total_res.matches_saved += sub_res.matches_saved
            total_res.records_saved += sub_res.records_saved
            total_res.timelines_saved += sub_res.timelines_saved
            self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def collect_batch_with_smurfs(
        self,
        player_list: list[tuple[str, str]],
        sources: list[str],
        count: int = 50,
    ) -> IngestionResult:
        """Collect batch for players and their smurf accounts."""
        self.tracker.set_running(total=len(player_list), player="Batch Smurfs")
        total_res = IngestionResult()

        for name, tag in player_list:
            self.tracker.set_player(f"{name}#{tag}")
            region = resolve_region(tag)
            try:
                api = riot_api.RiotAPI(region)
                main_puuid = api.get_puuid(name, tag)
                if not main_puuid:
                    self.tracker.add_log(f"Could not resolve PUUID for {name}#{tag}")
                    self.tracker.step(1)
                    continue

                self.player_repo.save_player(main_puuid, name, tag)
                accounts = [{"puuid": main_puuid, "server": region, "gamename": name, "tagline": tag}]

                for acc in accounts:
                    sub_res = self.collect_player_matches(
                        acc["puuid"], region=acc["server"], count=count
                    )
                    total_res.matches_saved += sub_res.matches_saved
                    total_res.records_saved += sub_res.records_saved
                    total_res.timelines_saved += sub_res.timelines_saved
            except Exception as e:
                logger.error("Error in batch with smurfs for %s#%s: %s", name, tag, e)
                total_res.errors += 1
            finally:
                self.tracker.step(1)

        self.tracker.set_completed()
        return total_res

    def run_crawler(
        self,
        target_matches: int = 1000,
        regions: list[str] = ["EUW", "KR", "NA"],
        days_back: int = 14,
        batch_size: int = 25,
        stop_event: Optional[threading.Event] = None,
    ) -> IngestionResult:
        """High-throughput crawler running continuously until target matches or stopped."""
        total_res = IngestionResult()
        self.tracker.set_running(total=target_matches, player="Continuous Crawler")
        start_time = int(time.time()) - (days_back * 86400)
        batch_counter = 0

        # Seed initial queue if empty
        if self.queue_repo.get_stats()["pending"] == 0:
            for region in regions:
                canon = resolve_region(region)
                api = riot_api.RiotAPI(canon)
                try:
                    entries = api.get_top_players(total_count=50)
                    puuids = [e["puuid"] for e in entries if e.get("puuid")]
                    self.queue_repo.enqueue_puuids(puuids, region=canon, priority=10)
                except Exception as e:
                    logger.warning("Failed seeding top players for %s: %s", canon, e)

        while (
            total_res.matches_saved < target_matches
            and not (stop_event and stop_event.is_set())
        ):
            items = self.queue_repo.pop_next(limit=1)
            if not items:
                logger.info("Frontier queue depleted. Refreshing seeds...")
                for region in regions:
                    canon = resolve_region(region)
                    api = riot_api.RiotAPI(canon)
                    try:
                        entries = api.get_top_players(total_count=30)
                        puuids = [e["puuid"] for e in entries if e.get("puuid")]
                        self.queue_repo.enqueue_puuids(puuids, region=canon, priority=5)
                    except Exception:
                        pass
                time.sleep(2)
                continue

            item = items[0]
            region = resolve_region(item.region)
            puuid = item.puuid

            if not puuid and item.game_name and item.tag_line:
                api = riot_api.RiotAPI(region)
                puuid = api.get_puuid(item.game_name, item.tag_line)

            if not puuid:
                self.queue_repo.mark_processed_by_id(item.id)
                continue

            sub_res = self.collect_player_matches(
                puuid, region=region, count=batch_size, start_time=start_time
            )
            self.queue_repo.mark_processed(puuid)
            self.queue_repo.mark_processed_by_id(item.id)

            total_res.matches_saved += sub_res.matches_saved
            total_res.records_saved += sub_res.records_saved
            total_res.timelines_saved += sub_res.timelines_saved
            self.tracker.step(sub_res.matches_saved)

            # Snowballing: enqueue newly discovered participants
            if sub_res.discovered_puuids:
                discovered_by_region: dict[str, list[str]] = {}
                for p, r in sub_res.discovered_puuids:
                    discovered_by_region.setdefault(r, []).append(p)
                for r, plist in discovered_by_region.items():
                    self.queue_repo.enqueue_puuids(plist, region=r, priority=0)

            batch_counter += 1
            if batch_counter % 10 == 0:
                with database.get_connection(self.db_path) as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        self.tracker.set_completed()
        return total_res
```

Update `core/collection/__init__.py` to export `CollectionService`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/collection/service.py core/collection/__init__.py tests/core/test_collection_service.py
git commit -m "feat(collection): implement unified CollectionService facade"
```

---

### Task 6: Modern CLI (`scripts/crawler.py`) & Backward-Compatibility Adapters

**Files:**
- Create: `scripts/crawler.py`
- Modify: `scripts/data_collector.py`
- Modify: `scripts/background_collector.py`
- Modify: `scripts/crawler_1m.py`
- Test: `tests/test_collection_database_selection.py`

**Interfaces:**
- Consumes: `core.collection.service.CollectionService`, `core.collection.regions.resolve_region`
- Produces:
  - CLI: `python scripts/crawler.py [--target] [--regions] [--seeds] [--status] [--db]`
  - Preserved APIs in `scripts.data_collector`:
    - `resolve_region`
    - `_build_training_record`
    - `collect_training_data`
    - `collect_top_leagues_data`
    - `collect_by_puuid`
    - `collect_batch_with_smurfs`
    - `batch_status`, `batch_lock`, `add_batch_log`

- [ ] **Step 1: Implement `scripts/crawler.py`**

Create `scripts/crawler.py`:
```python
#!/usr/bin/env python3
"""Unified high-throughput crawler and match collection CLI."""
import argparse
import os
import signal
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import database
from core.collection import CollectionService, resolve_region


def print_status(db_path: str | None = None):
    stats = database.get_db_stats(db_path=db_path) if db_path else database.get_db_stats()
    svc = CollectionService(db_path=db_path)
    q_stats = svc.queue_repo.get_stats()

    print("\n--- NNriot Unified Collection & Database Status ---")
    print(f"Total Matches:         {stats.get('matches', 0):,}")
    print(f"Training Dataset:      {stats.get('training_records', 0):,}")
    print(f"Total Players:         {stats.get('players', 0):,}")
    print(f"Queue Total:           {q_stats['total']:,}")
    print(f"Queue Pending:         {q_stats['pending']:,}")
    print(f"Queue Processed:       {q_stats['processed']:,}")
    print("----------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="NNriot Unified Collector & Crawler")
    parser.add_argument("--target", type=int, default=1000, help="Target new matches to collect")
    parser.add_argument("--regions", type=str, default="EUW,KR,NA", help="Comma-separated regions")
    parser.add_argument("--days-back", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--batch-size", type=int, default=25, help="Matches per player fetch")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated Name#Tag seeds")
    parser.add_argument("--top", action="store_true", help="Crawl Challenger & GM top leagues")
    parser.add_argument("--status", action="store_true", help="Display collection status and exit")
    parser.add_argument("--db", type=str, default=None, help="Target SQLite database path")
    args = parser.parse_args()

    if args.status:
        print_status(args.db)
        return

    service = CollectionService(db_path=args.db)
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]

    if args.seeds:
        seed_list = []
        for s in args.seeds.split(","):
            if "#" in s:
                name, tag = s.split("#", 1)
                seed_list.append((name.strip(), tag.strip()))
        print(f"Collecting {len(seed_list)} seed players...")
        service.collect_seed_players(seed_list, matches_per_player=args.batch_size, days_back=args.days_back)
        return

    if args.top:
        print(f"Collecting top league players in {regions}...")
        service.collect_top_leagues(regions=regions, total_players_per_region=100, matches_per_player=args.batch_size)
        return

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        print("\nStopping crawler gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting continuous crawler towards +{args.target} matches in {regions}...")
    service.run_crawler(
        target_matches=args.target,
        regions=regions,
        days_back=args.days_back,
        batch_size=args.batch_size,
        stop_event=stop_event,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Refactor `scripts/data_collector.py` as a delegation adapter**

Modify `scripts/data_collector.py`:
- Delegate `resolve_region` to `core.collection.regions.resolve_region`.
- Delegate `_build_training_record` to `MatchIngestor(db_path).build_training_record`.
- Delegate `collect_training_data`, `collect_top_leagues_data`, `collect_by_puuid`, `collect_batch_with_smurfs` to `CollectionService(db_path)`.
- Connect `batch_status`, `batch_lock`, and `add_batch_log` to `get_default_tracker()`.

- [ ] **Step 3: Refactor `scripts/background_collector.py` and `scripts/crawler_1m.py`**

Refactor both scripts to instantiate `CollectionService` and invoke `service.run_crawler()`.

- [ ] **Step 4: Run existing database selection tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_collection_database_selection.py tests/test_collector_game_mode_guard.py tests/test_timeline_collection.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/crawler.py scripts/data_collector.py scripts/background_collector.py scripts/crawler_1m.py
git commit -m "refactor(scripts): provide unified crawler CLI and backward-compatibility shims"
```

---

### Task 7: Flask Routes Refactoring

**Files:**
- Modify: `app/routes/history.py`
- Modify: `app/routes/api.py`
- Modify: `app/routes/lolpros.py`
- Test: `tests/test_predict_api.py`
- Test: `tests/test_task_runner.py`

**Interfaces:**
- Consumes: `core.collection.CollectionService`, `core.collection.resolve_region`
- Produces: Clean Flask route handlers without direct imports from `scripts/`.

- [ ] **Step 1: Refactor `app/routes/history.py`**

- Replace `from scripts import data_collector` with `from core.collection import CollectionService, resolve_region`.
- In `api_player_history` (lines 253-285): replace inline match fetching and record creation with `service.collect_player_matches(puuid, region=region, count=count)`.
- In `/api/history/collect`: submit task calling `service.collect_seed_players([(game_name, tag_line)], count)`.
- In `/api/history/collect_batch`: submit task calling `service.collect_seed_players(seeds, 10)`.
- In `/api/batch/collect`: submit task calling `service.collect_batch_with_smurfs(player_list, sources, count)`.
- In `/api/batch/status`: return `service.tracker.to_dict()`.

- [ ] **Step 2: Refactor `app/routes/api.py` and `app/routes/lolpros.py`**

- In `app/routes/api.py`:
  - Replace `from scripts.data_collector import resolve_region` with `from core.collection import resolve_region`.
  - In `auto_collect_task`: call `CollectionService(db_path=db_path).collect_seed_players(seeds, matches_per_player=10)`.
  - In `/api/monitor/metrics`: read `DEFAULT_REGIONS` etc. from `core.collection.service`.
- In `app/routes/lolpros.py`:
  - Replace `data_collector.collect_by_puuid(...)` with `CollectionService(db_path=db_path).collect_by_puuid_accounts(accounts_to_collect)`.

- [ ] **Step 3: Run full Flask API and task runner tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_collection_database_selection.py tests/test_predict_api.py tests/test_task_runner.py -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add app/routes/history.py app/routes/api.py app/routes/lolpros.py
git commit -m "refactor(routes): switch Flask endpoints to CollectionService facade"
```

---

### Task 8: Verification & Cleanup

**Files:**
- Modify: `scripts/live_game.py`
- Modify: `scripts/backfill_timelines.py`
- Test: All collector and integration tests

- [ ] **Step 1: Clean remaining imports from `scripts.data_collector` across the codebase**

Update `scripts/live_game.py` and `scripts/backfill_timelines.py` to import `resolve_region` from `core.collection`.

- [ ] **Step 2: Verify git grep shows no web route imports from `scripts`**

Run: `git grep "from scripts" app/`
Expected: Empty output.

- [ ] **Step 3: Run comprehensive test suite**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/core/test_collection_regions_models.py tests/core/test_crawl_queue_repository.py tests/core/test_collection_tracker.py tests/core/test_match_ingestor.py tests/core/test_collection_service.py tests/test_collection_database_selection.py tests/test_collector_game_mode_guard.py tests/test_timeline_collection.py tests/test_predict_api.py tests/test_task_runner.py -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/live_game.py scripts/backfill_timelines.py
git commit -m "chore(cleanup): eliminate legacy scripts imports and verify clean boundaries"
```
