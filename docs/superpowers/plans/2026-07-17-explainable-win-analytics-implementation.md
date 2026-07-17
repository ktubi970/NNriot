# Explainable Win Analytics and French Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate, rebuildable `analytics.db` that produces statistically defensible retrospective gameplay evidence and exposes it through a constrained interactive French chatbot.

**Architecture:** A standalone `analytics/` package reads `training_data.db` through a query-only SQLite connection, materializes normalized match/team/player facts, resolves allowlisted cohorts, and produces immutable evidence packets. Interpretable modeling and the existing NN are downstream candidate/discovery tools only; a Flask blueprint and deterministic French renderer expose validated evidence without allowing generated SQL or untraceable numbers.

**Tech Stack:** Python 3.10+, SQLite, NumPy, SciPy, scikit-learn, joblib, Flask, Jinja, vanilla JavaScript, pytest.

## Global Constraints

- Keep `training_data.db` as the operational source and open it read-only with `mode=ro` and `PRAGMA query_only=ON`.
- Store all derived facts and evidence in a separate `analytics.db`; never migrate the operational schema for this feature.
- Restrict phase 1 to retrospective final Match-V5 data; timeline-derived 5/10/15-minute analysis remains unavailable.
- Default to `CLASSIC`, ranked solo queue `420`, and the newest patch with at least 500 eligible matches; widen backward one patch at a time to at most three patches and disclose widening.
- Require at least 500 matches, 100 distinct players, and 100 events in each compared group.
- Require a 95% confidence interval excluding the null, Benjamini-Hochberg `q <= 0.05`, practical effect of at least 3 win-rate points or `|d_z| >= 0.20`, and consistent direction in at least two temporal folds.
- Label evidence strong only at 2,000 matches, `q <= 0.01`, at least 5 win-rate points or `|d_z| >= 0.50`, and consistent direction in every available temporal fold.
- Show automatic inter-region differences only when every displayed region has at least 500 matches and 100 wins/losses, the patch-adjusted interaction has `q <= 0.05`, the contrast is at least 3 points or 0.20 standardized units, and the recent-period sensitivity direction agrees.
- Use seeded 2,000-resample bootstrap intervals and seeded two-sided permutation tests; record method version and seed in every evidence packet.
- Make every numeric token in a chat answer traceable to an evidence ID; qualitative narrative may not introduce digits or factual comparisons.
- Ban target/forbidden fields from explanatory matrices and label all phase-1 model outputs retrospective.
- Keep the current NN discovery-only until a clean independent evaluation exists; it cannot publish evidence or recommendations directly.
- Preserve all existing collection, training, prediction, API, and page contracts.
- Preserve the dirty worktree and stage only paths named by the current task.

---

## Planned File Structure

| Path | Responsibility |
| --- | --- |
| `analytics/__init__.py` | Stable public package exports and package version. |
| `analytics/config.py` | Environment-backed paths, thresholds, seeds, and method versions. |
| `analytics/types.py` | Frozen request, cohort, estimate, evidence, and rendered-answer types. |
| `analytics/registry.py` | Allowlisted metrics, aliases, units, feature leakage class, and comparison method. |
| `analytics/db.py` | Analytical schema, connections, metadata, transactions, and evidence cache. |
| `analytics/source.py` | Read-only source connection, raw JSON decoding, and source fingerprinting. |
| `analytics/extract.py` | Match-V5 normalization into match/team/player rows and stable issue codes. |
| `analytics/build.py` | Full rebuild, incremental update, quality gate, atomic replacement, and build report. |
| `analytics/cohorts.py` | Filter validation, default/widened cohort resolution, SQL predicates, and counts. |
| `analytics/statistics.py` | Paired effects, binary effects, bootstrap, permutation, intervals, and FDR. |
| `analytics/evidence.py` | Robustness checks, recommendation thresholds, stable evidence IDs, and serialization. |
| `analytics/engine.py` | Registered analysis execution, candidate-family correction, caching, and abstention. |
| `analytics/regions.py` | Patch-adjusted regional contrasts and automatic-display gate. |
| `analytics/model_data.py` | Named blue-minus-red matrix construction and leakage enforcement. |
| `analytics/splits.py` | Chronological and player-disjoint splits. |
| `analytics/interpretable.py` | Regularized logistic model, selection, evaluation, calibration, and reports. |
| `analytics/nn_adapter.py` | Discovery-only NN candidate contract and independent validation gate. |
| `analytics/planner.py` | Deterministic French intent parsing and validated follow-up context. |
| `analytics/renderer.py` | Deterministic evidence prose, narrative adapter, and numeric trace validation. |
| `analytics/service.py` | Chat orchestration, freshness checks, evidence retrieval, and health. |
| `analytics/cli.py`, `analytics/__main__.py` | Explicit build, update, model, benchmark, and health commands. |
| `app/routes/analytics.py` | Analytics blueprint and JSON error mapping. |
| `templates/analytics.html` | French chat page and accessible evidence cards. |
| `static/analytics.css`, `static/analytics.js` | Page styling and browser interaction. |
| `tests/analytics/factories.py` | Deterministic Match-V5, source-DB, mart, evidence, and model fixtures. |
| `tests/analytics/` | Focused unit, integration, model, API, and performance tests. |

### Stable interfaces shared across tasks

```python
AnalysisRequest.from_dict(data: dict) -> AnalysisRequest
AnalysisRequest.to_dict() -> dict
CohortResolver.resolve(request: AnalysisRequest) -> ResolvedCohort
AnalysisEngine.run(request: AnalysisRequest, question: str,
                   cohort: ResolvedCohort | None = None) -> list[EvidencePacket]
EvidencePacket.to_dict() -> dict
AnalyticsService.chat(message: str, context: dict | None, filters: dict) -> ChatResponse
build_analytics(source_path: str, analytics_path: str, incremental: bool = False) -> BuildReport
train_and_evaluate(analytics_path: str, artifact_dir: str) -> ModelReport
```

## Task 1: Foundation Types, Configuration, and Metric Registry

**Files:**
- Create: `analytics/__init__.py`
- Create: `analytics/config.py`
- Create: `analytics/types.py`
- Create: `analytics/registry.py`
- Create: `tests/analytics/__init__.py`
- Create: `tests/analytics/test_registry.py`
- Modify: `requirements.txt`
- Modify: `web_requirements.txt`

**Interfaces:**
- Consumes: No feature code.
- Produces: `AnalyticsConfig`, `AnalysisRequest`, `AnalysisFilters`, `ResolvedCohort`, `Estimate`, `EvidencePacket`, `MetricSpec`, `METRICS`, `get_metric()`.

- [ ] **Step 1: Write registry and serialization tests**

```python
# tests/analytics/test_registry.py
import pytest

from analytics.registry import FeatureClass, MetricKind, get_metric
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


def test_registry_exposes_named_final_game_metric():
    gold = get_metric("gold")
    assert gold.column == "gold"
    assert gold.kind is MetricKind.CONTINUOUS
    assert gold.feature_class is FeatureClass.FINAL_GAME_RETROSPECTIVE
    assert gold.unit == "or"


def test_registry_rejects_target_field():
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        get_metric("win")


def test_request_round_trip_is_canonical():
    request = AnalysisRequest(
        intent=Intent.EXPLAIN_METRIC,
        metric="vision_score",
        filters=AnalysisFilters(regions=("EUW",), queue_ids=(420,)),
    )
    assert AnalysisRequest.from_dict(request.to_dict()) == request
```

- [ ] **Step 2: Run the focused test and verify the missing-package failure**

Run: `python -m pytest tests/analytics/test_registry.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'analytics'`.

- [ ] **Step 3: Create the public types and registry**

```python
# analytics/config.py
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AnalyticsConfig:
    source_path: str = os.getenv("NNRIOT_DB_PATH", "training_data.db")
    analytics_path: str = os.getenv("NNRIOT_ANALYTICS_DB_PATH", "analytics.db")
    seed: int = int(os.getenv("NNRIOT_ANALYTICS_SEED", "20260716"))
    bootstrap_resamples: int = 2000
    min_matches: int = 500
    min_players: int = 100
    min_group_events: int = 100
    strong_matches: int = 2000
    method_version: str = "1.0.0"
    extractor_version: str = "1.0.0"
```

```python
# analytics/types.py
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Intent(str, Enum):
    RANK_CONDITIONS = "rank_conditions"
    EXPLAIN_METRIC = "explain_metric"
    COMPARE_REGIONS = "compare_regions"
    COMPARE_PATCHES = "compare_patches"
    SHOW_EVIDENCE = "show_evidence"
    SHOW_LIMITATIONS = "show_limitations"


class AbstentionCode(str, Enum):
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    UNSUPPORTED_FILTER = "UNSUPPORTED_FILTER"
    TIMELINE_DATA_UNAVAILABLE = "TIMELINE_DATA_UNAVAILABLE"
    ESPORT_TEAM_IDENTITY_UNAVAILABLE = "ESPORT_TEAM_IDENTITY_UNAVAILABLE"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    INSUFFICIENT_PLAYERS = "INSUFFICIENT_PLAYERS"
    SPARSE_COMPARISON_GROUP = "SPARSE_COMPARISON_GROUP"
    EFFECT_NOT_ROBUST = "EFFECT_NOT_ROBUST"
    ANALYTICS_DB_STALE = "ANALYTICS_DB_STALE"
    SOURCE_DATA_INVALID = "SOURCE_DATA_INVALID"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_OUTPUT_REJECTED = "LLM_OUTPUT_REJECTED"


@dataclass(frozen=True)
class AnalysisFilters:
    regions: tuple[str, ...] = ()
    patches: tuple[str, ...] = ()
    queue_ids: tuple[int, ...] = (420,)
    date_start_ms: int | None = None
    date_end_ms: int | None = None
    champion: str | None = None
    role: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnalysisFilters":
        data = data or {}
        return cls(
            regions=tuple(data.get("regions", ())),
            patches=tuple(data.get("patches", ())),
            queue_ids=tuple(int(v) for v in data.get("queue_ids", (420,))),
            date_start_ms=data.get("date_start_ms"),
            date_end_ms=data.get("date_end_ms"),
            champion=data.get("champion"),
            role=data.get("role"),
        )


@dataclass(frozen=True)
class AnalysisRequest:
    intent: Intent
    metric: str | None = None
    filters: AnalysisFilters = field(default_factory=AnalysisFilters)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRequest":
        return cls(
            intent=Intent(data["intent"]),
            metric=data.get("metric"),
            filters=AnalysisFilters.from_dict(data.get("filters")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["intent"] = self.intent.value
        return result


@dataclass(frozen=True)
class ResolvedCohort:
    mode: str
    queue_ids: tuple[int, ...]
    patches: tuple[str, ...]
    regions: tuple[str, ...]
    date_start_ms: int | None
    date_end_ms: int | None
    champion: str | None
    role: str | None
    widened: bool
    matches: int
    players: int


@dataclass(frozen=True)
class Estimate:
    metric: str
    value: float
    unit: str
    baseline: float
    effect_size: float
    confidence_interval_95: tuple[float, float]
    p_value: float
    q_value: float | None = None
    events_by_group: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePacket:
    evidence_id: str
    question: str
    analysis_request: dict[str, Any]
    cohort: dict[str, Any]
    sample: dict[str, Any]
    estimate: dict[str, Any]
    method: dict[str, Any]
    robustness: dict[str, Any]
    recommendation: dict[str, Any]
    limitations: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

```python
# analytics/registry.py
from dataclasses import dataclass
from enum import Enum


class MetricKind(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class FeatureClass(str, Enum):
    PRE_MATCH = "pre_match"
    IN_GAME_TEMPORAL = "in_game_temporal"
    FINAL_GAME_RETROSPECTIVE = "final_game_retrospective"
    TARGET_FORBIDDEN = "target_forbidden"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    column: str
    kind: MetricKind
    unit: str
    feature_class: FeatureClass
    aliases: tuple[str, ...]


def _continuous(name: str, unit: str, *aliases: str) -> MetricSpec:
    return MetricSpec(name, name, MetricKind.CONTINUOUS, unit,
                      FeatureClass.FINAL_GAME_RETROSPECTIVE, tuple(aliases))


def _binary(name: str, column: str, *aliases: str) -> MetricSpec:
    return MetricSpec(name, column, MetricKind.BINARY, "points de victoire",
                      FeatureClass.FINAL_GAME_RETROSPECTIVE, tuple(aliases))


METRICS = {
    spec.name: spec for spec in (
        _continuous("kills", "eliminations", "kills", "éliminations"),
        _continuous("deaths", "morts", "deaths", "morts"),
        _continuous("assists", "assistances", "assists", "assistances"),
        _continuous("gold", "or", "gold", "or"),
        _continuous("champion_damage", "dégâts", "damage", "dégâts"),
        _continuous("vision_score", "points de vision", "vision"),
        _continuous("lane_cs", "sbires de voie", "cs", "farm"),
        _continuous("jungle_cs", "monstres de jungle", "jungle farm"),
        _continuous("barons", "barons", "baron"),
        _continuous("dragons", "dragons", "dragon"),
        _continuous("towers", "tours", "tour", "tours"),
        _continuous("inhibitors", "inhibiteurs", "inhibiteur"),
        _continuous("rift_heralds", "hérauts", "héraut"),
        _continuous("atakhan", "Atakhan", "atakhan"),
        _binary("first_blood", "first_blood", "premier sang", "first blood"),
        _binary("first_baron", "first_baron", "premier baron"),
        _binary("first_dragon", "first_dragon", "premier dragon"),
        _binary("first_tower", "first_tower", "première tour"),
        _binary("first_inhibitor", "first_inhibitor", "premier inhibiteur"),
    )
}


def get_metric(name: str) -> MetricSpec:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise ValueError(f"UNSUPPORTED_FILTER: metric={name}") from exc
```

Add `__version__ = "1.0.0"` and exports to `analytics/__init__.py`. Add `scipy>=1.10.0` and `joblib>=1.3.0` to both requirements files because analytics imports them directly.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/analytics/test_registry.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the foundation**

```powershell
git add analytics/__init__.py analytics/config.py analytics/types.py analytics/registry.py tests/analytics/__init__.py tests/analytics/test_registry.py requirements.txt web_requirements.txt
git commit -m "feat: define analytics contracts and registry"
```

## Task 2: Analytical Schema, Metadata, and Evidence Cache

**Files:**
- Create: `analytics/db.py`
- Create: `tests/analytics/test_db.py`

**Interfaces:**
- Consumes: `EvidencePacket` from Task 1.
- Produces: `connect_analytics()`, `initialize_schema()`, `get_meta()`, `set_meta()`, `EvidenceRepository.get()`, `EvidenceRepository.put()`.

- [ ] **Step 1: Write schema and cache tests**

```python
# tests/analytics/test_db.py
import sqlite3

from analytics.db import EvidenceRepository, connect_analytics, initialize_schema


def test_schema_has_required_fact_tables(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_schema(str(path))
    with connect_analytics(str(path), readonly=True) as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"analytics_meta", "match_facts", "team_facts", "player_facts",
            "extraction_issues", "evidence_cache"} <= names


def test_evidence_cache_is_scoped_by_dataset_and_method(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_schema(str(path))
    repo = EvidenceRepository(str(path))
    repo.put("ev-1", "dataset-a", "request-a", "1.0.0", {"value": 1})
    assert repo.get("dataset-a", "request-a", "1.0.0") == {"value": 1}
    assert repo.get("dataset-b", "request-a", "1.0.0") is None
```

- [ ] **Step 2: Verify the tests fail on the missing database module**

Run: `python -m pytest tests/analytics/test_db.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'analytics.db'`.

- [ ] **Step 3: Implement the schema and repository**

Create `analytics/db.py` with this complete `SCHEMA_SQL`; keep column order aligned with the positional insert statements in Task 4:

```sql
CREATE TABLE IF NOT EXISTS analytics_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS match_facts (
    match_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    platform_prefix TEXT NOT NULL,
    game_version TEXT NOT NULL,
    patch TEXT NOT NULL,
    queue_id INTEGER,
    game_creation_ms INTEGER NOT NULL,
    game_duration_s INTEGER NOT NULL,
    winner_team_id INTEGER NOT NULL CHECK (winner_team_id IN (100, 200)),
    participant_count INTEGER NOT NULL CHECK (participant_count = 10),
    source_season_id INTEGER
);
CREATE TABLE IF NOT EXISTS team_facts (
    match_id TEXT NOT NULL REFERENCES match_facts(match_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL CHECK (team_id IN (100, 200)),
    side TEXT NOT NULL CHECK (side IN ('BLUE', 'RED')),
    win INTEGER NOT NULL CHECK (win IN (0, 1)),
    kills INTEGER NOT NULL,
    deaths INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    gold INTEGER NOT NULL,
    champion_damage INTEGER NOT NULL,
    vision_score REAL NOT NULL,
    lane_cs INTEGER NOT NULL,
    jungle_cs INTEGER NOT NULL,
    barons INTEGER NOT NULL,
    dragons INTEGER NOT NULL,
    towers INTEGER NOT NULL,
    inhibitors INTEGER NOT NULL,
    rift_heralds INTEGER NOT NULL,
    atakhan INTEGER NOT NULL,
    first_blood INTEGER NOT NULL CHECK (first_blood IN (0, 1)),
    first_baron INTEGER NOT NULL CHECK (first_baron IN (0, 1)),
    first_dragon INTEGER NOT NULL CHECK (first_dragon IN (0, 1)),
    first_tower INTEGER NOT NULL CHECK (first_tower IN (0, 1)),
    first_inhibitor INTEGER NOT NULL CHECK (first_inhibitor IN (0, 1)),
    PRIMARY KEY (match_id, team_id)
);
CREATE TABLE IF NOT EXISTS player_facts (
    match_id TEXT NOT NULL REFERENCES match_facts(match_id) ON DELETE CASCADE,
    puuid TEXT NOT NULL,
    team_id INTEGER NOT NULL CHECK (team_id IN (100, 200)),
    side TEXT NOT NULL CHECK (side IN ('BLUE', 'RED')),
    win INTEGER NOT NULL CHECK (win IN (0, 1)),
    champion TEXT NOT NULL,
    role TEXT NOT NULL,
    kills INTEGER NOT NULL,
    deaths INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    gold INTEGER NOT NULL,
    champion_damage INTEGER NOT NULL,
    vision_score REAL NOT NULL,
    lane_cs INTEGER NOT NULL,
    jungle_cs INTEGER NOT NULL,
    PRIMARY KEY (match_id, puuid)
);
CREATE TABLE IF NOT EXISTS extraction_issues (
    match_id TEXT NOT NULL,
    issue_code TEXT NOT NULL,
    field TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    PRIMARY KEY (match_id, issue_code, field, extractor_version)
);
CREATE TABLE IF NOT EXISTS evidence_cache (
    evidence_id TEXT PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    method_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_version, request_hash, method_version)
);
CREATE INDEX IF NOT EXISTS idx_match_patch_region_queue ON match_facts(patch, region, queue_id);
CREATE INDEX IF NOT EXISTS idx_match_creation ON match_facts(game_creation_ms);
CREATE INDEX IF NOT EXISTS idx_match_region ON match_facts(region);
CREATE INDEX IF NOT EXISTS idx_team_match_win ON team_facts(match_id, win);
CREATE INDEX IF NOT EXISTS idx_player_puuid ON player_facts(puuid);
CREATE INDEX IF NOT EXISTS idx_player_champion_role ON player_facts(champion, role);
CREATE INDEX IF NOT EXISTS idx_evidence_lookup ON evidence_cache(dataset_version, request_hash, method_version);
```

Use this connection and repository behavior:

```python
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


@contextmanager
def connect_analytics(path: str, readonly: bool = False):
    if readonly:
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def initialize_schema(path: str) -> None:
    with connect_analytics(path) as conn:
        conn.executescript(SCHEMA_SQL)


class EvidenceRepository:
    def __init__(self, path: str):
        self.path = path

    def get(self, dataset_version: str, request_hash: str,
            method_version: str) -> dict | None:
        with connect_analytics(self.path, readonly=True) as conn:
            row = conn.execute(
                "SELECT evidence_json FROM evidence_cache "
                "WHERE dataset_version=? AND request_hash=? AND method_version=?",
                (dataset_version, request_hash, method_version),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, evidence_id: str, dataset_version: str, request_hash: str,
            method_version: str, packet: dict) -> None:
        payload = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        with connect_analytics(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evidence_cache "
                "(evidence_id,dataset_version,request_hash,method_version,evidence_json,created_at) "
                "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                (evidence_id, dataset_version, request_hash, method_version, payload),
            )

    def get_by_id(self, evidence_id: str) -> dict | None:
        with connect_analytics(self.path, readonly=True) as conn:
            row = conn.execute(
                "SELECT evidence_json FROM evidence_cache WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None
```

Add `get_meta(path, key) -> str | None`, `set_meta(conn, key, value) -> None`, and `all_meta(path) -> dict[str, str]` using parameterized SQL.

- [ ] **Step 4: Run schema tests**

Run: `python -m pytest tests/analytics/test_db.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the analytical database layer**

```powershell
git add analytics/db.py tests/analytics/test_db.py
git commit -m "feat: add analytics mart schema"
```

## Task 3: Read-only Source and Match-V5 Extraction

**Files:**
- Create: `analytics/source.py`
- Create: `analytics/extract.py`
- Create: `tests/analytics/conftest.py`
- Create: `tests/analytics/factories.py`
- Create: `tests/analytics/test_extract.py`

**Interfaces:**
- Consumes: source `matches(match_id, raw_json, game_mode, game_creation_time, game_duration, season_id)`.
- Produces: `iter_source_matches()`, `source_fingerprint()`, `extract_match() -> ExtractionResult`.

- [ ] **Step 1: Add a realistic ten-participant fixture and extraction tests**

Define `classic_match(match_id="EUW1_1", queue_id=420, patch="16.14", created_ms=1)` in `tests/analytics/factories.py`. It returns metadata plus ten participants split 5/5, unique PUUIDs, `teamPosition` values `TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY`, and two teams with `champion`, `baron`, `dragon`, `tower`, `inhibitor`, `riftHerald`, and `atakhan` objectives. Define `write_source_db(path, matches)` in the same file; it creates only `schema_version(version)` and the six operational `matches` columns used by `analytics.source`, encoding raw JSON with the same gzip/base64 format as `database.save_match()`. `tests/analytics/conftest.py` exposes these helpers as `classic_match`, `source_db_factory`, and `source_db` fixtures.

```python
# tests/analytics/test_extract.py
from analytics.extract import ExtractionError, extract_match


def test_extracts_one_match_two_teams_ten_players(classic_match):
    result = extract_match("EUW1_1", classic_match)
    assert result.match["region"] == "EUW"
    assert result.match["patch"] == "16.14"
    assert len(result.teams) == 2
    assert len(result.players) == 10
    assert sum(row["win"] for row in result.teams) == 1
    assert result.teams[0]["gold"] == sum(
        p["gold"] for p in result.players if p["team_id"] == 100
    )


def test_rejects_non_classic_without_partial_rows(classic_match):
    classic_match["info"]["gameMode"] = "ARAM"
    try:
        extract_match("EUW1_2", classic_match)
    except ExtractionError as exc:
        assert exc.code == "UNSUPPORTED_GAME_MODE"
    else:
        raise AssertionError("ARAM must be rejected")


def test_rejects_duplicate_puuid(classic_match):
    classic_match["info"]["participants"][1]["puuid"] = \
        classic_match["info"]["participants"][0]["puuid"]
    try:
        extract_match("EUW1_3", classic_match)
    except ExtractionError as exc:
        assert exc.code == "DUPLICATE_PARTICIPANT"
    else:
        raise AssertionError("duplicate PUUID must be rejected")
```

- [ ] **Step 2: Run tests and verify the extractor is missing**

Run: `python -m pytest tests/analytics/test_extract.py -q`

Expected: FAIL during collection for `analytics.extract`.

- [ ] **Step 3: Implement query-only source access and deterministic decoding**

```python
# analytics/source.py
from contextlib import contextmanager
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3


@contextmanager
def connect_source(path: str):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
    finally:
        conn.close()


def decode_raw_json(value: str) -> dict:
    if value.startswith("{") or value.startswith("["):
        return json.loads(value)
    return json.loads(gzip.decompress(base64.b64decode(value)).decode("utf-8"))


def iter_source_matches(path: str, after_match_id: str | None = None):
    sql = "SELECT match_id,raw_json FROM matches WHERE game_mode='CLASSIC'"
    params: tuple[str, ...] = ()
    if after_match_id is not None:
        sql += " AND match_id > ?"
        params = (after_match_id,)
    sql += " ORDER BY match_id"
    with connect_source(path) as conn:
        for row in conn.execute(sql, params):
            yield row["match_id"], decode_raw_json(row["raw_json"])


def source_fingerprint(path: str, extractor_version: str) -> dict[str, str]:
    digest = hashlib.sha256(extractor_version.encode("utf-8"))
    with connect_source(path) as conn:
        schema = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        rows = conn.execute(
            "SELECT match_id,game_creation_time FROM matches "
            "WHERE game_mode='CLASSIC' ORDER BY match_id"
        )
        count = 0
        maximum = 0
        for match_id, created in rows:
            digest.update(f"{match_id}:{created or 0}\n".encode("utf-8"))
            count += 1
            maximum = max(maximum, int(created or 0))
    return {"dataset_version": digest.hexdigest(), "source_schema_version": str(schema),
            "source_match_count": str(count), "source_max_creation_ms": str(maximum)}


def source_snapshot(path: str) -> dict[str, str]:
    """Cheap request-time freshness token for the append-only operational DB."""
    with connect_source(path) as conn:
        schema = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        count, maximum = conn.execute(
            "SELECT COUNT(*),COALESCE(MAX(game_creation_time),0) FROM matches "
            "WHERE game_mode='CLASSIC'"
        ).fetchone()
    return {"source_schema_version": str(schema), "source_match_count": str(count),
            "source_max_creation_ms": str(maximum)}
```

- [ ] **Step 4: Implement normalization and stable extraction errors**

`analytics/extract.py` must define:

```python
from dataclasses import dataclass


REGIONS = {"NA1": "NA", "EUW1": "EUW", "EUN1": "EUN", "KR": "KR"}
ROLES = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


class ExtractionError(ValueError):
    def __init__(self, code: str, field: str):
        super().__init__(f"{code}: {field}")
        self.code = code
        self.field = field


@dataclass(frozen=True)
class ExtractionResult:
    match: dict
    teams: tuple[dict, dict]
    players: tuple[dict, ...]


def _objective(team: dict, key: str) -> tuple[int, int]:
    value = team.get("objectives", {}).get(key, {})
    return int(value.get("kills", 0)), int(bool(value.get("first", False)))


def _region(match_id: str) -> tuple[str, str]:
    prefix = match_id.split("_", 1)[0].upper()
    return prefix, REGIONS.get(prefix, "OTHER")


def _patch(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ExtractionError("INVALID_GAME_VERSION", "info.gameVersion")
    return f"{int(parts[0])}.{int(parts[1])}"
```

`extract_match()` validates `gameMode == "CLASSIC"`, exactly two teams with IDs 100/200, exactly ten participants with five per team, unique non-empty PUUIDs, and exactly one winner. It then creates player rows from `kills`, `deaths`, `assists`, `goldEarned`, `totalDamageDealtToChampions`, `visionScore`, `totalMinionsKilled`, and `neutralMinionsKilled`; creates team totals from those player rows; and creates objective/first-objective columns from the Riot team objects. Unknown `teamPosition` becomes `UNKNOWN` and is not an extraction failure.

- [ ] **Step 5: Run extractor tests**

Run: `python -m pytest tests/analytics/test_extract.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Commit source extraction**

```powershell
git add analytics/source.py analytics/extract.py tests/analytics/conftest.py tests/analytics/factories.py tests/analytics/test_extract.py
git commit -m "feat: normalize match facts for analytics"
```

## Task 4: Atomic Full Build, Incremental Update, and CLI

**Files:**
- Create: `analytics/build.py`
- Create: `analytics/cli.py`
- Create: `analytics/__main__.py`
- Create: `tests/analytics/test_build.py`
- Modify: `tests/analytics/conftest.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Tasks 2-3 database, source, and extractor interfaces.
- Produces: `BuildReport`, `build_analytics()`, `python -m analytics build|update|health`.

- [ ] **Step 1: Write integration tests for rebuild, idempotency, and source immutability**

```python
# tests/analytics/test_build.py
import hashlib
import sqlite3

from analytics.build import build_analytics
from analytics.db import connect_analytics


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_build_is_idempotent_and_does_not_write_source(source_db, tmp_path):
    target = tmp_path / "analytics.db"
    before = _sha(source_db)
    first = build_analytics(str(source_db), str(target))
    second = build_analytics(str(source_db), str(target))
    assert _sha(source_db) == before
    assert first.inserted_matches == second.inserted_matches == 1
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
    assert report.inserted_matches == 1
```

- [ ] **Step 2: Run integration tests and verify the builder is missing**

Run: `python -m pytest tests/analytics/test_build.py -q`

Expected: FAIL during collection for `analytics.build`.

- [ ] **Step 3: Implement transactional insertion and atomic rebuild**

```python
# analytics/build.py
from dataclasses import asdict, dataclass
import sqlite3
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .config import AnalyticsConfig
from .db import connect_analytics, initialize_schema, set_meta
from .extract import ExtractionError, extract_match
from .source import iter_source_matches, source_fingerprint


@dataclass(frozen=True)
class BuildReport:
    eligible_matches: int
    inserted_matches: int
    issue_counts: dict[str, int]
    unexplained_failures: int
    dataset_version: str
    quality_gate_passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _insert_result(conn, result) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO match_facts VALUES "
        "(:match_id,:region,:platform_prefix,:game_version,:patch,:queue_id,"
        ":game_creation_ms,:game_duration_s,:winner_team_id,:participant_count,"
        ":source_season_id)", result.match,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO team_facts VALUES "
        "(:match_id,:team_id,:side,:win,:kills,:deaths,:assists,:gold,"
        ":champion_damage,:vision_score,:lane_cs,:jungle_cs,:barons,:dragons,"
        ":towers,:inhibitors,:rift_heralds,:atakhan,:first_blood,:first_baron,"
        ":first_dragon,:first_tower,:first_inhibitor)", result.teams,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_facts VALUES "
        "(:match_id,:puuid,:team_id,:side,:win,:champion,:role,:kills,:deaths,"
        ":assists,:gold,:champion_damage,:vision_score,:lane_cs,:jungle_cs)",
        result.players,
    )
```

For a full build, create a temporary database beside the target, initialize it, insert all rows in batches of 500 matches, write metadata/build report, close it, and replace the target with `os.replace()`. For incremental mode, verify `extractor_version` matches, skip existing `match_id` values using a set loaded from `match_facts`, append new rows, recompute the source fingerprint, and clear `evidence_cache` because the dataset version changed. Record expected `ExtractionError` instances in `extraction_issues`; count other exceptions as unexplained. Refuse to publish the full-build temporary file when unexplained failures exceed 0.5% of eligible matches.

- [ ] **Step 4: Add explicit CLI commands**

```python
# analytics/cli.py
import argparse
import json
from .build import build_analytics
from .config import AnalyticsConfig
from .db import all_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analytics")
    parser.add_argument("--source", default=AnalyticsConfig().source_path)
    parser.add_argument("--analytics", default=AnalyticsConfig().analytics_path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("update")
    sub.add_parser("health")
    args = parser.parse_args(argv)
    if args.command in {"build", "update"}:
        report = build_analytics(args.source, args.analytics,
                                 incremental=args.command == "update")
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0 if report.quality_gate_passed else 2
    print(json.dumps(all_meta(args.analytics), indent=2, ensure_ascii=False))
    return 0
```

`analytics/__main__.py` imports `main` and exits with its return code. Add `*.db-wal`, `*.db-shm`, and `analytics_artifacts/` to `.gitignore`.

- [ ] **Step 5: Run builder and CLI tests**

Run: `python -m pytest tests/analytics/test_build.py -q`

Expected: `2 passed`.

Run: `python -m analytics --help`

Expected: usage containing `build`, `update`, and `health`.

- [ ] **Step 6: Commit the mart builder**

```powershell
git add analytics/build.py analytics/cli.py analytics/__main__.py tests/analytics/conftest.py tests/analytics/test_build.py .gitignore
git commit -m "feat: build analytics mart atomically"
```

## Task 5: Cohort Validation and Default Resolution

**Files:**
- Create: `analytics/cohorts.py`
- Create: `tests/analytics/test_cohorts.py`
- Modify: `tests/analytics/conftest.py`
- Modify: `tests/analytics/factories.py`

**Interfaces:**
- Consumes: `AnalysisRequest`, `ResolvedCohort`, registry fields, `analytics.db`.
- Produces: `CohortResolver.resolve()`, `CohortResolver.where_clause()`.

- [ ] **Step 1: Write default, widening, filter, and underpowered tests**

Extend `tests/analytics/factories.py` with `seed_fact_rows(path, patch_counts, region_shifts=None)`. It initializes the analytical schema and inserts deterministic two-team/ten-player matches with unique match IDs, at least 100 repeating PUUIDs, queue 420, increasing creation times, and winner/loser metrics derived from the requested shift. Expose `cohort_db`, `cohort_db_underpowered_latest`, and `cohort_db_tiny` in `conftest.py` with patch counts `{16.14: 500}`, `{16.14: 200, 16.13: 400}`, and `{16.14: 100, 16.13: 100, 16.12: 100}` respectively.

```python
# tests/analytics/test_cohorts.py
import pytest

from analytics.cohorts import AnalyticsAbstention, CohortResolver
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


def test_default_selects_latest_powered_ranked_patch(cohort_db):
    cohort = CohortResolver(str(cohort_db)).resolve(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    )
    assert cohort.queue_ids == (420,)
    assert cohort.patches == ("16.14",)
    assert cohort.widened is False


def test_default_widens_at_most_three_patches(cohort_db_underpowered_latest):
    cohort = CohortResolver(str(cohort_db_underpowered_latest)).resolve(
        AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    )
    assert cohort.patches == ("16.14", "16.13")
    assert cohort.widened is True


def test_rejects_unknown_region(cohort_db):
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold",
                              AnalysisFilters(regions=("MARS",)))
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        CohortResolver(str(cohort_db)).resolve(request)


def test_abstains_when_three_patches_remain_underpowered(cohort_db_tiny):
    with pytest.raises(AnalyticsAbstention) as caught:
        CohortResolver(str(cohort_db_tiny)).resolve(
            AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
        )
    assert caught.value.code == "INSUFFICIENT_MATCHES"
```

- [ ] **Step 2: Verify the cohort tests fail**

Run: `python -m pytest tests/analytics/test_cohorts.py -q`

Expected: FAIL during collection for `analytics.cohorts`.

- [ ] **Step 3: Implement allowlisted SQL construction and patch widening**

```python
# analytics/cohorts.py
from .config import AnalyticsConfig
from .db import connect_analytics
from .registry import get_metric
from .types import AnalysisRequest, ResolvedCohort


SUPPORTED_REGIONS = {"NA", "EUW", "EUN", "KR", "OTHER"}
SUPPORTED_ROLES = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "UNKNOWN"}


class AnalyticsAbstention(RuntimeError):
    def __init__(self, code: str, details: dict | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


class CohortResolver:
    def __init__(self, path: str, config: AnalyticsConfig | None = None):
        self.path = path
        self.config = config or AnalyticsConfig(analytics_path=path)

    def _validate(self, request: AnalysisRequest) -> None:
        if request.metric is not None:
            get_metric(request.metric)
        if set(request.filters.regions) - SUPPORTED_REGIONS:
            raise ValueError("UNSUPPORTED_FILTER: region")
        if request.filters.role and request.filters.role not in SUPPORTED_ROLES:
            raise ValueError("UNSUPPORTED_FILTER: role")
        if any(queue != 420 for queue in request.filters.queue_ids):
            raise ValueError("UNSUPPORTED_FILTER: queue")

    def resolve(self, request: AnalysisRequest) -> ResolvedCohort:
        self._validate(request)
        patches = request.filters.patches
        widened = False
        if not patches:
            with connect_analytics(self.path, readonly=True) as conn:
                available = [row[0] for row in conn.execute(
                    "SELECT patch FROM match_facts WHERE queue_id=420 "
                    "GROUP BY patch ORDER BY MAX(game_creation_ms) DESC"
                )]
            selected: list[str] = []
            for patch in available[:3]:
                selected.append(patch)
                matches, players = self._counts(request, tuple(selected))
                if matches >= self.config.min_matches and players >= self.config.min_players:
                    break
            patches = tuple(selected)
            widened = len(patches) > 1
        matches, players = self._counts(request, patches)
        if matches < self.config.min_matches:
            raise AnalyticsAbstention("INSUFFICIENT_MATCHES", {"matches": matches})
        if players < self.config.min_players:
            raise AnalyticsAbstention("INSUFFICIENT_PLAYERS", {"players": players})
        return ResolvedCohort("CLASSIC", request.filters.queue_ids, tuple(patches),
                              request.filters.regions, request.filters.date_start_ms,
                              request.filters.date_end_ms, request.filters.champion,
                              request.filters.role, widened, matches, players)
```

Implement `where_clause(request, patches, alias="m") -> tuple[str, list]` using only hard-coded column fragments and `?` parameters. Implement `_counts()` as `COUNT(DISTINCT m.match_id)` plus `COUNT(DISTINCT p.puuid)`; join `player_facts` only when player count or champion/role filters require it. Never interpolate user values or user field names.

- [ ] **Step 4: Run cohort tests**

Run: `python -m pytest tests/analytics/test_cohorts.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit cohort resolution**

```powershell
git add analytics/cohorts.py tests/analytics/conftest.py tests/analytics/factories.py tests/analytics/test_cohorts.py
git commit -m "feat: resolve powered analytics cohorts"
```

## Task 6: Statistical Primitives and Multiple-testing Correction

**Files:**
- Create: `analytics/statistics.py`
- Create: `tests/analytics/test_statistics.py`

**Interfaces:**
- Consumes: NumPy arrays and the configured seed/resample count.
- Produces: `continuous_effect()`, `binary_effect()`, `benjamini_hochberg()`.

- [ ] **Step 1: Write known-effect, null-effect, sparse-cell, and FDR tests**

```python
# tests/analytics/test_statistics.py
import numpy as np
import pytest

from analytics.statistics import binary_effect, benjamini_hochberg, continuous_effect


def test_continuous_effect_recovers_known_paired_difference():
    differences = np.array([2.0, 3.0, 4.0, 5.0] * 150)
    result = continuous_effect(differences, seed=7, resamples=2000)
    assert result.value == pytest.approx(3.5)
    assert result.confidence_interval_95[0] > 0
    assert result.p_value <= 0.01
    assert result.effect_size > 0.2


def test_binary_effect_uses_one_match_once():
    exposure = np.array([1] * 300 + [0] * 300)
    outcome = np.array([1] * 210 + [0] * 90 + [1] * 150 + [0] * 150)
    result = binary_effect(exposure, outcome)
    assert result.value == pytest.approx(0.20)
    assert result.events_by_group == {"exposed": 300, "comparison": 300}


def test_binary_effect_rejects_sparse_group():
    with pytest.raises(ValueError, match="SPARSE_COMPARISON_GROUP"):
        binary_effect(np.array([1] * 99 + [0] * 501), np.zeros(600), min_group=100)


def test_bh_is_monotone_in_original_order():
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04]
    )
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `python -m pytest tests/analytics/test_statistics.py -q`

Expected: FAIL during collection for `analytics.statistics`.

- [ ] **Step 3: Implement seeded paired and binary effects**

```python
# analytics/statistics.py
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import fisher_exact, norm


@dataclass(frozen=True)
class StatisticalResult:
    value: float
    baseline: float
    effect_size: float | None
    confidence_interval_95: tuple[float, float]
    p_value: float
    events_by_group: dict[str, int]
    odds_ratio: float | None = None


def continuous_effect(differences: np.ndarray, seed: int,
                      resamples: int = 2000) -> StatisticalResult:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("SOURCE_DATA_INVALID")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    permuted = np.empty(resamples, dtype=float)
    chunk_size = 64
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        indices = rng.integers(0, len(values), size=(size, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        permuted[start:stop] = np.abs((signs * values).mean(axis=1))
    ci = tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))
    observed = abs(float(values.mean()))
    p_value = (1.0 + float(np.count_nonzero(permuted >= observed))) / (resamples + 1.0)
    std = float(values.std(ddof=1))
    dz = float(values.mean()) / std if std > 1e-12 else None
    return StatisticalResult(float(values.mean()), 0.0, dz, ci, p_value,
                             {"matches": len(values)})


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = norm.ppf(0.975)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def binary_effect(exposure: np.ndarray, outcome: np.ndarray,
                  min_group: int = 100) -> StatisticalResult:
    x = np.asarray(exposure, dtype=int)
    y = np.asarray(outcome, dtype=int)
    if x.shape != y.shape or not set(np.unique(x)) <= {0, 1} or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("SOURCE_DATA_INVALID")
    exposed = y[x == 1]
    comparison = y[x == 0]
    if min(len(exposed), len(comparison)) < min_group:
        raise ValueError("SPARSE_COMPARISON_GROUP")
    a, b = int(exposed.sum()), int(len(exposed) - exposed.sum())
    c, d = int(comparison.sum()), int(len(comparison) - comparison.sum())
    exposed_ci, comparison_ci = _wilson(a, len(exposed)), _wilson(c, len(comparison))
    difference = a / len(exposed) - c / len(comparison)
    newcombe = (exposed_ci[0] - comparison_ci[1], exposed_ci[1] - comparison_ci[0])
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return StatisticalResult(difference, c / len(comparison), difference,
                             newcombe, float(p_value),
                             {"exposed": len(exposed), "comparison": len(comparison)},
                             float(odds_ratio))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted
```

- [ ] **Step 4: Run statistical tests twice to prove deterministic results**

Run: `python -m pytest tests/analytics/test_statistics.py -q && python -m pytest tests/analytics/test_statistics.py -q`

Expected: both runs report `4 passed`.

- [ ] **Step 5: Commit the statistical primitives**

```powershell
git add analytics/statistics.py tests/analytics/test_statistics.py
git commit -m "feat: add reproducible evidence statistics"
```

## Task 7: Evidence Construction, Thresholds, Robustness, and Analysis Engine

**Files:**
- Create: `analytics/evidence.py`
- Create: `analytics/engine.py`
- Create: `tests/analytics/test_evidence.py`
- Create: `tests/analytics/test_engine.py`
- Modify: `tests/analytics/conftest.py`
- Modify: `tests/analytics/factories.py`

**Interfaces:**
- Consumes: resolved cohorts, registry specs, statistical results, cache repository.
- Produces: `classify_recommendation()`, `build_evidence()`, `AnalysisEngine.run()`.

- [ ] **Step 1: Write exact threshold and stability tests**

Add `powered_analytics_db` to `conftest.py` using deterministic facts for three patches and four regions, with at least 2,100 matches, at least 300 distinct PUUIDs, known nonzero effects for `gold` and `vision_score`, and null effects for at least one registered metric. This single fixture backs evidence, engine, region, model-data, service, and benchmark tests without reading the real corpus.

```python
# tests/analytics/test_evidence.py
from analytics.evidence import classify_recommendation


def test_moderate_requires_all_statistical_and_practical_gates():
    result = classify_recommendation(matches=800, players=300,
        group_counts=(300, 500), ci=(0.02, 0.08), q_value=0.01,
        effect_size=0.25, win_rate_difference=None,
        temporal_directions=(1, 1, -1))
    assert result == {"allowed": True, "strength": "moderate",
                      "wording_key": "association_moderate"}


def test_strong_requires_all_available_temporal_folds():
    result = classify_recommendation(matches=2500, players=900,
        group_counts=(1200, 1300), ci=(0.51, 0.70), q_value=0.005,
        effect_size=0.60, win_rate_difference=None,
        temporal_directions=(1, 1, -1))
    assert result["strength"] == "moderate"


def test_ci_touching_null_abstains():
    result = classify_recommendation(matches=2500, players=900,
        group_counts=(1200, 1300), ci=(0.0, 0.70), q_value=0.001,
        effect_size=0.60, win_rate_difference=None,
        temporal_directions=(1, 1, 1))
    assert result["allowed"] is False
```

```python
# tests/analytics/test_engine.py
from analytics.engine import AnalysisEngine
from analytics.types import AnalysisFilters, AnalysisRequest, Intent


def test_engine_applies_bh_to_complete_candidate_family(powered_analytics_db):
    packets = AnalysisEngine(str(powered_analytics_db)).run(
        AnalysisRequest(Intent.RANK_CONDITIONS),
        question="Comment gagne-t-on ?",
    )
    assert len(packets) == 19
    assert all(packet.estimate["q_value"] is not None for packet in packets)
    assert all(packet.provenance["dataset_version"] for packet in packets)


def test_same_dataset_request_and_seed_reproduce_evidence_id(powered_analytics_db):
    engine = AnalysisEngine(str(powered_analytics_db))
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "gold")
    first = engine.run(request, question="L'or est-il associé à la victoire ?")[0]
    second = engine.run(request, question="L'or est-il associé à la victoire ?")[0]
    assert first.evidence_id == second.evidence_id
    assert first.to_dict() == second.to_dict()


def test_compare_patches_returns_contrast_not_two_unrelated_rankings(powered_analytics_db):
    request = AnalysisRequest(Intent.COMPARE_PATCHES, "gold",
                              AnalysisFilters(patches=("16.13", "16.14")))
    packet = AnalysisEngine(str(powered_analytics_db)).run(
        request, question="L'effet de l'or a-t-il changé ?"
    )[0]
    assert packet.estimate["metric"] == "gold_patch_contrast"
    assert packet.sample["events_by_group"]["16.13"] >= 100
    assert packet.sample["events_by_group"]["16.14"] >= 100
```

- [ ] **Step 2: Run the tests and verify evidence modules are missing**

Run: `python -m pytest tests/analytics/test_evidence.py tests/analytics/test_engine.py -q`

Expected: FAIL during collection for `analytics.evidence`.

- [ ] **Step 3: Implement recommendation classification exactly once**

```python
# analytics/evidence.py
import hashlib
import json
from .types import EvidencePacket


def classify_recommendation(*, matches: int, players: int,
        group_counts: tuple[int, ...], ci: tuple[float, float], q_value: float,
        effect_size: float | None, win_rate_difference: float | None,
        temporal_directions: tuple[int, ...]) -> dict:
    excludes_null = ci[0] > 0 or ci[1] < 0
    practical = ((win_rate_difference is not None and abs(win_rate_difference) >= 0.03)
                 or (effect_size is not None and abs(effect_size) >= 0.20))
    nonzero = [direction for direction in temporal_directions if direction != 0]
    stable_two = len(nonzero) >= 2 and max(nonzero.count(1), nonzero.count(-1)) >= 2
    moderate = (matches >= 500 and players >= 100 and min(group_counts) >= 100
                and excludes_null and q_value <= 0.05 and practical and stable_two)
    if not moderate:
        return {"allowed": False, "strength": "none", "wording_key": "descriptive_only"}
    direction = 1 if sum(nonzero) > 0 else -1
    strong_practical = ((win_rate_difference is not None and abs(win_rate_difference) >= 0.05)
                        or (effect_size is not None and abs(effect_size) >= 0.50))
    strong = (matches >= 2000 and q_value <= 0.01 and strong_practical
              and len(nonzero) == len(temporal_directions)
              and all(value == direction for value in nonzero))
    return {"allowed": True, "strength": "strong" if strong else "moderate",
            "wording_key": "association_strong" if strong else "association_moderate"}


def stable_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

`build_evidence()` fills every field in the approved JSON contract. It sets unavailable statistical values to `None` plus a limitation, computes `source_match_ids_hash` from sorted match IDs, and computes the cache request hash from `{"analysis_request": request.to_dict(), "question": question.strip()}`. It computes `evidence_id` from dataset version, that request hash, method version, seed, and estimate. `generated_at` is read from the cached packet on replay so the same question/request remains byte-for-byte stable; a different wording may reuse the same lower-level arrays but receives its own packet and question field.

- [ ] **Step 4: Implement one-observation-per-match engine queries**

For continuous metrics, query one joined winner/loser row per `match_id` and pass `winner_metric - loser_metric` to `continuous_effect()`. For binary metrics, query one blue-team row per match, pass the blue exposure and blue win target to `binary_effect()`. Use `CohortResolver.where_clause()` and the metric registry column, never request text. Split ordered matches into three chronological folds, recompute the sign in each fold, apply BH to every candidate in the predeclared family, then construct/cache packets only after corrected q-values are available.

For `COMPARE_PATCHES`, require exactly two explicitly selected patches, compute the metric effect independently within each patch using the same one-row-per-match method, bootstrap the difference between patch effects in seeded chunks, permute patch labels for the contrast p-value, and expose the result as `<metric>_patch_contrast`. Both patches must independently meet 500 matches, 100 players, and 100 group events; otherwise abstain instead of widening an explicit comparison.

`AnalysisEngine.run()` resolves the cohort internally when one is not supplied and supports:

```python
FAMILIES = {
    "rank_conditions": tuple(METRICS),
    "explain_metric": None,
    "compare_regions": None,
    "compare_patches": None,
}
```

It raises `AnalyticsAbstention("SPARSE_COMPARISON_GROUP")` for sparse binary cells and returns descriptive evidence with `recommendation.allowed == False` for effects that fail robustness.

- [ ] **Step 5: Run evidence and engine tests**

Run: `python -m pytest tests/analytics/test_evidence.py tests/analytics/test_engine.py -q`

Expected: `6 passed`.

- [ ] **Step 6: Commit evidence generation**

```powershell
git add analytics/evidence.py analytics/engine.py tests/analytics/conftest.py tests/analytics/factories.py tests/analytics/test_evidence.py tests/analytics/test_engine.py
git commit -m "feat: generate thresholded evidence packets"
```

## Task 8: Robust Inter-region Comparison

**Files:**
- Create: `analytics/regions.py`
- Create: `tests/analytics/test_regions.py`
- Modify: `analytics/engine.py`

**Interfaces:**
- Consumes: metric observations grouped by patch and region.
- Produces: `compare_regions() -> RegionalComparison`, `regional_display_allowed()`.

- [ ] **Step 1: Write powered and non-robust region tests**

```python
# tests/analytics/test_regions.py
import numpy as np

from analytics.regions import RegionalObservation, compare_regions


def test_patch_adjusted_region_difference_passes_all_gates():
    rows = []
    rng = np.random.default_rng(3)
    for region, shift in (("EUW", 0.35), ("KR", 0.0)):
        for patch in ("16.13", "16.14"):
            for index in range(600):
                rows.append(RegionalObservation(region, patch,
                    shift + rng.normal(0, 1), index, None))
    result = compare_regions(rows, seed=11, resamples=2000)
    assert result.display_allowed is True
    assert result.q_value <= 0.05
    assert abs(result.standardized_contrast) >= 0.20


def test_region_ranking_is_hidden_when_one_region_is_small():
    rows = [RegionalObservation("EUW", "16.14", 0.4, i, None)
            for i in range(600)]
    rows += [RegionalObservation("KR", "16.14", 0.0, i, None)
             for i in range(499)]
    result = compare_regions(rows, seed=11, resamples=2000)
    assert result.display_allowed is False
    assert "INSUFFICIENT_MATCHES" in result.limitations
```

- [ ] **Step 2: Verify region tests fail**

Run: `python -m pytest tests/analytics/test_regions.py -q`

Expected: FAIL during collection for `analytics.regions`.

- [ ] **Step 3: Implement patch-standardized contrasts and sensitivity gate**

```python
# analytics/regions.py
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np


@dataclass(frozen=True)
class RegionalObservation:
    region: str
    patch: str
    value: float
    game_creation_ms: int
    outcome: int | None


@dataclass(frozen=True)
class RegionalComparison:
    region_effects: dict[str, float]
    standardized_contrast: float
    confidence_interval_95: tuple[float, float]
    p_value: float
    q_value: float
    display_allowed: bool
    limitations: tuple[str, ...]


def _patch_standardized(rows):
    by_patch = defaultdict(list)
    for row in rows:
        by_patch[row.patch].append(row)
    patch_weights = {patch: len(values) for patch, values in by_patch.items()}
    total = sum(patch_weights.values())
    regions = sorted({row.region for row in rows})
    effects = {}
    for region in regions:
        weighted = []
        for patch, values in by_patch.items():
            selected = [row.value for row in values if row.region == region]
            if selected:
                weighted.append((patch_weights[patch] / total, float(np.mean(selected))))
        effects[region] = sum(weight * value for weight, value in weighted) / sum(
            weight for weight, _ in weighted
        )
    return effects
```

`compare_regions()` requires exactly two requested/displayed regions per contrast, resamples matches within patch and region 2,000 times, and permutes region labels within patch for the interaction p-value. It standardizes the effect using the pooled within-region standard deviation. It computes recent-period sensitivity from the newest half of `game_creation_ms`; `outcome` is `None` for continuous comparisons and the blue-win target for binary comparisons. It returns `display_allowed=True` only when both regions meet counts, each has at least 100 binary outcomes of both classes when the method is binary, q/effect gates pass, the CI excludes zero, and the recent direction matches. `AnalysisEngine` applies BH across all requested region pairs before rendering any rank.

- [ ] **Step 4: Run region and regression tests**

Run: `python -m pytest tests/analytics/test_regions.py tests/analytics/test_engine.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit regional evidence**

```powershell
git add analytics/regions.py analytics/engine.py tests/analytics/test_regions.py
git commit -m "feat: gate patch-adjusted regional evidence"
```

## Task 9: Named Model Matrix and Leakage Enforcement

**Files:**
- Create: `analytics/model_data.py`
- Create: `tests/analytics/test_model_data.py`
- Modify: `tests/analytics/conftest.py`

**Interfaces:**
- Consumes: `match_facts`, `team_facts`, and `player_facts`.
- Produces: `ModelDataset`, `load_model_dataset()`, `assert_allowed_features()`.

- [ ] **Step 1: Write side-swap, feature-name, and leakage tests**

```python
# tests/analytics/test_model_data.py
import numpy as np
import pytest

from analytics.model_data import (MODEL_FEATURES, assert_allowed_features,
                                  load_model_dataset, rows_to_matrix)


def test_matrix_is_one_blue_minus_red_row_per_match(powered_analytics_db):
    dataset = load_model_dataset(str(powered_analytics_db))
    assert dataset.x.shape == (dataset.match_ids.shape[0], len(MODEL_FEATURES))
    assert set(np.unique(dataset.y)) <= {0, 1}
    assert "win" not in dataset.feature_names
    assert "winner_team_id" not in dataset.feature_names


def test_swapping_sides_negates_features_and_target():
    blue = {name: float(index + 1) for index, name in enumerate(MODEL_FEATURES)}
    red = {name: 0.0 for name in MODEL_FEATURES}
    forward = rows_to_matrix([("m1", blue, red, 1, "EUW", "16.14", 1)])
    swapped = rows_to_matrix([("m1", red, blue, 0, "EUW", "16.14", 1)])
    assert np.allclose(forward.x, -swapped.x)
    assert forward.y[0] == 1 - swapped.y[0]


def test_forbidden_feature_fails_before_fit():
    with pytest.raises(ValueError, match="TARGET_FORBIDDEN"):
        assert_allowed_features(("gold_delta", "win"))
```

- [ ] **Step 2: Verify the model-data tests fail**

Run: `python -m pytest tests/analytics/test_model_data.py -q`

Expected: FAIL during collection for `analytics.model_data`.

- [ ] **Step 3: Implement the immutable named retrospective matrix**

```python
# analytics/model_data.py
from dataclasses import dataclass
import numpy as np
from .db import connect_analytics


MODEL_FEATURES = (
    "kills", "deaths", "assists", "gold", "champion_damage", "vision_score",
    "lane_cs", "jungle_cs", "barons", "dragons", "towers", "inhibitors",
    "rift_heralds", "atakhan", "first_blood", "first_baron", "first_dragon",
    "first_tower", "first_inhibitor",
)
FORBIDDEN_FEATURES = {"win", "winner_team_id", "target", "winner_label"}


@dataclass(frozen=True)
class ModelDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    match_ids: np.ndarray
    creation_ms: np.ndarray
    regions: np.ndarray
    patches: np.ndarray
    match_puuids: tuple[frozenset[str], ...]


def assert_allowed_features(names: tuple[str, ...]) -> None:
    forbidden = set(names) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError("TARGET_FORBIDDEN: " + ",".join(sorted(forbidden)))
    unknown = set(names) - {f"{name}_delta" for name in MODEL_FEATURES}
    if unknown:
        raise ValueError("UNREGISTERED_MODEL_FEATURE: " + ",".join(sorted(unknown)))


def rows_to_matrix(rows) -> ModelDataset:
    x = np.asarray([[blue[name] - red[name] for name in MODEL_FEATURES]
                    for _, blue, red, _, _, _, _ in rows], dtype=float)
    names = tuple(f"{name}_delta" for name in MODEL_FEATURES)
    assert_allowed_features(names)
    return ModelDataset(
        x=x,
        y=np.asarray([blue_win for _, _, _, blue_win, _, _, _ in rows], dtype=int),
        feature_names=names,
        match_ids=np.asarray([match_id for match_id, *_ in rows]),
        creation_ms=np.asarray([created for *_, created in rows], dtype=np.int64),
        regions=np.asarray([region for _, _, _, _, region, _, _ in rows]),
        patches=np.asarray([patch for _, _, _, _, _, patch, _ in rows]),
        match_puuids=tuple(frozenset() for _ in rows),
    )
```

Keep `rows_to_matrix` at module level as shown. `load_model_dataset()` queries one blue and one red team per match, orders by `(game_creation_ms, match_id)`, loads participant PUUID sets with one additional indexed query, checks finite values, and returns no `win` or winner indicator inside `x`. Region and patch stay separate as control variables.

- [ ] **Step 4: Run model-data tests**

Run: `python -m pytest tests/analytics/test_model_data.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the leakage-safe model dataset**

```powershell
git add analytics/model_data.py tests/analytics/conftest.py tests/analytics/test_model_data.py
git commit -m "feat: build named retrospective model matrix"
```

## Task 10: Chronological and Player-disjoint Interpretable Evaluation

**Files:**
- Create: `analytics/splits.py`
- Create: `analytics/interpretable.py`
- Create: `tests/analytics/test_splits.py`
- Create: `tests/analytics/test_interpretable.py`
- Modify: `analytics/cli.py`
- Modify: `tests/analytics/conftest.py`

**Interfaces:**
- Consumes: `ModelDataset`.
- Produces: `chronological_split()`, `player_disjoint_split()`, `train_and_evaluate() -> ModelReport`.

- [ ] **Step 1: Write exact split and baseline tests**

Expose `model_db` from the same deterministic fact factory with 4,000 chronologically ordered matches, both outcome classes in every temporal segment, a known positive gold coefficient, a null vision coefficient, and repeated PUUIDs that make the player-disjoint test set underpowered at a 500-match threshold.

```python
# tests/analytics/test_splits.py
import numpy as np

from analytics.splits import chronological_split, player_disjoint_split


def test_chronological_split_is_70_15_15_without_shuffle():
    train, validation, test = chronological_split(np.arange(100))
    assert train.tolist() == list(range(70))
    assert validation.tolist() == list(range(70, 85))
    assert test.tolist() == list(range(85, 100))


def test_player_disjoint_split_has_no_shared_puuid():
    puuids = (frozenset({"a", "bridge"}), frozenset({"c"}),
              frozenset({"d"}), frozenset({"bridge", "z"}))
    split = player_disjoint_split(puuids, min_test_matches=1)
    train_players = set().union(*(puuids[index] for index in split.train))
    test_players = set().union(*(puuids[index] for index in split.test))
    assert train_players.isdisjoint(test_players)
    assert 0 in split.discarded
```

```python
# tests/analytics/test_interpretable.py
import json

from analytics.interpretable import train_and_evaluate


def test_report_includes_baselines_calibration_and_untouched_test(model_db, tmp_path):
    report = train_and_evaluate(str(model_db), str(tmp_path / "artifacts"))
    assert report.dataset_version
    assert set(report.metrics) >= {"majority", "mean_probability", "logistic"}
    for name in ("log_loss", "brier", "calibration_error", "roc_auc"):
        assert name in report.metrics["logistic"]
    assert "gold_delta*kills_delta" in report.coefficients
    assert len(report.coefficient_intervals["gold_delta*kills_delta"]) == 2
    assert report.direction_agreement["gold_delta"] in {"agrees", "conditional_only"}
    assert report.split_counts["test"] > 0
    saved = json.loads((tmp_path / "artifacts" / "model_report.json").read_text())
    assert saved["retrospective"] is True


def test_grouped_result_reports_underpowered_instead_of_fallback(model_db, tmp_path):
    report = train_and_evaluate(str(model_db), str(tmp_path / "artifacts"),
                                min_grouped_test_matches=500)
    assert report.player_disjoint["available"] is False
    assert report.player_disjoint["code"] == "INSUFFICIENT_MATCHES"
```

- [ ] **Step 2: Run model tests and verify modules are missing**

Run: `python -m pytest tests/analytics/test_splits.py tests/analytics/test_interpretable.py -q`

Expected: FAIL during collection for `analytics.splits`.

- [ ] **Step 3: Implement chronological and connected-component player splits**

```python
# analytics/splits.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IndexSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class GroupedSplit:
    train: np.ndarray
    test: np.ndarray
    discarded: np.ndarray
    available: bool
    code: str | None


def chronological_split(order: np.ndarray) -> IndexSplit:
    indices = np.asarray(order, dtype=int)
    first = int(len(indices) * 0.70)
    second = int(len(indices) * 0.85)
    return IndexSplit(indices[:first], indices[first:second], indices[second:])


def player_disjoint_split(match_puuids: tuple[frozenset[str], ...],
                          min_test_matches: int = 500) -> GroupedSplit:
    target_test = max(min_test_matches, int(len(match_puuids) * 0.15))
    test_start = max(0, len(match_puuids) - target_test)
    test = list(range(test_start, len(match_puuids)))
    test_players = set().union(*(match_puuids[index] for index in test)) if test else set()
    train, discarded = [], []
    for index in range(test_start):
        if match_puuids[index] & test_players:
            discarded.append(index)
        else:
            train.append(index)
    available = len(test) >= min_test_matches and bool(train)
    return GroupedSplit(np.asarray(train), np.asarray(test), np.asarray(discarded),
                        available, None if available else "INSUFFICIENT_MATCHES")
```

- [ ] **Step 4: Implement fixed-candidate logistic selection and final evaluation**

Use a scikit-learn `ColumnTransformer` that standardizes named numeric deltas and one-hot encodes region and patch controls, followed by `LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000, random_state=seed)`. Pre-register only three gameplay interactions—`gold_delta*kills_delta`, `vision_score_delta*dragons_delta`, and `lane_cs_delta*towers_delta`—and create them by name before the transformer; do not run unrestricted polynomial expansion. Select `C` from `(0.01, 0.1, 1.0, 10.0)` using validation log loss only. Refit on train+validation with the selected `C`, then evaluate the untouched final 15% once.

```python
# analytics/interpretable.py
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class ModelReport:
    dataset_version: str
    retrospective: bool
    selected_c: float
    split_counts: dict[str, int]
    metrics: dict[str, dict[str, float | list[float]]]
    coefficients: dict[str, float]
    coefficient_intervals: dict[str, list[float]]
    direction_agreement: dict[str, str]
    player_disjoint: dict


def calibration_error(y_true, probabilities, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if right == 1.0:
            mask |= probabilities == 1.0
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(error)


def evaluate_probabilities(y_true, probabilities) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "calibration_error": calibration_error(y_true, probabilities),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
```

Add seeded 2,000-match bootstrap confidence intervals for every metric and every coefficient, majority and train-mean probability baselines, coefficient/odds-ratio export with transformed feature names, dataset-version and seed metadata, and the player-disjoint sensitivity result. Compare every base-feature coefficient direction with the paired descriptive effect; record `agrees`, `conditional_only`, or `descriptive_unavailable`, and never promote `conditional_only` to a recommendation. Write `interpretable_model.joblib` and `model_report.json` atomically under the requested artifact directory. Do not print or persist a model performance claim when the final test has fewer than 500 matches; return an underpowered report instead.

- [ ] **Step 5: Add `train-model` CLI command**

`python -m analytics --analytics analytics.db train-model --artifacts analytics_artifacts` calls `train_and_evaluate()`, prints the complete report JSON, and returns exit code 2 when the final test is underpowered.

- [ ] **Step 6: Run model tests**

Run: `python -m pytest tests/analytics/test_splits.py tests/analytics/test_interpretable.py -q`

Expected: `4 passed`.

- [ ] **Step 7: Commit interpretable evaluation**

```powershell
git add analytics/splits.py analytics/interpretable.py analytics/cli.py tests/analytics/conftest.py tests/analytics/test_splits.py tests/analytics/test_interpretable.py
git commit -m "feat: evaluate interpretable retrospective model"
```

## Task 11: Discovery-only NN Adapter and Independent Validation Gate

**Files:**
- Create: `analytics/nn_adapter.py`
- Create: `tests/analytics/test_nn_adapter.py`

**Interfaces:**
- Consumes: injectable NN discovery function and `AnalysisEngine` validation callback.
- Produces: `DiscoveryCandidate`, `NNDiscoveryAdapter.discover()`, `CandidateValidationGate.validate()`.

- [ ] **Step 1: Write tests that prevent NN evidence bypass**

```python
# tests/analytics/test_nn_adapter.py
from analytics.nn_adapter import CandidateValidationGate, DiscoveryCandidate, NNDiscoveryAdapter


def test_current_checkpoint_is_always_labeled_discovery_only():
    adapter = NNDiscoveryAdapter(lambda rows: [{"metric": "gold", "score": 0.8}],
                                 checkpoint="model_v2.keras")
    candidate = adapter.discover([{"match_id": "m1"}])[0]
    assert candidate.discovery_only is True
    assert candidate.checkpoint_independently_validated is False


def test_candidate_cannot_validate_on_discovery_matches():
    candidate = DiscoveryCandidate("gold", {}, ("m1", "m2"), True, False)
    gate = CandidateValidationGate(lambda request, excluded: {"allowed": True})
    result = gate.validate(candidate, validation_match_ids={"m2", "m3"})
    assert result.allowed is False
    assert result.code == "DISCOVERY_VALIDATION_OVERLAP"


def test_candidate_only_returns_registered_engine_evidence():
    candidate = DiscoveryCandidate("gold", {"region": "EUW"}, ("m1",), True, False)
    gate = CandidateValidationGate(
        lambda request, excluded: {"evidence_id": "ev-1", "allowed": True}
    )
    result = gate.validate(candidate, validation_match_ids={"m2", "m3"})
    assert result.allowed is True
    assert result.evidence_id == "ev-1"
```

- [ ] **Step 2: Verify the adapter tests fail**

Run: `python -m pytest tests/analytics/test_nn_adapter.py -q`

Expected: FAIL during collection for `analytics.nn_adapter`.

- [ ] **Step 3: Implement the narrow discovery contract**

```python
# analytics/nn_adapter.py
from dataclasses import dataclass
from typing import Callable
from .registry import get_metric


@dataclass(frozen=True)
class DiscoveryCandidate:
    metric: str
    filters: dict
    discovery_match_ids: tuple[str, ...]
    discovery_only: bool
    checkpoint_independently_validated: bool


@dataclass(frozen=True)
class CandidateValidation:
    allowed: bool
    code: str | None
    evidence_id: str | None


class NNDiscoveryAdapter:
    def __init__(self, predict_candidates: Callable, checkpoint: str):
        self.predict_candidates = predict_candidates
        self.checkpoint = checkpoint

    def discover(self, rows: list[dict]) -> tuple[DiscoveryCandidate, ...]:
        match_ids = tuple(sorted({row["match_id"] for row in rows}))
        candidates = []
        for raw in self.predict_candidates(rows):
            get_metric(raw["metric"])
            candidates.append(DiscoveryCandidate(raw["metric"], raw.get("filters", {}),
                match_ids, True, False))
        return tuple(candidates)


class CandidateValidationGate:
    def __init__(self, validate_with_engine: Callable):
        self.validate_with_engine = validate_with_engine

    def validate(self, candidate: DiscoveryCandidate,
                 validation_match_ids: set[str]) -> CandidateValidation:
        if set(candidate.discovery_match_ids) & validation_match_ids:
            return CandidateValidation(False, "DISCOVERY_VALIDATION_OVERLAP", None)
        result = self.validate_with_engine(
            {"metric": candidate.metric, "filters": candidate.filters},
            set(candidate.discovery_match_ids),
        )
        return CandidateValidation(bool(result.get("allowed")),
                                   None if result.get("allowed") else "EFFECT_NOT_ROBUST",
                                   result.get("evidence_id"))
```

The adapter receives no evidence repository and exposes no rendering method. Document in the class docstring that `model_v2.keras` is post-match, hashed, trained on all currently marked rows, and therefore cannot carry an independent performance claim.

- [ ] **Step 4: Run NN boundary tests**

Run: `python -m pytest tests/analytics/test_nn_adapter.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the NN safety boundary**

```powershell
git add analytics/nn_adapter.py tests/analytics/test_nn_adapter.py
git commit -m "feat: constrain neural discovery candidates"
```

## Task 12: Deterministic French Planner and Safe Follow-up Context

**Files:**
- Create: `analytics/planner.py`
- Create: `tests/analytics/test_planner.py`

**Interfaces:**
- Consumes: user message, optional prior `AnalysisRequest`, explicit filters.
- Produces: validated `AnalysisRequest` or `AnalyticsAbstention`.

- [ ] **Step 1: Write supported-intent, follow-up, and refusal tests**

```python
# tests/analytics/test_planner.py
import pytest

from analytics.cohorts import AnalyticsAbstention
from analytics.planner import FrenchPlanner
from analytics.types import Intent


def test_parses_metric_and_region_without_sql():
    request = FrenchPlanner().plan("La vision aide-t-elle à gagner en EUW ?")
    assert request.intent is Intent.EXPLAIN_METRIC
    assert request.metric == "vision_score"
    assert request.filters.regions == ("EUW",)


def test_region_followup_only_replaces_region():
    planner = FrenchPlanner()
    first = planner.plan("La vision aide-t-elle à gagner en EUW ?")
    second = planner.plan("Et en KR ?", context=first.to_dict())
    assert second.metric == "vision_score"
    assert second.filters.regions == ("KR",)


def test_evidence_followup_keeps_validated_request():
    planner = FrenchPlanner()
    first = planner.plan("La vision aide-t-elle à gagner en EUW ?")
    second = planner.plan("Montre les preuves", context=first.to_dict())
    assert second.intent is Intent.SHOW_EVIDENCE
    assert second.metric == "vision_score"
    assert second.filters.regions == ("EUW",)


def test_parses_explicit_patch_comparison():
    request = FrenchPlanner().plan(
        "Compare l'or entre les patchs 16.13 et 16.14"
    )
    assert request.intent is Intent.COMPARE_PATCHES
    assert request.filters.patches == ("16.13", "16.14")


@pytest.mark.parametrize("message,code", [
    ("Qui gagnait à 10 minutes ?", "TIMELINE_DATA_UNAVAILABLE"),
    ("Décris le style de jeu de G2", "ESPORT_TEAM_IDENTITY_UNAVAILABLE"),
    ("Prouve que prendre le dragon cause la victoire", "UNSUPPORTED_INTENT"),
    ("Donne-moi une stratégie de pari certaine", "UNSUPPORTED_INTENT"),
])
def test_refuses_unsupported_claims(message, code):
    with pytest.raises(AnalyticsAbstention) as caught:
        FrenchPlanner().plan(message)
    assert caught.value.code == code


def test_context_cannot_contain_sql():
    with pytest.raises(ValueError, match="UNSUPPORTED_FILTER"):
        FrenchPlanner().plan("Et en KR ?", context={"sql": "DROP TABLE match_facts"})
```

- [ ] **Step 2: Verify planner tests fail**

Run: `python -m pytest tests/analytics/test_planner.py -q`

Expected: FAIL during collection for `analytics.planner`.

- [ ] **Step 3: Implement ordered, allowlisted French parsing**

```python
# analytics/planner.py
import re
import unicodedata
from .cohorts import AnalyticsAbstention
from .registry import METRICS
from .types import AnalysisFilters, AnalysisRequest, Intent


REGION_PATTERNS = {"EUW": r"\beuw\b", "EUN": r"\beun\b",
                   "NA": r"\bna\b", "KR": r"\bkr\b|\bcor[eé]e\b"}
TIMELINE_PATTERN = re.compile(r"\b(5|10|15)\s*minutes?\b|\bearly\b", re.I)
ESPORT_PATTERN = re.compile(r"\b(team|équipe|g2|fnatic|t1|gentle mates|karmine)\b", re.I)
CAUSAL_PATTERN = re.compile(r"\b(cause|causal|garantit|prouve|certain|pari)\b", re.I)
PATCH_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\b")
FILTER_FIELDS = {"regions", "patches", "queue_ids", "date_start_ms",
                 "date_end_ms", "champion", "role"}


def _normalized(message: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", message.lower())
                   if not unicodedata.combining(char))


def _contains_alias(text: str, alias: str) -> bool:
    value = re.escape(_normalized(alias))
    return re.search(rf"(?<!\w){value}(?!\w)", text) is not None


class FrenchPlanner:
    def plan(self, message: str, context: dict | None = None,
             filters: dict | None = None) -> AnalysisRequest:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("UNSUPPORTED_INTENT: empty message")
        if context is not None and set(context) - {"intent", "metric", "filters"}:
            raise ValueError("UNSUPPORTED_FILTER: context")
        if context is not None and set(context.get("filters", {})) - FILTER_FIELDS:
            raise ValueError("UNSUPPORTED_FILTER: context.filters")
        if filters is not None and set(filters) - FILTER_FIELDS:
            raise ValueError("UNSUPPORTED_FILTER: filters")
        if TIMELINE_PATTERN.search(message):
            raise AnalyticsAbstention("TIMELINE_DATA_UNAVAILABLE")
        if ESPORT_PATTERN.search(message):
            raise AnalyticsAbstention("ESPORT_TEAM_IDENTITY_UNAVAILABLE")
        if CAUSAL_PATTERN.search(message):
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")
        previous = AnalysisRequest.from_dict(context) if context else None
        normalized = _normalized(message)
        if ("preuve" in normalized or "limite" in normalized) and previous is None:
            raise AnalyticsAbstention("UNSUPPORTED_INTENT")
        regions = tuple(region for region, pattern in REGION_PATTERNS.items()
                        if re.search(pattern, normalized, re.I))
        patches = tuple(dict.fromkeys(PATCH_PATTERN.findall(normalized)))
        metric = next((spec.name for spec in METRICS.values()
                       if any(_contains_alias(normalized, alias)
                              for alias in spec.aliases)), None)
        if previous and normalized.startswith(("et ", "qu'en ", "et en ")):
            metric = metric or previous.metric
        if previous and ("preuve" in normalized or "limite" in normalized):
            metric = previous.metric
        intent = Intent.EXPLAIN_METRIC if metric else Intent.RANK_CONDITIONS
        if "preuve" in normalized:
            intent = Intent.SHOW_EVIDENCE
        elif "limite" in normalized:
            intent = Intent.SHOW_LIMITATIONS
        elif len(patches) == 2:
            intent = Intent.COMPARE_PATCHES
        if len(regions) >= 2 or "compare" in normalized or "difference" in normalized:
            if len(patches) != 2:
                intent = Intent.COMPARE_REGIONS
        explicit = AnalysisFilters.from_dict(filters)
        previous_filters = previous.filters if previous else AnalysisFilters()
        resolved = AnalysisFilters(
            regions=regions or explicit.regions or previous_filters.regions,
            patches=patches or explicit.patches or previous_filters.patches,
            queue_ids=explicit.queue_ids,
            date_start_ms=(explicit.date_start_ms if explicit.date_start_ms is not None
                           else previous_filters.date_start_ms),
            date_end_ms=(explicit.date_end_ms if explicit.date_end_ms is not None
                         else previous_filters.date_end_ms),
            champion=explicit.champion or previous_filters.champion,
            role=explicit.role or previous_filters.role,
        )
        return AnalysisRequest(intent, metric, resolved)
```

Champion and role filters arrive only through the structured `filters` object; metric aliases come from the registry. The parser never constructs SQL and never treats an unknown noun as a field. `SHOW_EVIDENCE` and `SHOW_LIMITATIONS` require a valid prior context; without it they abstain with `UNSUPPORTED_INTENT`.

- [ ] **Step 4: Run planner tests**

Run: `python -m pytest tests/analytics/test_planner.py -q`

Expected: `9 passed`.

- [ ] **Step 5: Commit the constrained planner**

```powershell
git add analytics/planner.py tests/analytics/test_planner.py
git commit -m "feat: parse safe French analytics intents"
```

## Task 13: Deterministic Renderer, Narrative Validation, and Chat Service

**Files:**
- Create: `analytics/renderer.py`
- Create: `analytics/service.py`
- Create: `tests/analytics/test_renderer.py`
- Create: `tests/analytics/test_service.py`
- Modify: `tests/analytics/conftest.py`

**Interfaces:**
- Consumes: planner requests, evidence packets, optional `NarrativeAdapter`.
- Produces: `render_answer()`, `validate_narrative()`, `AnalyticsService.chat()`, `AnalyticsService.health()`.

- [ ] **Step 1: Write numeric trace, fallback, stale-mart, and end-to-end tests**

Add `sample_evidence` to `conftest.py` as a complete `EvidencePacket` with a 600-match EUW cohort, non-null interval/q/effect values, two agreeing temporal folds, a stable 64-character evidence ID, and dataset/source hashes. No test may construct a partial packet that bypasses required contract fields.

```python
# tests/analytics/test_renderer.py
from analytics.renderer import render_answer, validate_narrative


def test_every_deterministic_number_maps_to_evidence(sample_evidence):
    rendered = render_answer([sample_evidence], narrative_adapter=None)
    assert rendered.answer
    assert rendered.numeric_sources
    assert all(source.evidence_id == sample_evidence.evidence_id
               for source in rendered.numeric_sources)


def test_narrative_with_digits_is_rejected():
    accepted, code = validate_narrative("L'effet est supérieur de 4 points.",
                                        allowed_evidence_ids={"ev-1"})
    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"


def test_narrative_with_unknown_evidence_reference_is_rejected():
    accepted, code = validate_narrative("Effet cohérent selon [ev-9].",
                                        allowed_evidence_ids={"ev-1"})
    assert accepted is False
    assert code == "LLM_OUTPUT_REJECTED"
```

```python
# tests/analytics/test_service.py
from analytics.service import AnalyticsService


def test_supported_chat_returns_evidence_and_dataset_version(powered_analytics_db, monkeypatch):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(service, "freshness", lambda: {"fresh": True,
        "source_dataset_version": "same", "dataset_version": "same"})
    response = service.chat(
        "La vision aide-t-elle à gagner en EUW ?", None, {}
    )
    assert response.success is True
    assert response.abstained is False
    assert response.evidence
    assert response.dataset_version


def test_timeline_question_abstains_explicitly(powered_analytics_db, monkeypatch):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(service, "freshness", lambda: {"fresh": True,
        "source_dataset_version": "same", "dataset_version": "same"})
    response = service.chat(
        "Qui gagnait à 10 minutes ?", None, {}
    )
    assert response.abstained is True
    assert response.warnings[0]["code"] == "TIMELINE_DATA_UNAVAILABLE"


def test_stale_mart_refuses_analysis(powered_analytics_db, monkeypatch):
    service = AnalyticsService(str(powered_analytics_db))
    monkeypatch.setattr(service, "freshness", lambda: {"fresh": False,
        "source_dataset_version": "new", "dataset_version": "old"})
    response = service.chat("Comment gagne-t-on ?", None, {})
    assert response.abstained is True
    assert response.warnings[0]["code"] == "ANALYTICS_DB_STALE"


def test_missing_mart_refuses_instead_of_raising(tmp_path):
    response = AnalyticsService(str(tmp_path / "missing.db")).chat(
        "Comment gagne-t-on ?", None, {}
    )
    assert response.abstained is True
    assert response.warnings[0]["code"] == "ANALYTICS_DB_STALE"
```

- [ ] **Step 2: Verify renderer/service tests fail**

Run: `python -m pytest tests/analytics/test_renderer.py tests/analytics/test_service.py -q`

Expected: FAIL during collection for `analytics.renderer`.

- [ ] **Step 3: Implement deterministic numeric rendering and narrative protocol**

```python
# analytics/renderer.py
from dataclasses import dataclass
import re
from typing import Protocol
from .types import EvidencePacket


NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?%?")
EVIDENCE_REF = re.compile(r"\[([A-Za-z0-9-]{4,64})\]")


class NarrativeAdapter(Protocol):
    def narrate(self, evidence: list[dict]) -> str: ...


@dataclass(frozen=True)
class NumericSource:
    token: str
    start: int
    end: int
    evidence_id: str


@dataclass(frozen=True)
class RenderedAnswer:
    answer: str
    numeric_sources: tuple[NumericSource, ...]
    warnings: tuple[dict, ...]


def validate_narrative(text: str, allowed_evidence_ids: set[str]) -> tuple[bool, str | None]:
    references = set(EVIDENCE_REF.findall(text))
    if references - allowed_evidence_ids:
        return False, "LLM_OUTPUT_REJECTED"
    without_references = EVIDENCE_REF.sub("", text)
    if NUMBER.search(without_references):
        return False, "LLM_OUTPUT_REJECTED"
    return True, None


def _format_packet(packet: EvidencePacket) -> tuple[str, tuple[NumericSource, ...]]:
    estimate = packet.estimate
    sample = packet.sample
    evidence_id = packet.evidence_id
    value = f"{estimate['value']:.2f}"
    low, high = estimate["confidence_interval_95"]
    matches = str(sample["matches"])
    text = (f"[{evidence_id}] Sur {matches} matchs, l'écart observé est de {value} "
            f"{estimate['unit']} (IC à 95 % : {low:.2f} à {high:.2f}; "
            f"q={estimate['q_value']:.3f}).")
    sources = tuple(NumericSource(match.group(), match.start(), match.end(), evidence_id)
                    for match in NUMBER.finditer(text))
    return text, sources


def validate_numeric_trace(answer: str, sources: tuple[NumericSource, ...]) -> None:
    expected = {(match.start(), match.end(), match.group()) for match in NUMBER.finditer(answer)}
    actual = {(source.start, source.end, source.token) for source in sources}
    if expected != actual:
        raise ValueError("UNTRACED_NUMERIC_TOKEN")
```

`render_answer()` sorts packets by recommendation strength then absolute practical effect, emits explicit association/causality wording and limitations, concatenates packet sections while offsetting every `NumericSource.start/end`, and calls `validate_numeric_trace()` on the final deterministic answer. It calls the optional narrative adapter with packet dictionaries only. If the adapter raises, returns claim digits outside an evidence reference, or cites an unknown evidence ID, append `LLM_OUTPUT_REJECTED` and keep the deterministic answer. When no adapter exists, do not warn: deterministic chat is a supported normal mode.

- [ ] **Step 4: Implement freshness-aware service orchestration**

```python
# analytics/service.py
from dataclasses import asdict, dataclass
from .cohorts import AnalyticsAbstention, CohortResolver
from .db import all_meta, EvidenceRepository
from .engine import AnalysisEngine
from .planner import FrenchPlanner
from .renderer import render_answer
from .source import source_snapshot
from .types import AnalysisRequest, Intent


@dataclass(frozen=True)
class ChatResponse:
    success: bool
    answer: str
    analysis_request: dict
    evidence: tuple[dict, ...]
    warnings: tuple[dict, ...]
    abstained: bool
    dataset_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class AnalyticsService:
    def __init__(self, analytics_path: str, source_path: str = "training_data.db",
                 narrative_adapter=None):
        self.analytics_path = analytics_path
        self.source_path = source_path
        self.planner = FrenchPlanner()
        self.engine = AnalysisEngine(analytics_path)
        self.narrative_adapter = narrative_adapter

    def health(self) -> dict:
        metadata = all_meta(self.analytics_path)
        freshness = self.freshness()
        return {"ok": bool(metadata) and freshness["fresh"], **metadata, **freshness}

    def freshness(self) -> dict:
        metadata = all_meta(self.analytics_path)
        source = source_snapshot(self.source_path)
        fresh = all(metadata.get(key) == value for key, value in source.items())
        return {"fresh": fresh, "dataset_version": metadata.get("dataset_version"),
                "source_snapshot": source}

    def get_evidence(self, evidence_id: str) -> dict | None:
        return EvidenceRepository(self.analytics_path).get_by_id(evidence_id)

    def chat(self, message: str, context: dict | None, filters: dict) -> ChatResponse:
        try:
            metadata = all_meta(self.analytics_path)
            dataset_version = metadata.get("dataset_version")
            if not self.freshness()["fresh"]:
                raise AnalyticsAbstention("ANALYTICS_DB_STALE")
            request = self.planner.plan(message, context, filters)
            execution_request = (AnalysisRequest.from_dict(context)
                                 if request.intent in {Intent.SHOW_EVIDENCE,
                                                       Intent.SHOW_LIMITATIONS}
                                 else request)
            cohort = CohortResolver(self.analytics_path).resolve(execution_request)
            packets = self.engine.run(execution_request, cohort=cohort, question=message)
            rendered = render_answer(packets, self.narrative_adapter)
            return ChatResponse(True, rendered.answer, execution_request.to_dict(),
                tuple(packet.to_dict() for packet in packets), rendered.warnings,
                False, dataset_version)
        except AnalyticsAbstention as exc:
            return ChatResponse(True, "Analyse non publiée : " + exc.code, {}, (),
                ({"code": exc.code, "details": exc.details},), True, dataset_version)
        except (OSError, sqlite3.Error):
            return ChatResponse(True, "Analyse non publiée : ANALYTICS_DB_STALE", {}, (),
                ({"code": "ANALYTICS_DB_STALE", "details": {}},), True, None)
```

`freshness()` uses the indexed/count-only `source_snapshot()` rather than recomputing the full dataset hash or decoding raw JSON. Because the operational collector inserts matches immutably with `INSERT OR IGNORE`, schema version, CLASSIC match count, and maximum creation time form the request-time staleness gate; full dataset hashing remains a build/update responsibility. `health()` catches missing/corrupt mart errors and returns `{"ok": false, "code": "ANALYTICS_DB_STALE"}`; `chat()` converts the same failures to a structured abstention. `get_evidence(evidence_id)` uses a parameterized repository lookup. Invalid request types remain `ValueError` so the HTTP layer can return 400; evidence insufficiency becomes a structured abstention with HTTP 200.

- [ ] **Step 5: Run renderer and service tests**

Run: `python -m pytest tests/analytics/test_renderer.py tests/analytics/test_service.py -q`

Expected: `7 passed`.

- [ ] **Step 6: Commit the chat core**

```powershell
git add analytics/renderer.py analytics/service.py tests/analytics/conftest.py tests/analytics/test_renderer.py tests/analytics/test_service.py
git commit -m "feat: render traceable French analytics chat"
```

## Task 14: Additive Flask Blueprint and API Contracts

**Files:**
- Create: `app/routes/analytics.py`
- Create: `tests/analytics/test_api.py`
- Modify: `app/__init__.py:15-64`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `AnalyticsService`.
- Produces: `/api/analytics/chat`, `/api/analytics/evidence/<evidence_id>`, `/api/analytics/health`.

- [ ] **Step 1: Write isolated blueprint tests and legacy regression assertion**

```python
# tests/analytics/test_api.py
from unittest.mock import MagicMock

from app import create_app


def test_chat_endpoint_preserves_response_contract(monkeypatch):
    service = MagicMock()
    service.chat.return_value.to_dict.return_value = {
        "success": True, "answer": "Association observée.",
        "analysis_request": {"intent": "explain_metric"}, "evidence": [],
        "warnings": [], "abstained": False, "dataset_version": "abc",
    }
    monkeypatch.setattr("app.routes.analytics.get_analytics_service", lambda: service)
    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().post("/api/analytics/chat", json={
        "message": "La vision aide-t-elle ?", "context": None, "filters": {}
    })
    assert response.status_code == 200
    assert response.get_json()["dataset_version"] == "abc"


def test_chat_rejects_arbitrary_context_shape(monkeypatch):
    service = MagicMock()
    service.chat.side_effect = ValueError("UNSUPPORTED_FILTER: context")
    monkeypatch.setattr("app.routes.analytics.get_analytics_service", lambda: service)
    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().post("/api/analytics/chat", json={
        "message": "Et en KR ?", "context": {"sql": "SELECT 1"}, "filters": {}
    })
    assert response.status_code == 400
    assert response.get_json()["code"] == "UNSUPPORTED_FILTER"


def test_existing_status_contract_still_exists(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert "model_loaded" in response.get_json()
```

- [ ] **Step 2: Run API tests and verify routes return 404**

Run: `python -m pytest tests/analytics/test_api.py -q`

Expected: first test FAIL because `/api/analytics/chat` is not registered.

- [ ] **Step 3: Implement lazy service creation and JSON error mapping**

```python
# app/routes/analytics.py
from flask import Blueprint, current_app, jsonify, request
from analytics.service import AnalyticsService


analytics_bp = Blueprint("analytics", __name__)


def get_analytics_service() -> AnalyticsService:
    service = current_app.extensions.get("analytics_service")
    if service is None:
        service = AnalyticsService(
            current_app.config["ANALYTICS_DB_PATH"],
            current_app.config["NNRIOT_DB_PATH"],
        )
        current_app.extensions["analytics_service"] = service
    return service


@analytics_bp.post("/api/analytics/chat")
def analytics_chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        return jsonify({"success": False, "code": "INVALID_REQUEST"}), 400
    try:
        response = get_analytics_service().chat(
            payload["message"], payload.get("context"), payload.get("filters", {})
        )
        return jsonify(response.to_dict())
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        return jsonify({"success": False, "code": code}), 400


@analytics_bp.get("/api/analytics/health")
def analytics_health():
    health = get_analytics_service().health()
    return jsonify(health), 200 if health.get("ok") else 503


@analytics_bp.get("/api/analytics/evidence/<evidence_id>")
def analytics_evidence(evidence_id):
    packet = get_analytics_service().get_evidence(evidence_id)
    if packet is None:
        return jsonify({"success": False, "code": "EVIDENCE_NOT_FOUND"}), 404
    return jsonify({"success": True, "evidence": packet})
```

In `create_app()`, set `app.config["NNRIOT_DB_PATH"]` and `app.config["ANALYTICS_DB_PATH"]` from `NNRIOT_DB_PATH` and `NNRIOT_ANALYTICS_DB_PATH`, import/register `analytics_bp`, and do not open or build `analytics.db` at startup. Add `NNRIOT_ANALYTICS_DB_PATH=analytics.db` and `NNRIOT_ANALYTICS_SEED=20260716` to `.env.example`.

- [ ] **Step 4: Run new and existing Flask tests**

Run: `python -m pytest tests/analytics/test_api.py tests/test_final_web_app.py tests/test_predict_api.py -q`

Expected: all tests pass with existing prediction response shapes unchanged.

- [ ] **Step 5: Commit the analytics API**

```powershell
git add app/routes/analytics.py app/__init__.py .env.example tests/analytics/test_api.py
git commit -m "feat: expose analytics chat blueprint"
```

## Task 15: French Chat Page and Evidence Cards

**Files:**
- Create: `templates/analytics.html`
- Create: `static/analytics.css`
- Create: `static/analytics.js`
- Create: `tests/analytics/test_ui.py`
- Modify: `app/routes/views.py:37-40`
- Modify: `templates/layout.html`

**Interfaces:**
- Consumes: the JSON endpoints from Task 14.
- Produces: `/analytics` page with visible cohort, warnings, evidence, freshness, and safe follow-ups.

- [ ] **Step 1: Write page and static-contract tests**

```python
# tests/analytics/test_ui.py
from app import create_app


def test_analytics_page_is_french_and_contains_required_regions():
    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().get("/analytics")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'lang="fr"' in html
    assert 'id="analytics-chat"' in html
    assert 'id="active-cohort"' in html
    assert 'id="evidence-list"' in html
    assert 'id="dataset-freshness"' in html
    assert "association" in html.lower()
    assert "causal" in html.lower()


def test_layout_links_to_analytics():
    app = create_app()
    app.config["TESTING"] = True
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'href="/analytics"' in html
```

- [ ] **Step 2: Verify the page test fails with 404**

Run: `python -m pytest tests/analytics/test_ui.py -q`

Expected: first test FAIL because `/analytics` is absent.

- [ ] **Step 3: Add the route and accessible page structure**

Add to `app/routes/views.py`:

```python
@views_bp.route("/analytics")
def analytics():
    """Serve the retrospective evidence chat."""
    return render_template("analytics.html", active_page="analytics")
```

`templates/analytics.html` extends `layout.html`, sets a French page container with `lang="fr"`, and includes:

```html
<main class="analytics-shell" lang="fr">
  <header>
    <p class="eyebrow">Analyse rétrospective avec preuves</p>
    <h1>Comment ces matchs sont-ils gagnés&nbsp;?</h1>
    <p>Les résultats décrivent des associations observées, pas des effets causaux.</p>
    <div id="dataset-freshness" role="status">Vérification des données…</div>
  </header>
  <section id="suggested-questions" aria-label="Questions suggérées">
    <button type="button">Comment gagne-t-on le plus souvent&nbsp;?</button>
    <button type="button">La vision est-elle associée à la victoire en EUW&nbsp;?</button>
    <button type="button">Existe-t-il une différence robuste entre EUW et KR&nbsp;?</button>
  </section>
  <aside id="active-cohort" aria-live="polite">Cohorte non résolue</aside>
  <section id="analytics-chat" aria-live="polite" aria-label="Conversation"></section>
  <form id="analytics-form">
    <label for="analytics-message">Votre question</label>
    <input id="analytics-message" name="message" autocomplete="off" required>
    <button type="submit">Analyser</button>
  </form>
  <section id="analytics-warnings" role="alert"></section>
  <section id="evidence-list" aria-label="Preuves statistiques"></section>
</main>
```

Add an Analytics navigation link in both navbar and footer. Keep existing link labels and routes unchanged.

- [ ] **Step 4: Implement browser state without arbitrary HTML injection**

`static/analytics.js` keeps only the validated `analysis_request` as follow-up context, sends JSON to `/api/analytics/chat`, and renders every user/server string with `textContent`. Evidence cards use `<details>` and expose sample size, effect, interval, q-value, temporal robustness, limitations, evidence ID, and dataset version. Disable the submit button during a request, restore it in `finally`, and display network/API errors in `analytics-warnings`. On load, call `/api/analytics/health` and show freshness without triggering a build.

`static/analytics.css` uses the existing CSS variables, provides keyboard-visible focus states, responsive layout at 760 px, readable evidence tables, distinct warning/abstention styling, and no external asset dependency.

- [ ] **Step 5: Run UI and full Flask tests**

Run: `python -m pytest tests/analytics/test_ui.py tests/test_final_web_app.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the chat UI**

```powershell
git add templates/analytics.html static/analytics.css static/analytics.js app/routes/views.py templates/layout.html tests/analytics/test_ui.py
git commit -m "feat: add French evidence chat page"
```

## Task 16: Real-data Build, Performance Benchmark, Documentation, and Completion Audit

**Files:**
- Create: `analytics/benchmark.py`
- Create: `tests/analytics/test_benchmark.py`
- Create: `docs/analytics.md`
- Modify: `analytics/cli.py`
- Modify: `README.md`
- Modify: `tests/analytics/conftest.py`

**Interfaces:**
- Consumes: the complete feature.
- Produces: repeatable benchmarks, real build/model reports, operating documentation, and a requirement audit.

- [ ] **Step 1: Write benchmark result and no-request-time-scan tests**

Add `powered_analytics_pair` to `conftest.py`. It returns the source DB used to build the existing `powered_analytics_db` plus that mart, so request-time freshness is genuinely true rather than mocked in the benchmark.

```python
# tests/analytics/test_benchmark.py
from analytics.benchmark import percentile, run_benchmark


def test_percentile_is_nearest_rank():
    assert percentile([0.01, 0.02, 0.03, 0.04, 0.05], 95) == 0.05


def test_benchmark_reports_cache_analysis_and_chat(powered_analytics_pair):
    source_db, analytics_db = powered_analytics_pair
    report = run_benchmark(str(analytics_db), str(source_db), iterations=5)
    assert set(report["p95_ms"]) == {"cached_evidence", "single_metric", "chat_no_llm"}
    assert report["iterations"] == 5
    assert report["context"]["platform"]
```

- [ ] **Step 2: Verify benchmark tests fail**

Run: `python -m pytest tests/analytics/test_benchmark.py -q`

Expected: FAIL during collection for `analytics.benchmark`.

- [ ] **Step 3: Implement reproducible latency measurement**

```python
# analytics/benchmark.py
from dataclasses import asdict
import math
import platform
import time
from .engine import AnalysisEngine
from .service import AnalyticsService
from .types import AnalysisRequest, Intent


def percentile(values: list[float], value: int) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(value / 100 * len(ordered)) - 1)
    return ordered[index]


def run_benchmark(path: str, source_path: str, iterations: int = 30) -> dict:
    request = AnalysisRequest(Intent.EXPLAIN_METRIC, "vision_score")
    engine = AnalysisEngine(path)
    service = AnalyticsService(path, source_path)
    engine.run(request, question="La vision aide-t-elle ?")
    timings = {"cached_evidence": [], "single_metric": [], "chat_no_llm": []}
    for _ in range(iterations):
        started = time.perf_counter()
        engine.cached(request, question="La vision aide-t-elle ?")
        timings["cached_evidence"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        engine.run_uncached(request, question="La vision aide-t-elle ?")
        timings["single_metric"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        service.chat("La vision aide-t-elle ?", None, {})
        timings["chat_no_llm"].append((time.perf_counter() - started) * 1000)
    return {
        "iterations": iterations,
        "p95_ms": {name: percentile(values, 95) for name, values in timings.items()},
        "context": {"platform": platform.platform(), "python": platform.python_version()},
    }
```

Expose explicit `AnalysisEngine.cached(request, question)` and `run_uncached(request, question)` methods so the benchmark never manipulates private cache state. Add `benchmark --iterations 30` to the CLI and pass the global `--source` path to `run_benchmark()`. It prints JSON and returns exit code 2 when cached evidence is at least 100 ms, a single metric is at least 2,000 ms, or chat excluding external narrative is at least 3,000 ms.

- [ ] **Step 4: Run the complete synthetic test suite before touching the real mart**

Run: `python -m pytest tests/analytics -q`

Expected: all analytics tests pass.

Run: `python -m pytest tests -q`

Expected: all existing and new tests pass; any pre-existing unrelated failure is recorded with command/output and is not hidden.

- [ ] **Step 5: Build the real `analytics.db` and inspect the quality report**

Run: `python -m analytics --source training_data.db --analytics analytics.db build > analytics_build_report.json`

Expected: exit code 0, `quality_gate_passed: true`, required metadata populated, no more than 0.5% unexplained eligible-match failures, and counts consistent with one match row, two team rows, and ten player rows per accepted match.

Run these read-only integrity queries:

```powershell
@'
import sqlite3
from pathlib import Path
conn = sqlite3.connect(Path('analytics.db').resolve().as_uri() + '?mode=ro', uri=True)
conn.execute('PRAGMA query_only=ON')
matches = conn.execute('SELECT COUNT(*) FROM match_facts').fetchone()[0]
teams = conn.execute('SELECT COUNT(*) FROM team_facts').fetchone()[0]
players = conn.execute('SELECT COUNT(*) FROM player_facts').fetchone()[0]
bad_teams = conn.execute('SELECT COUNT(*) FROM (SELECT match_id FROM team_facts GROUP BY match_id HAVING COUNT(*) <> 2)').fetchone()[0]
bad_players = conn.execute('SELECT COUNT(*) FROM (SELECT match_id FROM player_facts GROUP BY match_id HAVING COUNT(*) <> 10)').fetchone()[0]
print({'matches': matches, 'teams': teams, 'players': players, 'bad_teams': bad_teams, 'bad_players': bad_players})
conn.close()
'@ | python -
```

Expected: `teams == matches * 2`, `players == matches * 10`, `bad_teams == 0`, `bad_players == 0`.

- [ ] **Step 6: Train/evaluate the real interpretable model and run benchmarks**

Run: `python -m analytics --analytics analytics.db train-model --artifacts analytics_artifacts > analytics_model_report.json`

Expected: chronological 70/15/15 counts, baselines, log loss, Brier score, calibration error, ROC-AUC with bootstrap intervals, named coefficients, dataset version, retrospective label, and either a valid player-disjoint result or `INSUFFICIENT_MATCHES` without fallback.

Run: `python -m analytics --analytics analytics.db benchmark --iterations 30 > analytics_benchmark_report.json`

Expected targets on the recorded hardware: cached p95 below 100 ms, single-metric p95 below 2 seconds, chat without external narrative p95 below 3 seconds. If a target fails, profile indexed SQL and cache keys, add a failing regression test, fix the bottleneck, and rerun the exact benchmark.

- [ ] **Step 7: Write operating and limitation documentation from measured reports**

`docs/analytics.md` must contain:

- exact build, update, health, model, benchmark, and Flask commands;
- the separate-database/read-only-source guarantee;
- schema and dataset-version descriptions;
- supported French intents and follow-up context shape;
- all statistical, practical, regional, and temporal thresholds;
- the association-not-causality warning;
- corpus sampling bias and absent rank limitation;
- the final-game retrospective label and forbidden pre-match claim;
- timeline, esports-team, and arbitrary-SQL refusal behavior;
- the current NN discovery-only limitation;
- actual accepted/excluded match counts and issue codes from `analytics_build_report.json`;
- actual model metrics and grouped-split status from `analytics_model_report.json`;
- actual hardware/context and p95 results from `analytics_benchmark_report.json`.

Add a concise Analytics section and link to `docs/analytics.md` in `README.md`. Do not commit `analytics.db`, model artifacts, or generated measurement JSON unless the user explicitly requests versioning of those reports.

- [ ] **Step 8: Run the final traceability and compatibility audit**

Run:

```powershell
python -m pytest tests/analytics -q
python -m pytest tests -q
python -m pytest test_components.py test_data.py test_improved_vector.py test_model_init.py -q
git diff --check
git status --short
```

Then manually verify this acceptance matrix against test names and measured reports:

| Requirement | Authoritative evidence |
| --- | --- |
| Separate rebuildable mart, source unchanged | `test_build.py`, build report, source checksum |
| Deterministic cohorts and widening | `test_cohorts.py`, evidence cohort field |
| Complete evidence/provenance | `test_evidence.py`, cached packet JSON |
| Regional display gate | `test_regions.py` |
| Statistical/practical/temporal thresholds | `test_statistics.py`, `test_evidence.py` |
| Chronological calibrated model | `test_interpretable.py`, real model report |
| Player-disjoint sensitivity | `test_splits.py`, real model report |
| NN cannot bypass evidence | `test_nn_adapter.py` |
| French numeric traceability | `test_renderer.py`, `test_service.py` |
| Explicit abstention | `test_planner.py`, `test_service.py`, `test_api.py` |
| No causal/timeline/pre-match/team claim | planner/renderer tests and `docs/analytics.md` |
| Existing behavior preserved | full existing test suite |
| Performance targets measured | real benchmark report |

- [ ] **Step 9: Commit documentation and benchmark tooling**

```powershell
git add analytics/benchmark.py analytics/cli.py tests/analytics/conftest.py tests/analytics/test_benchmark.py docs/analytics.md README.md
git commit -m "docs: operationalize explainable analytics"
```

## Execution Notes

- Each task is a reviewer gate and should be implemented in order because later interfaces consume earlier ones.
- Use fresh synthetic SQLite fixtures for unit/integration tests; only Task 16 opens the real corpus for a full build and measured evaluation.
- Keep bootstrap arrays memory-bounded on the real corpus by processing resamples in deterministic chunks while preserving the same seeded result.
- Before every commit, inspect `git diff --cached --name-only` and confirm it contains only the paths listed in that task.
- If existing dirty changes overlap a planned file, preserve them, inspect the combined diff, and stop for user direction only when the overlap cannot be safely isolated.
