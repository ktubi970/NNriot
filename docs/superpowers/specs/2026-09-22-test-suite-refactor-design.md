# NNriot Test Suite Refactor Design

**Status:** Approved in chat on 2026-09-22

**Scope:** Pytest configuration, test-suite structure, test hygiene, collection performance, Riot rate-limiter testability, and Python CI.

## Problem statement

The current suite mixes executable verification scripts, benchmarks, unit tests, integration tests, and ML smoke tests in a flat `tests/` directory. Several modules import Flask or TensorFlow during pytest collection, two collected functions return values instead of asserting, a third (`test_model_init.py`) has the same defect, and the rate-limiter test waits for real wall-clock time.

Measured on the repository's Python 3.12 environment before the refactor:

- `.venv312\Scripts\python.exe -m pytest --collect-only -q`: 293 tests in 20.17 seconds; 23.14 seconds wall time.
- Importing the complete test dependency set initializes TensorFlow/oneDNN before tests run.
- The system Python 3.14 environment is incomplete and is not a valid verification environment.
- `.github/workflows/ci.yml` already exists, but runs Python 3.10/3.11 and executes the full pytest suite on pull requests. It does not provide the requested fast/full split.

## Goals

1. Collect the suite in less than 3 seconds on the repository's Python 3.12 environment.
2. Make every collected test assert observable behavior or use pytest's explicit skip/raises mechanisms.
3. Keep heavy imports out of module import time.
4. Organize tests by domain while preserving focused files and the existing `tests/analytics/` structure.
5. Eliminate real sleeping from rate-limiter tests.
6. Make `pytest -m "not slow"` the pull-request path and run the full suite after pushes to `main` or `master`.

## Non-goals

- No production API redesign beyond the injectable clock/sleeper in `RiotRateLimiter`.
- No change to ML architecture, model outputs, database schema, Flask route contracts, or analytics algorithms.
- No consolidation merely to reduce file count. Files stay separate when merging would create a large mixed-responsibility module or introduce symbol collisions.
- Historical implementation plans are not rewritten when test paths move.

## Architecture decisions

### 1. Domain packages, not a flat suite

`tests/` becomes a Python package and gains `core/`, `api/`, `ml/`, `scripts/`, and `integration/` packages. `tests/analytics/` remains in place and receives only an `__init__.py` plus the import/marker changes needed for deterministic collection.

The requested canonical filenames are used as anchors. Supporting files remain focused instead of being forced into a single file. In particular, the four backfill test modules remain distinct because combining them would exceed 400 lines and would silently collide on `test_backfill_idempotent`.

| Current path | Target path |
| --- | --- |
| `tests/test_analytics_wal_mode.py` | `tests/analytics/test_wal_mode.py` |
| `tests/test_api_telemetry_and_errors.py` | `tests/api/test_telemetry.py` |
| `tests/test_api_train_and_stream.py` | `tests/api/test_train_and_stream.py` |
| `tests/test_disclaimed_markets.py` | `tests/api/test_disclaimed_markets.py` |
| `tests/test_final_web_app.py` | `tests/api/test_web_app.py` |
| `tests/test_player_history_real.py` | `tests/api/test_player_history.py` |
| `tests/test_predict_api.py` | `tests/api/test_predict_api.py` |
| `tests/test_predictor_ui.py` | `tests/api/test_predictor_ui.py` |
| `tests/test_services.py` | `tests/api/test_services.py` |
| `tests/test_task_runner.py` | `tests/api/test_task_runner.py` |
| `tests/test_warmup_ttl.py` | `tests/api/test_warmup_ttl.py` |
| `tests/test_database_v11_migration.py` | `tests/core/test_database.py` |
| `tests/test_db_config_harmony.py` | `tests/core/test_db_config.py` |
| `tests/test_db_purge_polluted.py` | `tests/core/test_db_purge.py` |
| `tests/test_excluded_modes.py` | `tests/core/test_excluded_modes.py` |
| `tests/test_feature_labels.py` | `tests/core/test_feature_labels.py` |
| `tests/test_pre_match_features.py` | `tests/core/test_pre_match_features.py` |
| `tests/test_riot_api_hardening.py` | `tests/core/test_riot_api.py` |
| `tests/test_riot_api_timeline.py` | `tests/core/test_riot_api_timeline.py` |
| `tests/test_riot_rate_limiter.py` | `tests/core/test_riot_rate_limiter.py` |
| `tests/test_model_calibration_guard.py` | `tests/ml/test_model_calibration.py` |
| `tests/test_multi_output_model.py` | `tests/ml/test_multi_output_model.py` |
| `tests/test_multi_output_trainer.py` | `tests/ml/test_multi_output_trainer.py` |
| `tests/test_pre_match_training.py` | `tests/ml/test_pre_match_training.py` |
| `tests/test_backfill_loop_guard.py` | `tests/scripts/test_backfill.py` |
| `tests/test_backfill_participants_script.py` | `tests/scripts/test_backfill_participants.py` |
| `tests/test_backfill_timelines.py` | `tests/scripts/test_backfill_timelines.py` |
| `tests/test_streaming_backfill.py` | `tests/scripts/test_backfill_streaming.py` |
| `tests/test_cleanup_malformed.py` | `tests/scripts/test_cleanup_malformed.py` |
| `tests/test_collector_game_mode_guard.py` | merged into `tests/scripts/test_data_collector.py` |
| `tests/test_followups.py` | merged into `tests/scripts/test_data_collector.py` |
| `tests/test_timeline_collection.py` | merged into `tests/scripts/test_data_collector.py` |
| `tests/test_migrate_winner_kills.py` | `tests/scripts/test_migrations.py` |
| `tests/test_resolve_pro_name.py` | `tests/scripts/test_import_liquipedia.py` |
| `tests/test_run_batch_import.py` | `tests/scripts/test_import_liquipedia_batch.py` |
| `tests/test_p5_smoke.py` | `tests/integration/test_pipeline_smoke.py` |
| `tests/test_data.py` | `tests/fixtures_data.py` |
| `tests/test_components.py` | `scripts/verify_components.py` |
| `tests/test_improved_vector.py` | `scripts/benchmarks/benchmark_vectorization.py` |
| `tests/test_model_init.py` | assertion-based test merged into `tests/ml/test_multi_output_trainer.py` |

The root `tests/` directory contains support files only (`__init__.py`, `fixtures_data.py`, and any future shared `conftest.py`), not collected `test_*.py` modules.

### 2. Explicit test taxonomy

`pytest.ini` declares four markers:

- `unit`: deterministic tests contained within one process and without external services.
- `integration`: tests that cross Flask, SQLite, scripts, or multiple production modules.
- `ml`: tests that exercise TensorFlow/Keras or ML training/model construction.
- `slow`: tests intentionally excluded from the pull-request fast path.

ML modules are marked `ml` and `slow`. Flask application modules are marked `integration` and `slow` because importing the current package creates a module-level app and can initialize the asynchronous trainer. The pipeline smoke test carries `integration`, `ml`, and `slow`. Lightweight core and pure script tests use `unit` or `integration` according to their real boundary.

### 3. Imports are execution dependencies, not collection dependencies

Test modules may import `pytest`, standard-library modules, and lightweight constants at module scope. Imports of the following move inside fixtures or test bodies:

- `tensorflow` and Keras-facing production modules;
- `app`, `app.*`, and the module-level Flask application;
- `core.model_factory` and `core.generate_graph`;
- `scripts.continuous_trainer`;
- NumPy/SciPy imports used only to build runtime fixtures.

Markers are applied before those imports execute. `pytest --collect-only` therefore discovers node IDs without initializing Flask, TensorFlow, the trainer, or oneDNN.

### 4. Manual diagnostics and benchmarks are not tests

`scripts/verify_components.py` remains an explicit human-operated diagnostic. Its functions are named `verify_*` and its process exit code reports failure. Optional dependencies use explicit diagnostic statuses instead of pytest semantics.

`scripts/benchmarks/benchmark_vectorization.py` keeps the verbose vector comparison/reporting behavior. Pytest receives small invariant tests in `tests/core/test_json_utils.py`, using `tests.fixtures_data` and direct assertions on shape, finiteness, non-zero features, and normalization.

### 5. Minimal dependency injection for time

The analyzer did not justify a general architecture pattern for the test tree. The single justified pattern is constructor-level dependency injection in `core.riot_api.RiotRateLimiter`:

```python
def __init__(
    self,
    per_second: int = 18,
    per_2min: int = 95,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
):
    ...
```

The production defaults preserve behavior while using a monotonic clock. Tests provide a virtual clock whose `sleep()` advances time immediately. No dependency-injection container or new abstraction hierarchy is introduced.

| File/symbol | Observed problem | Benefit | Cost |
| --- | --- | --- | --- |
| `core/riot_api.py:RiotRateLimiter` | Hidden wall clock and sleeper force a real one-second wait and can be affected by clock adjustments. | Deterministic zero-wait tests and monotonic elapsed time. | Two keyword-only constructor dependencies and two instance attributes. |
| `tests/` package boundaries | 40 root test files mix unrelated domains. | Predictable ownership, marker placement, and focused test execution. | Mechanical path/import updates. |
| Flask/ML test imports | Module import initializes application and C++ runtime during collection. | Collection becomes discovery-only. | Fixtures/tests perform local imports and some signatures change. |

No Repository, Strategy, Adapter, Facade, or service layer is added. Existing repositories and application factory remain unchanged.

### 6. CI split

`.github/workflows/test.yml` uses Python 3.12 and installs `requirements-dev.txt` (which already includes `requirements.txt`). The fast job runs on pull requests and pushes with `pytest -m "not slow"`. A full job runs only for pushes to `main` or `master` and executes unfiltered `pytest`.

The existing `ci.yml` becomes the lint workflow and stops running pytest, preventing an outdated Python 3.10/3.11 full-suite duplicate on pull requests. The Playwright workflow remains unchanged.

## Error handling and test semantics

- Expected exceptions use `with pytest.raises(...)`.
- Missing optional dependencies in manual scripts are diagnostic outcomes; collected tests use `pytest.importorskip` only when the dependency is genuinely optional.
- Collected tests never return booleans, dictionaries, or status strings.
- `PytestReturnNotNoneWarning` is not filtered and is promoted to an error during final verification.

## Acceptance evidence

All measurements use `.venv312\Scripts\python.exe` on Windows and Python 3.12 in CI:

1. `pytest --collect-only -q` exits 0 and wall time is below 3.0 seconds.
2. `pytest -m "not slow" -q` exits 0.
3. `pytest -q -W error::pytest.PytestReturnNotNoneWarning` exits 0.
4. `rg -n "sys\.path\.insert" tests` returns no matches.
5. No `tests/test_*.py` files remain at the test root.
6. `git diff --check` reports no whitespace errors.
