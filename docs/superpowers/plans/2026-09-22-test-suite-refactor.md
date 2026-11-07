# NNriot Test Suite Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor NNriot's pytest suite into domain packages, remove false tests and real waits, reduce collection below three seconds, and add a Python 3.12 fast/full CI split.

**Architecture:** Preserve production behavior and focused test boundaries while making collection discovery-only. Heavy Flask/ML dependencies load inside marked test execution, manual diagnostics leave the pytest tree, and `RiotRateLimiter` receives only the two injectable time dependencies required for deterministic tests.

**Tech Stack:** Python 3.12, pytest, Flask, TensorFlow/Keras, SQLite, GitHub Actions, PowerShell.

**Spec:** `docs/superpowers/specs/2026-09-22-test-suite-refactor-design.md`

## Global Constraints

- Use `.venv312\Scripts\python.exe` for every local pytest command.
- Preserve all current application, database, model, and API behavior except switching `RiotRateLimiter` elapsed-time measurement from `time.time` to `time.monotonic`.
- Preserve every existing collected test behavior unless it is a false test; replace false tests with direct assertions or move the manual behavior out of `tests/`.
- Keep `tests/analytics/` in place; only add package metadata, markers, or lazy imports required by this spec.
- Do not add a dependency-injection container or new production abstraction layer.
- `pytest --collect-only -q` must complete in less than 3.0 seconds.
- `pytest -m "not slow" -q` and the complete unfiltered suite must both pass.
- Do not suppress `pytest.PytestReturnNotNoneWarning`.

---

### Task 1: Configure pytest and establish package boundaries

**Files:**
- Modify: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/api/__init__.py`
- Create: `tests/ml/__init__.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/analytics/__init__.py`

**Interfaces:**
- Consumes: The existing pytest invocation and Python import system.
- Produces: Importable test packages and registered `slow`, `ml`, `integration`, and `unit` markers.

- [ ] **Step 1: Capture the current configuration and collection baseline**

Run:

```powershell
Get-Content -Raw pytest.ini
$watch = [System.Diagnostics.Stopwatch]::StartNew()
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q
$watch.Stop()
$watch.Elapsed.TotalSeconds
```

Expected: 293 tests are discovered; the measured baseline is approximately 20–25 seconds and therefore fails the final three-second budget.

- [ ] **Step 2: Replace `pytest.ini` with the explicit discovery and marker contract**

```ini
[pytest]
pythonpath = .
testpaths = tests
markers =
    slow: tests intentionally excluded from the fast pull-request suite
    ml: tests that initialize or exercise TensorFlow/Keras and ML models
    integration: tests crossing Flask, SQLite, scripts, or multiple modules
    unit: deterministic tests contained within one process
```

The old `filterwarnings = ignore::pytest.PytestReturnNotNoneWarning` entry must be absent.

- [ ] **Step 3: Add empty `__init__.py` files to every test package**

Create the seven files listed above. Do not add imports or package-level side effects.

- [ ] **Step 4: Verify configuration parsing and package imports**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest --markers
& .\.venv312\Scripts\python.exe -c "import tests, tests.analytics; print('test-packages-ok')"
```

Expected: all four custom markers appear and `test-packages-ok` is printed.

- [ ] **Step 5: Commit the configuration slice**

```powershell
git add pytest.ini tests/__init__.py tests/core/__init__.py tests/api/__init__.py tests/ml/__init__.py tests/scripts/__init__.py tests/integration/__init__.py tests/analytics/__init__.py
git commit -m "test: define pytest packages and markers"
```

### Task 2: Move tests into domain packages without changing behavior

**Files:**
- Move: all root `tests/test_*.py` modules except the four special cases handled in Task 3
- Modify: imports affected by the moves

**Interfaces:**
- Consumes: Package directories from Task 1.
- Produces: Domain-specific test paths with no `sys.path.insert(...)` boilerplate.

- [ ] **Step 1: Move API and analytics tests**

Use `git mv` for the following exact mappings:

```text
tests/test_analytics_wal_mode.py        -> tests/analytics/test_wal_mode.py
tests/test_api_telemetry_and_errors.py  -> tests/api/test_telemetry.py
tests/test_api_train_and_stream.py      -> tests/api/test_train_and_stream.py
tests/test_disclaimed_markets.py        -> tests/api/test_disclaimed_markets.py
tests/test_final_web_app.py             -> tests/api/test_web_app.py
tests/test_player_history_real.py       -> tests/api/test_player_history.py
tests/test_predict_api.py               -> tests/api/test_predict_api.py
tests/test_predictor_ui.py              -> tests/api/test_predictor_ui.py
tests/test_services.py                  -> tests/api/test_services.py
tests/test_task_runner.py               -> tests/api/test_task_runner.py
tests/test_warmup_ttl.py                -> tests/api/test_warmup_ttl.py
```

- [ ] **Step 2: Move core and ML tests**

```text
tests/test_database_v11_migration.py    -> tests/core/test_database.py
tests/test_db_config_harmony.py         -> tests/core/test_db_config.py
tests/test_db_purge_polluted.py         -> tests/core/test_db_purge.py
tests/test_excluded_modes.py            -> tests/core/test_excluded_modes.py
tests/test_feature_labels.py            -> tests/core/test_feature_labels.py
tests/test_pre_match_features.py        -> tests/core/test_pre_match_features.py
tests/test_riot_api_hardening.py        -> tests/core/test_riot_api.py
tests/test_riot_api_timeline.py         -> tests/core/test_riot_api_timeline.py
tests/test_riot_rate_limiter.py         -> tests/core/test_riot_rate_limiter.py
tests/test_model_calibration_guard.py   -> tests/ml/test_model_calibration.py
tests/test_multi_output_model.py        -> tests/ml/test_multi_output_model.py
tests/test_multi_output_trainer.py      -> tests/ml/test_multi_output_trainer.py
tests/test_pre_match_training.py        -> tests/ml/test_pre_match_training.py
```

- [ ] **Step 3: Move script and integration tests**

```text
tests/test_backfill_loop_guard.py         -> tests/scripts/test_backfill.py
tests/test_backfill_participants_script.py -> tests/scripts/test_backfill_participants.py
tests/test_backfill_timelines.py          -> tests/scripts/test_backfill_timelines.py
tests/test_streaming_backfill.py           -> tests/scripts/test_backfill_streaming.py
tests/test_cleanup_malformed.py            -> tests/scripts/test_cleanup_malformed.py
tests/test_collector_game_mode_guard.py     -> tests/scripts/test_data_collector.py
tests/test_migrate_winner_kills.py          -> tests/scripts/test_migrations.py
tests/test_resolve_pro_name.py              -> tests/scripts/test_import_liquipedia.py
tests/test_run_batch_import.py              -> tests/scripts/test_import_liquipedia_batch.py
tests/test_p5_smoke.py                      -> tests/integration/test_pipeline_smoke.py
```

- [ ] **Step 4: Merge the two additional data-collector modules into `test_data_collector.py`**

Append the `_match` fixture builder and three `_build_training_record` tests from `tests/test_followups.py` to `tests/scripts/test_data_collector.py`. Move `test_get_untrained_records_returns_dict` to `tests/core/test_database.py`, because it asserts the database deserialization contract rather than collector behavior. Append `test_fetch_timelines_default_false` from `tests/test_timeline_collection.py` to `tests/scripts/test_data_collector.py`. Delete the two source files through the patch after their tests are present at the targets.

Retain the exact existing assertions. Rename `_match` to `_collector_match` to make its scope explicit.

- [ ] **Step 5: Remove every path mutation from the moved suite**

Delete imports used only for path manipulation and every statement matching:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Do not remove `os` or `sys` when the module uses them for actual behavior.

- [ ] **Step 6: Verify discovery paths and absence of path mutations**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q
rg -n "sys\.path\.insert" tests
Get-ChildItem tests -File -Filter 'test_*.py'
```

Expected: collection succeeds; `rg` returns no matches; only the four Task 3 source files can remain at the root before Task 3.

- [ ] **Step 7: Commit the structural slice**

```powershell
git add tests
git commit -m "test: organize suite by domain"
```

### Task 3: Separate diagnostics and benchmarks from real tests

**Files:**
- Move/modify: `tests/test_components.py` -> `scripts/verify_components.py`
- Move/modify: `tests/test_improved_vector.py` -> `scripts/benchmarks/benchmark_vectorization.py`
- Move/modify: `tests/test_data.py` -> `tests/fixtures_data.py`
- Modify/delete: `tests/test_model_init.py`
- Create: `tests/core/test_json_utils.py`
- Modify: `tests/core/test_database.py`
- Modify: `tests/ml/test_multi_output_trainer.py`

**Interfaces:**
- Consumes: Existing component checks, vector benchmark, and fixture builders.
- Produces: Explicit manual scripts plus assertion-based pytest coverage.

- [ ] **Step 1: Write invariant tests for JSON vectorization**

Create `tests/core/test_json_utils.py` with this behavioral shape before moving the benchmark:

```python
import numpy as np
from scipy import sparse

from core import json_utils
from tests.fixtures_data import get_extended_test_data


def _non_zero_values(vector):
    return vector.data if sparse.issparse(vector) else vector[vector != 0]


def test_enhanced_vectorization_preserves_batch_shape_and_finite_values():
    matches = get_extended_test_data()
    vectors = json_utils.enhanced_json_to_vector(matches, dim=512)

    assert vectors.shape == (len(matches), 512)
    assert np.count_nonzero(_non_zero_values(vectors)) > 0
    assert np.isfinite(_non_zero_values(vectors)).all()


def test_sparse_vectorization_returns_non_empty_csr_matrix():
    matches = get_extended_test_data()
    vectors = json_utils.create_sparse_vector(matches, dim=512, sparsity=0.02)

    assert sparse.isspmatrix_csr(vectors)
    assert vectors.shape == (len(matches), 512)
    assert 0 < vectors.nnz < vectors.shape[0] * vectors.shape[1]
```

Run before adding `tests/fixtures_data.py` at its final path:

```powershell
& .\.venv312\Scripts\python.exe -m pytest tests/core/test_json_utils.py -q
```

Expected: collection fails with `ModuleNotFoundError: tests.fixtures_data`, proving the new tests depend on the intended fixture module.

- [ ] **Step 2: Rename the fixture helper and make the invariant tests green**

Move `tests/test_data.py` to `tests/fixtures_data.py`. Replace both `import test_data` statements in the benchmark source with:

```python
from tests import fixtures_data
```

and replace `test_data.get_extended_test_data()` with `fixtures_data.get_extended_test_data()`.

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest tests/core/test_json_utils.py -q
```

Expected: both invariant tests pass.

- [ ] **Step 3: Move and rename the manual component verifier**

Move the file to `scripts/verify_components.py`. Rename all collected-looking functions from `test_*` to `verify_*`, update the `tests = [...]` function table, and keep `main()` returning `failed == 0`. Preserve the command-line exit behavior:

```python
if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
```

Move the database save/fetch assertions into `tests/core/test_database.py` using `tmp_path` and direct assertions on the stored row and required stats keys. Do not copy the try/except or boolean returns.

- [ ] **Step 4: Move the vector report to the benchmark namespace**

Create `scripts/benchmarks/__init__.py`, move the source to `scripts/benchmarks/benchmark_vectorization.py`, and rename `test_vectorization_improvements()` to `benchmark_vectorization()`. Update its `__main__` call accordingly. The report may return its vector dictionary because it is no longer collected by pytest.

- [ ] **Step 5: Replace the false trainer-init test with an assertion-based slow test**

Move the behavior into `tests/ml/test_multi_output_trainer.py`:

```python
@pytest.mark.slow
@pytest.mark.ml
def test_continuous_trainer_initializes_all_output_heads():
    from core.feature_labels import ALL_LABEL_KEYS
    from scripts.continuous_trainer import ContinuousTrainer

    trainer = ContinuousTrainer()

    assert set(trainer.model.output_names) == set(ALL_LABEL_KEYS)
```

Delete `tests/test_model_init.py`. Do not catch initialization errors; a real failure must fail pytest.

- [ ] **Step 6: Verify that no collected test returns a status value**

Run:

```powershell
rg -n "return (True|False|\"SKIP\")" tests -g '*.py'
& .\.venv312\Scripts\python.exe -m pytest tests/core/test_json_utils.py tests/core/test_database.py --collect-only -q
```

Expected: the search returns no test-status returns and targeted collection succeeds.

- [ ] **Step 7: Commit the hygiene slice**

```powershell
git add scripts/verify_components.py scripts/benchmarks tests
git commit -m "test: separate diagnostics from assertions"
```

### Task 4: Add markers and defer heavy imports

**Files:**
- Modify: `tests/api/*.py`
- Modify: `tests/ml/*.py`
- Modify: `tests/integration/test_pipeline_smoke.py`
- Modify: `tests/analytics/test_api.py`
- Modify: `tests/analytics/test_ui.py`
- Modify: any collected module reported by the import-time verification command

**Interfaces:**
- Consumes: Marker declarations from Task 1.
- Produces: Discovery-only module imports and a reliable `not slow` selection.

- [ ] **Step 1: Apply module-level marker taxonomy before production imports**

Use these exact module-level declarations:

```python
# tests/api/*.py
import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

# tests/ml/*.py
import pytest
pytestmark = [pytest.mark.ml, pytest.mark.slow]

# tests/integration/test_pipeline_smoke.py
import pytest
pytestmark = [pytest.mark.integration, pytest.mark.ml, pytest.mark.slow]
```

For `tests/analytics/test_api.py` and `tests/analytics/test_ui.py`, use:

```python
pytestmark = [pytest.mark.integration, pytest.mark.slow]
```

Do not duplicate `import pytest` when it already exists.

- [ ] **Step 2: Move Flask imports behind fixtures or test bodies**

Remove every module-scope import matching `from app ...`, `import app`, or `from app.* ...` in `tests/api/`, `tests/analytics/test_api.py`, and `tests/analytics/test_ui.py`.

Use local-import fixtures such as:

```python
@pytest.fixture
def app_modules():
    from app import app as flask_app
    from app import core as app_core

    flask_app.config["TESTING"] = True
    return flask_app, app_core
```

Tests that monkeypatch `app_core` accept `app_modules`, unpack it, and retain their existing assertions. Fixtures needing only an application use a local `from app import create_app` inside the fixture body.

- [ ] **Step 3: Move TensorFlow and TensorFlow-facing imports behind test execution**

Remove module-scope imports of `tensorflow`, `core.model_factory`, `core.generate_graph`, and `scripts.continuous_trainer` from `tests/ml/`. For repeated access, use fixtures:

```python
@pytest.fixture(scope="module")
def trainer_module():
    from scripts import continuous_trainer
    return continuous_trainer


@pytest.fixture(scope="module")
def tensorflow_module():
    import tensorflow
    return tensorflow
```

Update each test signature and replace former module globals with the fixture objects. Move NumPy/SciPy imports used only for runtime fixture construction into helpers or test bodies.

- [ ] **Step 4: Remove the manual exception anti-pattern**

In `tests/analytics/test_benchmark.py`, replace the try/except/else block with:

```python
with pytest.raises(ValueError):
    percentile([], 95)
```

Keep a `match=` expression if the production exception has a stable contract message; otherwise assert only the exception type.

- [ ] **Step 5: Prove marker selection does not execute slow bodies**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q -m "not slow"
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q -m ml
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q -m integration
```

Expected: each command exits 0 and selects the intended categories.

- [ ] **Step 6: Measure and enforce the collection budget**

Run:

```powershell
$watch = [System.Diagnostics.Stopwatch]::StartNew()
& .\.venv312\Scripts\python.exe -m pytest --collect-only -q
$code = $LASTEXITCODE
$watch.Stop()
"collection_seconds=$([math]::Round($watch.Elapsed.TotalSeconds, 3))"
exit $code
```

Expected: exit 0 and `collection_seconds` below `3.000`.

If the threshold is still exceeded, capture import timing without changing production code:

```powershell
& .\.venv312\Scripts\python.exe -X importtime -m pytest --collect-only -q 2> pytest-importtime.log
Get-Content pytest-importtime.log | Sort-Object { [int](($_ -split '\|')[0] -replace '\D','') } -Descending | Select-Object -First 30
```

For any remaining test-only module-scope import of NumPy, SciPy, scikit-learn, Flask, TensorFlow, or an NNriot module that transitively imports them, move that import into the fixture/test that uses it and rerun the stopwatch command. Delete `pytest-importtime.log` after the threshold is met.

- [ ] **Step 7: Commit the collection-performance slice**

```powershell
git add tests pytest.ini
git commit -m "perf(test): defer heavy collection imports"
```

### Task 5: Replace real rate-limit waiting with a virtual clock

**Files:**
- Modify: `tests/core/test_riot_rate_limiter.py`
- Modify: `core/riot_api.py:21-59`

**Interfaces:**
- Consumes: `RiotRateLimiter(per_second, per_2min)` public constructor.
- Produces: Backward-compatible keyword-only `clock` and `sleep` callables.

- [ ] **Step 1: Write the virtual-clock regression test**

Replace wall-clock assertions with:

```python
class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_advances_virtual_clock_when_second_limit_is_reached():
    clock = VirtualClock()
    limiter = RiotRateLimiter(
        per_second=2,
        per_2min=10,
        clock=clock,
        sleep=clock.sleep,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == pytest.approx(1.05)
    assert clock.now == pytest.approx(1.05)
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest tests/core/test_riot_rate_limiter.py::test_rate_limiter_advances_virtual_clock_when_second_limit_is_reached -q
```

Expected: FAIL with `TypeError` because `RiotRateLimiter.__init__` does not accept `clock` or `sleep`.

- [ ] **Step 3: Inject the minimal time dependencies**

In `core/riot_api.py`, import `Callable` from `collections.abc` and change the constructor to:

```python
def __init__(
    self,
    per_second: int = 18,
    per_2min: int = 95,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
):
    self.per_second = per_second
    self.per_2min = per_2min
    self._clock = clock
    self._sleep = sleep
    self.lock = threading.Lock()
    self.timestamps_1s = collections.deque()
    self.timestamps_120s = collections.deque()
```

Inside `acquire()`, replace `time.time()` with `self._clock()` and replace the limiter's `time.sleep(wait_time)` with `self._sleep(wait_time)`. Do not alter retry sleeps elsewhere in `RiotAPI`.

- [ ] **Step 4: Verify GREEN and zero wall-clock delay**

Run:

```powershell
$watch = [System.Diagnostics.Stopwatch]::StartNew()
& .\.venv312\Scripts\python.exe -m pytest tests/core/test_riot_rate_limiter.py -q
$watch.Stop()
$watch.Elapsed.TotalSeconds
```

Expected: all rate-limiter tests pass and the file completes well below one second after interpreter startup.

- [ ] **Step 5: Commit the deterministic-time slice**

```powershell
git add core/riot_api.py tests/core/test_riot_rate_limiter.py
git commit -m "test: inject virtual clock into riot limiter"
```

### Task 6: Verify the fast suite and remove remaining false-test warnings

**Files:**
- Modify: only test modules identified by the warning/error output

**Interfaces:**
- Consumes: Structured suite, markers, and lazy imports from Tasks 1–5.
- Produces: A green pull-request test selection with return-value warnings treated as failures.

- [ ] **Step 1: Run the fast suite with return warnings promoted to errors**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest -m "not slow" -q -W error::pytest.PytestReturnNotNoneWarning
```

Expected: exit 0 with no `PytestReturnNotNoneWarning`.

- [ ] **Step 2: Correct any remaining return-value test at the behavior boundary**

For each warning, remove the returned status and assert the observable result. Use:

```python
with pytest.raises(ExpectedException):
    operation_that_must_fail()
```

for expected failures, and `pytest.skip("dependency unavailable")` only when the dependency is intentionally optional for the selected environment. Do not catch `Exception` to convert failures into booleans.

- [ ] **Step 3: Re-run the fast suite and targeted package suites**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest tests/core tests/scripts tests/analytics -m "not slow" -q -W error::pytest.PytestReturnNotNoneWarning
& .\.venv312\Scripts\python.exe -m pytest -m "not slow" -q -W error::pytest.PytestReturnNotNoneWarning
```

Expected: both commands exit 0.

- [ ] **Step 4: Commit the fast-suite slice if files changed**

```powershell
git add tests
git commit -m "test: enforce assertion-based fast suite"
```

### Task 7: Add the Python 3.12 fast/full GitHub Actions split

**Files:**
- Create: `.github/workflows/test.yml`
- Modify: `.github/workflows/ci.yml`
- Preserve: `.github/workflows/playwright.yml`

**Interfaces:**
- Consumes: `requirements-dev.txt` and pytest markers.
- Produces: Fast tests on pull requests/pushes and full tests on protected-branch pushes.

- [ ] **Step 1: Create `.github/workflows/test.yml`**

Use this workflow:

```yaml
name: Python tests

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  fast:
    name: Fast pytest suite
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install development dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements-dev.txt
      - name: Run tests excluding slow cases
        env:
          TF_CPP_MIN_LOG_LEVEL: "3"
        run: python -m pytest -m "not slow" -q

  full:
    name: Full pytest suite
    if: github.event_name == 'push'
    needs: fast
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install development dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements-dev.txt
      - name: Run complete test suite
        env:
          TF_CPP_MIN_LOG_LEVEL: "3"
        run: python -m pytest -q
```

- [ ] **Step 2: Convert the existing `ci.yml` test job to Python 3.12 lint only**

Keep its checkout and flake8 commands, set a single `python-version: "3.12"`, rename the job `lint`, and remove the `Run unit tests with pytest` step. This prevents duplicate full-suite execution on pull requests while preserving the existing quality check.

- [ ] **Step 3: Validate workflow syntax and intended commands**

Run:

```powershell
& .\.venv312\Scripts\python.exe -c "import yaml, pathlib; [yaml.safe_load(pathlib.Path(p).read_text(encoding='utf-8')) for p in ('.github/workflows/ci.yml', '.github/workflows/test.yml')]; print('workflow-yaml-ok')"
rg -n "python-version:.*3\.12|pytest -m|pytest -q|requirements-dev\.txt" .github/workflows/ci.yml .github/workflows/test.yml
```

If PyYAML is unavailable, use Ruby's built-in parser when present:

```powershell
ruby -e "require 'yaml'; YAML.load_file('.github/workflows/ci.yml'); YAML.load_file('.github/workflows/test.yml'); puts 'workflow-yaml-ok'"
```

Expected: YAML parsing succeeds; `test.yml` contains both pytest commands and both jobs install `requirements-dev.txt`; `ci.yml` contains no pytest command.

- [ ] **Step 4: Commit the CI slice**

```powershell
git add .github/workflows/ci.yml .github/workflows/test.yml
git commit -m "ci: split fast and full pytest runs"
```

### Task 8: Run the complete acceptance gate

**Files:**
- Verify: repository-wide diff and final test tree

**Interfaces:**
- Consumes: All prior tasks.
- Produces: Fresh evidence for every acceptance criterion.

- [ ] **Step 1: Verify the final tree and static hygiene**

Run:

```powershell
Get-ChildItem tests -Recurse -File | Select-Object -ExpandProperty FullName
Get-ChildItem tests -File -Filter 'test_*.py'
rg -n "sys\.path\.insert|PytestReturnNotNoneWarning" tests pytest.ini
git diff --check
```

Expected: domain packages and `fixtures_data.py` are present; no root `test_*.py` files exist; no `sys.path.insert` or warning filter exists; `git diff --check` is silent.

- [ ] **Step 2: Re-measure collection three times**

Run:

```powershell
1..3 | ForEach-Object {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & .\.venv312\Scripts\python.exe -m pytest --collect-only -q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "collection failed" }
    $watch.Stop()
    [math]::Round($watch.Elapsed.TotalSeconds, 3)
}
```

Expected: every run is below 3.0 seconds. Report all three values rather than only the fastest.

- [ ] **Step 3: Run the exact CI fast command**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest -m "not slow" -q
```

Expected: exit 0.

- [ ] **Step 4: Run the full suite with false-test warnings treated as errors**

Run:

```powershell
& .\.venv312\Scripts\python.exe -m pytest -q -W error::pytest.PytestReturnNotNoneWarning
```

Expected: exit 0, no failures, no errors, and no `PytestReturnNotNoneWarning`.

- [ ] **Step 5: Review the final diff and working tree**

Run:

```powershell
git status --short
git diff --stat main...HEAD
git diff main...HEAD -- pytest.ini core/riot_api.py tests scripts .github/workflows
```

Expected: only the approved test-suite, rate-limiter, scripts, documentation, and workflow changes are present.

- [ ] **Step 6: Commit any final verification-only corrections**

```powershell
git add pytest.ini core/riot_api.py tests scripts .github/workflows docs/superpowers
git commit -m "test: complete suite hygiene refactor"
```

Skip this commit when the working tree is already clean.

