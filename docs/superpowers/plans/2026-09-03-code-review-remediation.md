# Code Review Remediation Plan (P0, P1, P2 + Final Coherence Check)

Implementation plan to resolve all discrepancies, bugs, and architecture debt discovered during the code review on branch `refactor/architecture-and-design-patterns`.

## User Review Required

> [!IMPORTANT]
> - **P0 Phase (Agent 1)**: Disambiguates `/api/players/search` parameter (`q` vs `query`), aligns `/api/predict/custom` response parsing keys (`blue_win_probability`), and separates the Jinja2 dashboard (`/dashboard`) from the React SPA Cockpit (`/`) so Playwright E2E suites run deterministically.
> - **P1 Phase (Agent 2)**: Decouples `PlayerService` and `SystemService` from HTTP protocol return tuples, protects endpoints against unhandled `request.get_json()` crashes with strict 400 Bad Request responses, and prevents destructive log loss on `/api/batch/status`.
> - **P2 Phase (Agent 3)**: Defers background TensorFlow loading until explicit server startup, resolves `sys.path` and Windows `cp1252` encoding in `tests/test_components.py`, and aligns `ModelFactory` with `VECTOR_DIM`.
> - **Final Coherence Phase (Agent 4)**: Executes comprehensive end-to-end coherence check, verifies build integrity (`npm run build`), executes full regression test suite, inspects git diff for clean code style and boundary isolation, and outputs a final sign-off report.

## Proposed Changes

Each phase is executed by a dedicated subagent:

---

### Phase 1: Priority P0 — Critical Discrepancies & Contract Mismatches (Agent P0)

#### [MODIFY] [api.py](file:///d:/projet/NNriot/app/routes/api.py)
- In `api_players_search`: accept both `request.args.get("q")` and `request.args.get("query")`.
- In `api_predict_custom`: provide `blue_win_prob` and `red_win_prob` alongside `blue_win_probability` for backwards compatibility.

#### [MODIFY] [api.ts](file:///d:/projet/NNriot/frontend/src/services/api.ts)
- In `searchPlayers`: send `?q=${encodeURIComponent(query)}`.
- In `predictMatch`: read `data.blue_win_probability ?? data.blue_win_prob ?? data.multi_output?.winner?.team_a`. Map expected kills from `data.multi_output?.kills`.

#### [MODIFY] [views.py](file:///d:/projet/NNriot/app/routes/views.py)
- Expose `@views_bp.route("/dashboard")` to explicitly serve the classic Jinja2 template `templates/index.html`.

#### [MODIFY] [dashboard.page.ts](file:///d:/projet/NNriot/e2e/pages/dashboard.page.ts)
- Update `goto()` to navigate to `/dashboard` so `dashboard.spec.ts` tests the legacy dashboard while `cyber_cockpit.spec.ts` tests `/` (the React SPA).

#### [MODIFY] [playwright.yml](file:///d:/projet/NNriot/.github/workflows/playwright.yml)
- Add `npm run build` prior to `npx playwright test`.

---

### Phase 2: Priority P1 — Architecture & Error Resilience (Agent P1)

#### [MODIFY] [player_service.py](file:///d:/projet/NNriot/app/services/player_service.py)
- Strip HTTP status codes (`return players, 200` -> `return players`).
- Raise `ValueError` on missing or invalid arguments instead of returning `{"error": ...}, 400`.

#### [MODIFY] [system_service.py](file:///d:/projet/NNriot/app/services/system_service.py)
- Strip HTTP status codes and fix stale fallback import to `from core import database`.

#### [MODIFY] [api.py](file:///d:/projet/NNriot/app/routes/api.py)
- Wrap service calls and handle `ValueError` as HTTP 400.
- Guard `request.get_json(silent=True) or {}` across `api_link_players` and `api_predict_custom`.

#### [MODIFY] [history.py](file:///d:/projet/NNriot/app/routes/history.py)
- Guard `request.get_json(silent=True) or {}` in `api_batch_collect`.
- Implement non-destructive rolling log buffer in `api_batch_status`.

#### [MODIFY] [api.spec.ts](file:///d:/projet/NNriot/e2e/api.spec.ts)
- Replace `expect([400, 422, 500]).toContain(...)` with strict `expect([400, 422]).toContain(...)`.

---

### Phase 3: Priority P2 — Tooling, Performance & Compatibility (Agent P2)

#### [MODIFY] [core.py](file:///d:/projet/NNriot/app/core.py)
- Remove module-level top-level execution of `init_trainer_async()`.

#### [MODIFY] [cli.py](file:///d:/projet/NNriot/cli.py)
- Call `init_trainer_async()` inside `run_web()` before starting Flask.

#### [MODIFY] [test_components.py](file:///d:/projet/NNriot/tests/test_components.py)
- Inject repository root to `sys.path`.
- Replace Unicode emoji glyphs (`\u23ed\ufe0f`, `\u26a0\ufe0f`) with ASCII equivalents (`[SKIP]`, `[WARN]`) to avoid Windows `cp1252` encoding crashes.

#### [MODIFY] [json_utils.py](file:///d:/projet/NNriot/core/json_utils.py)
- Update `if __name__ == '__main__':` import from `import test_data` to `from tests import test_data`.

#### [MODIFY] [model_factory.py](file:///d:/projet/NNriot/core/model_factory.py)
- Import `VECTOR_DIM` from `core.feature_labels` as default for `input_dim`.

#### [MODIFY] [config.py](file:///d:/projet/NNriot/core/config.py)
- Add `http://localhost:5173` and `http://127.0.0.1:5173` to default `CORS_ORIGINS`.

---

### Phase 4: Final Coherence & Whole-Branch Integration (Agent Coherence Check)

**Role:** Autonomous Senior Code Auditor & Systems Integrator  
**Mission:** Thoroughly review the complete branch state across all layers (Core, App, Services, Routes, Frontend, E2E, CI), verify that all four priorities form a seamless system, ensure zero regressions, and generate a final sign-off report.

#### Scope of Audit:
1. **API Contract Verification**:
   - Verify that all endpoints in `frontend/src/services/api.ts` accurately map to routes in `app/routes/`.
   - Ensure both `q` and `query` parameters respond identically.
   - Ensure custom prediction response matches frontend types with zero fallback degradation.
2. **Architecture & Boundary Validation**:
   - Confirm domain services (`PlayerService`, `SystemService`) contain zero HTTP concerns (no `jsonify`, no status code tuples).
   - Ensure repositories in `core/repository.py` are cleanly decoupled from database implementation.
   - Confirm no circular dependencies or unintended module-level side effects exist.
3. **Build & Test Suite Execution**:
   - Run `npm run build` to confirm production asset compilation.
   - Run Python test suite (`pytest`) to confirm 100% tests passing without failures or errors.
   - Run `tests/test_components.py` standalone to confirm Windows `cp1252` console compatibility.
4. **Git Diff Hygiene & Documentation**:
   - Check `git diff` for dead code, temporary artifacts, or inconsistent comments.
   - Ensure `README.md` and documentation accurately reflect modern module paths (`core/`, `scripts/`, `app/services/`).

---

## Verification Plan

### Automated Tests
1. **P0 Agent verification**:
   - `& ".\.venv312\Scripts\python.exe" -m pytest tests/test_predict_api.py -v`
   - `& ".\.venv312\Scripts\python.exe" -c "from app import app; c = app.test_client(); r=c.get('/dashboard'); assert r.status_code==200; print('OK')"`
2. **P1 Agent verification**:
   - `& ".\.venv312\Scripts\python.exe" -m pytest tests/test_services.py tests/test_predict_api.py -v`
3. **P2 Agent verification**:
   - `& ".\.venv312\Scripts\python.exe" tests/test_components.py`
   - `& ".\.venv312\Scripts\python.exe" -m pytest tests/test_feature_labels.py tests/test_multi_output_trainer.py -v`
4. **Agent 4 Coherence Check verification**:
   - `npm run build`
   - `& ".\.venv312\Scripts\python.exe" -m pytest`
   - Review git status and unified diff
