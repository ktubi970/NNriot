# Serving Guards and Swap Diagnostics Implementation Plan

> **For agentic workers:** Use test-driven development for the reproduced defects and independent review before completion.

**Goal:** Close the calibration/validation bypasses identified in the approved audit, correct swap diagnostics, adapt legacy tests, and pass the targeted suite.

**Architecture:** Keep PredictionService as the inference facade. Derive verification state only from checked artifacts, bind serving approval to the complete evaluated configuration, and preserve the distinction between draft symmetry and map-side bias.

**Tech Stack:** Python, TensorFlow/Keras, pytest, JSON manifests, Flask.

**Spec:** AGENTS.md and the service/measurement requirements in `docs/superpowers/specs/2026-10-16-project-stabilization-design.md`, limited to the defects authorized in this chat.

## Constraints and decisions

- Preserve all pre-existing changes; work in the current checkout because the audited service and its dependencies include uncommitted files. Scoped pre-edit copies are under `scratch/serving-guards-baseline`.
- No operational database, checkpoint, calibration, candidate manifest or promotion changes. Tests use temporary artifacts and databases.
- Reject manifest-supplied internal fields. An unavailable calibration remains null and forces abstention.
- Missing verified independent validation always forces abstention, including services injected in memory.
- A serving report must bind checkpoint, calibration, feature sources, split/cohort, and the exact decision policy. A changed threshold invalidates actionability.
- No invented quality threshold or model promotion. Structural evidence checks do not establish the honesty of externally supplied measurements.
- Run the user-requested targeted suite plus directly affected tests; do not expand into unrelated stabilization work.

## Tasks

- [x] **Artifact and decision gates** — add failing cases in `tests/test_serving_validation_gates.py`, implement in `core/serving_contract.py` and `core/prediction_service.py`, document the report schema in `docs/model_serving/README.md`. Cover reserved JSON keys, missing calibration/manifests/validation, incomplete or stale evidence, policy changes, a genuinely valid positive fixture, and abstention response metadata.
- [x] **Calibration fit verification** — isolate failed/missing/non-boolean optimizer success in `tests/test_model_calibration.py`, then tighten `core/calibration.py` to match the save contract.
- [x] **Swap diagnostics and coverage** — add 14D/18D/46D invariant coverage and executable diagnostic tests; fix training diagnostics to inspect the draft head on validation data, retain winner side bias, use actual mirror predictions for the compiled-model probability, and clarify affected docstrings.
- [x] **HTTP and legacy tests** — replace obsolete trainer mocks with the facade, keep positive calibrated/validated cases explicit, and verify missing calibration/validation yields abstention through both endpoints.
- [x] **Integration and review** — run the original 13-file targeted suite with new regression files and affected training tests; inspect the incremental diff against the baseline, resolve review findings, and record exact outcomes.

## Review focus

Reserved/internal fields cannot manufacture trust; evidence remains bound after temperature or threshold changes; in-memory and HTTP paths agree on abstention; valid evidence can still authorize predictions; swap checks never remove a legitimate side bias.

## Verification outcome

Integrated targeted suite: 280 passed. After the review-driven history-window guard, the 13 directly affected suites: 146 passed. No failures; counts overlap. See `docs/model_serving/serving_guards_verification_20261018.md` for exact commands, warnings and scope.
