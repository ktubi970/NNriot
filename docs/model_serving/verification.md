# Verification record — 9 October 2026

The delivered comparison is [the reproducible run](../../runs/served_model_comparison_20261009_complete/report.md). Its **exploratory** classification is unchanged by passing software tests. No checkpoint is promoted or independently validated for actionable serving.

## Executed commands and results

```powershell
.\.venv312\Scripts\python.exe -m scripts.formalize_served_model
.\.venv312\Scripts\python.exe -m scripts.compare_served_candidates --replay runs/served_model_comparison_20261009_verified --output-dir runs/served_model_comparison_20261009_complete
```

Both commands exited 0. Formalization inventories 22 checkpoints (including the existing tree) and writes exact contracts and candidate manifests. Comparison freezes 10,000 historical rows, 500 validation rows and 2,000 test rows; all 2,000 test matches are in the intersection of usable models. No weights, calibration, blend or thresholds are fitted on this test. The delivered protocol binds current evaluation source hashes and records dependency versions. The older frozen source used for this command lacked that binding, which the command explicitly reported before recording the current runtime.

To replay the delivered, source-bound run without reading a changing database, use a new output directory:

```powershell
.\.venv312\Scripts\python.exe -m scripts.compare_served_candidates --replay runs/served_model_comparison_20261009_complete --output-dir runs/new_replay
```

The final targeted suite uses test-only TEMP/TMP, leaving the operational source out of test fixtures:

```powershell
$env:TEMP = 'D:\projet\NNriot\scratch\serving-tests'
$env:TMP = $env:TEMP
.\.venv312\Scripts\python.exe -m pytest tests/test_serving_contract.py tests/test_pre_match_history_serving.py tests/test_common_cohort_evaluation.py tests/test_serving_http.py tests/test_predict_api.py tests/test_draft_multi_output_and_serving.py tests/test_final_web_app.py tests/test_shapley_credit.py tests/test_equivariant_model.py tests/test_deepsets_layers.py tests/test_selective_infer.py tests/test_config_cli.py tests/test_features_pipeline.py tests/test_point_in_time_timeline.py -q
```

**Final result after the regional-context correction: 153 passed, 65 Keras/NumPy deprecation warnings, exit 0, in 118.53 seconds.**

An earlier concurrent run reported 150 passes and two 30-second subprocess import timeouts in `test_config_cli.py`; isolated retries passed in 25.17 seconds. A full retry still reported one import timeout (151 passes), so concurrency alone was not assumed to explain it. The eager TensorFlow import in the route module was replaced by a lazy facade wrapper, and the factory test now asserts TensorFlow is not imported. Both factory tests then passed in 11.76 seconds. No test was skipped and no timeout was increased; the full suite was rerun after this fix. Keras/NumPy emitted 65 deprecation warnings in serialization tests. This is the executed targeted scope, not a claim that every repository test passes.

`compileall -q` passed for the new contract/history/inventory/metrics/plots modules, configuration, service, encoder, both CLI modules, HTTP facade/routes and new tests. `git diff --check` passed on files edited for this task. The whole-workspace check also reports preexisting whitespace in `core/pcgrad.py` and `tests/test_equivariant_model.py`; those unrelated edits were preserved.

## Artifact and live-service evidence

[artifact_checks.json](artifact_checks.json) records verified source, cohort, inventory, prediction and evaluation-runtime hashes. The 24,000 shared prediction/failure rows from the successful predecessor replay are **exactly identical**, in the same order. The final replay also restores canonical NA/EUW region categories to the original tree encoder, confirmed against stored pre-match metadata. This input correction did not change its predictions on this cohort. The delivered file additionally retains 30,000 explicit exclusions from unsupported contracts, for 54,000 rows total. There are 27 probability-variant coverage entries: 22 inventoried candidates, two probability baselines and three calibrated variants. No 14D/18D test feature extraction failed. Two 14D candidates have zero inference coverage because their calibration schema/fit bindings fail verification; 15 unsupported legacy/spatial candidates have zero feature and inference coverage with reasons retained.

| Artifact | SHA-256 |
|---|---|
| Reference checkpoint | `b19429597f8b9747aa413b42cdefdd0739556378ab692ee25904c12e98f19f0f` |
| Cohort JSONL | `530a3ca4226620bb1fcb539a64dd9c9f1e5add322c0596db6c07ef9d794ca7cc` |
| Frozen source snapshot | `47bb50e2709b6b41c4729b664c0585c90f217fc19438c3da4bbd199c84022ec7` |
| Final predictions | `fe1015c46617430ec5c2fadedba815563360343e8508f907ab2e8a95f9a4b274` |
| Final metrics | `d4b829649b46e5fe6d86b74e2348a61daa296ef4906694d306d69fb66f86e33f` |

A read-only live smoke on the first frozen test match compared the actual database history provider with frozen replay at the **same explicit creation cutoff**, for both 14D and 18D: player priors were exactly equal. The real v3 `PredictionService` loaded its manifest/checkpoint and returned raw Red probability 0.5607417226, calibrated probabilities null, missing `draft_advantage` null and `UNCALIBRATED` abstention. Its one-match output differed from comparison batch inference by only `5.96e-8`. Evidence is in [live_smoke.json](live_smoke.json). The first smoke attempt omitted the frozen creation cutoff and correctly evaluated history at current time instead; that invalid comparison was corrected, not treated as a service defect.

Independent review reproduced the missing-participant-index divergence (live prior 0.5 versus replay 0.5454545). The added regression failed before the raw fallback and passed afterward, with both empty and complete indexes. Review also checked calibration rejection, timeline availability, shared creation/window/result-end rules and initial/intersection coverage. No operational DB migration, checkpoint modification or remote action was performed.

## Workspace attribution and remaining limits

[preexisting-status.txt](preexisting-status.txt) records the initial working tree. Preexisting frontend archival, architecture, PCGrad, timeline/GPU training, spatial experiments and their tests were retained. In overlapping configuration/service/encoder/docs files, this task extends that local work; the total Git diff is not solely attributable to this task.

Added for this task: `core/serving_contract.py`, `core/pre_match_history.py`, `core/candidate_inventory.py`, `core/common_cohort_evaluation.py`, `core/evaluation_plots.py`, the formalization/comparison CLI modules, contract/history/evaluation/HTTP tests and their facade fixture, the serving documentation/manifests, and comparison runs. Extended: `core/config.py`, `core/prediction_service.py`, legacy encoder deserialization, HTTP facade/routes, existing serving/API/factory tests, README, PROJECT_SUMMARY, migration documentation, `.env.example` and the Plotly dependency. `SUPERSEDED.md` marks diagnostic comparison predecessors.

The schema checked from `schema_version` is 14 (`PRAGMA user_version` is 0). React/Vite remains in `archive/frontend-2026-10-09/`. Current source bindings do not supply unknown historical training semantics. v3 has no bound training split/calibration; v4 timelines used an unverified training availability policy; the tree's recorded partitions are later than this cohort; spatial encoder provenance is missing. Historical ingestion times and patch-specific static profile availability are unknown. Results and temporal bootstrap intervals remain exploratory. Actual service action coverage is 0% until calibration and independent validation gates are satisfied.
