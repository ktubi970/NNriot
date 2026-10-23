# Serving guards and swap verification — 18 October 2026

The approved audit corrections are implemented. Calibration and independent
serving approval now come from checked artifacts. Missing or changed evidence
forces abstention, including the in-memory service and HTTP status paths.

## Implemented contract

- Manifest JSON cannot supply verifier-owned underscore-prefixed fields.
- Calibration loading requires literal `fit.optimizer_success: true` and retains
  the existing checkpoint, split, feature, class and fit checks.
- Serving approval binds the checkpoint, calibration, runtime sources, dataset,
  chronological disjoint split, exact held-out test cohort and versioned decision
  policy. Missing bindings or incomplete reports are rejected.
- In-memory flags cannot establish calibration. Model/temperature changes revoke
  calibration; missing or changed manifest, thresholds or history-window settings
  revoke serving approval. Calibrated probabilities remain available when only
  independent validation is unavailable.
- Prediction and `/api/status` use the same verification method.
- Training diagnoses `draft_advantage` on validation for both DeepSets formats,
  with explicit transformation and tolerance `1e-5`. The final test is evaluated
  once. Legacy heads report the draft check as unavailable.
- Compiled-model diagnostics read the actual mirrored winner probability for
  both evidential and ordinary softmax heads. Importing the script opens no model.
- Fresh-model tests cover permutations and fixed-side composition swaps for
  14/18/46-dimensional inputs, with and without side bias. Perspective inversion
  is tested separately.
- The two obsolete serving tests now use the real facade and checked synthetic
  artifacts, replacing only Keras deserialization. Endpoint parity checks compare
  actual encoded tensors and analytically expected temperature scaling.

## Red/green evidence

Before source correction, regression runs reproduced 21 serving-gate failures,
6 optimizer-success failures, 7 swap-diagnostic failures, 3 HTTP-status failures,
and 3 history-window failures. Positive controls were retained. The history-window
case was found during independent review and then corrected and re-reviewed.

All commands below used `.venv312`, CPU inference with CUDA disabled and two
TensorFlow threads, `PYTHONDONTWRITEBYTECODE=1`, and isolated TEMP/TMP directories
under `scratch/instruction-fixes`. No operational database or checkpoint was used
as test output.

### Integrated targeted suite

```powershell
$tests = @(
  'tests/test_deepsets_layers.py',
  'tests/test_equivariant_model.py',
  'tests/test_selective_classification.py',
  'tests/test_selective_infer.py',
  'tests/test_calibrated_serving.py',
  'tests/test_model_calibration.py',
  'tests/test_model_calibration_guard.py',
  'tests/test_training_metrics_calibration.py',
  'tests/test_serving_contract.py',
  'tests/test_serving_http.py',
  'tests/test_draft_multi_output_and_serving.py',
  'tests/test_predict_api.py',
  'tests/test_final_web_app.py',
  'tests/test_serving_validation_gates.py',
  'tests/test_swap_contract.py',
  'tests/test_symmetry_diagnostics.py',
  'tests/test_shapley_credit.py',
  'tests/test_pre_match_training.py',
  'tests/test_pre_match_run.py'
)
./.venv312/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short @tests
```

**280 passed, 440 warnings, exit 0, 157.83 seconds.**
Log: `scratch/instruction-fixes/targeted-final/pytest.log`.

### Final history-window correction and affected suites

After the integrated run, the final history-window correction was verified with:

```powershell
$tests = @(
  'tests/test_serving_history_policy.py',
  'tests/test_serving_validation_gates.py',
  'tests/test_serving_contract.py',
  'tests/test_calibrated_serving.py',
  'tests/test_model_calibration_guard.py',
  'tests/test_predict_api.py',
  'tests/test_final_web_app.py',
  'tests/test_draft_multi_output_and_serving.py',
  'tests/test_shapley_credit.py',
  'tests/test_serving_http.py',
  'tests/test_pre_match_history_serving.py',
  'tests/test_frozen_history.py',
  'tests/test_aligned_parity.py'
)
./.venv312/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short @tests
```

**146 passed, 155 warnings, exit 0, 35.63 seconds.**
Log: `scratch/instruction-fixes/history-policy-final/pytest.log`.
These counts overlap and must not be summed as unique tests.

Warnings are Keras/NumPy `__array__` deprecations and optimizer-variable loading
warnings in existing model-factory tests. Scoped `git diff --check` passed.

## Review and limits

The source/test changes were inspected against pre-edit snapshots in
`scratch/serving-guards-baseline`, preserving the existing dirty checkout. The
independent review's history-window finding is closed by the final guard and its
regression cases. There are no unresolved findings within this correction scope.

Artifact binding verifies structural consistency; it does not authenticate an
external report author's measurements or establish predictive quality. Existing
candidates retain their validation status. No operational artifacts or historical
source hashes were rewritten, and no candidate was promoted. The whole repository
test suite was not run; the commands above state the verified scope.
