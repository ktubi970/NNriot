# Independent review — ordered matchup integration

Final code-review checkpoint, 2026-09-27. **No unresolved actionable code finding remains in the reviewed scope.** This is a code-review conclusion, not a claim that the full test suite or final benchmark process has completed.

Reviewed the approved plan/spec, initial worktree status, core extractor and result-end reader, new provenance and feature-contract APIs, shared training preparation, all four consumers, integrated benchmark, and new tests (including untracked files). The existing player-prior/selective/GPU work was not treated as this implementation diff. No implementation file was changed by the reviewer.

## Findings resolved and re-reviewed

| Finding | Resolution inspected |
| --- | --- |
| P2: v3 ignored an explicit missing cache | `train_deepsets_v3` now requires an existing file before streaming. Regression asserts no streaming and no output creation. |
| P2: general CLI ignored cache without manifest | CLI rejects `--splits-cache` unless `--split-manifest` is supplied, before DB access. Regression added. |
| P2: corrupt DEFLATE bypassed explicit exclusion | Reader catches `zlib.error`; valid-plus-corrupt fixture verifies valid result retained and invalid counter incremented. Parent recorded RED then 27-test GREEN evidence. |
| P2: ordered raw candidate rejected without calibration | Ensemble supports absence and records `calibration_applied=false`. Present ordered calibration requires a temperature and matching contract hash; probability application validates finite positive temperature. Optional calibration behavior is preserved. |
| P2: new ordered manifest could recertify changed priors | `validate_replay_feature_provenance` checks a present source companion contract against split and complete enriched hashes. Both general CLI and v3 call it after restoration and before result-end resolution/output. Prior-only mutation regression covers both paths without a supplied end artifact. Historical manifests without a contract remain supported. |

## Guarantees examined

- Ordered extraction ingests only train events whose explicit end is strictly before query creation. It retains row indexing and extracts original/swap at the same tracker state. Low-level tests exercise own/future labels, holdout labels, equal timestamps and permutations.
- Complete enriched hashes include attached priors and labels; result-end provenance binds exact train membership/creation, units, policy and diagnostics. Source errors and total exclusion fail explicitly.
- Result-end SQLite access uses a separate mode=ro connection, query_only and no migration.
- All four consumers use the shared extractor. No old train-prefill loop remains in them. Neural weights are 5 for v2/v3 and 15 for the general ordered branch; tree weight is separately fixed at 15.
- Ensemble identifies inputs by name, accepts v3 macro inputs independently of list order, uses saved swap matrices and selects blend weights on validation only.
- The 8-dimensional training route keeps its historical extractor and does not require result ends. Its new test actually trains through that route.
- Three-neural integration tests now observe real PCGrad batches, compare delivered matchup deltas to shared extraction, check v3 macro shape, and save/reload/predict finite outputs. The batch comparison sorts per-lane deltas to account for epoch shuffle; it does not independently certify every row/label pairing, but inspected production shuffling uses the same indices for inputs and targets.
- A real fourth-consumer test now trains a v2 checkpoint and executes actual HGB/LightGBM, verifies separate extraction calls/weights, reads the report, rejects occupied-output rerun and checks source hashes.
- New output directories and exclusive sidecars protect old artifacts. General training retains its existing per-artifact protection in addition to an ordered-directory guard.
- Benchmark uses fixed A3 parameters and verifies original matrix hashes before score interpretation, with identical-library probability comparison at absolute tolerance 1e-12.

## Evidence independently inspected

- `architecture.txt`: **28 passed**.
- `contracts_final.txt`: **53 passed**.
- `ensemble_contracts_final.txt` and exit code: **48 passed**, exit **0**, including the real fourth-consumer test. Warnings are reported by Keras/NumPy.
- Durable neural artifacts in `smoke_v2_1415/{v2,v3,general}`: reviewer independently read the three contracts, calibration and metrics files. All calibration feature hashes match actual contract bytes, all saved metric numbers are finite, and all three checkpoint files are present. Representations/weights are tensors/5, v3/5 and tensors/15 respectively. This inspection does not itself rerun training or predictions.
- Durable `ensemble_v2_1500/ensemble_comparison.json` and rerun refusal log: real evaluation produced a report with NN weight 5 and tree weight 15; rerun fails with FileExistsError.
- `benchmark_summary.json` from the first completed integrated 45k run: all three variants have exact A3 original matrix hashes; every split has zero classification differences and maximum probability difference **0.0** under matching library versions. Ordered test accuracy 0.5755555556; neutral 0.5734814815; greedy 0.5305185185. This is reproduction evidence, not evidence of an active-model improvement.
- `protected-verification.json`: **22 files checked, no changed hashes**.
- `diff_check.exitcode`: **0**.

## Remaining verification limits at this checkpoint

- `full_suite.txt` had no terminal result when inspected; do not claim the entire repository suite passed until its completion/exit code is available. In particular the final training-spy and legacy8 test execution must be confirmed in that run or its dedicated result.
- The final 45k benchmark rerun is still pending. It was requested because the first process started before diagnostics schema hardening (`excluded`), so the earlier exact-prediction evidence does not replace validation of the final sidecar schema. Confirm final completion and comparison results.
- The parent reported a failed exploratory general `--no-pcgrad` smoke due to a pre-existing winner_uncertainty target mismatch, then successful normal PCGrad training. This broader branch is not certified by the successful ordered PCGrad paths; retain the limitation if discussing it.
- Result end is an optimistic availability proxy; ingestion lag and attached player-prior temporal validity remain outside this change. No serving activation or improved production accuracy is established.

The parent should append final job outcomes to the delivery evidence. Code review can close independently of those pending verification gates.

## Late targeted compatibility check — review reopened pending one fix

6. **P2 — Cache-trained models cannot use the promised exact DB fallback (`scripts/evaluate_ensemble.py:119-121`).** The evaluator first restores DB records when no discovered/explicit cache is available, verifies the complete enriched hash successfully, then rejects because the training contract contains a source pickle hash and the pickle does not exist. This makes the specified DB fallback impossible for ordinary v2 cache-trained models, even with byte-equivalent canonical enriched data. Track whether a cache was actually loaded and enforce its file SHA only in that branch. Always retain restored-manifest and complete enriched hash checks. Explicit missing cache requests must still fail. Report the cache hash as null when DB was the source. Add exact-DB-fallback acceptance and altered-prior rejection tests with a populated source_cache_sha256 contract. No implementation change made by reviewer.

This additional targeted finding supersedes the earlier no-open-findings statement until its correction is inspected.

## Final scoped re-review — late P2 resolved

The DB fallback fix has now been independently inspected. `cache_was_loaded` records the chosen source; training-cache SHA verification only runs for an actually loaded pickle, and the report emits null for DB reconstruction. Enriched and split hash checks remain mandatory. New coverage accepts an exact reconstructed cohort, rejects a prior-only mutation before output creation, and rejects an explicitly missing cache before DB loading. **Finding 6 is resolved; code review closes with no unresolved actionable findings.** This verdict supersedes the reopened status above.

Updated execution evidence: independently read `full_suite.txt` and `full_suite.exitcode`: **851 passed, 1974 warnings, exit 0**. That process loaded code before the last localized DB fallback correction, so its result validates the earlier complete state, including neural training spies and legacy8. Parent reports RED/GREEN targeted tests for the final correction; full evaluator revalidation is running in `ensemble_post_review.txt` and had no terminal output when inspected. Confirm its final exit/result before final delivery; repeating the entire full suite is not required solely for this scoped correction once its affected evaluator suite passes.

The only other remaining execution gate is completion of the final 45k benchmark with the hardened diagnostics sidecar. The first completed benchmark already reproduced A3 matrices and probabilities exactly; do not represent that as completion of the currently running final process. No new implementation or scope expansion is requested by this review.

## Delivery closure — all remaining execution gates complete

Independently read the final evidence, without launching additional tests:

- `ensemble_post_review.txt` / `.exitcode`: **12 passed, 62 warnings, exit 0**, after the DB fallback correction.
- `benchmark_final_45000.exitcode`: **0**. Final `runs/ordered_matchup_integration_validation_final/metrics.json` has `completed=true`; greedy, ordered and neutral all match A3 feature hashes, with **0.0** maximum probability difference and **0** classification differences across their split comparisons.
- Final end diagnostics contain the hardened field: requested/found/valid **31,500**, invalid/missing/excluded **0**.

**Final verdict: no unresolved actionable code finding and no remaining blocked execution gate from this review.** These completed results supersede all earlier pending-gate notes in this document. The parent is additionally reopening the final end artifact through the resolver as a delivery consistency check.

Practical limits remain: neural checks are short synthetic CPU smoke runs, not GPU qualification or model-quality evidence; the pre-existing general `--no-pcgrad` winner_uncertainty target mismatch is outside the validated normal PCGrad route; recorded end does not establish API ingestion availability, and attached player-prior temporal validity is not newly certified. The benchmark reproduces A3 and does not establish a new serving-model accuracy gain. No active model was promoted by this work.
