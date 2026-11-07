# Served model contract and common-cohort comparison

> Execution: inline, in the existing checkout to preserve the user's local training and frontend archive changes. The detailed user request is the specification and authorizes implementation through verification.

**Goal:** Make the served checkpoint and feature semantics explicit, reject incompatible artifacts, and deliver an honest reproducible comparison.

**Architecture:** A versioned contract and hashed model manifest sit behind PredictionService. A read-only result-end ordered history provider supplies pre-match priors. Evaluation freezes membership before inference, keeps calibration separate from test, reports failures before intersecting predictions, and never promotes from exploratory results.

**Constraints:** Preserve existing local edits and archive. No database migration or training on final test. Unknown provenance stays unknown. Composition and side-aware outcome remain separate; a legacy checkpoint does not acquire new heads merely because source code changed.

- [x] Contract: exact 14/18 layouts, strict roles, supplied prior lengths and availability; reject unsupported spatial encoder provenance; validate model shapes, probabilities, checkpoint/calibration hashes.
- [x] Service: deterministic default checkpoint, explicit manifest override, raw/calibrated fields and abstention when calibration unavailable; HTTP routes use facade with explicit errors.
- [x] History: replay only results ending strictly before prediction time, use Riot team IDs rather than participant row order, include historical timeline priors; test overlapping matches and equality boundary.
- [x] Evaluation: inventory direct candidate artifacts and splits, freeze recent cohort JSONL with labels/patch/timestamps and SHA, run identical matches and baselines, retain per-match failures, metrics, Wilson risk/coverage and paired temporal block bootstrap.
- [x] Verification: regression tests for contract/calibration/history/service, existing permutation and draft/side-bias tests; independent review of final changes.
- [x] Documentation: schema_version table v14, explicit legacy reference limitations, commands/results and frontend archive status; separate original status and task artifacts.

Review focus: artifact tampering; 18D with absent timeline history; malformed/duplicate roles; overlapping historical results; missing winner head or invalid probability; missing spatial encoder provenance; preexisting training evidence not linked to current checkpoint.
