# Common recent cohort — exploratory comparison

exploratory; no validated winner or automatic promotion

Test: 2000 initial matches, 2000 common evaluable matches; 2026-09-16T20:52:51.592000+00:00 to 2026-09-17T11:38:52.620000+00:00.
Cohort SHA-256: `530a3ca4226620bb1fcb539a64dd9c9f1e5add322c0596db6c07ef9d794ca7cc`.

| Candidate / probability | Initial coverage | Accuracy | AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| coinflip50:raw | 100.0% | 0.5245 | 0.5000 | 0.6931 | 0.2500 | 0.0245 |
| blue_prior:raw | 100.0% | 0.5245 | 0.5000 | 0.6920 | 0.2494 | 0.0016 |
| candidate_deepsets_v3:raw | 100.0% | 0.5245 | 0.5415 | 0.6994 | 0.2528 | 0.0547 |
| candidate_deepsets_v4_compiled_gpu:raw | 100.0% | 0.5185 | 0.5343 | 0.6939 | 0.2503 | 0.0408 |
| candidate_deepsets_v4_compiled_gpu:calibrated | 100.0% | 0.5185 | 0.5343 | 0.6931 | 0.2499 | 0.0369 |
| candidate_deepsets_v4_timeline:raw | 100.0% | 0.5195 | 0.5327 | 0.6949 | 0.2508 | 0.0339 |
| candidate_deepsets_v4_timeline:calibrated | 100.0% | 0.5195 | 0.5327 | 0.6933 | 0.2500 | 0.0296 |
| candidate_deepsets_v4_timeline_gpu:raw | 100.0% | 0.5210 | 0.5373 | 0.6922 | 0.2495 | 0.0257 |
| candidate_deepsets_v4_timeline_gpu:calibrated | 100.0% | 0.5210 | 0.5373 | 0.6932 | 0.2500 | 0.0321 |
| tree_baseline:raw | 100.0% | 0.5035 | 0.5067 | 0.7217 | 0.2630 | 0.0946 |

The configured reference remains candidate_deepsets_v3. No model is declared superior or ready for actionable serving.

Risk/coverage, equal conviction thresholds, Wilson intervals, and paired chronological block bootstrap intervals are in metrics.json. Predictions and failures remain visible on the initial cohort; no failed feature row is silently dropped.

Training provenance: see protocol.json eligibility per candidate. Split dates alone do not prove training feature availability. v3 lacks a checkpoint-bound split and calibration. Spatial candidates lack encoder provenance. The stored tree is evaluated only through its original encoder; temporal overlap is reported.

Calibration artifacts are existing validation fits, not fitted on this recent test. Different calibration cohorts and repaired result-end priors make these exploratory comparisons. Current static champion profiles lack historical patch provenance. Player overlap and dependencies across bootstrap blocks limit uncertainty estimates.

Reproduce from immutable source:

`python -m scripts.compare_served_candidates --replay "D:\projet\NNriot\runs\served_model_comparison_20261009_final" --output-dir runs/served_model_comparison_replay`
