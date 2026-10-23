# Review package — native training/serving parity

Branch codex/training-serving-parity, base and HEAD 3bb93c7. The checkout was already dirty; no mixed commits were made. Review implementation.patch (local snapshot → current) rather than git diff HEAD, which includes unrelated prior work. The baseline copies in baseline/ are the exact source before native implementation. Do not edit files, Git state or processes. Do not dispatch subagents.

Plan: docs/superpowers/plans/2026-10-09-training-serving-parity-and-frozen-dataset.md
Spec: docs/superpowers/specs/2026-10-09-training-serving-parity-design.md
Ledger decisions: progress.md, every Ruling line.
This delivery package includes the single Important review fix; original reviewed input is review-package-before-fix.md and reviewed-before-fix.patch. No second review was dispatched.

Code review template: C:/Users/ktubi/.codex/plugins/cache/claude-plugins-official/superpowers/6.4.1/skills/requesting-code-review/code-reviewer.md

Evidence: final-tests.log (16 targeted suites, existing warnings; exact final count in the log), task-*-tests.log and task-6-admission-tests.log, payload-read-red.log / payload-read-green.log, history-only-red.log / history-only-green.log, source-window-final-tests.log, verify-review.log and live-audit-review.log. Dataset evidence: runs/aligned_dataset_20261009_reviewed/{dataset,artifact_checks,replay_verification,live_panel,live_parity,source_inventory}.json. Narrative report: docs/model_serving/aligned_dataset_verification.md. Large facts and tensors stay under runs; don't reread them just to duplicate the completed whole-dataset replay (148458 exact comparisons).

Changed existing files:
- core/features/timeline_metrics.py
- core/prediction_service.py
- core/pre_match_history.py
- core/serving_contract.py
- scripts/train_deepsets_v3.py
- scripts/train_deepsets_v4_timeline.py
- docs/pre_match_training.md
- docs/model_serving/README.md

New files:
- core/frozen_history.py
- core/aligned_features.py
- core/aligned_partitions.py
- core/aligned_artifacts.py
- core/aligned_training.py
- scripts/prepare_aligned_dataset.py
- tests/aligned_dataset_fixture.py
- tests/test_frozen_history.py
- tests/test_aligned_partitions.py
- tests/test_aligned_artifacts.py
- tests/test_aligned_training.py
- tests/test_aligned_parity.py
- docs/model_serving/aligned_dataset_verification.md
- docs/model_serving/aligned_dataset_execution.md
- scratch/audit_aligned_live.py

Formalization regenerated the existing docs/model_serving/{feature_contracts.v1,reference.manifest}.json and manifests/ bindings. Those are dependencies originating in the earlier serving task, not newly created architectures. Checkpoint and calibration files were not written by this task, serving remains unverified. No real pilot training or predictive test metric was computed. 18D real data remains explicitly blocked because recent source has no timelines. Test is reserved for this future experiment, not claimed globally research-unseen.

Review Focus (verbatim from plan):
1. Partie commencée plus tôt mais terminée après le cutoff, ou terminée exactement au cutoff : priors inchangés jusqu'à sa disponibilité stricte — tâche 1.
2. Créations ex æquo et fin tardive à une frontière train/validation/test : regrouper les créations et purger le résultat indisponible — tâche 2.
3. Index participants vide/incomplet, ou résultat historique sans label ML : parité des priors avec le service ; absence de label ML historique sans exclusion du résultat Riot — tâches 1 et 5.
4. Dataset récent sans timelines, fichier modifié, fichier référencé hors du répertoire : refus explicite, aucune reconstruction silencieuse — tâches 3 et 4.
5. Cache legacy fourni avec le dataset aligné, ou accès aux labels test pendant un pilot : refus avant apprentissage et aucune ouverture du test — tâche 4.

Assess realistic inputs beyond the spec too, classify Critical/Important/Minor by actual user effect. Give file/line and concrete reproducer for actionable findings. List every behavior considered but declined to judge with its reason; use an explicitly empty list if none. Fresh review, read-only, no re-review dispatches. Don't rerun the entire expensive regression or trigger training. If a bounded probe is useful, use Python with -B and scratch fixture, never modify operational SQLite or the frozen run.
