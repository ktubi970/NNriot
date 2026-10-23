# Exécution Native — dataset aligné

Les six tâches fonctionnelles sont réalisées sur `codex/training-serving-parity`.
Le journal, les snapshots de départ et les diffs attribuables sont conservés
dans [execution/](../../runs/aligned_dataset_20261009_reviewed/execution/).
Base et HEAD : `3bb93c7` ; aucun commit, merge ou push mélangeant les dépendances
préexistantes. Le répertoire de travail et le journal ignoré sont conservés.

Réception : [aligned_dataset_verification.md](aligned_dataset_verification.md).
Suite finale : **164 passed, 325 warnings in 800.66s (0:13:20)**, exit 0. Replay : 148 458 comparaisons exactes.
SQLite : les mêmes 20 IDs fixés avant comparaison, priors et tenseurs exacts.
Source et tenseurs 14D identiques aux snapshots précédents ; nouveaux hashes
de runtime dans le manifeste final, snapshots antérieurs préservés.

Une seule [revue indépendante](../../runs/aligned_dataset_20261009_reviewed/execution/final-review.md)
a été réalisée par gpt-6-astra avec contexte neuf. Son Important sur les
timelines sans observations a été reproduit par sept cas RED, corrigé dans
une seule passe et vérifié GREEN (36 tests), puis par la suite finale.
Pas de seconde revue. Aucun Critical ou Minor étayé n'a été rapporté.

## Décisions consignées, dans leur ordre

- Ruling: Travail dans le checkout D:/projet/NNriot sur codex/training-serving-parity — périmètre approuvé et dépendances locales non commitées; préserver ces dépendances prime une copie incomplète dans un worktree — coût: isolation Git limitée.
- Ruling: Les helpers Bash échouent sous le sandbox Windows (NtCreateDirectoryObject); utiliser leur équivalent Python/PowerShell, mêmes briefs et ledger — coût: comptabilisation manuelle des étapes.
- Ruling: Ne pas créer de commits mêlant les nouvelles modifications aux dépendances du chantier précédent non commitées; conserver un diff par rapport au snapshot de départ pour la revue — coût: intégration Git à effectuer séparément.
- Task 3: Ruling: CLI prints coverage summary and keeps all per-match failures in availability.json/dataset.json — prevents unbounded console output with 50k unavailable 18D matches — cost: detailed causes require reading the saved file.
- Task 3: Ruling: preserve train/validation/test JSON insertion order in model.split.json — existing experiment reader requires that order; canonical sorting remains for other artifacts — cost: this companion file uses a dedicated serialization option.
- Task 5: Ruling: exclude target creations tied to the last warmup creation — strict global history would otherwise have fewer than 10000 preceding rows in the frozen snapshot — cost: marginal loss of tied targets, recorded as history_context_incomplete. RED observed [20,30] instead of [30].
- Task 5: Ruling: load each NPZ array once per partition rather than once per record — NumPy lazily reloads arrays; repeated reads scale quadratically — cost: holds the three partition arrays in memory. RED observed 27 accesses instead of 9.
- Task 6: Ruling: Select only creation/ID headers in the global SQLite window before retrieving raw payloads; sort compact facts in Python — the real 61000-row run spent 18 minutes in a temp payload sort before its first row; a bounded probe read 19 payloads for a limit of 3, RED observed — cost: compact fact sort in memory, same temporal population and minima. Owned read jobs stopped before any dataset directory was created; restart uses the same arguments.
- Task 6 diagnosis: faulthandler repeatedly stops at SQLite execute; real schema places raw_json before game_mode/creation, causing blob-overflow reads during mode filtering. General header-only sort was insufficient. Ruling refinement for Task 6 source-read ruling: use the covering mode rowid index to filter creation-index headers for ordinary tables; preserve mode fallback for views/WITHOUT ROWID — identical global creation/ID window, no DB/index writes — cost: an extra in-memory rowid set. SQLite EXPLAIN confirms covering mode and range-creation indexes, without raw payload sort. Auxiliary mmap and sandbox comparisons did not justify runtime changes; mmap remains unchanged. Dataset directory still absent; interrupted read attempts kept in workspace logs.
- Task 6 source-read ruling refinement: use pure covering-index rowid/time headers, resolve match-ID ordering only for the boundary tie, fetch chosen payloads by rowid; an explicit read transaction covers all statements. Use SQLite internal temp memory to avoid disk spills; no source/schema writes. View/WITHOUT ROWID fallback retained. Full 61000 covering-header count measured 7.95s, versus header+match-ID traversals still waiting after minutes. Cost: bounded rowid/temp memory and dependency on the already installed SQLite JSON1 virtual table, used only with generated integer IDs. Mmap and external execution provided no fix and are not adopted.
- Task 6 history-only gate RED: SQLite authorizer forbidding training_dataset data reads reproduces unintended historical dependency (DatabaseError); historical priors use Riot results only. Ruling: DatabaseHistoryProvider requests a new opt-in history_only reader mode, leaving cohort/dataset callers unchanged, and uses a bounded 64 MiB SQLite page cache — removes unnecessary ML payload reads and avoids page-cache thrashing during the fixed panel — cost: one explicit reader option and up to 64 MiB per connection. Initial dataset/source/148458 tensor comparisons remain archived; any new runtime binding will be prepared into a fresh directory from the same frozen source and partitions, then replayed, never rewritten in place.
- Final: Ruling: strict timeline observations are opt-in on the shared serving/aligned history path; the legacy extractor keeps its permissive default — existing legacy resilience tests intentionally accept missing measurements and must not silently become a new certified pipeline — cost: legacy exploratory paths remain uncertified and permissive.
- Final: Ruling: predictive quality and hyperparameters set aside by reviewer remain a future experiment — this plan explicitly prepares data without real training/test metrics — cost: no measured predictive gain from this delivery.
- Final: Ruling: timeline acquisition/merge set aside by reviewer remains a separate task — no acquisition/migration is authorized here and recent source has zero timelines — cost: recent 18D remains blocked.
- Final: Ruling: historical global test secrecy set aside by reviewer is not claimed — stored acquisition/research history cannot establish it; the reservation is for the future experiment only — cost: unknown earlier research exposure may limit independence.
- Final: Ruling: production promotion/calibration set aside by reviewer remains unverified with current checkpoint — no new independently validated model exists in scope — cost: service retains abstention and no actionable validation.
- Final: Ruling: coordinated replacement of artifacts and manifest hashes set aside by reviewer is not claimed detectable — local SHA bindings ensure integrity relative to that manifest, not a signed trust anchor — cost: no cryptographic authenticity against a coordinated replacement.

## Minor différés

Aucun.

Les cinq points que le reviewer a laissés hors jugement disposent chacun
d'une décision explicite ci-dessus. Leurs limites ne sont pas des résultats
prédictifs certifiés. Les commandes et hashes du run figurent dans le rapport
de réception ; le [journal complet](../../runs/aligned_dataset_20261009_reviewed/execution/progress.md)
conserve les observations RED/GREEN et les tentatives interrompues.
