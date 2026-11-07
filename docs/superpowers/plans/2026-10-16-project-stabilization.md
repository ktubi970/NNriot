# NNriot Project Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger les défauts établis, livrer une application vérifiée et rendre la décision ML reproductible sans contourner l'abstention.

**Architecture:** Conserver les couches données/features/modèles/décision/service et la façade existante. Adapter les contrats de fusion par version, corriger les fixtures et diagnostics obsolètes, puis vérifier l'intégration réelle et la provenance des évaluations.

**Tech Stack:** Python, SQLite, TensorFlow/Keras, pytest, Flask, React/TypeScript, Vite, Playwright ; environnement local de référence `.venv312`.

**Spec:** `docs/superpowers/specs/2026-10-16-project-stabilization-design.md`.

## Global Constraints

- Préserver tous les changements préexistants ; aucune opération sur une base réelle pour valider les correctifs.
- Schéma courant v14, ledger `schema_version` ; fusion intermédiaire, sans publication automatique.
- Anti-symétrie draft, biais directionnel winner, permutation intra-équipe ; tolérance des nouveaux diagnostics `1e-5`.
- Toute route utilise `PredictionService` ; aucune recommandation sans calibration et validation indépendante.
- Aucun test supprimé/xfail pour masquer un échec ; aucun ancien test ML recyclé en validation indépendante.
- Aucun redémarrage/arrêt de collecte, entraînement long, remplacement du checkpoint ou push dans la phase de planification.
- Les seeds 42–46 et cinq époques constituent le protocole proposé, à verrouiller avant les expériences.

## Review Focus

- WAL committé ou écriture concurrente : sortie cohérente ou refus explicite, entrées intactes (lot 2).
- Queue v14 avec PUUID seul, identifiants locaux en collision ou région legacy inconnue : aucune perte silencieuse (lot 2).
- Calibration valide mais manifeste absent, seuil modifié ou preuve ancienne : abstention/erreur, aucune action non validée (lot 3).
- Ancien modèle sans tête draft ou nouveau winner avec biais : diagnostic honnête sans fausse violation (lot 4).
- Test déjà consommé présenté `reserved` ou données nouvelles insuffisantes : refus avant entraînement (lot 6).

## Ordre et parallélisation

Lot 0 d'abord. Lots 1–2 (données) en séquence ; lots 3 (service) et 4
(diagnostics) peuvent avancer en parallèle dans des périmètres distincts.
Lot 5 après intégration 1–4. Lot 6 après stabilisation des contrats de service.
Lot 7 ferme la stabilisation logicielle ; lot 8 est l'expérience ML séparée.
Ne pas lancer plusieurs suites TensorFlow lourdes simultanément sur cette machine.

## Lot 0 — État de référence et protection du travail existant

**Fichiers :** preuves nouvelles sous `scratch/stabilization-20261016/` ; aucune source modifiée.

- [ ] Relever HEAD/branche, diff suivi, inventaire des sources non suivies, hashes des manifests/checkpoints de référence et versions des dépendances. Ne jamais copier/imprimer `.env`.
- [ ] Réutiliser un checkout isolé approprié ou en créer un avec `using-git-worktrees`. Le peupler explicitement avec les modifications et sources non suivies nécessaires : un worktree créé depuis HEAD seul ne contient pas le travail des 9–11 octobre.
- [ ] Conserver les DB/checkpoints volumineux à leur emplacement ; les tests créent leurs propres artefacts. Pour une expérience ultérieure, utiliser des chemins explicites en lecture seule.
- [ ] Relever les fichiers de la campagne de collecte active pour éviter de les inclure dans les correctifs ; ne pas modifier son code ou son état.
- [ ] Rejouer une fois les tests historiquement en échec avec TEMP/TMP dédiés ; conserver la liste actuelle. Les 36 cas historiques sont une référence, pas une limite au nombre de défauts à traiter.

Commande initiale :

```powershell
$testTemp = Join-Path (Get-Location) 'scratch/stabilization-tests'
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
.venv312/Scripts/python.exe -m pytest tests/analytics/test_extract.py tests/test_merge_databases.py tests/test_database_v11_migration.py tests/test_calibrated_serving.py tests/test_model_calibration_guard.py tests/test_pre_match_training.py -q --tb=short
```

Résultat attendu : causes reproduites et attribuées, sans contact avec les données réelles.

## Lot 1 — Fixtures de schéma et migrations

**Fichiers :** créer `tests/database_schema_fixture.py` et `tests/test_database_schema_compatibility.py` ; modifier `tests/analytics/test_extract.py`, `tests/test_database_v11_migration.py`, `tests/test_merge_databases.py`.

**Interface produite :** `create_schema_db(path: Path, version: int) -> Path`, limitée aux schémas 12, 13, 14, construite en appliquant les migrations existantes jusqu'à la version demandée dans une base temporaire.

- [ ] Paramétrer les sept cas analytics sur 12/13/14 : version attendue explicite et mêmes assertions indépendantes sur comptes/MAX. Confirmer l'échec actuel des attentes v12 sur fixture courante.
- [ ] Vérifier le ledger, les colonnes/indices de queue et la table spatiale après migration réelle v12→v14 ; une seconde initialisation doit conserver données et version.
- [ ] Corriger la fixture de queue par insertions à colonnes nommées. Utiliser `SCHEMA_VERSION` seulement pour les scénarios « version courante », pas comme preuve unique de correction d'une migration.
- [ ] Exécuter `python -m pytest tests/analytics/test_extract.py tests/test_database_v11_migration.py tests/test_database_schema_compatibility.py -q` ; attendre zéro échec.
- [ ] Revoir puis commiter uniquement ce lot ; ne pas modifier l'extracteur analytics sans nouvelle preuve de défaut.

## Lot 2 — Fusion v12/v13/v14 sans perte

**Fichiers :** créer `core/database_merge_schema.py` ; modifier `scripts/merge_databases.py`, `tests/test_merge_databases.py`, `docs/database_merge.md`.

**Interfaces :** `read_merge_schema(conn: sqlite3.Connection) -> MergeSchema` ; descripteur immuable avec `version`, `table_columns`, `primary_keys`. Étendre `merge_databases(*, source, target, output, report, dry_run=False, legacy_queue_region=None)` sans casser les appels existants ; ajouter `--legacy-queue-region`.

- [ ] Ajouter des tests des six couples source≤cible ; rejeter les trois déclassements, schémas malformés et versions futures avant création de sortie.
- [ ] Faire utiliser le descripteur à la validation, aux compteurs, fingerprints, recherches de clés et contrôles de conservation ; rapport séparant versions source/cible/sortie.
- [ ] Adapter la queue : cible prioritaire, IDs source réalloués, PUUID seul permis, PUUID distincts non fusionnés par nom, ambiguïtés archivées. Tester chaque cas, y compris région absente et conversion `added_at→created_at`.
- [ ] Préserver les BLOBs spatiaux cibles octet par octet. Archiver les embeddings source sans provenance, conflits et orphelins avec raisons distinctes et BLOB base64 réversible. Inclure toutes les lignes dans les comptes/fingerprints.
- [ ] Rejouer répétabilité, dry-run sans artefact, WAL committé, concurrence, conservation canonique, intégrité SQLite et foreign keys. Aucun simple relâchement du garde v12.
- [ ] Exécuter `python -m pytest tests/test_merge_databases.py tests/test_database_schema_compatibility.py tests/analytics/test_extract.py -q` ; zéro échec. Vérifier le décodage intégral de l'archive dans les tests.
- [ ] Documenter matrice, région, spatial et protocole de publication séparée ; revue puis commit dédié.

## Lot 3 — Contrat de service et abstention complète

**Fichiers :** créer `core/serving_evidence.py`, `scripts/build_serving_evidence.py`, `tests/test_serving_evidence.py` ; modifier `core/prediction_service.py`, `core/serving_contract.py`, `tests/serving_fixture.py`, `tests/test_calibrated_serving.py`, `tests/test_model_calibration_guard.py`, `tests/test_serving_contract.py`, `tests/test_serving_http.py` ; ajuster les tests API consommateurs de la fixture. Documenter dans `docs/model_serving/README.md`.

**Interfaces :** conserver les signatures de `PredictionService` et des endpoints. Étendre la preuve de service avec `evidence_schema_version`, hashes checkpoint/calibration/split/cohorte/features, `decision_policy`, protocole et résultat d'acceptation ; preuves anciennes insuffisantes restent non validantes.

**Producteur/consommateur partagés :** `build_serving_evidence(*, protocol_path: Path, evaluation_path: Path, lock_path: Path, output_path: Path) -> dict` dans `core/serving_evidence.py`, exposé par le CLI homonyme avec ces quatre arguments. `validate_serving_evidence(evidence: dict, manifest: dict) -> dict` est utilisé par `load_model_manifest`. Le producteur vérifie les hashes/usage de cohorte, applique exclusivement les critères verrouillés avant évaluation et émet `validated_for_serving`, `descriptive` ou `rejected` ; politique absente/inconnue, bornes manquantes ou résultat non fini ne peuvent produire `validated_for_serving`. Il n'effectue ni fit ni inférence.

- [ ] Reproduire un service en mémoire calibré, sans manifeste et à forte conviction : exiger `UNVALIDATED_MODEL`, `is_actionable=False`, `abstained=True`.
- [ ] Remplacer les injections de `global_trainer` par la façade, avec cinq rôles canoniques par équipe. Vérifier HTTP 200 + probabilités brutes + probabilités calibrées nulles + `UNCALIBRATED` quand seule la calibration manque ; conserver 503 pour artefact invalide.
- [ ] Tester calibration appliquée exactement une fois : raw `[0.9, 0.1]`, T=2 donne `[0.75, 0.25]`, têtes auxiliaires inchangées. Une calibration seule ne rend pas la réponse actionnable.
- [ ] Supprimer la condition qui permet à l'absence de manifeste d'éviter le contrôle de validation. Les fixtures calibrées sans preuve deviennent explicitement abstentionnistes.
- [ ] Lier la validation aux artefacts et à la politique exacte ; tests de remplacement de calibration, split, cohorte, features et seuil malgré checkpoint inchangé. Rejeter une revendication `validated` incohérente ; ne jamais accepter un simple booléen de fixture comme preuve dans le chargement réel.
- [ ] Tester avec artefacts synthétiques le parcours complet producteur de preuve → manifeste → chargement de façade, puis invalider la politique ou la calibration et exiger le refus. Tester aussi des critères absents, définis après ouverture du test ou non satisfaits : aucune validation actionnable.
- [ ] Vérifier qu'un override de seuil ne réutilise pas une validation d'une politique différente. Les tests purs de la règle de conviction restent indépendants de cette autorisation de service.
- [ ] Exécuter `python -m pytest tests/test_serving_evidence.py tests/test_calibrated_serving.py tests/test_model_calibration_guard.py tests/test_serving_contract.py tests/test_serving_http.py tests/test_predict_api.py tests/test_final_web_app.py tests/test_draft_multi_output_and_serving.py tests/test_shapley_credit.py tests/test_selective_infer.py -q`.
- [ ] Revue puis commit ; aucune promotion du v3 ou des candidats pilotes.

## Lot 4 — Diagnostics draft/winner exacts

**Fichiers :** modifier `scripts/train_pre_match_model.py`, `tests/test_pre_match_training.py`, éventuellement les assertions communes de `tests/test_equivariant_model.py` ; documentation `docs/pre_match_training.md`.

**Interface :** helper `_equivariance_diagnostics(model, inputs, batch_size: int) -> dict` dans le script. Rapport explicite `symmetry_target`, `draft_swap_max_abs_error`, `draft_mirror_max_abs_error`, `permutation_max_abs_error`, `winner_swap_complementarity_required` et état de disponibilité.

- [ ] Ajouter une régression avec biais de côté non nul : draft swap/miroir valides, winner miroir égal à la transformation du biais appris pour la tête employée, pas à une constante 0,5 ou 0,525. Tolérance `1e-5`. Ajouter un faux modèle à draft volontairement asymétrique et vérifier que le diagnostic échoue.
- [ ] Mesurer les sorties brutes de `draft_advantage` et les permutations sur validation ; ne pas attribuer la complémentarité au winner avec biais. Inverser le signe de la macro au swap et la mettre à zéro au miroir ; conserver toutes les autres entrées requises. Pour le winner évidential, normaliser `softplus([-b,b])+1` ; pour le softmax classique, utiliser `P(A)=1/(1+exp(2b))`, conformément au builder.
- [ ] Si le champ historique `anti_symmetry_guarantee` est conservé, le lier explicitement au diagnostic draft ; sans tête draft, utiliser null + raison, jamais True par défaut.
- [ ] Tester les profils 14D, 18D, 46D et la sérialisation/recharge dans les tests d'invariants, sans rendre le 46D éligible au service.
- [ ] Exécuter `python -m pytest tests/test_pre_match_training.py tests/test_equivariant_model.py tests/test_deepsets_layers.py tests/test_aligned_training.py tests/test_pcgrad.py -q` ; revoir le JSON produit, puis commit.

## Lot 5 — Intégration réelle du cockpit

**Fichiers :** créer `tests/serving_artifact_fixture.py`, `scripts/run_integration_fixture.py`, `playwright.integration.config.ts`, `e2e/cockpit_integration.spec.ts` ; modifier `package.json` et `.github/workflows/playwright.yml`.

**Interfaces :** `build_serving_fixture(root: Path, mode: str) -> dict[str, str]` produit les chemins de DB/checkpoint/manifeste temporaires ; modes `calibrated_unvalidated`, `uncalibrated`, `invalid_artifact`. CLI locale `python -m scripts.run_integration_fixture --root <temp> --port <port> --mode <mode>` ; nouvelle commande `npm run test:e2e:integration`.

- [ ] Créer un petit modèle Keras et des sidecars réels dans le dossier temporaire, sans entraînement long ni accès Riot. Démarrer `create_app(TESTING=True)` sur loopback pour éviter le trainer et les collecteurs ; servir la build réelle `static/dist`.
- [ ] Ajouter un premier parcours navigateur sans interception des routes d'inférence : draft valide → API → façade → modèle → affichage des probabilités et de l'abstention.
- [ ] Couvrir modèle non calibré, modèle calibré non validé et artefact invalide ; vérifier l'absence de probabilités inventées et la propagation des erreurs. Vérifier également historique/joueur depuis SQLite temporaire.
- [ ] Garder la suite UI simulée desktop/tablette/mobile. Isoler ports, chemins et nettoyage des processus lancés par la fixture ; ne pas toucher aux services utilisateur existants.
- [ ] Exécuter `npm run build`, `npm run typecheck`, `npx tsc --noEmit`, `npm run test:unit`, `npm run test:e2e`, `npm run test:e2e:integration` ; zéro échec.
- [ ] Ajouter Python/dépendances nécessaires au job d'intégration ; conserver traces d'échec. Revue et commit.

## Lot 6 — Protection cumulative des tests ML et preuve reproductible

**Fichiers :** créer `core/evaluation_registry.py`, `tests/test_evaluation_registry.py`, `scripts/evaluate_aligned_candidates.py`, `tests/test_aligned_evaluation.py` ; modifier `scripts/prepare_aligned_dataset.py`, `tests/test_aligned_partitions.py`, `tests/test_aligned_artifacts.py` ; ajouter un registre documentaire versionné dans `docs/model_serving/evaluation_registry.json`.

**Interfaces :** `load_protected_evaluations(path: Path) -> dict` valide un registre de références hashées ; `protected_test_ids(registry: dict) -> set[str]` calcule l'union des tests consommés. `claim_test_evaluation(registry_path: Path, *, cohort_sha256: str, match_ids: list[str], protocol_sha256: str, checkpoint_sha256s: list[str]) -> str` enregistre l'usage atomique et retourne son ID. Étendre `prepare(db, output, after_cohort, *, source_limit=61000, evaluation_registry=None)` et le CLI avec `--evaluation-registry`. L'évaluateur `evaluate_locked_candidates(*, dataset_dir: Path, lock_path: Path, registry_path: Path, output_dir: Path) -> dict` appelle ce registre avant toute inférence test ; ses métriques réutilisent `core/common_cohort_evaluation.py` et les contrôles du pilote, sans réentraîner ni recalibrer.

- [ ] Recenser les tests historiques et `runs/aligned_comparison_20261011/test_usage.json`, vérifier leurs hashes/IDs et enregistrer leur usage sans réécrire les manifests figés.
- [ ] Tester qu'un ID consommé est exclu des trois partitions cibles, même si son ancien manifeste annonce `reserved`. Historique antérieur autorisé uniquement selon la politique PIT existante, sans sélection de labels du test.
- [ ] Tester registre incomplet/introuvable/altéré et manque de 3 000 nouvelles cibles : refus explicite avant entraînement, avec effectifs et déficit. Les anciens appels exploratoires restent identifiés comme tels et ne produisent pas de validation indépendante.
- [ ] Brancher le registre sur l'évaluateur réutilisable : verrou exclusif et écriture atomique avant la première inférence/consultation des labels pour évaluation. Après interruption ou échec, la cohorte reste consommée. Tester double lancement concurrent, interruption et reprise sous le même protocole ; la reprise conserve le même usage et ne peut revendiquer une nouvelle indépendance.
- [ ] Vérifier partitions disjointes, nouveau test postérieur à la dernière cohorte évaluée du registre, dates de fin strictement antérieures aux frontières, parité des colonnes 14D communes, seed/budget verrouillés et absence de lecture des labels test pendant sélection/calibration.
- [ ] Exécuter `python -m pytest tests/test_evaluation_registry.py tests/test_aligned_evaluation.py tests/test_aligned_partitions.py tests/test_aligned_artifacts.py tests/test_aligned_parity.py tests/test_common_cohort_evaluation.py -q` ; revue puis commit.

## Lot 7 — Validation globale et consolidation

**Fichiers :** `.github/workflows/ci.yml`, `.github/workflows/playwright.yml`, `.gitignore`, `README.md`, `PROJECT_SUMMARY.md`, `docs/model_serving/verification.md` et nouveau compte rendu de stabilisation.

- [ ] Exécuter la suite Python complète sur la révision intégrée : `python -m pytest tests/ --doctest-modules -q --tb=short`. Résoudre les échecs supplémentaires par cause ; ne pas s'arrêter aux seuls 36 cas historiques.
- [ ] Exécuter lint bloquant de CI sur le périmètre source/test intentionnel ; exclure environnements, archives de référence et sorties générées, sans cacher du code actif. Conserver les versions Python déjà supportées et ajouter la 3.12 utilisée pour les preuves locales si les dépendances passent.
- [ ] Rejouer la chaîne frontend du lot 5 après intégration. Ne déclarer la CI distante verte qu'après son exécution réelle ; sinon distinguer clairement la preuve locale.
- [ ] Mettre à jour le résumé avec le pilote du 11 octobre et les limites actuelles ; marquer les anciens rapports comme historiques par un ajout daté plutôt que réécrire leurs résultats.
- [ ] Pour les manifests de référence liés aux sources modifiées, vérifier la compatibilité puis produire une nouvelle formalisation traçable. Garder calibration/provenance/validation inconnues lorsqu'elles le sont ; ne pas simplement apposer un nouveau hash et annoncer une validation.
- [ ] Trier sources/tests/docs à versionner et sorties régénérables à ignorer. Ne pas ignorer tout `runs/` : conserver les preuves utiles et leurs métadonnées ; ne supprimer aucun artefact préexistant.
- [ ] Commits par lot, inventaire final des changements préexistants restants, `git diff --check` sur les modifications de stabilisation et revue indépendante finale. État propre de la branche isolée, sans prétendre nettoyer le checkout utilisateur partagé.

Acceptation logicielle : zéro échec dans les suites prévues, contrats d'abstention
vérifiés et preuves attachées à une même révision. Aucun lancement ML long n'est
nécessaire pour clore cette étape.

## Lot 8 — Confirmation ML et décision de service

**Fichiers :** nouveau protocole daté sous `docs/model_serving/`, nouveaux artefacts sous `runs/` ; réutiliser les scripts alignés existants. Toute nouvelle validation de service doit satisfaire le contrat du lot 3.

- [ ] Vérifier les nouvelles données disponibles depuis la collecte avec une capture stable en lecture seule. Ne pas interrompre ni reconfigurer une campagne existante.
- [ ] Figer train/validation/test et exclusions du lot 6. Minimum proposé : 3 000 validation et 3 000 nouveaux tests ; produire le déficit exact si impossible. L'ancien déficit de 1 656 n'est plus applicable après consommation des 1 344 matchs pilotes.
- [ ] Verrouiller le protocole avant entraînement : 14D/18D, seeds 42–46, cinq époques, même budget et mêmes données ; meilleure époque et calibration sur validation ; sélection du candidat par log-loss moyenne de validation, puis seed retenue par log-loss de validation, sans consulter le test. Fixer dans ce protocole les objectifs risque/couverture et la méthode statistique d'acceptation, ou déclarer explicitement l'expérience descriptive et inéligible à une promotion.
- [ ] Exécuter les entraînements séquentiellement sur GPU, vérifier leurs invariants et hashes, puis verrouiller le candidat avant l'ouverture unique du test. Conserver tous les runs, y compris défavorables.
- [ ] Comparer baseline, scores bruts/calibrés, variation entre seeds, delta apparié et incertitude temporelle. Les résultats par seuil fixe restent descriptifs ; aucun seuil optimisé sur test.
- [ ] Appliquer uniquement les critères d'acceptation fixés avant l'ouverture du test ; s'ils n'étaient pas définis, conserver `unverified` et demander une autre cohorte pour toute future validation de service. La réussite des tests logiciels ou un meilleur point estimé ne suffit pas.
- [ ] Produire la décision avec `scripts.build_serving_evidence` du lot 3 : candidat insuffisant/inconclusif → maintien de l'abstention ; candidat satisfaisant → paquet de preuves lié à tous les artefacts, comparaison en parallèle avec le service courant et procédure de retour arrière.
- [ ] Préparer le changement de référence comme étape distincte et réversible. Ce plan ne prescrit aucune promotion automatique ni publication distante.

## Clôture et compte rendu

Rapporter séparément : défauts corrigés, tests exécutés et non exécutés, état des
preuves ML, modèle réellement configuré, travaux préexistants conservés et
dépendances externes restantes. Une conclusion ML inconclusive ne doit pas
être renommée « corrigée » pour satisfaire artificiellement « fix all ».
