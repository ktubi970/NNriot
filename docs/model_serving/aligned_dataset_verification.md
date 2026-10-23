# Réception du dataset aligné entraînement/service

Réception du 9 octobre 2026. Le dataset **14D est prêt pour une future expérience**, avec parité de features vérifiée. Le contrat 18D est bloqué par l'absence de timelines dans la source récente. Aucun entraînement réel, métrique prédictive du test réservé ou changement de modèle servi n'a été effectué dans ce chantier.

Actualisation du 11 octobre 2026 : l'[audit de `training_data.db`](timeline18_coverage_audit_20261011.md) trouve 97 894 cibles exploitables en 18D sur 97 916. Cette autre base ne contient aucun des 49 486 identifiants du présent bundle ; ce résultat ne modifie donc pas son statut 18D. Aucun refreeze ni entraînement n'a été effectué lors de cet audit.

## Source et sélection

Source SQLite en lecture seule : `D:\projet\NNriot\runs\db_merge_20260924\consolidated.db`. Inventaire enregistré dans [source_inventory.json](../../runs/aligned_dataset_20261009_reviewed/source_inventory.json). Le snapshot contient **61,000 créations CLASSIC**, de 2026-09-15T23:03:48.025000+00:00 à 2026-09-24T14:06:27.690000+00:00. Les 10 000 premières lignes constituent le contexte historique et ne sont pas des cibles. Les créations ex æquo à la dernière ligne de warmup sont également exclues. Population cible : queues 420/440, cinq rôles valides par équipe, vainqueur Riot cohérent avec les labels ML multitâches.

Le cutoff exploratoire antérieur est **2026-09-17T11:38:52.620000+00:00**. Le SHA-256 du cohort antérieur est `530a3ca4226620bb1fcb539a64dd9c9f1e5add322c0596db6c07ef9d794ca7cc` ; ses 2,000 IDs test sont exclus des cibles. Le nouveau test commence strictement après ce cutoff.

| Partition | Cibles | Première création UTC | Dernière création UTC |
|---|---:|---|---|
| train | 43,486 | 2026-09-17T00:58:59.947000+00:00 | 2026-09-23T07:56:59.919000+00:00 |
| validation | 3,000 | 2026-09-23T08:18:01.718000+00:00 | 2026-09-23T23:24:57.184000+00:00 |
| test | 3,000 | 2026-09-23T23:38:33.848000+00:00 | 2026-09-24T14:06:27.690000+00:00 |

Les créations identiques restent groupées. Le train est purgé de toute partie finissant à partir du début de validation ; la validation de toute partie finissant à partir du début du test. Les minima de 3 000 cibles validation et test ont été conservés. Causes enregistrées dans [exclusions.json](../../runs/aligned_dataset_20261009_reviewed/exclusions.json) :

| Cause | Lignes |
|---|---:|
| previous_test | 1,198 |
| result_not_available_at_partition_start | 310 |
| team_a roles must contain each canonical role exactly once | 5 |
| team_b roles must contain each canonical role exactly once | 1 |

## Population, sans lecture de métriques prédictives

| Partition | Attribut connu avant match | Effectifs |
|---|---|---|
| train | queue_id | 420: 43,486 |
| train | region | EUW: 1,130, LA1: 3, LA2: 83, NA: 42,270 |
| train | patch | 16.18.817.5716: 43,486 |
| validation | queue_id | 420: 3,000 |
| validation | region | LA1: 1, LA2: 9, NA: 2,990 |
| validation | patch | 16.18.817.5716: 47, 16.19.820.7193: 2,953 |
| test | queue_id | 420: 3,000 |
| test | region | LA2: 1, NA: 2,999 |
| test | patch | 16.19.820.7193: 726, 16.19.821.7343: 2,274 |

## Contrats et audits

| Contrat | Statut | Éligibles/total | Couverture | Causes |
|---|---|---:|---:|---|
| nnriot.players14.v1 | ready | 49,486/49,486 | 100.0% |  |
| nnriot.players18.v1 | blocked | 0/49,486 | 0.0% | FeatureUnavailableError: 18D timeline history source unavailable; cannot replace missing source with cold defaults (49,486) |

Les tensors sont float32 : équipes `(N, 5, 14)` et macro `(N, 15)`. Le replay complet a vérifié **148,458 comparaisons exactes** par `np.array_equal`, à partir des faits figés, sans relire la base. Il contrôle également formes, dtype, valeurs finies, chemins internes et SHA-256. Le test réservé est rejoué pour vérifier ses features ; aucune accuracy, log-loss ou sélection d'hyperparamètres n'en est extraite. Voir [replay_verification.json](../../runs/aligned_dataset_20261009_reviewed/replay_verification.json).

Un panel déterministe de **20 cibles train/validation**, fixé avant comparaison avec des positions régulièrement espacées incluant les extrêmes, a été rejoué au même cutoff via `DatabaseHistoryProvider`. Priors et tenseurs sont exactement égaux sur les 20 cibles. Provenance vérifiée : même création, fenêtre globale de 10 000 créations antérieures, dernier résultat strictement antérieur au cutoff. Voir [live_panel.json](../../runs/aligned_dataset_20261009_reviewed/live_panel.json), [live_parity.json](../../runs/aligned_dataset_20261009_reviewed/live_parity.json) et [script d'audit](../../runs/aligned_dataset_20261009_reviewed/live_audit.py).

Les tests des trainers ont branché les tenseurs alignés jusqu'au premier batch et contrôlé la clôture/calibration sur validation. L'ouverture des faits sources et des fichiers du test réservé est interdite par les tests du loader. Le trainer v4 refuse la disponibilité 18D bloquée avant de créer un modèle. Les chemins legacy restent exploratoires.

## Commandes exécutées et vérifications

```powershell
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset inventory --db training_data.db --db runs/training_data_23h_20260922_191434.db --db runs/db_merge_20260924/consolidated.db
.\.venv312\Scripts\python.exe -u scratch/trace_aligned_freeze.py freeze --db runs/db_merge_20260924/consolidated.db --after-cohort runs/served_model_comparison_20261009_complete/cohort.jsonl --output-dir runs/aligned_dataset_20261009 --source-limit 61000
.\.venv312\Scripts\python.exe -u scratch/refreeze_aligned.py
.\.venv312\Scripts\python.exe -u scratch/refreeze_aligned.py --from-dir runs/aligned_dataset_20261009_verified --output-dir runs/aligned_dataset_20261009_reviewed
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset verify --dataset-dir runs/aligned_dataset_20261009_reviewed
.\.venv312\Scripts\python.exe scratch/audit_aligned_live.py --dataset-dir runs/aligned_dataset_20261009_reviewed
.\.venv312\Scripts\python.exe -m scripts.formalize_served_model
$env:TEMP='D:\projet\NNriot\scratch\aligned-tests'
$env:TMP=$env:TEMP
.\.venv312\Scripts\python.exe -m pytest tests/test_frozen_history.py tests/test_aligned_partitions.py tests/test_aligned_artifacts.py tests/test_aligned_training.py tests/test_aligned_parity.py tests/test_serving_contract.py tests/test_pre_match_history_serving.py tests/test_serving_http.py tests/test_predict_api.py tests/test_config_cli.py tests/test_ordered_training_integration.py tests/test_train_deepsets_v4_timeline.py tests/test_equivariant_model.py tests/test_deepsets_layers.py tests/test_timeline_metrics.py tests/test_point_in_time_timeline.py -q
```

Résultat de la suite ciblée : **164 passed, 325 warnings in 800.66s (0:13:20)**, exit 0. Ceci couvre les 16 suites listées, sans prétendre que tous les tests du dépôt ont été exécutés. Les avertissements Keras/NumPy déjà présents sont conservés dans le journal.

Le gel initial sous `runs/aligned_dataset_20261009` a exécuté la sous-commande `freeze` via le wrapper diagnostique `scratch/trace_aligned_freeze.py`, avec les mêmes paramètres ; il ajoute uniquement une trace de pile périodique et un journal de progression. Les premières lectures ont été interrompues avant création d'un dataset, pour corriger le coût de la sélection SQLite. La fenêtre finale utilise les index couvrant les rowids et dates, résout les IDs ex æquo à sa frontière et lit les payloads retenus en ordre de rowid, dans une seule transaction en lecture seule. Les temporaires SQLite sont en mémoire ; aucune écriture ni création d'index sur la source.

Après suppression des lectures ML inutiles dans `DatabaseHistoryProvider`, un premier refreeze a créé `runs/aligned_dataset_20261009_verified`. La revue indépendante finale a ensuite reproduit un défaut sur des timelines partielles : des mesures absentes étaient remplacées par zéro puis comptées comme observations 18D. La correction exige maintenant dix participantFrames contenant les quatre mesures numériques finies requises, ainsi que les listes d'événements observées. Une liste vide ou une mesure numérique égale à zéro reste une observation valide. Les timelines incomplètes ne peuvent plus rendre 18D disponible ; les résultats Riot restent utilisables en 14D. Le chemin legacy permissif est conservé explicitement et ne reçoit aucune certification nouvelle. Sept cas de régression ont été observés RED, puis la vérification ciblée a obtenu 36 passes GREEN, incluant le writer et SQLite. Aucune seconde revue n'a été lancée.

La deuxième commande `refreeze_aligned.py` a créé le répertoire final `runs/aligned_dataset_20261009_reviewed` à partir des faits et records vérifiés du snapshot précédent, sans relire SQLite. SHA des faits, identifiants/ordre de partitions et fingerprint de données sont identiques ; les trois NPZ 14D gardent exactement les mêmes SHA-256 (assertions exécutées). Le manifeste final lie le runtime corrigé et le SHA du manifeste précédent. Les deux snapshots précédents restent archivés, avec leurs audits d'origine. Replay complet et panel SQLite ont été relancés pour le runtime final. Pour préparer une nouvelle expérience, exécuter directement `python -m scripts.prepare_aligned_dataset freeze` avec les arguments de la commande diagnostique ci-dessus et un répertoire neuf ; aucun wrapper n'est nécessaire au fonctionnement du CLI.

Les 22 entrées d'inventaire et manifestes ont été régénérées avec les sources de features actuelles. Le checkpoint de référence reste lié à `b19429597f8b9747aa413b42cdefdd0739556378ab692ee25904c12e98f19f0f` et le statut de service reste `unverified`. Les calibrations et checkpoints existants n'ont pas été réécrits. Les anciens protocoles de comparaison gardent leurs hashes d'origine ; leurs replays doivent refuser le nouveau runtime.

Ce refus a également été vérifié sur le protocole réel par `python -m scripts.compare_served_candidates --replay runs/served_model_comparison_20261009_complete --output-dir scratch/aligned-tests/legacy-runtime-refusal` : `ContractError: Frozen evaluation runtime source SHA-256 mismatch`, exit 1 attendu, avant toute comparaison de modèles. Aucun fichier de prédictions ou métriques n'a été écrit dans ce répertoire scratch. SHA du protocole identique avant/après : `face72e773913ff63af3305fdb1ae5abbfbf0220369199e53cab585e2b0d6355`.

## Empreintes des artefacts

[dataset.json](../../runs/aligned_dataset_20261009_reviewed/dataset.json) lie partitions, tenseurs, faits sources, disponibilité, règles et hashes de code. [artifact_checks.json](../../runs/aligned_dataset_20261009_reviewed/artifact_checks.json) enregistre les contrôles ainsi que les preuves SQLite supplémentaires.

| Artefact | SHA-256 |
|---|---|
| `availability.json` | `c44db98cdd7fe66c3967a3bfa7941ed6fd76c277b8e7141e2b7e7454cfba704f` |
| `exclusions.json` | `d975f64e4c8e17ae2febb1d0b31495203d85c448d489972d304d608149d0d7ac` |
| `model.split.json` | `9c97f4d3416e8176270b6a65650da30953612d226798dd8b0ece8f629d79abcb` |
| `sealed_test.json` | `5c556e0314362db1acfd0963d725d9ecdc8ea4c830758df0a8f6dd34cc20400e` |
| `sealed_test14.npz` | `7f2a0de4dc0f471c2661c1cb37d3d352a58d1d6adcff75e2d1f4a9f0f06ce851` |
| `source_rows.json.gz` | `a3392790b27eba57fcb50f44d0e61d104cd16e64fab7453f07225b34fd9235bf` |
| `train.json` | `ca8b3d1c9a03960e5c58913b57aef6241345fe5bc024f7ad3aa422592c92e5ab` |
| `train14.npz` | `0cbf98c5a687d645fcf1be44054f0f85baf42e82dcc268a74e4d8fe91e709969` |
| `validation.json` | `1484d853b4a4a9a94d9ea0a4d425d6bea3d9186725779bc7d88b7d25ee3fa54d` |
| `validation14.npz` | `ecfd1390a4b07bed15e7333970fbf308dd6541f83930cc101ab8cded6b2e1aa4` |
| `dataset.json` | `a164d584bbc4fd4a9499c45a74a19eb78dde8faa00c55e3ee02db64ae8818f45` |
| `live_panel.json` | `8af24261c6e02e0acad4ac6ee12d2c10a34beac8e65f39980dec209783b78a58` |
| `live_parity.json` | `a23ffba48cd9c192ae303f4bd7bfb65e79622b2536b1dc9f70add1355a771c78` |
| `live_audit.py` | `233c7662dc2595477e30ccd84e1cce127c413612e728a3eab2c8834d000ef385` |
| `replay_verification.json` | `bf1c570bab19b90f866cf4811581ddc57cc95e0521ddb14dcacdab862fe982f4` |
| `source_inventory.json` | `955a8d798701fcb036589b8632070e51854e9eb35802841f6df2395569f120c3` |
| `freeze_driver.py` | `0aa72de25f219d61ed6f9a8fe0a5117bd24b04f2e5e33381ce744eae88d0d7dc` |
| `refreeze_driver.py` | `4aa961f894d256d4e24922f12f7d6013a6d2001470c48d35a63363641d0a7590` |

## Limites et étape suivante

Les temps d'acquisition historiques ne sont pas stockés : disponibilité présumée à la fin Riot du résultat. Les profils statiques des champions sont hashés, mais leur disponibilité historique par patch n'est pas certifiée. Le test est réservé à la future expérience de ce chantier ; cela ne prouve pas que toute la base a toujours été inconnue lors de recherches antérieures. Les fichiers de faits contiennent les résultats Riot : la réservation est une discipline d'expérience.

La suite logique est un protocole d'expérience 14D utilisant train/validation pour entraîner, calibrer et choisir, puis une seule évaluation finale réservée. Les commandes de pilots sont documentées dans [pre_match_training.md](../pre_match_training.md) et n'ont pas été exécutées ici. Le 18D récent exige une acquisition de timelines utilisable également au service, dans un chantier séparé ; ni remplissage de zéros ni fusion implicite avec une autre base.

Le [bilan d’exécution Native](aligned_dataset_execution.md) conserve les décisions, la revue indépendante, les baselines et le diff attribuable.
