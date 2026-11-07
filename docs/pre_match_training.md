# Premier entraînement pré-match

Pour les nouveaux entraînements DeepSets v3/v4, utiliser le dataset aligné au
service décrit dans la section « Dataset aligné 14D/18D » ci-dessous.

Le parcours général est `scripts/train_pre_match_model.py`. Il charge une
base explicitement choisie en lecture seule et produit un modèle candidat,
un manifeste des partitions et un rapport d'évaluation. Il ne remplace pas
le modèle chargé par l'application.

Depuis la racine du dépôt, sous PowerShell :

```powershell
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv312\Scripts\python.exe scripts/train_pre_match_model.py `
  --db runs/training_data_23h_20260922_191434.db `
  --samples 1000 --epochs 1 --batch-size 64
```

La base du run de collecte est un exemple local : adapter `--db` à la base
souhaitée. Ce paramètre est obligatoire. Sans `--output`, le script crée un
répertoire daté distinct sous `runs/pre_match_.../`. Pour choisir le nom :

```powershell
.\.venv312\Scripts\python.exe scripts/train_pre_match_model.py `
  --db runs/training_data_23h_20260922_191434.db `
  --samples 1000 --epochs 1 --batch-size 64 `
  --output runs/premier_pilote/model.keras
```

Un checkpoint ou un fichier de rapport déjà présent provoque un refus. Une
relance, y compris après un échec partiel, doit utiliser un nouveau chemin.
Le chemin actif `scripts/model_v2.keras` est interdit pour ce parcours.

## Données et protocole

- `--samples` borne les candidats les plus récents par **date du match**, pas
  par date d'ingestion, et concerne le total des trois partitions.
- Le chargeur exige des matchs CLASSIC, une date entière positive, deux équipes
  de cinq champions, les 18 labels valides et un winner cohérent. Les candidats
  invalides sont ignorés et leur nombre est journalisé ; ils ne sont pas
  remplacés par une recherche sans limite dans les matchs plus anciens.
- Les entrées du modèle sont limitées aux champions, rôles et contexte de jeu.
  L'or final, le KDA et les statistiques de carrière sans date sont retirés.
- Environ 70 % des matchs servent à l'apprentissage, 15 % à la validation et
  15 % au test. Les matchs avec le même timestamp restent ensemble, ce qui
  peut modifier légèrement ces proportions. Il faut au moins 20 matchs
  valides et trois timestamps distincts.
- `--split-strategy {chronological,patch}` (défaut : `chronological`) :
  - `chronological` : découpe temporelle 70/15/15 standard.
  - `patch` : holdout par patch. Le patch le plus récent (`major.minor`) est affecté au test,
    le précédent à la validation, et tous les antérieurs à l'entraînement.
    Si moins de 3 patchs distincts sont présents, un repli automatique sur le découpage chronologique
    est appliqué avec `fallback_reason: 'fewer_than_three_patches'`.
    Les versions manquantes ou inconnues provoquent une erreur explicite.
    Tout chevauchement temporel entre les plages de patchs lève une erreur `ValueError`.
- `--split-manifest PATH` :
  - Rejoue fidèlement et de manière immuable la cohorte de matchs spécifiée dans le fichier manifeste JSON.
  - Mutuellement exclusif avec une stratégie de découpe explicite.
  - `--samples` est ignoré lorsqu'un manifeste est fourni.
- `--calibrate` (optionnel, désactivé par défaut) :
  - Ajuste une mise à l'échelle par température post-hoc sur les probabilités du `winner` sur la partition de validation (au moins 20 matchs et les deux classes présentes requis).
  - Émet un sidecar `model.calibration.json` cryptographiquement lié aux poids exacts du modèle et au manifeste.
  - Conserve les métriques brutes sous `metrics` et expose les métriques après calibrage sous `calibrated_metrics`.
- Le modèle part de nouveaux poids. Seule la partition d'apprentissage est
  utilisée pour les mises à jour. La log-loss du winner sur validation choisit
  le checkpoint ; le test est évalué une fois après cette sélection.
- La référence prédit une probabilité constante, égale à la fréquence de
  `winner=1` dans l'apprentissage. Elle n'est jamais ajustée sur le test.

## Artifacts à lire

- `model.keras` : checkpoint candidat sélectionné sur validation.
- `model.split.json` : manifeste (schéma 2) contenant le chemin source, les IDs et timestamps des trois partitions,
  l'empreinte `dataset_sha256` des features nettoyées et labels, ainsi que l'empreinte `split_sha256` de la découpe.
  Ce manifeste permet de rejouer exactement la même cohorte (IDs, timestamps, partitions) de façon immuable.
- `model.calibration.json` : présent uniquement si `--calibrate` est spécifié. Contient la température optimale ajustée sur validation, les métriques d'ajustement et les empreintes liant l'artefact au checkpoint et au manifeste.
- `model.metrics.json` : paramètres, versions, matériel, historique, époque
  retenue, accuracy, log-loss, Brier, ROC AUC et erreur de calibration par
  partition, avec les mêmes métriques pour la référence, lié au split hash.
  Si calibré, contient également `calibrated_metrics`.

La colonne positive est `winner=1` (équipe B/rouge). Un `accuracy_lift` positif
ou un `log_loss_improvement` positif indique un gain sur la référence dans la
partition considérée. ROC AUC vaut `null` si cette partition ne contient qu'une
classe. L'erreur de calibration, le quantile ECE, la décomposition de Brier et la pente de calibration (`calibration_slope`) sont des estimations diagnostiques descriptives. Elles ne prouvent pas qu'un calibrateur a été ajusté et ne constituent pas un critère automatique de promotion du modèle. Sans `--calibrate`, aucun calibrateur n'est ajusté et le rapport conserve `is_calibrated: false`. L'ajustement d'une température n'établit pas d'utilité prédictive supplémentaire.

## Limites du pilote

1 000 matchs et une époque vérifient surtout l'exécution et la traçabilité.
Ils ne démontrent pas une capacité prédictive utile. Des joueurs peuvent être
présents dans plusieurs partitions : ce protocole mesure un holdout temporel,
pas une généralisation à des joueurs entièrement nouveaux. Seul le winner est
évalué ; les autres sorties actives ne sont pas validées par ce rapport.

Les têtes `kills_odd` et `first_to_*_kills` conservent un poids nul. Leur présence
dans le checkpoint ne signifie pas qu'elles ont appris. Le rapport liste les
têtes actives et désactivées.

Ne pas utiliser le flag historique `is_trained` pour reconstruire le test de
ce run : sa définition est le manifeste. `python cli.py train` lance désormais
ce parcours candidat avec la base configurée, ou avec `--db` pour une base
explicite. `python cli.py train-continuous` conserve la boucle historique,
qui n'utilise pas ce protocole. Le flag `is_calibrated` du chargeur historique
de l'application reste insuffisant pour juger la qualité d'un modèle. Une
promotion vers l'application est une étape séparée, après validation du modèle
et de son contrat de features.

Sur Windows natif, vérifier le champ `gpu_devices` du rapport : une liste vide
signifie que ce run utilise le CPU. Les variables de threads ci-dessus limitent
la concurrence ; elles n'activent pas le GPU.

Pour utiliser la carte NVIDIA, suivre le [parcours GPU sous WSL2](gpu_training.md) et lancer les scripts avec `scripts/train_gpu.ps1`.

## Modèle arborescent de référence (Tree Baseline)

Un modèle baseline tabulaire basé sur `HistGradientBoostingClassifier` peut être entraîné via `scripts/train_tree_baseline.py`. Il utilise les mêmes découpes et le même contrat de features à 24 colonnes catégorielles (champions et rôles des 10 participants, plus mode, version, saison et région).

```powershell
.\.venv312\Scripts\python.exe scripts/train_tree_baseline.py `
  --db runs/training_data_23h_20260922_191434.db `
  --samples 5000 --output runs/tree_pilot
```

Pour comparer rigoureusement un modèle arborescent à un candidat neuronal existant sur la **même cohorte exactement**, utiliser `--compare-to` :

```powershell
.\.venv312\Scripts\python.exe scripts/train_tree_baseline.py `
  --db runs/training_data_23h_20260922_191434.db `
  --compare-to runs/pre_match_.../model.metrics.json `
  --output runs/tree_comparison
```

Dans ce mode, le manifeste référencé par le rapport neuronal est rejoué, les empreintes de données et de partitions (`dataset_sha256`, `split_sha256`) sont strictement vérifiées, et un fichier `comparison.json` est généré comparant les métriques de la référence constante, du modèle arborescent et du réseau neuronal.

## Bundle candidat et service en production

Chaque exécution d'entraînement candidat produit un bundle indissociable composé de 4 artefacts :
- `model.keras` : Poids et graphe Keras natif multi-têtes.
- `model.split.json` : Manifeste de partitionnement figé (schema 2) liant les identifiants et empreintes SHA-256.
- `model.calibration.json` : Artefact de calibration de température (optionnel, lié cryptographiquement au checkpoint et au manifeste).
- `model.metrics.json` : Rapport d'évaluation final (brut et calibré).

### Activation d'un bundle

Pour activer un bundle explicitement validé :
1. Choisir un bundle complet et immuable dans un répertoire de run (ne pas copier manuellement les fichiers individuellement).
2. Définir la variable d'environnement `NNRIOT_CHECKPOINT_PATH` avec le chemin absolu vers le fichier `.keras` de ce bundle.
3. Redémarrer l'application. Au démarrage, `ModelFactory` vérifie l'intégrité de l'architecture et charge les artefacts frères (`.split.json`, `.calibration.json`).
4. Les checkpoints historiques ou dépourvus d'un fichier `.calibration.json` valide restent disponibles pour l'entraînement mais sont intentionnellement marqués `is_calibrated: false` et refusés pour la prédiction calibrée (HTTP 503 `uncalibrated: true`).

### Invalidation par mutation

L'exécution de la boucle d'entraînement continu (`cli.py train-continuous` ou `ContinuousTrainer.run_training_step`) invalide la calibration immédiatement avant toute mise à jour des poids (`train_on_batch`) :
- L'artefact `model.calibration.json` est supprimé du disque.
- L'attribut mémoire `is_calibrated` est basculé à `False`.
- Le service de prédiction reste indisponible jusqu'à ce qu'un nouveau bundle évalué et calibré soit explicitement déployé.
- La calibration de température ne s'applique qu'à la tête `winner` ; elle ne valide ni ne calibre les têtes auxiliaires de régression ou d'événements.

## Matchups chronologiques pour les entraînements candidats

Les quatre consommateurs `train_deepsets_v2.py`, `train_deepsets_v3.py`,
`train_pre_match_model.py` (branche 14 dimensions) et `evaluate_ensemble.py`
utilisent l'extracteur partagé `core.features.ordered_matchup`. Un résultat du
train n'entre dans le tracker que si sa fin explicite est strictement antérieure
à la création de la cible. L'original et son swap sont extraits au même état ;
les résultats validation/test n'alimentent jamais ce tracker.

Fournir `--result-end-times` avec un fichier versionné de fins associé à la même
cohorte et aux mêmes priors, ou une base contenant les timestamps explicites.
Le défaut `--missing-result-end error` arrête avant l'entraînement si une fin du
train manque. `--missing-result-end exclude` exclut uniquement ces événements et
inscrit les compteurs dans les artefacts ; l'absence totale de fins valides reste
une erreur. Aucune fin n'est estimée à partir de la durée.

Exemples (choisir chaque fois un dossier neuf) :

```powershell
& '.venv312/Scripts/python.exe' scripts/train_deepsets_v2.py --splits-cache runs/candidate_selective_prod/splits_cache_45000.pkl --db-path runs/db_merge_20260924/consolidated.db --output-dir runs/ordered_v2_new
& '.venv312/Scripts/python.exe' scripts/train_deepsets_v3.py --splits-cache runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl --manifest-path runs/candidate_selective_prod/model.split.json --db-path runs/db_merge_20260924/consolidated.db --output-dir runs/ordered_v3_new
& '.venv312/Scripts/python.exe' scripts/evaluate_ensemble.py --model runs/ordered_v2_new/model.keras --db runs/db_merge_20260924/consolidated.db --splits-cache runs/candidate_selective_prod/splits_cache_45000.pkl --output-dir runs/ordered_ensemble_new
& '.venv312/Scripts/python.exe' scripts/benchmark_ordered_matchups.py --splits-cache runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl --manifest runs/candidate_selective_prod/model.split.json --database runs/db_merge_20260924/consolidated.db --output runs/ordered_matchup_integration_validation
```

Le script général accepte `--splits-cache` avec `--split-manifest` pour rejouer
les records enrichis exacts. Les caches historiques restent en lecture seule.
Un cache incompatible avec le manifeste ou une provenance incompatible avec les
priors attachés sont refusés ; un manifeste seul ne reconstitue pas ces priors.
Les nouveaux runs enrichis ajoutent `model.features.json` et
`model.result_end_times.json`. Le contrat conserve la politique
`ordered_matchup_v1`, les dimensions, les poids (5 pour v2/v3, 15 pour la branche
générale et les arbres), les empreintes et les exclusions. La calibration est
liée à son empreinte. Les entrées 8 dimensions conservent leur extracteur
historique. L'évaluation d'un ancien modèle 14 dimensions sans contrat demande
un réentraînement.

Cette intégration ne renforce pas la garantie temporelle des priors joueurs déjà
attachés aux caches. La fin de match reste un proxy optimiste de disponibilité,
car l'heure d'ingestion API n'est pas enregistrée. Ces bundles enrichis ont pour
périmètre l'entraînement et l'évaluation. Leur activation dans PredictionService
exigerait un contrat d'inférence et un état historique compatibles ; elle ne fait
pas partie de ce changement. Le benchmark de mai est un rejeu de non-régression,
sans réglage ni preuve d'amélioration d'un réseau en production.

Pour le rejeu enrichi du script général avec calibration :

```powershell
& '.venv312/Scripts/python.exe' scripts/train_pre_match_model.py --db runs/db_merge_20260924/consolidated.db --feature-version deepsets_v2 --split-manifest runs/candidate_selective_prod/model.split.json --splits-cache runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl --output runs/ordered_general_new/model.keras --calibrate
```

La calibration du script général reste optionnelle. L'évaluation d'ensemble
indique explicitement si elle a été appliquée ; un fichier de calibration présent
mais lié à un autre contrat est refusé. Une absence de calibration ne transforme
pas les probabilités brutes en probabilités calibrées.


## Dataset aligné 14D/18D

Le service demande une lecture `history_only` des faits Riot : il ne consulte
pas les données ML pour calculer les priors. Les autres lecteurs conservent
leurs labels habituels. La fenêtre globale est sélectionnée via les index de
mode/création et les ex æquo par ID, dans une transaction unique en lecture
seule ; temporaires SQLite en mémoire et cache de pages plafonné à 64 MiB
par connexion.

Le contrat `nnriot.serving_aligned_dataset.v1` fige une seule source SQLite
ouverte en lecture seule. Les cibles sont les queues 420/440, avec cinq rôles
canoniques par équipe, labels multitâches valides et vainqueur cohérent avec
Riot. Les 10 000 premières créations du snapshot servent au contexte ; les
créations ex æquo à sa frontière ne deviennent pas des cibles. Pour chaque
cible, le replay utilise exactement les 10 000 créations CLASSIC strictement
antérieures, puis uniquement les résultats terminés avant sa création.
Les résultats historiques sans label ML restent utilisables.

```powershell
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset inventory --db training_data.db --db runs/training_data_23h_20260922_191434.db --db runs/db_merge_20260924/consolidated.db
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset freeze --db runs/db_merge_20260924/consolidated.db --after-cohort runs/served_model_comparison_20261009_complete/cohort.jsonl --output-dir runs/aligned_dataset_20261009_reviewed --source-limit 61000
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset verify --dataset-dir runs/aligned_dataset_20261009_reviewed
```

Les sorties exigent un répertoire neuf. Le test compte au moins 3 000 cibles
créées après le cohort exploratoire antérieur ; la validation en compte au
moins 3 000. Les créations ex æquo restent ensemble. Un résultat train dont
la fin atteint le début de validation est purgé ; la même règle s'applique
entre validation et test. Les IDs du test exploratoire antérieur sont exclus
des nouvelles cibles. Les exclusions et les purges sont enregistrées.

`dataset.json` lie les hashes de code, les règles, la disponibilité par contrat
et les SHA-256 de chaque fichier. `source_rows.json.gz` conserve les faits
compactés ; `model.split.json` conserve les IDs et créations ; `train.json`
et `validation.json` portent leurs labels. `sealed_test.json` et
`sealed_test14.npz` réservent le test pour la prochaine expérience. Les
fichiers `train14.npz`/`validation14.npz` contiennent les équipes float32
`(N,5,14)` et le macro float32 `(N,15)`. Les fichiers 18D n'existent que si
leur disponibilité est complète sur les mêmes partitions. Les causes
individuelles figurent dans `availability.json`.

Les deux bases récentes inventoriées ne contiennent pas de timelines.
Le dataset récent peut être disponible en 14D tout en étant bloqué en 18D.
Le trainer v4 refuse ce statut ; les anciennes timelines d'une autre base
ne sont pas ajoutées implicitement. Leur acquisition récente est un chantier
séparé. Ni les defaults de cold start ni un padding ne remplacent une source
18D absente.

Après fixation du protocole de la prochaine expérience, commandes de pilot :

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v3 --dataset-dir runs/aligned_dataset_20261009_reviewed --output-dir runs/aligned_v3_pilot --epochs 1 --batch-size 64 --seed 42
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v4_timeline --dataset-dir runs/aligned_dataset_20261009_reviewed --output-dir runs/aligned_v4_pilot --epochs 1 --batch-size 64 --seed 42
```

La commande v4 exige `availability["nnriot.players18.v1"].status == "ready"`.
Dans ce mode, les trainers rejettent les caches/manifests legacy explicites,
construisent des poids neufs, utilisent uniquement train/validation et
retournent avant toute métrique test. La calibration stricte est ajustée sur
validation avec au moins 20 observations et les deux classes ; une validation
insuffisante ne produit aucun faux sidecar calibré. `model.metrics.json`
lie le checkpoint au manifeste du dataset et conserve `test_status=reserved`
et `validation_status=unverified`.

Le replay d'audit peut vérifier les features et hashes du test, sans mesurer
accuracy/log-loss ou distribuer les labels. Cette réservation est une
discipline d'expérience : les faits Riot archivés contiennent les résultats.
L'absence de timestamps d'acquisition impose l'hypothèse rétrospective
« résultats observables à la fin de partie ». Les profils statiques de
champions conservent leurs limites de provenance historique. La parité de
features ne démontre ni gain prédictif ni éligibilité à la production.
Voir [l'audit de réception](model_serving/aligned_dataset_verification.md).
