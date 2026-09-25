# Premier entraînement pré-match

Le parcours recommandé est `scripts/train_pre_match_model.py`. Il charge une
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
