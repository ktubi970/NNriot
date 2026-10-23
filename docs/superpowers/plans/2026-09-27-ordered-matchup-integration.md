# Ordered Matchup Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Brancher l'extraction chronologique validée sur les quatre scripts réels d'entraînement et d'évaluation, avec provenance, compatibilité explicite et tests de fonctionnement.

**Architecture:** Un extracteur partagé sous `core/features/` reste pur et local en mémoire. La couche données lit les fins explicites et `core/experiment.py` gère leur provenance et le contrat des nouveaux modèles. Les scripts consomment ces interfaces ; les entrées neuronales historiques à 8 dimensions restent indépendantes du nouveau tracker.

**Tech Stack:** Python 3.12, NumPy, SQLite, TensorFlow/Keras, scikit-learn, pytest ; environnement `.venv312`, aucune installation ajoutée.

**Spec:** `docs/superpowers/specs/2026-09-27-ordered-matchup-integration-design.md`.

## Global Constraints

- Préserver les modifications déjà présentes dans le workspace.
- Aucune nouvelle dépendance, migration SQLite ou modification des anciens caches.
- SQLite en lecture seule ; ne pas utiliser une connexion qui active WAL/commit pour cette lecture.
- Préserver les dimensions 8, 14, 232 et 15 des représentations concernées.
- Préserver l'anti-symétrie des probabilités et l'invariance aux permutations de joueurs.
- Ne jamais recalculer les swaps d'anciennes lignes avec le tracker final.
- Aucun label validation/test dans le tracker, aucune sélection sur les scores test.
- Aucun écrasement de checkpoint, rapport, manifeste ou calibration existant.
- Ne pas revendiquer un gain du modèle actif sur la base du benchmark tabulaire.
- Aucun changement à PredictionService, aux routes HTTP ou au modèle activé dans ce plan.

## Review Focus

1. Cache correct selon le manifeste mais priors différents : le nouveau hash enrichi doit détecter la différence (tâche 2).
2. Ancien cache sans fins, JSON tronqué, fichier de fins d'un autre run : erreur explicite, jamais de tracker neutre implicite (tâches 1–2).
3. Modèle 14 dimensions sans contrat, poids 5 confondu avec 15, v3 avec entrée macro : rejet ou extraction conforme au contrat (tâches 3–5).
4. Des permutations et matchs simultanés traversent deux splits : aucune contamination ni désalignement label/feature (tâches 1 et 4–5).
5. Relance sur un dossier occupé ou évaluation d'un checkpoint actif : aucune écriture dans les anciens artefacts (tâches 3–5).

## État initial et ordre de travail

Plan rédigé le 27 septembre 2026 sur HEAD `24e7bd9ff410994ec8a895f1f1e4b28ffaea0119`.
Relire `git status --short` et `AGENTS.md` à l'exécution : d'autres travaux
sélectifs, GPU et timelines sont présents. L'extracteur A3 et ses résultats sont
non suivis dans cet état ; une nouvelle worktree ne les recopiera pas
automatiquement. Ne pas partir d'une worktree où ces prérequis manquent. Ne pas
faire `git reset`, `git clean`, restauration globale ni commit de tous les fichiers.

Les tâches 1 puis 2/3 définissent les contrats. Les tâches 4 et 5 peuvent être
déléguées en parallèle après stabilisation de ces contrats, avec propriété de
fichiers distincte. Une revue indépendante vérifie ensuite l'ensemble. Les
sections suivantes décrivent des travaux à exécuter, pas des validations déjà
réalisées. La commande Python de référence est `& '.venv312/Scripts/python.exe'`.

---

### Task 1: Promouvoir l'extracteur pur et le lecteur de fins

**Files:**
- Create: `core/features/ordered_matchup.py`.
- Modify: `core/point_in_time_dataset.py`.
- Create: `tests/test_ordered_matchup_features.py`, `tests/test_result_end_times.py`.
- Read/preserve: les deux modules et deux fichiers de tests expérimentaux A3.

**Interfaces:**

- `core.features.ordered_matchup.extract_ordered_features(splits, result_end_times,
  *, representation='tabular', matchup_weight=15.0)` retourne
  `(features_by_split, swapped_by_split, diagnostics)` ; contrat du prototype.
- `core.point_in_time_dataset.load_result_end_times(db_path, rows)` retourne
  `(dict[match_id, end_ms], diagnostics)` ; clés des diagnostics :
  `requested`, `found`, `invalid`, `valid`, `missing`.

- [ ] Copier les tests comportementaux A3 vers les nouveaux fichiers, remplacer
  leur import dynamique par les imports de `core`, puis constater l'échec faute
  de fonctions. Conserver les originaux pour la reproductibilité des anciens runs.
- [ ] Ajouter le cas d'une archive gzip tronquée (inclure `EOFError` dans les
  erreurs d'entrée traitées), le schéma sans `raw_json` et deux matchs de même
  création dans des splits distincts. Le lecteur classe les métadonnées
  inutilisables ; il ne transforme pas une erreur d'accès SQLite en succès vide.
- [ ] Reprendre l'algorithme exact du prototype : événements train triés par fin,
  `end < query`, sorties à leur index d'origine, original/swap au même état,
  tracker neuf. Préserver les diagnostics, `float32`, les trois représentations
  et l'exclusion bas niveau des fins absentes/incohérentes.
- [ ] Reprendre la lecture du benchmark dans la couche données : URI `mode=ro`,
  `PRAGMA query_only=ON`, chunks d'IDs, JSON ou gzip/base64, création base/cache
  identique, `creation <= start < end`, fenêtre maximale de 24 h comme A3.
  Ne pas modifier le comportement de `load_enriched_records` ni reconstruire
  les priors attachés ; la lecture des fins est une opération séparée.
- [ ] Exécuter et vérifier le code de retour :

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_ordered_matchup_features.py tests/test_result_end_times.py tests/test_features_pipeline.py tests/test_extended_player_priors.py tests/test_point_in_time_dataset.py -q --tb=short
```

Livrable : le correctif causal est disponible depuis `core`, sans branchement
encore revendiqué aux scripts. Les anciens scripts de recherche restent intacts.

### Task 2: Résoudre les métadonnées et certifier leur provenance

**Files:** Modify `core/experiment.py`; create `tests/test_matchup_provenance.py`.

**Interfaces:** ajouter sans changer les fonctions de hash existantes :

- `compute_enriched_dataset_sha256(splits) -> str` : hash du JSON canonique
  complet, identifiant/création/feature_json/labels_json.
- `resolve_result_end_times(splits, *, db_path=None, artifact_path=None,
  missing='error')` retourne `(ends, provenance)` ; `ends` est directement
  consommable par l'extracteur de tâche 1.
- `write_result_end_times_artifact(path, splits, ends, provenance) -> str` :
  écriture exclusive d'un JSON schema_version=1, retourne le SHA256 du fichier.

Le fichier associé contient : `schema_version`, `timestamp_unit='ms'`,
`split_sha256`, `enriched_dataset_sha256`, `availability_assumption`,
`train_matches` (ID, création, fin ou null), `missing_policy`, diagnostics.
Le résolveur vérifie l'unicité, l'ensemble exact des IDs train, leurs créations,
le hash des splits et le hash enrichi complet. Les poids appartiennent au
contrat de features, pas au fichier de faits temporels réutilisable.

- [ ] Écrire le test qui modifie uniquement un prior attaché : le hash historique
  reste identique, le hash enrichi diffère et le fichier associé est refusé.
  Exemple de l'assertion, sur une fixture contenant les trois splits :

```python
original = compute_enriched_dataset_sha256(splits)
changed = copy.deepcopy(splits)
changed['train'][0]['feature_json']['team_a'][0]['priors'][0] += 0.01
assert compute_enriched_dataset_sha256(changed) != original
```

- [ ] Tester défaut `error`, exclusion partielle explicite et exclusion totale
  refusée. Tester un ID/une création/unité/fin falsifiés, `NaN`, doubles IDs,
  `artifact_path` et `db_path` absents, et conflit entre provenance et cache.
- [ ] Implémenter le hash complet sans changer `compute_dataset_sha256`.
  Sérialiser les splits dans `SPLIT_NAMES`, rows dans leur ordre vérifié,
  `sort_keys=True`, `allow_nan=False`, séparateurs canoniques.
- [ ] Implémenter le résolveur : si fichier fourni, valider puis l'utiliser ;
  sinon charger les fins du train depuis la base explicitement fournie via
  tâche 1. Ne pas réécrire ni « réparer » automatiquement un fichier invalide.
  Sans fin valide pour chaque train en mode `error`, lever `ValueError` avec
  compte et exemples limités d'IDs ; sous `exclude`, conserver les compteurs.
- [ ] Écrire le JSON exclusivement dans le nouveau run. Les anciens fichiers
  plats A3 ne sont pas des contrats versionnés et ne doivent pas être acceptés
  silencieusement. Ne pas réécrire le pickle source.
- [ ] Exécuter :

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_matchup_provenance.py tests/test_experiment.py tests/test_result_end_times.py -q --tb=short
```

### Task 3: Définir le contrat des nouveaux modèles

**Files:** Modify `core/experiment.py`; create `tests/test_matchup_feature_contract.py`.

**Interfaces:**

- `write_feature_contract(checkpoint_path, contract) -> str` valide et écrit
  `Path(checkpoint_path).with_suffix('.features.json')` sans écrasement ; retour :
  SHA256 des octets enregistrés.
- `read_feature_contract(checkpoint_path) -> dict` lit et valide ce même fichier ;
  absence ou incompatibilité provoquent une erreur explicite.

Champs obligatoires : `schema_version=1`, `policy='ordered_matchup_v1'`,
`representation` (`tensors` ou `v3`), `player_feature_dim=14`, `macro_feature_dim`
(0 ou 15), `matchup_weight`, `global_weight=50.0`, `split_sha256`,
`enriched_dataset_sha256`, `result_end_times_sha256`, `availability_assumption`,
`missing_policy`, diagnostics, `artifact_scope='training_evaluation'`.
Ajouter `source_cache_sha256` lorsqu'un pickle existe. Enregistrer aussi les
hashes des modules d'extraction et du registre de champions pour l'audit ;
la compatibilité est définie par la version de politique et les dimensions,
pas par l'égalité arbitraire du hash du fichier Python après un commentaire.

- [ ] Tester aller-retour, version inconnue, poids non positif/non fini,
  dimensions incompatibles, JSON invalide, absence et écrasement refusé.
  Commencer par cette régression, qui doit échouer tant que l'API n'existe pas :

```python
def test_missing_feature_contract_fails_explicitly(tmp_path):
    from core.experiment import read_feature_contract
    with pytest.raises(FileNotFoundError):
        read_feature_contract(tmp_path / 'model.keras')
```

- [ ] Tester que les poids 5 et 15 sont distincts dans le contrat et qu'un
  contrat v3 exige `macro_feature_dim=15`.
- [ ] Implémenter les deux fonctions et la validation interne commune. Produire
  un hash stable à partir des octets réellement écrits. Le JSON de calibration
  des nouveaux modèles reçoit `feature_contract_sha256` ; un ancien artefact
  n'est jamais requalifié comme calibré pour la nouvelle politique.
- [ ] Exécuter :

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_matchup_feature_contract.py tests/test_experiment.py -q --tb=short
```

### Task 4: Brancher les trois entraînements

**Files:** Modify `scripts/train_deepsets_v2.py`, `scripts/train_deepsets_v3.py`,
`scripts/train_pre_match_model.py`; create `tests/test_ordered_training_integration.py`.

**Interfaces:** conserver les arguments existants et ajouter des arguments
nommés `result_end_times=None`, `missing_result_end='error'`, plus `db_path=None`
pour v2. Côté CLI : `--result-end-times`, `--missing-result-end {error,exclude}`,
et `--db-path` pour v2. Le script général utilise déjà `source_db`, v3 `db_path`.
Pour le rejeu enrichi du script général, ajouter `--splits-cache` optionnel
associé à `--split-manifest` ; valider le manifeste puis le hash complet.

- [ ] Écrire des tests passant par les fonctions d'entraînement qui vérifient
  la propagation des matrices ordonnées, des poids et des diagnostics ; inverser
  un résultat propre/futur dans les fixtures. Un espion sur `model.fit` ou les
  batches vérifie les tableaux reçus, pas seulement qu'un helper a été appelé.
- [ ] Ajouter un test prouvant que le chemin neuronal 8 dimensions fonctionne
  sans fin de match et conserve `batch_extract_team_tensors`. Ajouter le rejeu
  du manifeste avec cache enrichi : aucun prior perdu ni split recalculé.
- [ ] Dans v3, lorsqu'un cache **et** `manifest_path` sont fournis, appeler
  `read_manifest` puis `restore_splits` avant de créer les nouveaux artefacts.
  Tester un cache cohérent en lui-même mais appartenant à un autre manifeste :
  refuser plutôt que produire un nouveau manifeste légitimant ce découpage.
- [ ] Supprimer les trois boucles de préremplissage. Utiliser :

```python
ends, provenance = resolve_result_end_times(
    splits, db_path=db_path, artifact_path=result_end_times,
    missing=missing_result_end,
)
x, _, matchup_diagnostics = extract_ordered_features(
    splits, ends, representation='tensors', matchup_weight=5.0,
)
# v3 : representation='v3', poids 5.0.
# train_model : branche feature_version == 'deepsets_v2' ou dim == 14,
# representation='tensors', poids 15.0, db_path=source_db.
```

  Dans le script général, normaliser des copies des records avec `_get_feat`
  pour conserver son contrat d'entrée. Ne pas reconstruire les priors à partir
  des labels holdout. Pour un manifeste enrichi sans cache et dont la branche
  historique perd les priors, refuser le rejeu en demandant le cache enrichi,
  plutôt que produire silencieusement une expérience différente.
- [ ] Valider entrées/provenance et réserver des sorties neuves avant le fit.
  Remplacer les manifestes manuels v2/v3 par `build_manifest` et enregistrer le
  contrat/fins/diagnostics. Leurs fichiers doivent être lisibles par
  `read_manifest` puis `restore_splits` sans affaiblir ces fonctions.
- [ ] Corriger les deux obstacles locaux déjà identifiés et couvrir leur branche :

```python
# v2 : après best_model.predict sur validation, avant fit_temperature.
raw_val_probs = np.asarray(val_preds['winner'])[:, 1]
# v3 : création de splits sans manifeste.
splits, split_meta = split_records(records, strategy='chronological')
```

- [ ] Réserver un nouveau dossier de sortie ; un deuxième lancement au même
  emplacement échoue avant toute écriture sur modèle/calibration/manifeste.
- [ ] Exécuter les tests rapides puis un entraînement réel d'une époque pour
  chacun des trois chemins affectés, avec une fixture synthétique de 140 matchs
  à rôles distincts, labels complets, deux classes dans chaque split et fins
  explicites. Le test sauvegarde/recharge le modèle et vérifie les probabilités
  finies, les contrats et la calibration. Utiliser le CPU si nécessaire ; ne
  pas remplacer cette vérification finale par un mock.

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_ordered_training_integration.py tests/test_pre_match_training.py tests/test_draft_pre_match_training.py tests/test_pre_match_run.py -q --tb=short
```

### Task 5: Brancher l'évaluation d'ensemble sans mélanger les politiques

**Files:** Modify `scripts/evaluate_ensemble.py`; create `tests/test_ordered_ensemble_integration.py`.

**Interfaces:** compléter `evaluate_ensemble(model_path, db_path, samples=45000,
*, splits_cache=None, result_end_times=None, missing_result_end='error',
output_dir=None)` et CLI avec les arguments correspondants, dont
`--splits-cache`. Sans `output_dir`, créer un run neuf
sous `runs/`; ne plus écrire près du checkpoint évalué.

- [ ] Écrire des tests : cache chargé mais hors manifeste ; modèle 14 sans
  contrat ; contrat poids 5 avec arbre poids 15 ; v3 `macro_clash` en première
  entrée ; calibration d'un autre contrat ; ancien modèle 8 sans contrat.
- [ ] Privilégier le cache explicitement fourni par `--splits-cache`, dont
  l'absence est une erreur. Sinon, garder la découverte historique et la lecture
  depuis la base, en vérifiant le hash enrichi du modèle 14 pour empêcher une
  reconstruction différente des priors. Restaurer/vérifier le manifeste **aussi
  sur la branche cache**. Détecter
  les entrées `team_a`, `team_b`, `macro_clash` par leurs noms et dimensions,
  jamais avec `model.inputs[0]` ni une valeur 8 de secours masquant une erreur.
- [ ] Le modèle 8 garde l'extracteur historique. Pour le modèle 14, lire le
  contrat, vérifier provenance, poids et calibration, puis extraire `tensors`
  ou `v3` au poids sauvegardé. Un ancien 14 sans contrat demande un
  réentraînement ; aucun nouveau mode de reproduction contaminé ici.
- [ ] Extraire séparément les arbres avec poids 15 et leurs swaps associés :

```python
x_tab, x_tab_swapped, diagnostics = extract_ordered_features(
    splits, ends, representation='tabular', matchup_weight=15.0,
)
x_tab_aug = np.concatenate([x_tab['train'], x_tab_swapped['train']])
y_aug = np.concatenate([y['train'], 1 - y['train']])
def _predict_symmetrized(model, split_name):
    p = model.predict_proba(x_tab[split_name])[:, 1]
    ps = model.predict_proba(x_tab_swapped[split_name])[:, 1]
    return 0.5 * (p + 1.0 - ps)
```

- [ ] Tester qu'aucune réextraction n'a lieu pendant `_predict_symmetrized`,
  que `p(A,B) + p(B,A) == 1` à la tolérance numérique existante, et que changer
  les labels des holdouts ne change aucune matrice. Pour ce dernier test,
  régénérer la provenance de la fixture modifiée afin de tester l'extraction ;
  avec l'ancien contrat, le refus pour hash incohérent est le comportement prévu.
  Conserver la sélection des
  poids de mélange sur validation, jamais sur test.
- [ ] Écrire le rapport, les hashes et les diagnostics dans le nouveau run.
  Tester que le hash du checkpoint, de ses anciens rapports et de sa
  calibration ne change pas après une évaluation ni une relance refusée.
- [ ] Ajouter un test réel de bout en bout du quatrième consommateur : utiliser
  le checkpoint 14 issu d'un entraînement court, son cache synthétique explicite
  et sa base de fins ; exécuter `evaluate_ensemble` avec `samples=140` dans un
  dossier neuf. Entraîner effectivement HGB et LightGBM avec leurs paramètres
  existants, prédire, symétriser, produire puis relire le rapport. Vérifier les
  valeurs finies et la conservation des artefacts source. Les tests mockés de
  contrats ne remplacent pas ce test ; si une bibliothèque déjà prévue par le
  dépôt est indisponible, signaler la validation manquante sans annoncer ce
  quatrième chemin comme validé.
- [ ] Exécuter :

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_ordered_ensemble_integration.py tests/test_matchup_feature_contract.py tests/test_matchup_provenance.py -q --tb=short
```

### Task 6: Rejouer le benchmark intégré et livrer les preuves

**Files:** Create `scripts/benchmark_ordered_matchups.py`; modify
`docs/pre_match_training.md`; append a new integration entry to
`docs/research-improvement/2026-09-27-accuracy.md` sans réécrire A1/A2/A3.

**Interfaces:** le nouveau benchmark reprend les arguments A3
`--splits-cache`, `--manifest`, `--database`, `--output` et les paramètres HGB
figés. Il importe l'extracteur de `core` et le lecteur partagé ; aucune
nouvelle copie de l'algorithme ordonné. La référence greedy reste strictement
cantonnée à ce benchmark de non-régression, jamais à un entraînement normal.

- [ ] Ajouter un test qui compare les sorties core/prototype sur la fixture
  causale pour les trois représentations et les deux swaps. Garder cette
  référence explicite tant que la politique reste `ordered_matchup_v1`.
- [ ] Rejouer les trois variantes A3 sur les splits figés dans un nouveau run :

```powershell
& '.venv312/Scripts/python.exe' scripts/benchmark_ordered_matchups.py --splits-cache runs/candidate_deepsets_v3/splits_cache_45000_v3.pkl --manifest runs/candidate_selective_prod/model.split.json --database runs/db_merge_20260924/consolidated.db --output runs/ordered_matchup_integration_validation
```

  Refuser un dossier existant ; choisir un autre nom neuf lors d'une relance.
  Vérifier les hashes de matrices contre A3 avant d'interpréter les scores.
  Sous versions identiques, attendre les mêmes classifications et des
  probabilités à tolérance numérique justifiée. Conserver accuracy/logloss,
  IDs, prédictions, diagnostics et codes de retour. Ne pas régler les modèles.
- [ ] Documenter les commandes nouvelles, le défaut strict, les anciens caches,
  l'absence de garantie temporelle nouvelle sur les priors joueurs et la
  différence entre intégration d'entraînement et activation d'un modèle.
- [ ] Exécuter les régressions architecturales puis la suite complète une fois
  les tâches précédentes vertes :

```powershell
& '.venv312/Scripts/python.exe' -m pytest tests/test_deepsets_layers.py tests/test_equivariant_model.py tests/test_features_pipeline.py tests/test_experiment.py -q --tb=short
& '.venv312/Scripts/python.exe' -m pytest -q --tb=short
git diff --check
```

- [ ] Revue indépendante du diff : chercher les anciens préremplissages dans
  les quatre scripts, les swaps recalculés à état final, les écrasements et les
  chemins d'exécution non testés. Corriger les problèmes imputables à ce chantier.
- [ ] Livrer un bilan donnant les fichiers effectivement branchés, les tests
  et entraînements courts exécutés, le run de comparaison, les limitations
  encore ouvertes et le fait que le modèle actif n'a pas été remplacé.

## Critères de fin et retour arrière

L'intégration est terminée seulement si les quatre consommateurs utilisent le
code partagé, les chemins 14/v3 et les chemins historiques 8 sont vérifiés,
les artefacts sont compatibles et traçables, les tests causaux passent et les
entraînements courts sauvegardent/rechargent leurs candidats. Un gain au-delà
de 58,83 % n'est pas requis pour intégrer une correction de contamination et
ne doit pas être annoncé sans nouvelle preuve.

Si une limite d'environnement empêche un entraînement, nommer précisément la
vérification manquante et déclarer l'intégration non entièrement validée ; ne
pas masquer l'absence par des tests mockés. Aucune activation automatique de
modèle n'est prévue. Restaurer uniquement les changements de ce chantier si
nécessaire, après comparaison au statut initial, en préservant les travaux
sélectifs/GPU/timelines et les artefacts historiques. Ne pas revenir au greedy
comme repli silencieux.
