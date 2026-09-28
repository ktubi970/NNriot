# Historiques spatiaux pour la prédiction avant match — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer les timelines antérieures en profils de positionnement et interactions d'équipe, puis mesurer leur gain prédictif sans fuite temporelle.

**Architecture:** Extraction pure et cache dérivé versionné ; historique interrogé à une date cible ; pipeline commun arbres/réseaux/service. Les modèles spatiaux sont des candidats séparés des checkpoints actuels, évalués par ablations chronologiques appariées.

**Tech Stack:** Python 3.12, SQLite, NumPy, scikit-learn/LightGBM déjà disponibles, TensorFlow/Keras, pytest ; pas de dépendance ajoutée sans besoin mesuré.

**Spec:** `docs/superpowers/specs/2026-09-27-universal-timelines-and-spatial-history-design.md`, sections communes et B.

## Global Constraints

- Tous les fichiers de données, caches, bases temporaires, rapports et modèles candidats restent sous `D:/projet/NNriot/runs/` ; aucun cache de données sur C:.
- Chaque accès reçoit un `db_path` explicite et une identité de base vérifiée ; aucune écriture ne dépend du répertoire courant.
- Une timeline historique n'est admissible que si `history_end_ms < target_start_ms` ; une fin inconnue est exclue de l'historique prédictif.
- Les modèles respectent l'anti-symétrie sous échange d'équipes et l'invariance sous permutation intra-équipe ; les masques suivent les mêmes transformations.
- Aucun test ne contacte Riot ; les clés ne figurent ni dans les journaux, ni dans les bases de coordination, ni dans les rapports.
- Aucun gain prédictif n'est affirmé sur la seule base d'un nombre de timelines collectées.

## Review Focus

1. Parties se chevauchant, même timestamp de début, fin inconnue ou fin égale au début cible : exclusion stricte — tâche 3.
2. Joueur inconnu, phase non atteinte, cadence irrégulière ou coordonnées contradictoires : masque explicite sans trajectoire inventée — tâches 1/2.
3. Permutations des joueurs, échange A/B et côté physique rouge/bleu : transformations distinctes et mêmes probabilités attendues — tâches 2/4/5.
4. Checkpoint ancien ou schéma/scaler incompatible : ancien chemin intact ou erreur explicite, jamais de dimensions complétées silencieusement — tâche 5.
5. Gain apparent causé par meilleure couverture ou fuite de calibration/test : cibles identiques, contrôle coverage_only et manifestes figés — tâches 6/7.

## Dépendances et préparation

Lire le plan A et la spec. Les contrats d'extraction peuvent être développés sur fixtures avant la fin de collecte ; l'évaluation réelle attend un corpus qualifié. Le plan B ne nécessite pas que tout le backfill soit terminé, mais exige un snapshot figé et une couverture mesurée.

À l'exécution, réévaluer les changements locaux et les interfaces effectivement intégrées. `core/player_priors.py`, `core/selective_classification.py`, `core/selective_infer.py` et des tests sont déjà modifiés par d'autres travaux : aucune réinitialisation, aucun commit englobant ces changements. Rebaser les modifications prévues sur leur version acceptée et comparer la baseline sur cet état exact.

Définir `$py='D:/projet/NNriot/.venv312/Scripts/python.exe'`. Tous les répertoires d'expérience ont un nom unique sous `D:/projet/NNriot/runs/spatial_history_<date>/`. Pour chaque invocation pytest, définir `PYTEST_ADDOPTS` avec un `--basetemp` neuf sur D: ; placer aussi les temporaires des processus enfants et de SQLite sur D:. Ne pas réutiliser un dossier contenant un corpus comme répertoire de test.

## Structure et interfaces communes

| Fichier | Responsabilité |
|---|---|
| `core/spatial/contracts.py` | Schéma, types, noms et dimensions immuables |
| `core/spatial/extraction.py` | Observations → résumé par joueur et match |
| `core/spatial/history.py` | Disponibilité temporelle et profils par joueur/rôle |
| `core/spatial/complementarity.py` | Agrégation symétrique des paires de profils |
| `core/spatial/artifacts.py` | Métadonnées et compatibilité des checkpoints |
| `core/database.py` | Accès au cache dérivé SQLite, centralisé |
| `core/features/pipeline.py` | Représentations partagées NN/tabulaire |
| `core/point_in_time_dataset.py` | Replay des fins et dataset comparable |
| `core/prediction_service.py` | Résolution des profils à la date cible et inférence |
| `scripts/build_spatial_cache.py` | Construction incrémentale et rapport de couverture |
| `scripts/benchmark_spatial_history.py` | Manifestes, variantes, métriques et décision |

Les types publics sont déclarés dès la tâche 1 :

```python
from dataclasses import dataclass
import numpy as np

SPATIAL_SCHEMA = 'spatial_v1'
SUMMARY_DIM = 54
STYLE_DIM = 114
COMPLEMENTARITY_DIM = 9

@dataclass(frozen=True)
class SpatialSummary:
    match_id: str
    puuid: str
    role: str
    champion: str
    team_id: int
    start_ms: int
    end_ms: int | None
    values: np.ndarray       # (54,)
    valid_mask: np.ndarray   # (54,), bool
    counts: np.ndarray      # (54,), dénominateurs
    quality: dict
    raw_sha256: str

@dataclass(frozen=True)
class PlayerStyle:
    values: np.ndarray      # (54,), valeurs lissées, zéro si manquantes
    valid_mask: np.ndarray  # (54,)
    metadata: np.ndarray    # (6,)
    history_count: int
    latest_history_end_ms: int | None

@dataclass(frozen=True)
class TemporalProtocol:
    mode: str               # 'frozen' ou 'replay'
    train_cutoff_ms: int
```

Les tableaux sont copiés en lecture seule dans les objets publics. Les schémas des artefacts incluent les noms ordonnés, pas seulement les dimensions.

## Task 1: Définir le schéma et extraire les résumés observés

**Files:** créer `core/spatial/__init__.py`, `contracts.py`, `extraction.py`, `tests/test_spatial_extraction.py`, `docs/spatial_features_v1.md`.

**Interfaces:** `extract_spatial_summaries(match: dict, timeline: dict) -> list[SpatialSummary]` ; `canonical_xy(x: float,y: float,team_id: int) -> tuple[float,float]`. Échec d'identité globale : `ValueError` avec motif sans données personnelles. Données locales manquantes : masques et compteurs, pas exception globale.

- [ ] Écrire des fixtures de trois phases avec dix participants, équipes physiques et timestamps explicites. Le test de côté ne dépend pas d'une API externe.

```python
def test_side_transform_is_not_prediction_team_order():
    import numpy as np
    from core.spatial.extraction import canonical_xy
    np.testing.assert_allclose(canonical_xy(3000, 6000, 100), (0.2, 0.4))
    np.testing.assert_allclose(canonical_xy(12000, 9000, 200), (0.2, 0.4))
```

- [ ] Constater RED avec `& $py -m pytest tests/test_spatial_extraction.py -q`.
- [ ] Implémenter les 18 mesures par phase **dans l'ordre de la spec** : grille 3×3 rangée par `v` puis `u`, extérieur, moyennes u/v, variances u/v, distances min allié/adversaire, fractions de proximité allié/adversaire. Distances divisées par 15000 ; seuil de proximité appliqué à 2000 unités brutes. Bornes des cases : gauche incluse, droite exclue sauf 1 inclus dans la dernière case.
- [ ] Calculer chaque dénominateur sur les seules observations admissibles. Pour les distances à l'allié/adversaire, exiger au moins une position de l'autre groupe au même timestamp ; exclure le joueur lui-même. Masque vrai à partir de trois observations ; variances avec `ddof=0`. Les frames avant 2 min et après 30 min ne contribuent pas aux mesures v1, mais restent dans le brut et la qualité.
- [ ] Tester NaN/inf, zéro réel, case extérieure, horodatages irréguliers, phase courte, doublons identiques/contradictoires, ordre du JSON, participants inversés et absence partielle d'alliés. Assert des valeurs et dénominateurs calculés à la main sur de petites fixtures. Vérifier le hash du JSON canonique et l'absence de mutation des payloads.
- [ ] Tests GREEN, documentation des limites de résolution puis commit `feat: extract versioned spatial summaries from observed frames`.

## Task 2: Construire le cache dérivé et qualifier la couverture

**Files:** modifier `core/database.py` ; créer `scripts/build_spatial_cache.py`, `tests/test_spatial_cache.py` ; compléter `docs/spatial_features_v1.md`.

**Interfaces:** `init_spatial_cache(cache_path: str) -> None`, `save_spatial_summaries(cache_path: str,summaries: list[SpatialSummary]) -> int`, `iter_spatial_summaries(cache_path: str) -> iterator`, `build_spatial_cache(source_db: str,cache_path: str,report_path: str,limit: int | None=None) -> dict`. Accès source en lecture seule.

- [ ] Écrire le test : reconstruire deux fois les mêmes matchs ne duplique rien ; remplacer le contenu source produit une nouvelle version identifiée par hash, sans écraser l'ancienne ; le cache ne devient pas la source canonique.

```python
def test_cache_round_trip_and_idempotence(cache_path, summaries):
    from core import database
    database.init_spatial_cache(cache_path)
    assert database.save_spatial_summaries(cache_path, summaries) == len(summaries)
    assert database.save_spatial_summaries(cache_path, summaries) == 0
    loaded = list(database.iter_spatial_summaries(cache_path))
    assert len(loaded) == len(summaries)
    assert loaded[0].raw_sha256 == summaries[0].raw_sha256
```

Définir `summaries` dans ce fichier par l'extracteur et les fixtures de la tâche 1 ; `cache_path` se situe sous le basetemp D:.

- [ ] Constater RED puis créer les tables `spatial_cache_meta`, `spatial_match_versions` et `spatial_player_summaries`. Clé de résumé `(match_id,puuid,spatial_schema,raw_sha256)`. `spatial_match_versions` désigne la version active cohérente avec le snapshot du brut ; ne jamais compter deux versions du même match dans l'historique.
- [ ] Indexer `(puuid,role,end_ms)` ; conserver dates, côté, qualité et dénominateurs. Traiter par lots bornés, commits courts et curseur de reprise. Détecter changement de snapshot/hash avant reprise ; aucune lecture de toutes les timelines en mémoire, aucun N+1 par joueur lors du benchmark.
- [ ] Conserver la date de réception quand elle est connue et celle d'extraction. Pour les anciennes lignes sans provenance de réception, laisser `first_received_at_ms` inconnu ; ne jamais l'antidater au jour du match. Le rapport distingue backtest rétrospectif et disponibilité réellement observée.
- [ ] Tester crash entre deux lots, timeline invalide, fin inconnue et base cible incorrecte. Le rapport inclut débits, disque, erreurs, couverture des phases et cadence ; le rapport de profondeur 1/5/10/20 est complété après la tâche 3.
- [ ] Vérifier intégrité/FK du cache et immutabilité de la source, puis commit `feat: build resumable spatial summary cache`.

## Task 3: Construire des historiques admissibles sans fuite

**Files:** créer `core/spatial/history.py`, `tests/test_spatial_history.py` ; modifier `core/point_in_time_dataset.py`, `tests/test_point_in_time_dataset.py` et les appels de `scripts/train_pre_match_model.py` qui chargent ce dataset. Adapter `core/player_priors.py` seulement après rapprochement des changements locaux.

**Interfaces:** `SpatialHistoryIndex(cache_path: str,protocol: TemporalProtocol)` ; `profile(puuid: str | None,role: str,as_of_ms: int) -> PlayerStyle` ; `advance_before(target_start_ms: int) -> None`. Les demandes doivent être non décroissantes dans le mode stream ; une requête plus ancienne utilise un index readonly séparé, jamais l'état courant déjà avancé.

- [ ] Écrire un test d'admissibilité avec trois parties finissant à `T-1`, `T`, `T+1`, une fin inconnue et la cible elle-même. Seule la première peut contribuer.

```python
def test_strict_end_cutoff(history_index):
    p = history_index.profile('player-1', 'MIDDLE', as_of_ms=1_800_000)
    assert p.history_count == 1
    assert p.latest_history_end_ms == 1_799_999
```

La fixture insère les cinq résumés décrits dans un cache réel. Ajouter un test modifiant arbitrairement les timelines futures : les features de la cible doivent rester bit à bit identiques.

- [ ] Constater RED puis utiliser une file ordonnée de fins : avant chaque cible, publier uniquement les événements `end_ms < start_ms`. Un match au début antérieur mais à fin postérieure reste en attente. Ne pas utiliser les résultats d'une cible pour ses propres priors. Utiliser le vrai `gameStartTimestamp`, avec fallback documenté sur création seulement si nécessaire et mesuré.
- [ ] Construire le profil sur les 20 dernières parties du même rôle. Pour chaque mesure disponible, moyenne pondérée par `2 ** (-age_days/30)`, lissée vers le prior de rôle disponible à cette date avec force 5. Si aucun prior de rôle n'existe, garder la moyenne individuelle ; sans observation individuelle, valeur zéro/masque nul. Métadonnées dans l'ordre déclaré ; âge du dernier match en jours, normalisé ensuite.
- [ ] Appliquer la même règle de fin aux priors agrégés de la baseline. `frozen` : cutoff `min(as_of_ms,train_cutoff_ms)` ; `replay` : cutoff `as_of_ms`. Les poids et transformations entraînées restent figés dans les deux cas. Produire un journal de provenance `latest_history_end_ms` et nombre de matchs utilisés sans exposer inutilement les PUUID dans les rapports publics.
- [ ] Tester frontières de splits, absence de joueurs, ordre d'ingestion inversé, rôles inconnus, deux requêtes concurrentes de dates différentes, identité de résultat index/stream et concordance avec une requête SQL de référence sur fixtures.
- [ ] Vérifier tests des priors existants et nouveaux tests, puis commit `feat: build spatial and aggregate priors from completed matches only`.

## Task 4: Exposer styles et complémentarité dans les deux familles de modèles

**Files:** créer `core/spatial/complementarity.py`, `tests/test_spatial_pipeline.py` ; modifier `core/features/pipeline.py`, `core/equivariant_model.py`, `scripts/train_tree_baseline.py`, `scripts/train_pre_match_model.py`, `tests/test_equivariant_model.py`, `tests/test_deepsets_layers.py`.

**Interfaces:** `team_complementarity(styles: list[PlayerStyle]) -> np.ndarray` de dimension 9 ; `DraftFeaturePipeline.extract_spatial_inputs(feature_dict: dict, *, styles: dict[str,PlayerStyle],include_complementarity: bool) -> dict[str,np.ndarray]` ; `extract_spatial_tabular` avec les mêmes arguments. `feature_version='spatial_v1'` sélectionne ces interfaces ; l'ancien `extract_tensors` reste à 14 dimensions.

- [ ] Tester les formes : team_a/team_b `(5,128)` ; si activée, deux entrées `team_a_complementarity` / `team_b_complementarity` de forme `(9,)`. Tabulaire : `ENRICHED_TABULAR_DIM + 10*114`, puis `+18` avec complémentarité. Les masques sont concaténés après les 54 valeurs puis les six métadonnées ; même ordre pour tous les chemins.
- [ ] Constater RED. Implémenter le chevauchement par paire/phase `sum(min(p_i,p_j))` sur les dix cases lorsque les deux distributions sont valides. Par équipe/phase : moyenne, écart-type `ddof=0`, nombre de paires valides divisé par 10. Pas de paire valide : trois zéros. Les paires sont non ordonnées ; aucun historique partagé de duo n'est supposé.
- [ ] Ajouter la branche commune d'équipe au builder neuronal avec `complementarity_dim=0` par défaut pour compatibilité. Quand activée, encoder et fusionner chaque vecteur de 9 valeurs avec l'embedding de son équipe à poids partagés ; les heads conservent leur construction anti-symétrique. La classification auxiliaire à trois classes inverse A/B et conserve la classe neutre.
- [ ] Pour les arbres, définir la transformation de swap sur les blocs de joueurs et d'équipes ; recalculer les différences de composition existantes au lieu d'inverser aveuglément tout le vecteur. Symétriser à l'inférence avant la calibration compatible.

```python
def test_spatial_predictions_swap_and_permute(spatial_predictor, match):
    import copy
    import numpy as np
    p = spatial_predictor(match)
    shuffled = copy.deepcopy(match)
    shuffled['team_a'].reverse()
    shuffled['team_b'] = shuffled['team_b'][2:] + shuffled['team_b'][:2]
    swapped = {**match, 'team_a': match['team_b'], 'team_b': match['team_a']}
    np.testing.assert_allclose(spatial_predictor(shuffled), p, atol=1e-6)
    np.testing.assert_allclose(spatial_predictor(swapped), 1-p, atol=1e-6)
```

La fixture `spatial_predictor` combine les vrais extracteurs et un petit modèle sans entraînement long, successivement NN et arbre symétrisé. Ajouter le même test avec masques absents, rôles dupliqués/inconnus et profils identiques. Canonisation stable de ces rôles ou pooling invariant ; ne pas laisser l'ordre d'entrée décider du résultat.

- [ ] Vérifier sauvegarde/rechargement du petit modèle et les invariances, puis commit `feat: add versioned spatial inputs to symmetric prediction models`.

## Task 5: Raccorder les artefacts et PredictionService

**Files:** créer `core/spatial/artifacts.py`, `core/legacy_prediction_adapter.py`, `tests/test_spatial_serving.py` ; modifier `core/prediction_service.py`, `core/model_factory.py`, `core/pre_match_dataset.py`, `scripts/train_pre_match_model.py`, `scripts/train_tree_baseline.py`, `app/routes/api.py`, `tests/test_predict_api.py`, `tests/test_model_calibration_guard.py` ; examiner `core/shapley_credit.py` et `templates/predictor.html` pour la compatibilité des réponses.

**Interfaces:** artefact `<checkpoint>.features.json` avec `feature_version`, noms/dimensions des entrées, `spatial_schema`, phases/grille/seuils, politique des manquants, protocole temporel, identifiant/hash du cache, paramètres et hash du normaliseur. `validate_spatial_artifact(metadata: dict,model_inputs: dict) -> None`. `PredictionService` accepte un fournisseur d'historique et `db_path` explicites ; `predict_match` lit `prediction_time_ms` et les PUUID optionnels.

- [ ] Tester qu'un checkpoint ancien produit toujours ses anciennes features ; un checkpoint spatial sans artefact ou de dimension erronée est rejeté. Une absence d'historique avec artefact valide reste supportée grâce aux masques.

```python
def test_unknown_player_is_not_a_version_error(spatial_service, request_without_ids):
    result = spatial_service.predict_match(request_without_ids)
    assert result['feature_version'] == 'spatial_v1'
    assert result['history_coverage']['players_with_history'] == 0
    assert 'is_actionable' in result
```

- [ ] Constater RED puis brancher le fournisseur uniquement dans la façade. La date est capturée une fois par requête ; rejeter une date future explicitement demandée, et utiliser une date passée pour les replays. Valider les PUUID sans les déduire du pseudo ; sans eux, profils absents. Le sanitizeur pré-match conserve seulement les identifiants/date nécessaires, jamais les statistiques du match cible.
- [ ] Raccorder réellement `/api/predict/custom` et `/api/predict` à la façade : conserver les PUUID reçus, champions et rôles ; déplacer les opérations de tenseurs/trainer hors des endpoints. Pour l'ancien trainer, `LegacyPredictionAdapter(trainer,lock).predict(match_dict: dict) -> dict` encapsule le chemin existant et son garde de calibration ; il est appelé par `PredictionService`, jamais directement par Flask. Garder les alias de probabilités attendus par l'interface et tester qu'une erreur de calibration reste bloquante. Ne pas synthétiser une statistique de victoire à partir du KDA.
- [ ] Appliquer le normaliseur entraîné aux seules valeurs valides ; replacer les valeurs masquées à zéro après transformation. Aucune adaptation du scaler en serving. Les métadonnées indiquent les historiques utilisés et le mode effectif ; un fallback de modèle, s'il est configuré, annonce sa version et utilise sa propre calibration.
- [ ] Supporter dans l'artefact la liste ordonnée des trois membres d'ensemble, leurs hashes et graines. Charger les membres compatibles, moyenner leurs probabilités, puis appliquer la calibration propre à cet ensemble ; un membre absent ou divergent rend l'artefact incompatible. Ne pas moyenner arbitrairement les diagnostics d'incertitude.
- [ ] Préserver probabilités calibrées, conviction, tier et abstention. Vérifier les anciennes routes sans IDs et celles avec IDs. Pour les diagnostics de crédit joueur, ne pas appeler l'ancien helper avec des entrées incomplètes : tant que le retrait d'un joueur ne recalcule pas sa complémentarité, retourner un diagnostic explicitement indisponible pour ce modèle ; ne pas produire d'explication fausse. Le calcul complet constitue une extension distincte.
- [ ] Ajouter un test d'équivalence offline/serving à même date, même cache et même artefact, puis commit `feat: serve spatial models with explicit history and artifact compatibility`.

## Task 6: Construire le benchmark d'ablation reproductible

**Files:** créer `scripts/benchmark_spatial_history.py`, `tests/test_spatial_benchmark.py`, `docs/spatial_benchmark.md` ; réutiliser `core/training_metrics.py`, `core/calibration.py` et les entraîneurs existants.

**Interfaces:** `create_spatial_experiment(source_db: str,cache_path: str,output_dir: str) -> dict` fige corpus, splits, protocole et budget ; `run_spatial_benchmark(manifest_path: str, *, evaluate_test: bool=False) -> dict`. L'évaluation du test exige un candidat sélectionné inscrit au manifeste ; une deuxième évaluation est refusée sauf nouvelle expérience avec nouveau bloc test.

- [ ] Écrire un test avec probabilités synthétiques identiques : les deltas doivent être zéro et la décision `keep_baseline`. Ajouter un candidat meilleur en moyenne mais à intervalle recouvrant zéro : pas de promotion.

```python
def test_no_lift_never_promotes():
    import numpy as np
    from scripts.benchmark_spatial_history import compare_predictions
    y = np.tile([0, 1], 3000)
    p = np.full(6000, 0.5)
    days = np.repeat(np.arange(12), 500)
    r = compare_predictions(y, p, p, utc_days=days, seed=17)
    assert r['delta_log_loss'] == 0
    assert r['decision'] == 'keep_baseline'
```

`compare_predictions(y,baseline,candidate,*,utc_days,seed) -> dict` est défini dans ce script et applique les métriques/seuils de la spec, indépendamment de l'entraînement.

- [ ] Constater RED puis figer IDs, vrais débuts/fins, hashes, cutoff des disponibilités, purge des matchs traversant les frontières et séparation 70/15/15. Diviser validation chronologiquement en 2/3 sélection et 1/3 calibration, avec purge des labels chevauchants. Interdire un cache reconstruit différemment sous le même manifeste.
- [ ] Comparer `baseline`, `coverage_only`, `styles`, `styles+complementarity` sur **toutes** les mêmes cibles, y compris sans historique. Même budget et graines par variante. La baseline utilise les mêmes priors temporellement corrigés. Effectuer les deux protocoles frozen/replay avec résultats séparés.
- [ ] Sélection et early stopping utilisent replay. Réutiliser les poids pour le contrôle frozen, sans entraînement supplémentaire : maximum initial de 144 entraînements pour les deux familles. Le smoke test mesure le coût de ce budget ; une réduction avant test doit rester identique entre variantes et figurer au manifeste.
- [ ] Définir avant résultats les six configurations par famille. Backend initial : `HistGradientBoostingClassifier` déjà utilisé ; `max_depth={3,6,None}` × `learning_rate={0.03,0.1}`, `max_iter=150`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `early_stopping=False`, mêmes indices catégoriels vérifiés sur les vecteurs étendus. NN : `latent_dim={32,64,128}` × `dropout_rate={0.1,0.3}`, `embed_dim=32`, batch 256, Adam `1e-3`, maximum 30 epochs et patience 5 sur la partie sélection. Figer les autres options à celles de la baseline acceptée. Trois graines 17/29/43 pour chaque configuration ; enregistrer versions exactes des bibliothèques.
- [ ] Pour les variantes NN, utiliser les mêmes formes et capacités : baseline = 14 features + zéros ; coverage_only = mêmes masques/métadonnées mais 54 valeurs spatiales nulles ; styles = profil complet et complémentarité nulle ; styles+complementarity = tous les blocs. Le modèle legacy accepté reste une référence additionnelle. Publier tailles et temps d'entraînement ; sélectionner sur validation un candidat apportant aussi un avantage sur le contrôle coverage_only.
- [ ] Normalisations et paramètres appris sur train ; sélection sur validation initiale ; température et seuils d'abstention sur validation finale ; test intact jusque-là. Chaque configuration est un ensemble à poids égaux des trois graines ; calibrer après moyenne des probabilités. Aucun choix de meilleur seed ni ajustement des poids d'ensemble sur test.
- [ ] Calculer log-loss, Brier, accuracy, AUC, calibration à 15 bins et risque/couverture. Bootstrap apparié par jour UTC, 2000 réplications, et sensibilité aux blocs de sept jours si possible. Rapporter toutes les strates de profondeur, régions, patches et inconnus ; ne pas présenter un gain sur les seuls mieux couverts comme un gain global.
- [ ] Vérifier les seuils de la spec, la taille minimale du test et le refus de réouverture du test ; commit `feat: compare spatial history candidates on frozen chronological experiments`.

## Task 7: Exécuter la qualification du corpus et les expériences candidates

**Files:** rapports uniquement sous un dossier neuf `D:/projet/NNriot/runs/spatial_history_<date>/`, documentation mise à jour après mesures.

- [ ] Figer un snapshot via backup SQLite après un lot terminé ; vérifier le manifeste et le journal, source inchangée et provenance des timelines. Ne pas bloquer la collecte réseau pendant un long scan de données sur le HDD.
- [ ] Construire le cache, puis publier les profondeurs 1/5/10/20, pourcentage de cibles avec les dix joueurs couverts, cadences, phases exploitables, disque, vitesse et taux d'erreurs. La disponibilité du joueur avant match est mesurée séparément de la disponibilité d'une timeline.
- [ ] Faire un smoke test sur un sous-ensemble chronologique, sans consulter le futur test final ; contrôler mémoire, débit et égalité des features offline/serving. Le dimensionnement complet dépend de cette mesure, pas d'une durée supposée.
- [ ] Exécuter les six configurations × trois graines prévues, d'abord arbres puis NN ; conserver les paramètres de la baseline et les budgets comparables. Publier les résultats validation et la sélection du candidat avant d'ouvrir le test final.
- [ ] Évaluer une fois le candidat choisi. Promotion proposée seulement si tous les seuils, invariances et contrôles de calibration sont satisfaits. Sinon rapporter `keep_baseline` ou `inconclusive`, avec raisons et effectifs.
- [ ] Préparer une validation prospective sur les nouvelles prédictions : journaliser à l'instant de prédiction la version, la couverture effectivement disponible, le cutoff et la probabilité, puis associer le résultat seulement après fin du match. Ce contrôle est distinct du backtest rétrospectif ; il ne provoque aucune collecte additionnelle par joueur ni promotion automatique.
- [ ] Revue indépendante du protocole, de la temporalité et du service. Livrer checkpoints candidats, métadonnées, rapports et procédure de retour arrière. La bascule en production est une action distincte sur le résultat concret, pas une conséquence automatique de l'entraînement.

## Critère d'acceptation

Le livrable est une chaîne reproductible capable de produire et servir des profils spatiaux historiques avec masques, provenance et garanties temporelles, accompagnée d'une comparaison honnête. Un benchmark concluant à l'absence de gain remplit l'objectif d'évaluation ; il ne justifie pas le remplacement de la baseline.

## Auto-revue de planification

Contrats de dimensions rapprochés (54 mesures, 114 dimensions de style, 128 entrées/joueur, 9 valeurs par équipe) ; frontière fin/début et disponibilité rétrospective distinguées ; route custom réellement raccordée ; checkpoints anciens, ensembles et diagnostics traités ; budget et seuils fixés avant test. Les cinq classes du Review Focus ont des tests assignés. Les commandes et tests décrits constituent du travail futur : ils n'ont pas été exécutés par la rédaction de ce plan.
