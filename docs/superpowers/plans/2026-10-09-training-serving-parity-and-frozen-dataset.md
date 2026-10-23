# Training/Serving Parity and Frozen Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer un dataset chronologique figé, ses adaptateurs v3/v4 et une preuve de parité exacte avec le service, avec un blocage explicite de 18D si les timelines manquent.

**Architecture:** Rejouer les faits historiques avec le même constructeur de priors et le même contrat d'extraction que le service. Séparer la sélection temporelle des cibles, le stockage des artefacts et les adaptateurs d'entraînement ; les trainers consomment uniquement train/validation. Garder la campagne comparative et l'ouverture du test final dans un plan d'expérience ultérieur.

**Tech Stack:** Python 3.12, SQLite en lecture seule, NumPy float32, TensorFlow 2.21/Keras 3.15.1, pytest ; aucune dépendance supplémentaire.

**Spec:** `docs/superpowers/specs/2026-10-09-training-serving-parity-design.md` ; lire ce document et le présent plan avant exécution.

**État d’exécution Native (9 octobre 2026) :** les six tâches fonctionnelles sont réalisées, avec 164 tests ciblés verts, 148 458 comparaisons de tenseurs exactes et 20/20 contrôles SQLite exacts. La revue indépendante finale est terminée : un problème Important sur les timelines partielles a été reproduit RED, corrigé et vérifié GREEN ; aucun Critical ou Minor étayé. Le snapshot final est `runs/aligned_dataset_20261009_reviewed`. Les six étapes de commit restent ouvertes par décision documentée : dépendances locales préexistantes non commitées, diff attribuable conservé depuis le snapshot de départ ; aucune intégration Git mélangée. Voir `docs/model_serving/aligned_dataset_verification.md` et `docs/model_serving/aligned_dataset_execution.md`.

## Global Constraints

- Répertoire de travail : `D:\projet\NNriot` ; commandes Python : `.\.venv312\Scripts\python.exe` ; Python 3.12.
- Aucune nouvelle dépendance ; utiliser SQLite, NumPy, TensorFlow/Keras et les modules déjà installés.
- Bases opérationnelles en lecture seule : URI `mode=ro`, `PRAGMA query_only=ON` ; aucune migration, fusion, collecte ou mise à jour `is_trained`.
- Préserver tous les changements locaux préexistants et `archive/frontend-2026-10-09/` ; aucun reset, nettoyage global ou commit de changements étrangers à la tâche.
- Dimensions exactes : contrats `nnriot.players14.v1` et `nnriot.players18.v1` ; entrées float32 `(N, 5, 14)` ou `(N, 5, 18)`, macro float32 `(N, 15)` ; aucun padding ni troncature.
- Rôles canoniques : `TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY` ; chaque équipe a cinq joueurs ; chaque identité connue apparaît une seule fois.
- Cutoff de features : `game_creation_time` en millisecondes UTC ; utiliser uniquement les résultats avec `result_end_ms < game_creation_time` ; timestamps d'acquisition inconnus, hypothèse explicitement déclarée.
- Historique : fenêtre globale de 10 000 créations CLASSIC strictement antérieures ; les labels ML historiques ne conditionnent pas les priors ; matchups neutres au service et dans le nouveau parcours d'entraînement.
- Population cible : queues Riot `420` et `440`, mode `CLASSIC`, rôles et labels valides ; l'historique conserve tous les CLASSIC, comme le service.
- Partitions communes définies avant les métriques : 3 000 cibles récentes pour test, 3 000 précédentes pour validation, créations ex æquo regroupées ; purger train avec fin >= début validation et validation avec fin >= début test.
- Le test réservé commence strictement après la dernière création du cohort exploratoire antérieur ; exclure ses identifiants test des cibles du nouveau dataset et enregistrer cette règle.
- Chaque fichier figé possède un SHA-256 ; le replay exige les mêmes hashes de sources de features ; toute incompatibilité échoue explicitement.
- Le parcours aligné charge et entraîne uniquement train/validation ; aucune lecture des labels test ni métrique test par les trainers ; aucun chargement de poids anciens.
- 18D exige une source de timelines historiques utilisable dans la fenêtre réelle ; absence de source = `FeatureUnavailableError`, avec causes et couverture, jamais une disponibilité inventée.
- Préserver `draft_advantage` antisymétrique et miroir 50/50 ; conserver `DirectionalSideBias` séparé dans `winner` ; aucun changement d'architecture dans ce chantier.
- Aucun changement automatique de checkpoint configuré, de calibration de production ou de statut `validated_for_serving`.

## Review Focus

1. Partie commencée plus tôt mais terminée après le cutoff, ou terminée exactement au cutoff : priors inchangés jusqu'à sa disponibilité stricte — tâche 1.
2. Créations ex æquo et fin tardive à une frontière train/validation/test : regrouper les créations et purger le résultat indisponible — tâche 2.
3. Index participants vide/incomplet, ou résultat historique sans label ML : parité des priors avec le service ; absence de label ML historique sans exclusion du résultat Riot — tâches 1 et 5.
4. Dataset récent sans timelines, fichier modifié, fichier référencé hors du répertoire : refus explicite, aucune reconstruction silencieuse — tâches 3 et 4.
5. Cache legacy fourni avec le dataset aligné, ou accès aux labels test pendant un pilot : refus avant apprentissage et aucune ouverture du test — tâche 4.

---

## Périmètre et carte des fichiers

Ce plan couvre une seule chaîne : faits figés → priors → tenseurs → entrées des trainers. Il n'ajoute aucune architecture, baseline ni collecte. Les travaux indépendants « acquisition de timelines récentes » et « expérience comparative et promotion » nécessiteront leurs propres plans.

| Fichier | Responsabilité |
| --- | --- |
| `core/frozen_history.py` (créer) | Provider de priors depuis les mêmes faits compactés que le provider SQLite ; index par identité sans modifier la fenêtre globale. |
| `core/aligned_features.py` (créer) | Extraction stricte partagée avec le service, sans state global appris. |
| `core/aligned_partitions.py` (créer) | Population cible, validation du label Riot et partitions purgées. |
| `core/aligned_artifacts.py` (créer) | Écriture exclusive, hashes, disponibilité 14/18, lecture train/validation. |
| `core/aligned_training.py` (créer) | Préparation des entrées des deux trainers et clôture d'un pilot sans test final. |
| `scripts/prepare_aligned_dataset.py` (créer) | Inventaire, préparation explicite et vérification/replay. |
| `core/pre_match_history.py:63` (modifier) | Conserver queue et labels dans le snapshot uniquement si demandé. |
| `core/prediction_service.py:103` (modifier) | Utiliser la fonction d'extraction commune, avec le pipeline du service actuel. |
| `core/serving_contract.py:29` (modifier) | Inclure l'extracteur commun dans les hashes de runtime. |
| `scripts/train_deepsets_v3.py:44` et `scripts/train_deepsets_v4_timeline.py:43` (modifier) | Nouveau mode `dataset_dir`, boucle d'optimisation existante, sortie avant évaluation test. |
| `tests/aligned_dataset_fixture.py` (créer) | Faits minimaux valides avec identités, rôles, queues, fins et labels multitâches. |
| `tests/test_frozen_history.py`, `tests/test_aligned_partitions.py`, `tests/test_aligned_artifacts.py`, `tests/test_aligned_training.py`, `tests/test_aligned_parity.py` (créer) | Tests correspondant aux cinq responsabilités et à l'audit final. |
| `docs/pre_match_training.md`, `docs/model_serving/README.md` (modifier) | Commandes, portée des garanties et disponibilité réelle. |

Lire aussi `core/features/ordered_matchup.py`, `core/ordered_training.py`, `core/point_in_time_dataset.py`, `core/experiment.py`, `core/calibration.py` et les tests existants des trainers. Le nouveau parcours est explicitement versionné ; les chemins exploratoires existants ne reçoivent pas de certification rétroactive.

## Commandes et discipline d'exécution

Toutes les commandes ci-dessous utilisent PowerShell à la racine du projet. Avant pytest :

```powershell
New-Item -ItemType Directory -Force -Path 'D:\projet\NNriot\scratch\aligned-tests' | Out-Null
$env:TEMP='D:\projet\NNriot\scratch\aligned-tests'
$env:TMP=$env:TEMP
git status --short
```

Conserver cet état initial dans le journal de travail. Les modifications existantes des trainers et du service appartiennent à l'utilisateur ou au chantier précédent. À chaque fin de tâche, examiner les hunks et créer un commit limité à cette tâche si les changements peuvent être isolés. Ne jamais ajouter intégralement un fichier déjà modifié sans contrôler son diff indexé.

```powershell
git diff -- core/pre_match_history.py scripts/train_deepsets_v3.py scripts/train_deepsets_v4_timeline.py
git add -p -- core/pre_match_history.py
git diff --cached --check
git diff --cached --stat
```

Ce bloc illustre la discipline Git ; chaque tâche fournit ses chemins réels. Si l'index contient du travail préexistant, ne pas créer de commit mêlé : conserver le diff de la tâche et signaler cette limite. Aucun commit n'est créé pendant la rédaction de ce plan.

### Task 1: Provider figé et extraction commune au service

**Files:** Create `core/frozen_history.py`, `core/aligned_features.py`, `tests/aligned_dataset_fixture.py`, `tests/test_frozen_history.py`; Modify `core/prediction_service.py:139`, `core/serving_contract.py:29`.

**Interfaces:**
- Consumes `decode_history_row(row) -> dict`, `build_target_history(records, match, *, as_of_ms: int) -> ResultOrderedHistory`, `FeatureContract.extract(match, pipeline, *, macro: bool) -> dict[str, np.ndarray]`.
- Produces `FrozenHistoryProvider(rows: list[dict], *, history_limit: int = 10000).enrich(match: dict, *, include_timeline: bool) -> dict`.
- Produces `extract_aligned_inputs(match: dict, contract_version: str, *, macro: bool = True, pipeline: DraftFeaturePipeline | None = None) -> dict[str, np.ndarray]`.
- Produces test fixture `source_row(mid: str, creation: int, end: int, winner: int = 0, *, timeline: bool = False) -> dict` and `target(row: dict) -> dict`.

- [x] **Step 1: Écrire les fixtures et les tests RED.** Utiliser les labels réels de `core.feature_labels.LABEL_KEYS` ; le bloc suivant définit entièrement la fixture, sans valeurs manquantes implicites.

```python
# tests/aligned_dataset_fixture.py
import json
from core.feature_labels import LABEL_KEYS
from core.pre_match_history import decode_history_row

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

def source_row(mid, creation, end, winner=0, *, timeline=False):
    parts = [dict(puuid=f'p{tid}_{slot}', participantId=i + 1,
                  teamId=tid, championName='Ahri' if tid == 100 else 'Lux',
                  teamPosition=ROLES[slot], win=(winner == (tid == 200)),
                  kills=2, deaths=1, assists=3, totalMinionsKilled=100,
                  neutralMinionsKilled=0, goldEarned=5000)
             for i, (tid, slot) in enumerate(
                 (tid, slot) for tid in (100, 200) for slot in range(5))]
    labels = dict.fromkeys(LABEL_KEYS, 0)
    labels.update(winner=winner, team_b_kill_lead=winner,
                  total_kills=20, team_a_kills=10, team_b_kills=10,
                  kill_handicap=0)
    metrics = {p['puuid']: dict(gd15=1., csd15=2., xpd15=3., fbi10=0.)
               for p in parts} if timeline else None
    return dict(match_id=mid, game_creation_time=creation,
                winner_label=winner, labels_json=json.dumps(labels),
                timeline_json=None, timeline_metrics=metrics,
                raw_json=json.dumps({'info': dict(
                    gameCreation=creation, gameStartTimestamp=creation + 1,
                    gameEndTimestamp=end, gameDuration=100,
                    gameVersion='16.18.1', gameMode='CLASSIC', queueId=420,
                    participants=parts)}))

def target(row):
    return decode_history_row(row)['feature_json']
```

```python
# tests/test_frozen_history.py
import numpy as np
import pytest
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from tests.aligned_dataset_fixture import source_row, target

def test_strict_end_and_global_window():
    rows = [source_row('NA1_1', 10, 100), source_row('NA1_2', 20, 30, 1)]
    query = target(source_row('NA1_q', 30, 130))
    frozen = FrozenHistoryProvider(rows, history_limit=1)
    at_end = frozen.enrich(query, include_timeline=False)
    assert at_end['priors_provenance']['history_matches'] == 0
    query['prediction_time_ms'] = 31
    later = frozen.enrich(query, include_timeline=False)
    assert later['priors_provenance']['history_matches'] == 1
    assert later['priors_provenance']['latest_result_end_ms'] == 30

def test_label_free_history_and_strict_shapes():
    old = source_row('NA1_1', 10, 20, timeline=True)
    old['winner_label'] = None
    old['labels_json'] = None
    query = target(source_row('NA1_2', 30, 130))
    for dim in (14, 18):
        enriched = FrozenHistoryProvider([old]).enrich(
            query, include_timeline=dim == 18)
        assert enriched['priors_provenance']['history_matches'] == 1
        x = extract_aligned_inputs(enriched, f'nnriot.players{dim}.v1')
        assert x['team_a'].shape == (1, 5, dim)
        assert x['macro_clash'].shape == (1, 15)
        assert all(v.dtype == np.float32 for v in x.values())

def test_creation_ties_are_not_history():
    old = source_row('NA1_1', 30, 40)
    query = target(source_row('NA1_2', 30, 140))
    assert FrozenHistoryProvider([old]).enrich(
        query, include_timeline=False)['priors_provenance']['history_matches'] == 0

def test_duplicate_sources_fail():
    row = source_row('NA1_1', 10, 20)
    with pytest.raises(ValueError, match='duplicate'):
        FrozenHistoryProvider([row, row])
```

- [x] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_frozen_history.py -q`. Attendu : erreur d'import des deux nouveaux modules.

- [x] **Step 3: Implémenter le provider et l'extraction.** Le provider trie par `(creation, match_id)` ; `bisect_left` exclut toutes les créations ex æquo. L'index par joueur ne remplace jamais la sélection globale des 10 000 positions.

```python
# core/frozen_history.py
from bisect import bisect_left
import copy
from core.pre_match_history import decode_history_row, build_target_history
from core.serving_contract import ContractError

class FrozenHistoryProvider:
    def __init__(self, rows, *, history_limit=10000):
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('history_limit must be a positive integer')
        self.rows = sorted(copy.deepcopy(rows), key=lambda r: (
            r['game_creation_time'], r['match_id']))
        ids = [r['match_id'] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate match_id')
        self.times = [r['game_creation_time'] for r in self.rows]
        self.history_limit = history_limit
        self.facts, self.positions, self.invalid = [], {}, []
        for index, row in enumerate(self.rows):
            try:
                fact = decode_history_row(row)
            except ContractError:
                fact = None
                self.invalid.append(index)
            self.facts.append(fact)
            if fact is not None:
                for puuid in {p.get('puuid') for p in fact['participants']} - {None}:
                    self.positions.setdefault(puuid, []).append(index)

    def enrich(self, match, *, include_timeline):
        as_of = match.get('prediction_time_ms')
        if type(as_of) is not int or as_of <= 0:
            raise ContractError('Frozen replay requires positive prediction_time_ms')
        stop = bisect_left(self.times, as_of)
        start = max(0, stop - self.history_limit)
        identities = {p.get('puuid') for k in ('team_a', 'team_b')
                      for p in match.get(k, []) if p.get('puuid')}
        indexes = set()
        for puuid in identities:
            positions = self.positions.get(puuid, [])
            indexes.update(positions[bisect_left(positions, start):
                                     bisect_left(positions, stop)])
        indexes.update(self.invalid[bisect_left(self.invalid, start):
                                    bisect_left(self.invalid, stop)])
        facts = [self.facts[i] for i in sorted(indexes)]
        history = build_target_history(facts, match, as_of_ms=as_of)
        enriched = history.enrich(match, include_timeline=include_timeline,
                                  as_of_ms=as_of)
        enriched['priors_provenance']['history_window_limit'] = self.history_limit
        return enriched
```

```python
# core/aligned_features.py
from core.features.pipeline import DraftFeaturePipeline
from core.features.matchup_matrix import LaneMatchupTracker
from core.serving_contract import FeatureContract

def extract_aligned_inputs(match, contract_version, *, macro=True, pipeline=None):
    actual = pipeline if pipeline is not None else DraftFeaturePipeline(
        matchup_tracker=LaneMatchupTracker())
    return FeatureContract(contract_version).extract(match, actual, macro=macro)
```

Dans `PredictionService.predict_match`, remplacer seulement la ligne d'extraction par :

```python
x_nn = extract_aligned_inputs(
    match_dict, self.contract.version,
    pipeline=self.feature_pipeline, macro='macro_clash' in input_names)
```

Ajouter l'import correspondant et `core/aligned_features.py` à `FEATURE_SOURCE_FILES`. Le provider figé ne fait pas partie du runtime HTTP, mais son fichier devra être hashé dans le manifeste du dataset.

- [x] **Step 4: Vérifier GREEN** avec `tests/test_frozen_history.py`, `tests/test_serving_contract.py` et `tests/test_pre_match_history_serving.py`. Attendu : priors stricts, dimensions exactes et aucun changement de probabilité causé par l'extraction commune.

- [ ] **Step 5: Revoir le diff et isoler le commit** `feat: share strict serving feature extraction with frozen replay`. Chemins : les cinq fichiers de cette tâche et `tests/aligned_dataset_fixture.py`. Pour les deux fichiers existants utiliser `git add -p`; examiner `git diff --cached` avant le commit.

### Task 2: Population cible et partitions purgées

**Files:** Create `core/aligned_partitions.py`, `tests/test_aligned_partitions.py`; Modify `core/pre_match_history.py:63` (`read_recent_rows`).

**Interfaces:**
- Extend `read_recent_rows(db_path, limit: int, *, before_ms: int | None = None, timelines: bool = True, puuids=None, include_labels: bool = False) -> list[dict]`.
- Produce `select_targets(rows: list[dict], *, excluded_ids: set[str], warmup_rows: int = 10000) -> tuple[list[dict], list[dict]]` ; tuples = cibles validées et exclusions `{match_id, reason}`.
- Produce `partition_targets(targets: list[dict], *, after_ms: int, validation_count: int = 3000, test_count: int = 3000) -> tuple[dict[str, list[dict]], list[dict]]` ; deuxième liste = purges.
- Target record fields : `match_id`, `game_creation_time`, `result_end_ms`, `feature_json`, `labels_json`, `queue_id`, `patch`, `region`.

- [x] **Step 1: Écrire les tests RED.** Les petits comptes sont exclusivement des fixtures ; la CLI de préparation réelle conserve les minima 3 000/3 000.

```python
import pytest
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

def test_bad_current_label_rejected_but_not_removed_from_history():
    row = source_row('NA1_1', 10, 20)
    row['winner_label'] = 1
    selected, rejected = select_targets([row], excluded_ids=set(), warmup_rows=0)
    assert selected == []
    assert rejected[0]['reason'] == 'winner_disagrees_with_riot'
    assert row['raw_json']

def test_ties_and_result_purge_at_boundaries():
    rows = [source_row(f'NA1_{i}', t, e) for i, (t, e) in enumerate(
        [(10, 19), (20, 31), (30, 39), (30, 39), (40, 51), (50, 59), (60, 69)])]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=0)
    parts, purged = partition_targets(targets, after_ms=40,
                                     validation_count=2, test_count=2)
    assert [r['game_creation_time'] for r in parts['train']] == [10]
    assert [r['game_creation_time'] for r in parts['validation']] == [30, 30]
    assert [r['game_creation_time'] for r in parts['test']] == [50, 60]
    assert {r['game_creation_time'] for r in purged} == {20, 40}

def test_insufficient_new_test_is_an_error():
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19) for i in range(4)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=0)
    with pytest.raises(ValueError, match='Insufficient'):
        partition_targets(targets, after_ms=35, validation_count=1, test_count=2)

def test_previous_test_id_excluded_and_unknown_queue_rejected():
    import json
    known = source_row('NA1_seen', 10, 20)
    unknown = source_row('NA1_unknown', 30, 40)
    raw = json.loads(unknown['raw_json']); raw['info'].pop('queueId')
    unknown['raw_json'] = json.dumps(raw)
    selected, errors = select_targets([known, unknown],
        excluded_ids={'NA1_seen'}, warmup_rows=0)
    assert selected == []
    assert {e['reason'] for e in errors} == {'previous_test', 'unsupported_queue'}
```

- [x] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_partitions.py -q`. Attendu : module absent.

- [x] **Step 3: Implémenter la sélection stricte et les coupures.** Réutiliser la validation des labels existante ; ne créer aucun défaut pour un label absent.

```python
# core/aligned_partitions.py
from core.pre_match_dataset import _decode, _valid_labels
from core.pre_match_history import decode_history_row
from core.serving_contract import ContractError, FeatureContract

def select_targets(rows, *, excluded_ids, warmup_rows=10000):
    if type(warmup_rows) is not int or warmup_rows < 0:
        raise ValueError('warmup_rows must be nonnegative')
    selected, rejected, seen = [], [], set()
    ordered = sorted(rows, key=lambda r: (r['game_creation_time'], r['match_id']))
    for index, row in enumerate(ordered):
        mid = row['match_id']
        if mid in seen:
            raise ValueError(f'duplicate match_id: {mid}')
        seen.add(mid)
        if index < warmup_rows:
            continue
        try:
            if mid in excluded_ids:
                raise ValueError('previous_test')
            rec = decode_history_row(row)
            info = rec['raw_info']
            if info.get('queueId') not in (420, 440):
                raise ValueError('unsupported_queue')
            if rec['result_end_ms'] is None:
                raise ValueError('invalid_result_end')
            FeatureContract('nnriot.players14.v1').validate_match(rec['feature_json'])
            parts = rec['participants']
            blue = [p for p in parts if p.get('teamId') == 100]
            red = [p for p in parts if p.get('teamId') == 200]
            if len(blue) != 5 or len(red) != 5 or any(
                    type(p.get('win')) is not bool for p in parts):
                raise ValueError('invalid_riot_winner')
            if len({p['win'] for p in blue}) != 1 or len({p['win'] for p in red}) != 1 or blue[0]['win'] == red[0]['win']:
                raise ValueError('invalid_riot_winner')
            winner = row['winner_label']
            if type(winner) is not int or winner != int(red[0]['win']):
                raise ValueError('winner_disagrees_with_riot')
            labels = _decode(row.get('labels_json'))
            if not _valid_labels(labels, winner):
                raise ValueError('invalid_multitask_labels')
            selected.append(dict(match_id=mid, game_creation_time=rec['game_creation_time'],
                result_end_ms=rec['result_end_ms'], feature_json=rec['feature_json'],
                labels_json=labels, queue_id=info['queueId'], patch=rec['patch'],
                region=rec['feature_json'].get('region')))
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(dict(match_id=mid, reason=str(exc)))
    return selected, rejected

def partition_targets(targets, *, after_ms, validation_count=3000, test_count=3000):
    if any(type(n) is not int or n < 1 for n in (validation_count, test_count)):
        raise ValueError('partition counts must be positive integers')
    if type(after_ms) is not int or after_ms < 0:
        raise ValueError('after_ms must be nonnegative UTC milliseconds')
    ordered = sorted(targets, key=lambda r: (r['game_creation_time'], r['match_id']))
    new = [r for r in ordered if r['game_creation_time'] > after_ms]
    if len(new) < test_count:
        raise ValueError('Insufficient new test targets')
    test_start = new[-test_count]['game_creation_time']
    before_test = [r for r in ordered if r['game_creation_time'] < test_start]
    if len(before_test) <= validation_count:
        raise ValueError('Insufficient train/validation targets')
    available_validation = [r for r in before_test if r['result_end_ms'] < test_start]
    if len(available_validation) <= validation_count:
        raise ValueError('Insufficient completed validation targets')
    validation_start = available_validation[-validation_count]['game_creation_time']
    splits = {'train': [], 'validation': [], 'test': []}
    purged = []
    for row in ordered:
        start = row['game_creation_time']
        name = 'test' if start >= test_start else 'validation' if start >= validation_start else 'train'
        cutoff = validation_start if name == 'train' else test_start if name == 'validation' else None
        if cutoff is not None and row['result_end_ms'] >= cutoff:
            purged.append(dict(match_id=row['match_id'], game_creation_time=start,
                               reason='result_not_available_at_partition_start'))
        else:
            splits[name].append(row)
    if not splits['train'] or len(splits['validation']) < validation_count or len(splits['test']) < test_count:
        raise ValueError('Insufficient targets after temporal purge')
    return splits, purged
```

Dans `read_recent_rows`, ajouter `include_labels=False`, garder `queueId` dans la liste des champs `small`, et choisir l'expression SQL des labels sans casser les anciennes fixtures :

```python
training_columns = {r[1] for r in conn.execute('PRAGMA table_info(training_dataset)')}
labels_expr = 't.labels_json' if include_labels and 'labels_json' in training_columns else 'NULL'
```

Ajouter `{labels_expr} AS labels_json` aux deux SELECT existants. Les données restent compactées par le même curseur SQLite. Dans le chemin live, la valeur par défaut ne charge aucun label multitâche.

- [x] **Step 4: Vérifier GREEN** sur `tests/test_aligned_partitions.py`, `tests/test_pre_match_history_serving.py`, `tests/test_point_in_time_dataset.py`. Inclure ce test SQLite de compactage dans la suite de partitions :

```python
def test_snapshot_labels_are_opt_in_and_queue_survives(tmp_path):
    import json
    import sqlite3
    from core.pre_match_history import read_recent_rows
    db = tmp_path / 'snapshot.db'
    row = source_row('NA1_1', 10, 20)
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER,
                                 game_mode TEXT, raw_json TEXT);
            CREATE TABLE training_dataset(match_id TEXT, winner_label INTEGER);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);
        ''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     (row['match_id'], 10, 'CLASSIC', row['raw_json']))
        conn.execute('INSERT INTO training_dataset VALUES(?,?)', (row['match_id'], 0))
    assert read_recent_rows(db, 10, include_labels=True)[0]['labels_json'] is None
    with sqlite3.connect(db) as conn:
        conn.execute('ALTER TABLE training_dataset ADD COLUMN labels_json TEXT')
        conn.execute('UPDATE training_dataset SET labels_json=?', (row['labels_json'],))
    assert read_recent_rows(db, 10)[0]['labels_json'] is None
    stored = read_recent_rows(db, 10, include_labels=True)[0]
    assert stored['labels_json'] == row['labels_json']
    assert json.loads(stored['raw_json'])['info']['queueId'] == 420
```

- [ ] **Step 5: Revoir et isoler le commit** `feat: freeze ranked targets with result-availability purges`, avec `core/aligned_partitions.py`, `tests/test_aligned_partitions.py` et seulement les nouveaux hunks de `core/pre_match_history.py`.

### Task 3: Artefacts figés, disponibilité 18D et CLI de préparation

**Files:** Create `core/aligned_artifacts.py`, `scripts/prepare_aligned_dataset.py`, `tests/test_aligned_artifacts.py`.

**Interfaces:**
- Consumes les interfaces des tâches 1–2 et `build_manifest(splits, *, source_db, split_metadata) -> dict`, `runtime_feature_sources() -> dict`, `sha256_file(path) -> str`.
- Produces `write_dataset(output: Path, rows: list[dict], splits: dict, *, source_db: str, selection: dict, exclusions: list[dict]) -> dict`.
- Produces `read_train_validation(dataset_dir: Path, contract_version: str) -> tuple[dict, dict, dict]` ; retour = tenseurs `{train, validation}`, records `{train, validation}`, manifeste du dataset.
- Produces `inspect_sources(paths: list[Path]) -> list[dict]`, `prepare(db: Path, output: Path, after_cohort: Path, *, source_limit: int = 61000) -> dict`, `verify(dataset_dir: Path) -> dict` et `main(argv=None) -> int`.

- [x] **Step 1: Écrire les tests RED d'artefacts.**

```python
import json
import numpy as np
import pytest
from pathlib import Path
from core.aligned_artifacts import write_dataset, read_train_validation
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

def tiny_bundle(tmp_path, *, timeline=False):
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19, i % 2,
                       timeline=timeline) for i in range(10)]
    targets, excluded = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, purged = partition_targets(targets, after_ms=60,
                                       validation_count=2, test_count=2)
    directory = tmp_path / 'dataset'
    manifest = write_dataset(directory, rows, splits, source_db='fixture.db',
        selection={'history_window_limit': 10000}, exclusions=excluded + purged)
    return directory, manifest

def test_missing_timelines_are_reported_and_18d_refused(tmp_path):
    directory, meta = tiny_bundle(tmp_path)
    x, records, loaded = read_train_validation(directory, 'nnriot.players14.v1')
    assert set(x) == set(records) == {'train', 'validation'}
    assert meta['availability']['nnriot.players18.v1']['status'] == 'blocked'
    with pytest.raises(ValueError, match='unavailable'):
        read_train_validation(directory, 'nnriot.players18.v1')

def test_exact_shared_membership_and_changed_tensor_rejected(tmp_path):
    directory, meta = tiny_bundle(tmp_path, timeline=True)
    for version in ('nnriot.players14.v1', 'nnriot.players18.v1'):
        x, records, _ = read_train_validation(directory, version)
        assert len(x['train']['team_a']) == len(records['train'])
    tensor = directory / 'train14.npz'
    tensor.write_bytes(tensor.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        read_train_validation(directory, 'nnriot.players14.v1')

def test_manifest_path_cannot_escape_dataset(tmp_path):
    directory, meta = tiny_bundle(tmp_path)
    meta['files']['train14.npz']['path'] = '../foreign.npz'
    (directory / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='outside dataset'):
        read_train_validation(directory, 'nnriot.players14.v1')

def test_occupied_output_is_preserved(tmp_path):
    directory, _ = tiny_bundle(tmp_path)
    sentinel = directory / 'keep.txt'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        write_dataset(directory, [], {}, source_db='unused', selection={}, exclusions=[])
    assert sentinel.read_text() == 'keep'
```

- [x] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_artifacts.py -q`. Attendu : module absent.

- [x] **Step 3: Implémenter le format et les contrôles.** Fichiers exacts : `source_rows.json.gz`, `model.split.json`, `train.json`, `validation.json`, `sealed_test.json`, `train14.npz`, `validation14.npz`, `sealed_test14.npz`, les trois fichiers `18.npz` si le contrat est disponible partout, `exclusions.json`, `availability.json`, `dataset.json`. La compression de source utilise `gzip.compress(..., mtime=0)`. `dataset.json` contient `schema_version: 1`, `kind: nnriot.serving_aligned_dataset.v1`, hashes des fichiers, hashes de runtime et availability. L'empreinte du manifeste lui-même est calculée par ses consommateurs ; ne pas créer de hash circulaire.

```python
# core/aligned_artifacts.py: noyau d'écriture
import gzip
import json
from pathlib import Path
import numpy as np
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from core.calibration import sha256_file
from core.experiment import build_manifest
from core.serving_contract import ContractError, runtime_feature_sources
from core.config import PROJECT_ROOT

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False, ensure_ascii=False) + '\n').encode('utf-8')

def dataset_runtime_sources():
    names = ('core/frozen_history.py', 'core/aligned_partitions.py',
             'core/aligned_artifacts.py', 'core/serving_contract.py',
             'scripts/prepare_aligned_dataset.py')
    return {**runtime_feature_sources(), **{
        name: sha256_file(PROJECT_ROOT / name) for name in names}}

def write_dataset(output, rows, splits, *, source_db, selection, exclusions):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provider = FrozenHistoryProvider(rows)
    files, availability = {}, {}
    def register(name):
        files[name] = {'path': name, 'sha256': sha256_file(output / name)}
    def save_json(name, value):
        with (output / name).open('xb') as handle:
            handle.write(canonical(value))
        register(name)
    (output / 'source_rows.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    register('source_rows.json.gz')
    for partition in ('train', 'validation', 'test'):
        prefix = 'sealed_test' if partition == 'test' else partition
        save_json(prefix + '.json', splits[partition])
    for dim in (14, 18):
        version = f'nnriot.players{dim}.v1'
        arrays, errors = {}, []
        total = sum(len(part) for part in splits.values())
        for partition, records in splits.items():
            extracted = []
            for record in records:
                try:
                    enriched = provider.enrich(record['feature_json'], include_timeline=dim == 18)
                    extracted.append(extract_aligned_inputs(enriched, version))
                except ContractError as exc:
                    errors.append({'match_id': record['match_id'], 'partition': partition,
                                   'reason': type(exc).__name__ + ': ' + str(exc)})
            if extracted:
                arrays[partition] = {name: np.concatenate([x[name] for x in extracted])
                                     for name in ('team_a', 'team_b', 'macro_clash')}
        availability[version] = {'status': 'blocked' if errors else 'ready',
            'eligible': total - len(errors), 'total': total,
            'coverage': (total - len(errors)) / total, 'errors': errors}
        if not errors:
            for partition, x in arrays.items():
                prefix = 'sealed_test' if partition == 'test' else partition
                name = f'{prefix}{dim}.npz'
                np.savez(output / name, **x)
                register(name)
    if availability['nnriot.players14.v1']['status'] != 'ready':
        save_json('availability.json', availability)
        raise ValueError('14D feature materialization unavailable; inspect availability.json')
    split_manifest = build_manifest(splits, source_db=source_db, split_metadata=selection)
    save_json('model.split.json', split_manifest)
    save_json('exclusions.json', exclusions)
    save_json('availability.json', availability)
    manifest = dict(schema_version=1, kind='nnriot.serving_aligned_dataset.v1',
        source_db=str(Path(source_db).resolve()), selection=selection,
        files=files, runtime_sources=dataset_runtime_sources(), availability=availability,
        split_sha256=split_manifest['split_sha256'],
        dataset_sha256=split_manifest['dataset_sha256'], test_status='reserved',
        availability_assumption='Riot results observable at result end; ingestion timestamps unknown')
    (output / 'dataset.json').write_bytes(canonical(manifest))
    return manifest

def checked_path(root, manifest, name):
    path = (root / manifest['files'][name]['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path outside dataset')
    if sha256_file(path) != manifest['files'][name]['sha256']:
        raise ValueError(f'SHA-256 mismatch: {name}')
    return path

def read_train_validation(dataset_dir, contract_version):
    root = Path(dataset_dir).resolve()
    manifest = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'nnriot.serving_aligned_dataset.v1':
        raise ValueError('unsupported aligned dataset')
    if manifest['runtime_sources'] != dataset_runtime_sources():
        raise ValueError('feature runtime SHA-256 mismatch')
    if manifest['availability'].get(contract_version, {}).get('status') != 'ready':
        raise ValueError(f'{contract_version} unavailable')
    dim = {'nnriot.players14.v1': 14, 'nnriot.players18.v1': 18}[contract_version]
    records, inputs = {}, {}
    for name in ('train', 'validation'):
        records[name] = json.loads(checked_path(root, manifest, name + '.json').read_text(encoding='utf-8'))
        with np.load(checked_path(root, manifest, f'{name}{dim}.npz'), allow_pickle=False) as data:
            inputs[name] = {key: data[key].copy() for key in data.files}
        expected = {'team_a': (len(records[name]), 5, dim),
                    'team_b': (len(records[name]), 5, dim),
                    'macro_clash': (len(records[name]), 15)}
        if set(inputs[name]) != set(expected) or any(
                inputs[name][key].shape != shape or inputs[name][key].dtype != np.float32
                or not np.isfinite(inputs[name][key]).all() for key, shape in expected.items()):
            raise ValueError(f'{name}: invalid tensor contract')
    return inputs, records, manifest
```

Le writer doit valider avant matérialisation que les trois partitions sont non vides et triées, contiennent des IDs uniques présents dans `rows`, et possèdent le cutoff exact dans `feature_json`. Appeler `build_manifest` avant les boucles pour bénéficier de sa validation structurale ; calculer les comptes seulement après cette validation. Ajouter le test suivant et le contrôle correspondant :

```python
def test_target_cutoff_must_match_row_creation(tmp_path):
    rows = [source_row(f'NA1_{i}', 10*i+10, 10*i+19) for i in range(10)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, _ = partition_targets(targets, after_ms=60, validation_count=2, test_count=2)
    splits['train'][0]['feature_json']['prediction_time_ms'] += 1
    with pytest.raises(ValueError, match='prediction cutoff'):
        write_dataset(tmp_path / 'bad', rows, splits, source_db='fixture', selection={}, exclusions=[])
```

```python
source_ids = {row['match_id'] for row in rows}
for partition in splits.values():
    for record in partition:
        if record['match_id'] not in source_ids:
            raise ValueError('target absent from source snapshot')
        if record['feature_json'].get('prediction_time_ms') != record['game_creation_time']:
            raise ValueError('prediction cutoff mismatch')
```

CLI : sous-commandes `inventory --db <path>` répétable, `freeze --db <path> --after-cohort <path> --output-dir <path> --source-limit 61000`, `verify --dataset-dir <path>`. `inventory` exécute uniquement les mêmes requêtes de comptage et de bornes que l'inventaire de la spec. La sélection `freeze` suit ce code ; aucun shuffle ni choix après lecture de métriques :

```python
def prepare(db, output, after_cohort, *, source_limit=61000):
    import json
    from core.pre_match_history import read_recent_rows
    from core.aligned_partitions import select_targets, partition_targets
    from core.aligned_artifacts import write_dataset
    from core.calibration import sha256_file
    if Path(output).exists():
        raise FileExistsError(f'occupied output directory: {output}')
    cohort = [json.loads(line) for line in Path(after_cohort).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if not cohort or any(type(r.get('game_creation_time')) is not int
            or r['game_creation_time'] <= 0
            or r.get('partition') not in ('history', 'validation', 'test')
            or not isinstance(r.get('match_id'), str) or not r['match_id'] for r in cohort):
        raise ValueError('invalid previous cohort')
    if len({r['match_id'] for r in cohort}) != len(cohort):
        raise ValueError('duplicate match_id in previous cohort')
    if not any(r['partition'] == 'test' for r in cohort):
        raise ValueError('previous cohort has no test partition')
    after_ms = max(row['game_creation_time'] for row in cohort)
    excluded_ids = {row['match_id'] for row in cohort if row['partition'] == 'test'}
    rows = read_recent_rows(db, source_limit, timelines=True, include_labels=True)
    targets, exclusions = select_targets(rows, excluded_ids=excluded_ids)
    splits, purged = partition_targets(targets, after_ms=after_ms)
    selection = dict(target_queues=[420, 440], mode='CLASSIC', source_limit=source_limit,
        warmup_rows=10000, history_window_limit=10000, after_ms=after_ms,
        validation_minimum=3000, test_minimum=3000,
        after_cohort_sha256=sha256_file(after_cohort), excluded_test_ids=sorted(excluded_ids),
        history_policy='result_end_strictly_before_start.v1',
        test_scope='reserved for this future experiment; prior research limitations declared')
    return write_dataset(output, rows, splits, source_db=str(db),
                         selection=selection, exclusions=exclusions + purged)
```

Le cohort existant est `cohort.jsonl`, une ligne JSON par match. Le hash du fichier et les IDs exclus sont figés dans `selection`. Inclure le test suivant dans `tests/test_aligned_artifacts.py` :

```python
@pytest.mark.parametrize('rows', [[], [{'match_id': 'NA1_1',
    'game_creation_time': True, 'partition': 'test'}]])
def test_invalid_previous_cohort_fails_before_database_read(tmp_path, rows):
    from scripts.prepare_aligned_dataset import prepare
    cohort = tmp_path / 'cohort.jsonl'
    cohort.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='previous cohort'):
        prepare(tmp_path / 'absent.db', tmp_path / 'output', cohort)
    assert not (tmp_path / 'output').exists()
```

`main` utilise les sous-commandes suivantes et retourne une cause précise. Ne pas importer TensorFlow pour cette CLI. Le bloc complet inclut `verify`, à enregistrer seulement lorsque la tâche 5 livre son implémentation.

```python
# scripts/prepare_aligned_dataset.py : imports et inventaire
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

def inspect_sources(paths):
    results = []
    for path in paths:
        path = Path(path).resolve()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.execute('PRAGMA query_only=ON')
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            count, first, last = conn.execute(
                'SELECT COUNT(*), MIN(game_creation_time), MAX(game_creation_time) FROM matches').fetchone()
            timelines = conn.execute('SELECT COUNT(*) FROM match_timelines').fetchone()[0] if 'match_timelines' in tables else 0
            results.append(dict(path=str(path), bytes=path.stat().st_size,
                matches=count, first_creation_ms=first, last_creation_ms=last,
                timeline_rows=timelines))
    return results

def main(argv=None):
    from core.serving_contract import ContractError
    parser = argparse.ArgumentParser(description='Prepare and audit serving-aligned datasets')
    commands = parser.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory')
    inventory.add_argument('--db', type=Path, action='append', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--db', type=Path, required=True)
    freeze.add_argument('--after-cohort', type=Path, required=True)
    freeze.add_argument('--output-dir', type=Path, required=True)
    freeze.add_argument('--source-limit', type=int, default=61000)
    replay = commands.add_parser('verify')
    replay.add_argument('--dataset-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inventory':
            result = inspect_sources(args.db)
        elif args.command == 'freeze':
            result = prepare(args.db, args.output_dir, args.after_cohort,
                             source_limit=args.source_limit)
        else:
            result = verify(args.dataset_dir)
        print(json.dumps(result, allow_nan=False, ensure_ascii=False))
        return 0
    except (ValueError, FileExistsError, FileNotFoundError, ContractError, sqlite3.Error) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
```

- [x] **Step 4: Vérifier GREEN** sur `tests/test_aligned_artifacts.py` et `tests/test_aligned_partitions.py`. Vérifier également `--help` et l'inventaire des trois bases de la spec, en conservant sa sortie JSON. Aucun freeze réel à cette étape.

- [ ] **Step 5: Revoir et isoler le commit** `feat: persist hash-bound aligned datasets and feature availability`, limité aux trois nouveaux fichiers de cette tâche.

### Task 4: Mode dataset aligné des trainers, calibration stricte et test réservé

**Files:** Create `core/aligned_training.py`, `tests/test_aligned_training.py`; Modify `scripts/train_deepsets_v3.py:44`, `scripts/train_deepsets_v4_timeline.py:43` et leurs blocs CLI.

**Interfaces:**
- Consumes `read_train_validation(dataset_dir, contract_version) -> (inputs, records, manifest)` et `checked_path(root, manifest, name) -> Path`.
- Produces `prepare_aligned_training(dataset_dir: Path, output_dir: Path, contract_version: str) -> tuple[dict, dict, dict]`.
- Produces `finish_aligned_pilot(*, output_dir: Path, dataset_dir: Path, dataset_manifest: dict, input_dim: int, validation_probabilities: np.ndarray, validation_labels: np.ndarray, history: list[dict], best_epoch: int, best_val_loss: float) -> dict`.
- Extend les deux fonctions train avec `dataset_dir: str | None = None` keyword-only ; CLI `--dataset-dir`.

- [x] **Step 1: Écrire les tests RED sur les consommateurs réels.**

```python
import json
from pathlib import Path
import pytest
from tests.test_aligned_artifacts import tiny_bundle

@pytest.mark.parametrize('name', ['train_deepsets_v3', 'train_deepsets_v4_timeline'])
def test_aligned_training_does_not_read_legacy_or_test(tmp_path, monkeypatch, name):
    import importlib
    trainer = importlib.import_module('scripts.' + name)
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name.startswith('sealed_test') or path.name == 'source_rows.json.gz':
            raise AssertionError('training attempted to open reserved outcomes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    def legacy_forbidden(*args, **kwargs):
        raise AssertionError('legacy data path used')
    monkeypatch.setattr(trainer, 'load_enriched_records', legacy_forbidden)
    if name == 'train_deepsets_v3':
        monkeypatch.setattr(trainer, 'prepare_ordered_training', legacy_forbidden)
    class FirstBatchReached(Exception):
        pass
    def stop(*args, **kwargs):
        raise FirstBatchReached
    monkeypatch.setattr(trainer, 'pcgrad_step', stop)
    with pytest.raises(FirstBatchReached):
        getattr(trainer, name)('unused.db', str(tmp_path / name),
                              dataset_dir=str(directory), epochs=1)

def test_aligned_and_legacy_cache_are_mutually_exclusive(tmp_path):
    from scripts.train_deepsets_v3 import train_deepsets_v3
    with pytest.raises(ValueError, match='exclusive'):
        train_deepsets_v3('unused.db', str(tmp_path / 'out'),
            dataset_dir='frozen', splits_cache='old.pkl')
    assert not (tmp_path / 'out').exists()
```

Inclure les tests de clôture suivants, pour validation insuffisante et validation à deux classes avec sidecar strict :

```python
import numpy as np
from core.aligned_training import finish_aligned_pilot, prepare_aligned_training
from core.calibration import load_calibration_artifact
from core.aligned_artifacts import write_dataset
from core.aligned_partitions import select_targets, partition_targets
from tests.aligned_dataset_fixture import source_row

@pytest.mark.parametrize('labels', [np.array([0,1,0,1]), np.zeros(20, dtype=int)])
def test_insufficient_calibration_has_no_fallback_or_test(tmp_path, labels):
    directory, meta = tiny_bundle(tmp_path)
    output = tmp_path / 'pilot'; output.mkdir()
    (output / 'model.keras').write_bytes(b'fixture-checkpoint')
    result = finish_aligned_pilot(output_dir=output, dataset_dir=directory,
        dataset_manifest=meta, input_dim=14, validation_probabilities=np.full(len(labels), .5),
        validation_labels=labels, history=[], best_epoch=1, best_val_loss=.7)
    assert result['calibration_status'] == 'insufficient_validation'
    assert result['calibrated'] is False
    assert result['test_status'] == 'reserved'
    assert 'test_raw' not in result and 'test_calibrated' not in result
    assert not (output / 'model.calibration.json').exists()

def test_completed_pilot_calibration_binds_dataset_manifest(tmp_path, monkeypatch):
    rows = [source_row(f'NA1_{i:03d}', 10*i+10, 10*i+19, i % 2) for i in range(70)]
    targets, _ = select_targets(rows, excluded_ids=set(), warmup_rows=1)
    splits, purged = partition_targets(targets, after_ms=400,
                                      validation_count=20, test_count=20)
    directory = tmp_path / 'dataset'
    meta = write_dataset(directory, rows, splits, source_db='fixture',
                         selection={}, exclusions=purged)
    output = tmp_path / 'pilot'
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name.startswith('sealed_test') or path.name == 'source_rows.json.gz':
            raise AssertionError('pilot accessed reserved outcomes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    _, records, loaded = prepare_aligned_training(directory, output, 'nnriot.players14.v1')
    (output / 'model.keras').write_bytes(b'fixture-checkpoint')
    labels = np.array([r['labels_json']['winner'] for r in records['validation']])
    result = finish_aligned_pilot(output_dir=output, dataset_dir=directory,
        dataset_manifest=loaded, input_dim=14, validation_probabilities=np.full(len(labels), .55),
        validation_labels=labels, history=[], best_epoch=1, best_val_loss=.7)
    assert result['calibrated'] is True
    assert load_calibration_artifact(output / 'model.keras', input_dim=14) is not None
```

- [x] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_training.py -q`. Attendu : argument `dataset_dir` inconnu et nouveau module absent.

- [x] **Step 3: Implémenter la préparation sans labels test et la clôture.** Copier le manifeste d'IDs ne nécessite pas la lecture des labels test. Le manifeste de calibration reste un compagnon exact `model.split.json`, vérifié par `save_calibration_artifact`.

```python
# core/aligned_training.py
import json
from pathlib import Path
import numpy as np
from core.aligned_artifacts import read_train_validation, checked_path, canonical
from core.calibration import fit_temperature, save_calibration_artifact, sha256_file

def prepare_aligned_training(dataset_dir, output_dir, contract_version):
    root, output = Path(dataset_dir), Path(output_dir)
    x, records, manifest = read_train_validation(root, contract_version)
    split_source = checked_path(root, manifest, 'model.split.json')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'model.split.json').write_bytes(split_source.read_bytes())
    return x, records, manifest

def finish_aligned_pilot(*, output_dir, dataset_dir, dataset_manifest, input_dim,
                         validation_probabilities, validation_labels, history,
                         best_epoch, best_val_loss):
    output = Path(output_dir)
    checkpoint = output / 'model.keras'
    probs = np.asarray(validation_probabilities, dtype=float)
    labels = np.asarray(validation_labels, dtype=int)
    if len(labels) < 20 or len(np.unique(labels)) != 2:
        fit = {'status': 'insufficient_validation', 'partition': 'validation',
               'count': len(labels)}
    else:
        fit = fit_temperature(probs, labels)
    calibrated = False
    if fit.get('status') == 'fitted' and fit.get('optimizer_success') is True:
        save_calibration_artifact(checkpoint_path=checkpoint,
            split_manifest_path=output / 'model.split.json', fit=fit, input_dim=input_dim)
        calibrated = True
    metrics = dict(player_feature_dim=input_dim, best_epoch=best_epoch,
        best_val_log_loss=float(best_val_loss), history=history,
        calibration_status=fit.get('status'), calibrated=calibrated,
        test_status='reserved', validation_status='unverified',
        dataset_manifest_sha256=sha256_file(Path(dataset_dir) / 'dataset.json'),
        checkpoint_sha256=sha256_file(checkpoint),
        split_sha256=dataset_manifest['split_sha256'])
    (output / 'model.metrics.json').write_bytes(canonical(metrics))
    return metrics
```

`fit_temperature` exige actuellement au moins 20 labels et deux classes, puis retourne `status='fitted'` avec `optimizer_success=True`. Le contrôle ci-dessus correspond à cette API vérifiée. Ne pas transformer une erreur technique de calibration en un artefact de température `1.0` prétendument calibré. Le smoke test de cette tâche utilise un modèle neuf ; aucune métrique test n'est créée.

Dans chaque trainer, sélectionner les données **avant** de créer le répertoire de sortie ; conserver la boucle actuelle d'entraînement. En mode aligné :

```python
# Entrée v3, avant cache, manifeste legacy et résolution des fins.
if dataset_dir is not None:
    if any(v is not None for v in (manifest_path, splits_cache, result_end_times)):
        raise ValueError('dataset_dir is exclusive with legacy inputs')
    x, splits, aligned_manifest = prepare_aligned_training(
        Path(dataset_dir), out_path, 'nnriot.players14.v1')
    active_partitions = ('train', 'validation')
else:
    active_partitions = ('train', 'validation', 'test')
```

```python
# Entrée v4, avant load_enriched_records et la création du répertoire.
if dataset_dir is not None:
    if manifest_path is not None:
        raise ValueError('dataset_dir is exclusive with legacy manifest')
    x, splits, aligned_manifest = prepare_aligned_training(
        Path(dataset_dir), out_path, 'nnriot.players18.v1')
    active_partitions = ('train', 'validation')
else:
    active_partitions = ('train', 'validation', 'test')
```

Placer les blocs legacy existants dans la branche `else` sans changer leurs calculs. Remplacer les comprehensions/itérations fixes des targets par `active_partitions`. v3 conserve son `_build_targets(splits['train'])` ; v4 utilise les deux partitions disponibles. Juste après le rechargement du meilleur checkpoint et la prédiction sur validation, avant l'ancienne calibration/test :

```python
if dataset_dir is not None:
    return finish_aligned_pilot(
        output_dir=out_path, dataset_dir=Path(dataset_dir),
        dataset_manifest=aligned_manifest, input_dim=14,  # 18 dans v4
        validation_probabilities=raw_val_probs,
        validation_labels=y['validation'], history=history,
        best_epoch=best_epoch, best_val_loss=best_val_loss)
```

L'entrée CLI v3 possède actuellement des défauts de cache et manifeste legacy. Pour `--dataset-dir`, les désactiver lorsqu'ils n'ont pas été explicitement fournis ; utiliser `default=None` au parseur et résoudre les anciens défauts uniquement dans la branche sans dataset. Le choix explicite de deux sources doit échouer. v4 doit respecter son hook `pcgrad_step` de test comme v3, sans changer le chemin compilé normal :

```python
if pcgrad_step is not _ORIGINAL_PCGRAD_STEP:
    losses = pcgrad_step(model, model.optimizer, xb, yb, loss_fns, loss_weights=LOSS_WEIGHTS)
    batch_loss = sum(losses.values())
else:
    step_losses = compiled_step(xb, yb)
    batch_loss = float(tf.add_n(list(step_losses.values())).numpy())
```

- [x] **Step 4: Vérifier GREEN** avec `tests/test_aligned_training.py`, `tests/test_ordered_training_integration.py`, `tests/test_train_deepsets_v4_timeline.py`. Exécuter les tests de clôture ci-dessus, dont le garde interdit explicitement l'ouverture des faits source et des fichiers `sealed_test*`.

- [ ] **Step 5: Revoir et isoler le commit** `feat: train aligned v3 and v4 pilots without opening reserved test`, avec les deux nouveaux fichiers et les seuls hunks nécessaires des deux trainers.

### Task 5: Replay intégral et preuve de parité avec SQLite

**Files:** Create `tests/test_aligned_parity.py`; Modify `scripts/prepare_aligned_dataset.py` pour `verify` et `core/aligned_artifacts.py` pour les contrôles complémentaires.

**Interfaces:**
- Consumes `checked_path`, `FrozenHistoryProvider`, `extract_aligned_inputs` et les fichiers figés de tâche 3.
- Produces `verify(dataset_dir: Path) -> dict` avec `status: verified`, hashes et nombre de comparaisons exactes ; ne calcule aucune métrique de victoire.
- Vérification de service : `DatabaseHistoryProvider(db_path, history_limit=10000).enrich(match, *, include_timeline: bool) -> dict` reste inchangée.

- [x] **Step 1: Écrire les tests RED de replay et de parité.**

```python
import copy
import numpy as np
import pytest
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from core.serving_contract import FeatureUnavailableError
from scripts.prepare_aligned_dataset import verify
from tests.test_aligned_artifacts import tiny_bundle
from tests.aligned_dataset_fixture import source_row, target

def test_frozen_replay_is_exact(tmp_path):
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    report = verify(directory)
    assert report['status'] == 'verified'
    assert report['tensor_comparisons'] > 0

def test_permutation_swap_and_target_outcomes_do_not_change_features():
    history = [source_row('NA1_1', 10, 20, timeline=True)]
    match = target(source_row('NA1_2', 30, 130))
    provider = FrozenHistoryProvider(history)
    for dim in (14, 18):
        version = f'nnriot.players{dim}.v1'
        original = provider.enrich(match, include_timeline=dim == 18)
        expected = extract_aligned_inputs(original, version)
        permuted = copy.deepcopy(original)
        for team in ('team_a', 'team_b'):
            permuted[team].reverse()
            for p in permuted[team]:
                p.update(kills=999999, win=False, goldEarned=999999)
        got = extract_aligned_inputs(permuted, version)
        for key in expected:
            assert np.array_equal(expected[key], got[key])
        swapped = dict(original, team_a=original['team_b'], team_b=original['team_a'])
        reverse = extract_aligned_inputs(swapped, version)
        assert np.array_equal(expected['team_a'], reverse['team_b'])
        assert np.array_equal(expected['team_b'], reverse['team_a'])
        assert np.array_equal(expected['macro_clash'], -reverse['macro_clash'])

def test_no_timeline_outside_global_window_can_rescue_18d():
    old = source_row('NA1_1', 10, 20, timeline=True)
    recent = source_row('NA1_2', 30, 40, timeline=False)
    match = target(source_row('NA1_3', 50, 150))
    enriched = FrozenHistoryProvider([old, recent], history_limit=1).enrich(
        match, include_timeline=True)
    with pytest.raises(FeatureUnavailableError):
        extract_aligned_inputs(enriched, 'nnriot.players18.v1')
```

- [x] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_parity.py -q`. Attendu : `verify` absent. Les autres tests peuvent déjà être GREEN et établissent la non-régression des tâches précédentes.

- [x] **Step 3: Implémenter le replay sans base.**

```python
def verify(dataset_dir):
    import gzip
    import json
    import numpy as np
    from core.aligned_artifacts import checked_path, dataset_runtime_sources
    from core.aligned_features import extract_aligned_inputs
    from core.frozen_history import FrozenHistoryProvider
    root = Path(dataset_dir).resolve()
    meta = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    if meta['runtime_sources'] != dataset_runtime_sources():
        raise ValueError('feature runtime SHA-256 mismatch')
    for name in meta['files']:
        checked_path(root, meta, name)
    rows = json.loads(gzip.decompress(checked_path(root, meta, 'source_rows.json.gz').read_bytes()))
    provider = FrozenHistoryProvider(rows)
    comparisons = 0
    for version, availability in meta['availability'].items():
        if availability['status'] != 'ready':
            continue
        dim = {'nnriot.players14.v1': 14, 'nnriot.players18.v1': 18}[version]
        for prefix in ('train', 'validation', 'sealed_test'):
            records = json.loads(checked_path(root, meta, prefix + '.json').read_text(encoding='utf-8'))
            with np.load(checked_path(root, meta, f'{prefix}{dim}.npz'), allow_pickle=False) as archive:
                for index, record in enumerate(records):
                    enriched = provider.enrich(record['feature_json'], include_timeline=dim == 18)
                    x = extract_aligned_inputs(enriched, version)
                    for key in ('team_a', 'team_b', 'macro_clash'):
                        if not np.array_equal(x[key][0], archive[key][index]):
                            raise ValueError(f'tensor mismatch: {record["match_id"]}/{key}/{dim}')
                        comparisons += 1
    return {'status': 'verified', 'tensor_comparisons': comparisons,
            'test_metrics_computed': False, 'availability': meta['availability']}
```

Cette commande d'audit a accès aux fichiers du test pour vérifier leurs hashes et leurs features, mais ne calcule ni accuracy, ni log-loss, ni histogramme de labels. La séparation train/validation du loader demeure stricte. Inclure le test de runtime altéré :

```python
def test_replay_refuses_changed_feature_runtime(tmp_path, monkeypatch):
    import core.aligned_artifacts as artifacts
    directory, _ = tiny_bundle(tmp_path)
    monkeypatch.setattr(artifacts, 'dataset_runtime_sources', lambda: {'changed': '0'*64})
    with pytest.raises(ValueError, match='runtime SHA-256'):
        verify(directory)
```

Pour la parité SQLite, utiliser la fixture complète ci-dessous. Elle réemploie le générateur Riot de `tests/test_point_in_time_timeline.py`, mais affecte les mêmes identités que le match raw. Paramétrer index vide/complet et 14/18 :

```python
@pytest.mark.parametrize('complete_index', [False, True])
@pytest.mark.parametrize('dim', [14, 18])
def test_sqlite_and_frozen_priors_and_tensors_are_equal(tmp_path, complete_index, dim):
    import json
    import sqlite3
    from core.pre_match_history import DatabaseHistoryProvider, read_recent_rows
    from tests.test_point_in_time_timeline import _create_sample_timeline
    db = tmp_path / 'parity.db'
    old = source_row('NA1_1', 1000, 1001000)
    raw = json.loads(old['raw_json'])
    raw['info']['gameDuration'] = 1000
    participants = raw['info']['participants']
    timeline = _create_sample_timeline()
    timeline['info']['participants'] = [
        {'participantId': p['participantId'], 'puuid': p['puuid'],
         'teamPosition': p['teamPosition']} for p in participants]
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE matches(match_id TEXT, game_creation_time INTEGER,
                                 game_mode TEXT, raw_json TEXT);
            CREATE TABLE training_dataset(match_id TEXT, winner_label INTEGER,
                                          labels_json TEXT);
            CREATE TABLE match_participants(match_id TEXT, puuid TEXT);
            CREATE TABLE match_timelines(match_id TEXT, timeline_json TEXT);
        ''')
        conn.execute('INSERT INTO matches VALUES(?,?,?,?)',
                     ('NA1_1', 1000, 'CLASSIC', json.dumps(raw)))
        # History is usable even with no ML label.
        conn.execute('INSERT INTO training_dataset VALUES(?,?,?)', ('NA1_1', None, None))
        conn.execute('INSERT INTO match_timelines VALUES(?,?)', ('NA1_1', json.dumps(timeline)))
        if complete_index:
            conn.executemany('INSERT INTO match_participants VALUES(?,?)',
                             [('NA1_1', p['puuid']) for p in participants])
    rows = read_recent_rows(db, 10000, timelines=True)
    query = target(source_row('NA1_query', 2000000, 2100000))
    a = FrozenHistoryProvider(rows).enrich(query, include_timeline=dim == 18)
    b = DatabaseHistoryProvider(db).enrich(query, include_timeline=dim == 18)
    assert [p['priors'] for team in ('team_a', 'team_b') for p in a[team]] == [
        p['priors'] for team in ('team_a', 'team_b') for p in b[team]]
    xa = extract_aligned_inputs(a, f'nnriot.players{dim}.v1')
    xb = extract_aligned_inputs(b, f'nnriot.players{dim}.v1')
    for key in xa:
        assert np.array_equal(xa[key], xb[key])
```

Les timestamps ci-dessus rendent la frame à 15 minutes disponible seulement après la fin du match historique. Inclure ces assertions dans le test de parité :

```python
for enriched in (a, b):
    provenance = enriched['priors_provenance']
    assert provenance['as_of_ms'] == 2000000
    assert provenance['history_window_limit'] == 10000
    assert provenance['latest_result_end_ms'] == 1001000
```

Les compteurs de lignes corrompues peuvent dépendre de l'optimisation de lecture ; ils ne sont pas des features et leur portée doit être décrite.

- [x] **Step 4: Vérifier GREEN** sur les cinq nouvelles suites et les invariants existants `tests/test_equivariant_model.py`, `tests/test_deepsets_layers.py`. Ne pas exiger une anti-symétrie de `winner` avec le biais de terrain actif ; l'exiger pour `draft_advantage` et vérifier son miroir 50/50.

- [ ] **Step 5: Revoir et isoler le commit** `test: certify frozen and SQLite feature parity`, limité aux fichiers de cette tâche.

### Task 6: Dataset réel, audit de disponibilité et documentation de réception

**Files:** Modify `docs/pre_match_training.md`, `docs/model_serving/README.md`; Regenerate `docs/model_serving/feature_contracts.v1.json`, `docs/model_serving/reference.manifest.json`, `docs/model_serving/manifests/` via le script existant ; Create les artefacts sous `runs/aligned_dataset_20261009/` et `docs/model_serving/aligned_dataset_verification.md`.

**Interfaces:**
- Consumes les trois sous-commandes `inventory`, `freeze`, `verify` et le nouveau paramètre `--dataset-dir` des trainers.
- Produces un snapshot réel 14D avec parité vérifiée, un rapport de disponibilité 18D, les commandes exactes et un bilan des limites.

- [x] **Step 1: Fixer les assertions de réception avant préparation réelle.** Ajouter ce test de CLI sur une fixture publiée, sans dépendre des bases de plusieurs Go :

```python
def test_verify_cli_has_no_prediction_metrics(tmp_path, capsys):
    from scripts.prepare_aligned_dataset import main
    directory, _ = tiny_bundle(tmp_path, timeline=True)
    assert main(['verify', '--dataset-dir', str(directory)]) == 0
    text = capsys.readouterr().out
    assert 'verified' in text
    assert 'test_metrics_computed' in text
    assert 'accuracy' not in text and 'log_loss' not in text
```

- [x] **Step 2: Exécuter la réception avant le run réel**, `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_artifacts.py tests/test_aligned_parity.py tests/test_aligned_training.py -q`. Un échec interdit de publier le dataset comme vérifié. Cette tâche ne demande pas de fabriquer un test rouge sur une documentation ou une commande déjà implémentée.

- [x] **Step 3: Préparer puis vérifier les artefacts réels.** Vérifier que `runs/aligned_dataset_20261009/` n'existe pas ; s'il existe, utiliser un suffixe neuf et reporter les chemins réellement utilisés dans la documentation. La base consolidée reste candidate jusqu'à la lecture du rapport de sélection.

```powershell
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset inventory --db training_data.db --db runs/training_data_23h_20260922_191434.db --db runs/db_merge_20260924/consolidated.db
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset freeze --db runs/db_merge_20260924/consolidated.db --after-cohort runs/served_model_comparison_20261009_complete/cohort.jsonl --output-dir runs/aligned_dataset_20261009 --source-limit 61000
.\.venv312\Scripts\python.exe -m scripts.prepare_aligned_dataset verify --dataset-dir runs/aligned_dataset_20261009
```

Si la purge ou la sélection ne laisse pas les minima définis, écrire le diagnostic et corriger la source/les données avant tout changement de protocole. Ne pas réduire discrètement les minima. Une source sans timelines doit publier `ready` pour 14D et `blocked` pour 18D, avec couverture 0 et les causes effectives.

Choisir un panel de 20 IDs de train/validation, déterministe et fixé avant comparaison : premières/dernières lignes et IDs espacés dans l'ordre temporel. Rejouer chacun au cutoff stocké avec la base source en lecture seule et `DatabaseHistoryProvider`, puis enregistrer hashes et comparaison exacte des priors/tenseurs. Si la base a évolué depuis le snapshot, déclarer cette différence et comparer sur une copie/source stable ; ne pas réécrire les tensors figés pour faire passer l'audit.

Ce chantier ne lance pas d'entraînement long. Les tests de tâche 4 établissent le branchement au premier batch et la clôture sans test. Documenter les commandes de futurs pilots, mais ne pas les exécuter sur le dataset réel avant le protocole d'expérience :

```powershell
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v3 --dataset-dir runs/aligned_dataset_20261009 --output-dir runs/aligned_v3_pilot --epochs 1 --batch-size 64 --seed 42
.\.venv312\Scripts\python.exe -m scripts.train_deepsets_v4_timeline --dataset-dir runs/aligned_dataset_20261009 --output-dir runs/aligned_v4_pilot --epochs 1 --batch-size 64 --seed 42
```

La deuxième commande doit refuser un dataset `blocked_18d` avant création du modèle ou apprentissage. Indiquer cette condition à côté de la commande.

- [x] **Step 4: Remettre les manifestes de service en cohérence et vérifier les régressions.**

```powershell
.\.venv312\Scripts\python.exe -m scripts.formalize_served_model
.\.venv312\Scripts\python.exe -m pytest tests/test_frozen_history.py tests/test_aligned_partitions.py tests/test_aligned_artifacts.py tests/test_aligned_training.py tests/test_aligned_parity.py tests/test_serving_contract.py tests/test_pre_match_history_serving.py tests/test_serving_http.py tests/test_predict_api.py tests/test_config_cli.py tests/test_ordered_training_integration.py tests/test_train_deepsets_v4_timeline.py tests/test_equivariant_model.py tests/test_deepsets_layers.py -q
```

Les checkpoints et calibrations existants ne changent pas. Le script de formalisation doit conserver `unverified` et l'abstention existante. Les anciens protocoles de comparaison gardent leurs hashes de runtime d'origine ; un replay avec le nouveau code doit refuser la divergence, pas réécrire leur historique.

Dans les deux documents existants, ajouter les commandes exactes, les schémas d'artefacts, la politique de purge, la logique de réservation test et la limitation 18D. Dans le rapport de réception, inclure les nombres réellement sélectionnés/exclus/purgés, dates UTC, régions/queues/patches, hashes, couverture par contrat, comparaison SQLite, commande pytest et résultat observé. Aucun chiffre de performance nouvelle n'est attendu.

- [ ] **Step 5: Revoir les artefacts et isoler le commit** `docs: record aligned dataset provenance and availability`. Inclure uniquement les docs/manifestes concernés ; ne pas ajouter la base SQLite ni des centaines de Mo de snapshot à Git. Conserver les gros artefacts sous `runs/` comme les runs existants et référencer leurs hashes dans la documentation.

## Autorelecture du plan

- Couverture : replay partagé = tâche 1 ; population/coupures = tâche 2 ; hashes et disponibilité = tâche 3 ; trainers/calibration/réservation = tâche 4 ; parité et invariants = tâche 5 ; données réelles et documentation = tâche 6.
- Les cinq classes de risque de Review Focus disposent de tests propriétaires indiqués ; les nouveaux tests de SQLite réutilisent un schéma Riot de timeline valide.
- Les signatures et noms de contrats sont communs à toutes les tâches ; les tableaux ont un ordre fixé par les records et aucun trainer ne filtre sa propre cohorte.
- Les blocs de code décrivent les nouvelles interfaces et les modifications minimales ; l'architecture réseau et les chemins legacy restent hors de la certification nouvelle.
- Point de décision matériel : les données actuelles permettent de préparer 14D ; 18D récent nécessite des timelines supplémentaires. Ce plan livre son diagnostic et le refus explicite, puis la collecte a son propre périmètre.

Autorelecture effectuée le 9 octobre 2026 : les blocs Python du plan ont été analysés avec `ast.parse`, sans erreur de syntaxe ; les contraintes globales ont été comparées à la spec et sont identiques. La fixture de labels et l'exemple de purges aux frontières ont été exécutés en mémoire et correspondent aux assertions proposées. Le chemin du cohort JSONL a été vérifié sur disque. Ces contrôles portent sur le plan ; les nouvelles suites d'implémentation restent à créer et exécuter.

## Handoff

Recommandation : **Native**, avec exécution des six tâches dans l'ordre puis une revue indépendante finale, car les interfaces et le contrat d'historique se transmettent d'une tâche à la suivante. La correction d'une fuite temporelle ou d'une mauvaise réservation test exige une revue attentive ; les tests des frontières doivent précéder les gros runs.

Faire relire le plan et choisir la méthode d'exécution avant l'implémentation, conformément au skill `superpowers:writing-plans`. Une fois la méthode choisie, utiliser `superpowers:executing-plans` pour Native ou `superpowers:subagent-driven-development` pour l'exécution avec sous-agents.
