### Task 1: Provider figé et extraction commune au service

**Files:** Create `core/frozen_history.py`, `core/aligned_features.py`, `tests/aligned_dataset_fixture.py`, `tests/test_frozen_history.py`; Modify `core/prediction_service.py:139`, `core/serving_contract.py:29`.

**Interfaces:**
- Consumes `decode_history_row(row) -> dict`, `build_target_history(records, match, *, as_of_ms: int) -> ResultOrderedHistory`, `FeatureContract.extract(match, pipeline, *, macro: bool) -> dict[str, np.ndarray]`.
- Produces `FrozenHistoryProvider(rows: list[dict], *, history_limit: int = 10000).enrich(match: dict, *, include_timeline: bool) -> dict`.
- Produces `extract_aligned_inputs(match: dict, contract_version: str, *, macro: bool = True, pipeline: DraftFeaturePipeline | None = None) -> dict[str, np.ndarray]`.
- Produces test fixture `source_row(mid: str, creation: int, end: int, winner: int = 0, *, timeline: bool = False) -> dict` and `target(row: dict) -> dict`.

- [ ] **Step 1: Écrire les fixtures et les tests RED.** Utiliser les labels réels de `core.feature_labels.LABEL_KEYS` ; le bloc suivant définit entièrement la fixture, sans valeurs manquantes implicites.

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

- [ ] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_frozen_history.py -q`. Attendu : erreur d'import des deux nouveaux modules.

- [ ] **Step 3: Implémenter le provider et l'extraction.** Le provider trie par `(creation, match_id)` ; `bisect_left` exclut toutes les créations ex æquo. L'index par joueur ne remplace jamais la sélection globale des 10 000 positions.

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

- [ ] **Step 4: Vérifier GREEN** avec `tests/test_frozen_history.py`, `tests/test_serving_contract.py` et `tests/test_pre_match_history_serving.py`. Attendu : priors stricts, dimensions exactes et aucun changement de probabilité causé par l'extraction commune.

- [ ] **Step 5: Revoir le diff et isoler le commit** `feat: share strict serving feature extraction with frozen replay`. Chemins : les cinq fichiers de cette tâche et `tests/aligned_dataset_fixture.py`. Pour les deux fichiers existants utiliser `git add -p`; examiner `git diff --cached` avant le commit.

