### Task 5: Replay intégral et preuve de parité avec SQLite

**Files:** Create `tests/test_aligned_parity.py`; Modify `scripts/prepare_aligned_dataset.py` pour `verify` et `core/aligned_artifacts.py` pour les contrôles complémentaires.

**Interfaces:**
- Consumes `checked_path`, `FrozenHistoryProvider`, `extract_aligned_inputs` et les fichiers figés de tâche 3.
- Produces `verify(dataset_dir: Path) -> dict` avec `status: verified`, hashes et nombre de comparaisons exactes ; ne calcule aucune métrique de victoire.
- Vérification de service : `DatabaseHistoryProvider(db_path, history_limit=10000).enrich(match, *, include_timeline: bool) -> dict` reste inchangée.

- [ ] **Step 1: Écrire les tests RED de replay et de parité.**

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

- [ ] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_parity.py -q`. Attendu : `verify` absent. Les autres tests peuvent déjà être GREEN et établissent la non-régression des tâches précédentes.

- [ ] **Step 3: Implémenter le replay sans base.**

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

- [ ] **Step 4: Vérifier GREEN** sur les cinq nouvelles suites et les invariants existants `tests/test_equivariant_model.py`, `tests/test_deepsets_layers.py`. Ne pas exiger une anti-symétrie de `winner` avec le biais de terrain actif ; l'exiger pour `draft_advantage` et vérifier son miroir 50/50.

- [ ] **Step 5: Revoir et isoler le commit** `test: certify frozen and SQLite feature parity`, limité aux fichiers de cette tâche.

