### Task 2: Population cible et partitions purgées

**Files:** Create `core/aligned_partitions.py`, `tests/test_aligned_partitions.py`; Modify `core/pre_match_history.py:63` (`read_recent_rows`).

**Interfaces:**
- Extend `read_recent_rows(db_path, limit: int, *, before_ms: int | None = None, timelines: bool = True, puuids=None, include_labels: bool = False) -> list[dict]`.
- Produce `select_targets(rows: list[dict], *, excluded_ids: set[str], warmup_rows: int = 10000) -> tuple[list[dict], list[dict]]` ; tuples = cibles validées et exclusions `{match_id, reason}`.
- Produce `partition_targets(targets: list[dict], *, after_ms: int, validation_count: int = 3000, test_count: int = 3000) -> tuple[dict[str, list[dict]], list[dict]]` ; deuxième liste = purges.
- Target record fields : `match_id`, `game_creation_time`, `result_end_ms`, `feature_json`, `labels_json`, `queue_id`, `patch`, `region`.

- [ ] **Step 1: Écrire les tests RED.** Les petits comptes sont exclusivement des fixtures ; la CLI de préparation réelle conserve les minima 3 000/3 000.

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

- [ ] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_partitions.py -q`. Attendu : module absent.

- [ ] **Step 3: Implémenter la sélection stricte et les coupures.** Réutiliser la validation des labels existante ; ne créer aucun défaut pour un label absent.

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

- [ ] **Step 4: Vérifier GREEN** sur `tests/test_aligned_partitions.py`, `tests/test_pre_match_history_serving.py`, `tests/test_point_in_time_dataset.py`. Inclure ce test SQLite de compactage dans la suite de partitions :

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

