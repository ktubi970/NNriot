### Task 3: Artefacts figés, disponibilité 18D et CLI de préparation

**Files:** Create `core/aligned_artifacts.py`, `scripts/prepare_aligned_dataset.py`, `tests/test_aligned_artifacts.py`.

**Interfaces:**
- Consumes les interfaces des tâches 1–2 et `build_manifest(splits, *, source_db, split_metadata) -> dict`, `runtime_feature_sources() -> dict`, `sha256_file(path) -> str`.
- Produces `write_dataset(output: Path, rows: list[dict], splits: dict, *, source_db: str, selection: dict, exclusions: list[dict]) -> dict`.
- Produces `read_train_validation(dataset_dir: Path, contract_version: str) -> tuple[dict, dict, dict]` ; retour = tenseurs `{train, validation}`, records `{train, validation}`, manifeste du dataset.
- Produces `inspect_sources(paths: list[Path]) -> list[dict]`, `prepare(db: Path, output: Path, after_cohort: Path, *, source_limit: int = 61000) -> dict`, `verify(dataset_dir: Path) -> dict` et `main(argv=None) -> int`.

- [ ] **Step 1: Écrire les tests RED d'artefacts.**

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

- [ ] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_artifacts.py -q`. Attendu : module absent.

- [ ] **Step 3: Implémenter le format et les contrôles.** Fichiers exacts : `source_rows.json.gz`, `model.split.json`, `train.json`, `validation.json`, `sealed_test.json`, `train14.npz`, `validation14.npz`, `sealed_test14.npz`, les trois fichiers `18.npz` si le contrat est disponible partout, `exclusions.json`, `availability.json`, `dataset.json`. La compression de source utilise `gzip.compress(..., mtime=0)`. `dataset.json` contient `schema_version: 1`, `kind: nnriot.serving_aligned_dataset.v1`, hashes des fichiers, hashes de runtime et availability. L'empreinte du manifeste lui-même est calculée par ses consommateurs ; ne pas créer de hash circulaire.

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

- [ ] **Step 4: Vérifier GREEN** sur `tests/test_aligned_artifacts.py` et `tests/test_aligned_partitions.py`. Vérifier également `--help` et l'inventaire des trois bases de la spec, en conservant sa sortie JSON. Aucun freeze réel à cette étape.

- [ ] **Step 5: Revoir et isoler le commit** `feat: persist hash-bound aligned datasets and feature availability`, limité aux trois nouveaux fichiers de cette tâche.

