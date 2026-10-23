### Task 4: Mode dataset aligné des trainers, calibration stricte et test réservé

**Files:** Create `core/aligned_training.py`, `tests/test_aligned_training.py`; Modify `scripts/train_deepsets_v3.py:44`, `scripts/train_deepsets_v4_timeline.py:43` et leurs blocs CLI.

**Interfaces:**
- Consumes `read_train_validation(dataset_dir, contract_version) -> (inputs, records, manifest)` et `checked_path(root, manifest, name) -> Path`.
- Produces `prepare_aligned_training(dataset_dir: Path, output_dir: Path, contract_version: str) -> tuple[dict, dict, dict]`.
- Produces `finish_aligned_pilot(*, output_dir: Path, dataset_dir: Path, dataset_manifest: dict, input_dim: int, validation_probabilities: np.ndarray, validation_labels: np.ndarray, history: list[dict], best_epoch: int, best_val_loss: float) -> dict`.
- Extend les deux fonctions train avec `dataset_dir: str | None = None` keyword-only ; CLI `--dataset-dir`.

- [ ] **Step 1: Écrire les tests RED sur les consommateurs réels.**

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

- [ ] **Step 2: Exécuter** `.\.venv312\Scripts\python.exe -m pytest tests/test_aligned_training.py -q`. Attendu : argument `dataset_dir` inconnu et nouveau module absent.

- [ ] **Step 3: Implémenter la préparation sans labels test et la clôture.** Copier le manifeste d'IDs ne nécessite pas la lecture des labels test. Le manifeste de calibration reste un compagnon exact `model.split.json`, vérifié par `save_calibration_artifact`.

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

- [ ] **Step 4: Vérifier GREEN** avec `tests/test_aligned_training.py`, `tests/test_ordered_training_integration.py`, `tests/test_train_deepsets_v4_timeline.py`. Exécuter les tests de clôture ci-dessus, dont le garde interdit explicitement l'ouverture des faits source et des fichiers `sealed_test*`.

- [ ] **Step 5: Revoir et isoler le commit** `feat: train aligned v3 and v4 pilots without opening reserved test`, avec les deux nouveaux fichiers et les seuls hunks nécessaires des deux trainers.

