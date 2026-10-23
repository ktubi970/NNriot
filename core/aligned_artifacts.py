# core/aligned_artifacts.py: noyau d'écriture
import gzip
import json
import logging
from pathlib import Path
import numpy as np
from core.aligned_features import extract_aligned_inputs
from core.frozen_history import FrozenHistoryProvider
from core.calibration import sha256_file
from core.experiment import build_manifest, read_manifest
from core.serving_contract import ContractError, runtime_feature_sources
from core.config import PROJECT_ROOT

def canonical(value, *, sort_keys=True):
    return (json.dumps(value, sort_keys=sort_keys, separators=(',', ':'),
                       allow_nan=False, ensure_ascii=False) + '\n').encode('utf-8')

def dataset_runtime_sources():
    names = ('core/frozen_history.py', 'core/aligned_partitions.py',
             'core/aligned_artifacts.py', 'core/serving_contract.py',
             'scripts/prepare_aligned_dataset.py')
    return {**runtime_feature_sources(), **{
        name: sha256_file(PROJECT_ROOT / name) for name in names}}

def write_dataset(output, rows, splits, *, source_db, selection, exclusions):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'occupied output directory: {output}')
    split_manifest = build_manifest(splits, source_db=source_db, split_metadata=selection)
    source_ids = {row['match_id'] for row in rows}
    for partition in splits.values():
        for record in partition:
            if record['match_id'] not in source_ids:
                raise ValueError('target absent from source snapshot')
            if record['feature_json'].get('prediction_time_ms') != record['game_creation_time']:
                raise ValueError('prediction cutoff mismatch')
    output.mkdir(parents=True, exist_ok=False)
    provider = FrozenHistoryProvider(rows)
    files, availability = {}, {}
    def register(name):
        files[name] = {'path': name, 'sha256': sha256_file(output / name)}
    def save_json(name, value, *, sort_keys=True):
        with (output / name).open('xb') as handle:
            handle.write(canonical(value, sort_keys=sort_keys))
        register(name)
    (output / 'source_rows.json.gz').write_bytes(gzip.compress(canonical(rows), mtime=0))
    register('source_rows.json.gz')
    for partition in ('train', 'validation', 'test'):
        prefix = 'sealed_test' if partition == 'test' else partition
        save_json(prefix + '.json', splits[partition])
    for dim in (14, 18):
        version = f'nnriot.players{dim}.v1'
        arrays, errors = {}, []
        processed = 0
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
                processed += 1
                if processed % 1000 == 0:
                    logging.getLogger(__name__).info('Materialized %s: %d/%d targets', version, processed, total)
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
    save_json('model.split.json', split_manifest, sort_keys=False)
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


def validate_tensor_inputs(inputs, count, dim, partition):
    expected = {'team_a': (count, 5, dim), 'team_b': (count, 5, dim),
                'macro_clash': (count, 15)}
    if set(inputs) != set(expected) or any(
            inputs[key].shape != shape or inputs[key].dtype != np.float32
            or not np.isfinite(inputs[key]).all() for key, shape in expected.items()):
        raise ValueError(f'{partition}: invalid tensor contract')

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
    split_manifest = read_manifest(checked_path(root, manifest, 'model.split.json'))
    if split_manifest['split_sha256'] != manifest['split_sha256'] or split_manifest['dataset_sha256'] != manifest['dataset_sha256']:
        raise ValueError('dataset split manifest mismatch')
    records, inputs = {}, {}
    for name in ('train', 'validation'):
        records[name] = json.loads(checked_path(root, manifest, name + '.json').read_text(encoding='utf-8'))
        membership = [{'match_id': r['match_id'], 'game_creation_time': r['game_creation_time']} for r in records[name]]
        if membership != split_manifest['splits'][name]:
            raise ValueError(f'{name}: record order differs from split manifest')
        with np.load(checked_path(root, manifest, f'{name}{dim}.npz'), allow_pickle=False) as data:
            inputs[name] = {key: data[key].copy() for key in data.files}
        validate_tensor_inputs(inputs[name], len(records[name]), dim, name)
    return inputs, records, manifest
