"""Inspect checkpoint metadata without loading or modifying model weights."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from core.calibration import sha256_file
from core.config import PROJECT_ROOT


def inspect_candidate(checkpoint: Path) -> dict:
    relative = checkpoint.resolve().relative_to(PROJECT_ROOT).as_posix()
    kind = 'keras' if checkpoint.suffix == '.keras' else 'tree'
    item = {'model_id': checkpoint.parent.name, 'checkpoint': {'path': relative, 'sha256': sha256_file(checkpoint)},
            'kind': kind, 'feature_contract': None, 'input_shapes': None, 'outputs': [],
            'training': {'status': 'unknown', 'source_db': None, 'bounds_ms': None, 'split_manifest_sha256': None},
            'calibration': None, 'independent_test_eligible': False, 'limitations': []}
    if kind == 'keras':
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read('config.json'))['config']
            item['saved_metadata'] = json.loads(archive.read('metadata.json'))
        shapes = {l['config']['name']: l['config'].get('batch_shape', l['config'].get('batch_input_shape'))
                  for l in data['layers'] if l['class_name'] == 'InputLayer'}
        item['input_shapes'] = shapes
        outs = data['output_layers']
        item['outputs'] = list(outs) if isinstance(outs, dict) else [str(o) for o in outs]
        dim = (shapes.get('team_a') or [None])[-1]
        item['feature_contract'] = {14: 'nnriot.players14.v1', 18: 'nnriot.players18.v1'}.get(dim)
        item['has_directional_side_bias'] = any(l['class_name'] == 'DirectionalSideBias' for l in data['layers'])
        if dim == 46:
            item['limitations'].append('Spatial RNN encoder training bounds/hash and historical embedding generation provenance absent; 46D disabled')
        elif not item['feature_contract']:
            item['limitations'].append('Legacy input representation has no audited serving contract; excluded explicitly')
    else:
        item['feature_contract'] = 'legacy.tree24.v1'
    split = checkpoint.with_suffix('.split.json')
    if split.is_file():
        data = json.loads(split.read_text(encoding='utf-8'))
        bounds = {k: {'count': len(v), 'min_ms': min(r['game_creation_time'] for r in v),
                      'max_ms': max(r['game_creation_time'] for r in v)} for k, v in data.get('splits', {}).items() if v}
        item['training'].update(status='split_membership_recorded_training_semantics_unverified',
                                source_db=data.get('source_db'), bounds_ms=bounds,
                                split_manifest_sha256=sha256_file(split), dataset_sha256=data.get('dataset_sha256'))
    else:
        item['limitations'].append('No checkpoint-bound training split artifact; training bounds unknown')
    calibration = checkpoint.with_suffix('.calibration.json')
    if calibration.is_file():
        item['calibration'] = {'path': calibration.resolve().relative_to(PROJECT_ROOT).as_posix(), 'sha256': sha256_file(calibration)}
    item['limitations'].append('Historical training feature provenance/PIT availability is not established by tensor shape or current source code')
    if kind == 'keras' and 'draft_advantage' not in item['outputs']:
        item['limitations'].append('Legacy checkpoint has no separate draft_advantage head; do not infer it from winner')
    return item


def inventory_candidates() -> list[dict]:
    # Top-level runs are candidates. Nested pytest/smoke checkpoints are fixtures.
    paths = sorted((PROJECT_ROOT/'runs').glob('*/*.keras'))
    paths += sorted((PROJECT_ROOT/'runs').glob('*/tree_baseline/*.joblib'))
    items = []
    for p in paths:
        try:
            items.append(inspect_candidate(p))
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            items.append({'model_id': p.parent.name, 'checkpoint': {'path': p.relative_to(PROJECT_ROOT).as_posix()}, 'error': str(exc)})
    return items


def serving_manifest(item: dict) -> dict:
    from core.serving_contract import runtime_feature_sources
    return {'schema_version': 1, 'model_id': item['model_id'], 'checkpoint': item['checkpoint'],
            'runtime_feature_sources': runtime_feature_sources(),
            'feature_contract': item['feature_contract'], 'feature_semantics_status': 'layout_verified_training_semantics_unverified',
            'training': item['training'], 'calibration': item['calibration'],
            'output_heads': item['outputs'], 'has_directional_side_bias': item.get('has_directional_side_bias'),
            'decision_policy': {'action_threshold': 0.04, 'high_threshold': 0.08, 'origin': 'existing defaults; not validated risk guarantees'},
            'serving_validation': {'status': 'unverified', 'independent_evaluation_report': None},
            'limitations': item['limitations']}
