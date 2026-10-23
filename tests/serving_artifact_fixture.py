"""Synthetic, fully bound serving artifacts; no operational model or data writes."""
import json
from pathlib import Path

from core.calibration import save_calibration_artifact, sha256_file
from core.experiment import build_manifest
from core.serving_contract import runtime_feature_sources


def write_json(path, data):
    Path(path).write_text(json.dumps(data, allow_nan=False), encoding='utf-8')


def build_serving_artifacts(tmp_path, *, calibrated=True, validated=True,
                            temperature=1.0, output_heads=None):
    """Build artifacts whose hashes are really checked; callers fake only Keras I/O."""
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'model.keras'
    checkpoint.write_bytes(b'synthetic checkpoint for serving-contract tests')
    split = checkpoint.with_suffix('.split.json')
    splits = {
        name: [{'match_id': f'{name}_{i}', 'game_creation_time': base + i}
               for i in range(25)]
        for name, base in (('train', 1000), ('validation', 2000), ('test', 3000))
    }
    split_data = build_manifest(splits, source_db=None,
                                split_metadata={'strategy': 'chronological'})
    write_json(split, split_data)
    calibration = None
    if calibrated:
        calibration = save_calibration_artifact(checkpoint, split, {
            'status': 'fitted', 'partition': 'validation', 'count': 25,
            'class_counts': [12, 13], 'temperature': temperature,
            'optimizer_success': True, 'boundary_hit': False,
            'log_loss_before': .69, 'log_loss_after': .65,
        }, input_dim=14)
    policy = {'rule': 'nnriot.conviction.v1', 'action_threshold': .04,
              'high_threshold': .08}
    manifest_data = {
        'schema_version': 1, 'model_id': 'synthetic-serving-test',
        'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': sha256_file(checkpoint)},
        'feature_contract': 'nnriot.players14.v1',
        'runtime_feature_sources': runtime_feature_sources(),
        'output_heads': output_heads or ['winner'],
        'training': {'status': 'synthetic_fixture',
                     'split_manifest_sha256': sha256_file(split),
                     'dataset_sha256': split_data['dataset_sha256']},
        'calibration': ({'path': str(calibration.resolve()), 'sha256': sha256_file(calibration)}
                        if calibration else None),
        'decision_policy': policy,
        'serving_validation': {'status': 'unverified', 'independent_evaluation_report': None},
    }
    report = None
    if validated and calibrated:
        report = root / 'serving-evaluation.json'
        write_json(report, {
            'schema_version': 1, 'status': 'validated_for_serving',
            'checkpoint_sha256': manifest_data['checkpoint']['sha256'],
            'calibration_sha256': manifest_data['calibration']['sha256'],
            'feature_contract': manifest_data['feature_contract'],
            'runtime_feature_sources': manifest_data['runtime_feature_sources'],
            'split_manifest_sha256': sha256_file(split),
            'dataset_sha256': split_data['dataset_sha256'],
            'decision_policy': policy,
            'evaluation': {'partition': 'test', 'count': 25,
                           'match_ids': [r['match_id'] for r in splits['test']],
                           'metrics': {'log_loss': .55, 'brier': .18}},
        })
        manifest_data['serving_validation'] = {
            'status': 'validated',
            'independent_evaluation_report': {'path': str(report.resolve()), 'sha256': sha256_file(report)},
        }
    manifest = root / 'serving.manifest.json'
    write_json(manifest, manifest_data)
    return {'checkpoint': checkpoint, 'manifest': manifest, 'split': split,
            'calibration': calibration, 'report': report}
