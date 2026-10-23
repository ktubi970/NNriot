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
