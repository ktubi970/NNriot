"""Fit the preregistered logistic and LightGBM baselines on train, selected and calibrated on validation.

Reads only train/validation through read_train_validation; the sealed test is never opened.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
DATASET = ROOT / 'runs/aligned_dataset_20261023_prospective'
SEED = 42


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    import numpy as np
    from core.aligned_artifacts import read_train_validation
    from core.tabular_baselines import fit_lightgbm, fit_logistic, save_baseline
    preparation = json.loads((OUT / 'preparation_result.json').read_text(encoding='utf-8'))
    assert preparation['status'] == 'verified' and preparation['test_metrics_computed'] is False
    assert digest(DATASET / 'dataset.json') == preparation['dataset_manifest_sha256']
    results = {}
    for dim in (14, 18):
        x, records, _ = read_train_validation(DATASET, f'nnriot.players{dim}.v1')
        y = {name: np.asarray([r['labels_json']['winner'] for r in records[name]], dtype=int) for name in records}
        for family, fit in (('logistic', fit_logistic), ('lightgbm', fit_lightgbm)):
            started = time.time()
            model = fit(x['train'], y['train'], x['validation'], y['validation'], seed=SEED)
            path = OUT / f'baseline_{family}{dim}.json'
            save_baseline(model, path)
            results[f'{family}{dim}'] = dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path),
                elapsed_seconds=round(time.time() - started, 2),
                selection={k: v for k, v in model['selection'].items() if k != 'params'},
                validation_log_loss_raw=model['validation_log_loss_raw'],
                temperature=model['calibration']['temperature'],
                validation_log_loss_calibrated=model['calibration']['log_loss_after'])
            print(f'{family}{dim}: {results[f"{family}{dim}"]}', flush=True)
    with (OUT / 'baselines_status.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(status='completed', test_status='reserved', seed=SEED, completed=results,
                       dataset_manifest_sha256=digest(DATASET / 'dataset.json'),
                       sources={n: digest(ROOT / n) for n in ('core/tabular_baselines.py', 'core/calibration.py',
                                                             'core/aligned_artifacts.py')} | {
                           Path(__file__).relative_to(ROOT).as_posix(): digest(__file__)},
                       finished_utc=datetime.now(timezone.utc).isoformat()), handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
