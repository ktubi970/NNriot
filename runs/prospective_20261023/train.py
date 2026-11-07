"""Run the six preregistered DeepSets trainings sequentially on the WSL GPU, never opening the test.

Resumable: a run whose checkpoint and metrics are already recorded in training_status.json is skipped.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASET = ROOT / 'runs/aligned_dataset_20261023_prospective'
SCRIPTS = {14: 'train_deepsets_v3.py', 18: 'train_deepsets_v4_timeline.py'}
# Interleaved so an interruption still leaves both dimensions with equal seeds.
ORDER = [(18, 42), (14, 42), (18, 43), (14, 43), (18, 44), (14, 44)]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, value):
    target = OUT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def command(dim, seed):
    cmd = ['wsl.exe', '-d', 'NNriot-GPU', '-u', 'root', '--cd', '/mnt/d/projet/NNriot',
           '--exec', 'env', 'TF_NUM_INTRAOP_THREADS=4', 'TF_NUM_INTEROP_THREADS=2',
           'OMP_NUM_THREADS=4', 'PYTHONUNBUFFERED=1', 'bash', 'scripts/run_with_cuda.sh',
           '/opt/nnriot-venv/bin/python', 'scripts/run_gpu_training.py', 'scripts/' + SCRIPTS[dim],
           '--dataset-dir', DATASET.relative_to(ROOT).as_posix(),
           '--output-dir', (OUT / f'deepsets{dim}_s{seed}').relative_to(ROOT).as_posix(),
           '--epochs', '5', '--batch-size', '64', '--learning-rate', '0.0008', '--seed', str(seed)]
    return cmd + (['--patience', '0'] if dim == 18 else [])


def main():
    preparation = json.loads((OUT / 'preparation_result.json').read_text(encoding='utf-8'))
    assert preparation['status'] == 'verified' and preparation['test_metrics_computed'] is False
    assert digest(DATASET / 'dataset.json') == preparation['dataset_manifest_sha256']
    manifest = json.loads((DATASET / 'dataset.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == manifest['selection']['protocol_sha256']
    names = set(manifest['runtime_sources']) | {
        'core/equivariant_model.py', 'core/deepsets_layers.py', 'core/generate_graph.py',
        'core/pcgrad.py', 'core/training_utils.py', 'core/aligned_training.py',
        'core/calibration.py', 'core/training_metrics.py', 'scripts/run_gpu_training.py',
        'scripts/run_with_cuda.sh', *('scripts/' + s for s in SCRIPTS.values())}
    hashes = {name: digest(ROOT / name) for name in sorted(names)}
    assert all(hashes[n] == h for n, h in manifest['runtime_sources'].items()), 'Feature runtime changed'
    plan = dict(protocol_sha256=digest(OUT / 'protocol.json'), dataset_manifest_sha256=digest(DATASET / 'dataset.json'),
                source_hashes=hashes, commands={f'{d}_{s}': command(d, s) for d, s in ORDER},
                test_policy='reserved during every training job')
    plan_path = OUT / 'training_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf-8'))
        assert {k: previous[k] for k in plan} == plan, 'Training plan changed; resume refused'
    else:
        save('training_plan.json', {**plan, 'created_utc': datetime.now(timezone.utc).isoformat()})
    status_path = OUT / 'training_status.json'
    completed = json.loads(status_path.read_text(encoding='utf-8'))['completed'] if status_path.exists() else {}
    for dim, seed in ORDER:
        key, output = f'{dim}_{seed}', OUT / f'deepsets{dim}_s{seed}'
        if key in completed:
            assert digest(output / 'model.keras') == completed[key]['checkpoint_sha256']
            continue
        assert not output.exists(), f'Refusing an incomplete earlier output: {output}'
        assert all(digest(ROOT / n) == h for n, h in hashes.items()), 'Sources changed during training'
        save('training_status.json', dict(status='running', current=key, completed=completed,
                                          started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Starting {dim}D seed {seed}', flush=True)
        started = time.time()
        with (OUT / f'deepsets{dim}_s{seed}.training.log').open('x', encoding='utf-8') as log:
            returncode = subprocess.call(command(dim, seed), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if returncode:
            raise RuntimeError(f'{key} failed with exit code {returncode}; inspect its training log')
        metrics = json.loads((output / 'model.metrics.json').read_text(encoding='utf-8'))
        assert metrics['test_status'] == 'reserved' and metrics['calibrated'] is True
        assert metrics['dataset_manifest_sha256'] == plan['dataset_manifest_sha256']
        assert metrics['split_sha256'] == manifest['split_sha256']
        assert metrics['checkpoint_sha256'] == digest(output / 'model.keras')
        completed[key] = dict(elapsed_seconds=round(time.time() - started, 2),
                              checkpoint_sha256=metrics['checkpoint_sha256'],
                              metrics_sha256=digest(output / 'model.metrics.json'),
                              best_epoch=metrics['best_epoch'], best_val_log_loss=metrics['best_val_log_loss'])
        print(f'Completed {key}: {completed[key]}', flush=True)
    save('training_status.json', dict(status='completed', completed=completed, test_status='reserved',
                                      finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (OUT / 'training_failure.json').write_text(json.dumps(dict(type=type(exc).__name__, message=str(exc))),
                                                   encoding='utf-8')
        raise
