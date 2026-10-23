"""Run the two predeclared GPU pilots sequentially, without opening the test."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASET = ROOT / 'runs/aligned_dataset_20261011_pilot'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, value):
    target = OUT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def main():
    preparation = json.loads((OUT / 'preparation_result.json').read_text(encoding='utf-8'))
    assert preparation['status'] == 'verified'
    assert digest(DATASET / 'dataset.json') == preparation['dataset_manifest_sha256']
    manifest = json.loads((DATASET / 'dataset.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == manifest['selection']['protocol_sha256']
    checks = json.loads((OUT / 'invariant_tests.json').read_text(encoding='utf-8'))
    assert checks['exit_code'] == 0, 'Required invariant tests did not pass'
    names = set(manifest['runtime_sources']) | {
        'core/equivariant_model.py', 'core/deepsets_layers.py', 'core/generate_graph.py',
        'core/pcgrad.py', 'core/training_utils.py', 'core/aligned_training.py',
        'core/calibration.py', 'core/training_metrics.py',
        'scripts/train_deepsets_v3.py', 'scripts/train_deepsets_v4_timeline.py',
        'scripts/run_gpu_training.py', 'scripts/run_with_cuda.sh'}
    hashes = {name: digest(ROOT / name) for name in sorted(names)}
    for name, expected in manifest['runtime_sources'].items():
        assert hashes[name] == expected
    commands = []
    for dim, script in ((14, 'train_deepsets_v3.py'), (18, 'train_deepsets_v4_timeline.py')):
        command = ['wsl.exe', '-d', 'NNriot-GPU', '-u', 'root', '--cd', '/mnt/d/projet/NNriot',
            '--exec', 'env', 'TF_NUM_INTRAOP_THREADS=4', 'TF_NUM_INTEROP_THREADS=2',
            'OMP_NUM_THREADS=4', 'PYTHONUNBUFFERED=1', 'bash', 'scripts/run_with_cuda.sh',
            '/opt/nnriot-venv/bin/python', 'scripts/run_gpu_training.py', 'scripts/' + script,
            '--dataset-dir', 'runs/aligned_dataset_20261011_pilot',
            '--output-dir', f'runs/aligned_comparison_20261011/{dim}d',
            '--epochs', '5', '--batch-size', '64', '--learning-rate', '0.0008', '--seed', '42']
        if dim == 18:
            command += ['--patience', '0']
        commands.append((dim, command))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(OUT / 'protocol.json'),
        dataset_manifest_sha256=digest(DATASET / 'dataset.json'), source_hashes=hashes,
        commands={str(dim): cmd for dim, cmd in commands}, test_policy='reserved during both training jobs')
    with (OUT / 'training_plan.json').open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2)
    results = {}
    for dim, command in commands:
        assert all(digest(ROOT / name) == expected for name, expected in hashes.items())
        output = OUT / f'{dim}d'
        assert not output.exists(), f'Refusing occupied output: {output}'
        save('training_status.json', dict(status='running', dimension=dim, completed=results,
            started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Starting {dim}D on verified CUDA; log={dim}d.training.log', flush=True)
        started = time.time()
        with (OUT / f'{dim}d.training.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            save(f'{dim}d.process.json', dict(wsl_pid=process.pid, command=command))
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f'{dim}D failed with exit code {returncode}; inspect training log')
        metrics = json.loads((output / 'model.metrics.json').read_text(encoding='utf-8'))
        assert metrics['test_status'] == 'reserved'
        assert metrics['dataset_manifest_sha256'] == plan['dataset_manifest_sha256']
        assert metrics['split_sha256'] == manifest['split_sha256']
        assert metrics['checkpoint_sha256'] == digest(output / 'model.keras')
        assert metrics['calibrated'] is True
        assert all(digest(ROOT / name) == expected for name, expected in hashes.items())
        results[str(dim)] = dict(status='completed', elapsed_seconds=round(time.time() - started, 2),
            checkpoint_sha256=metrics['checkpoint_sha256'], metrics_sha256=digest(output / 'model.metrics.json'),
            best_epoch=metrics['best_epoch'], best_val_log_loss=metrics['best_val_log_loss'])
        print(f'Completed {dim}D: {results[str(dim)]}', flush=True)
    save('training_status.json', dict(status='completed', completed=results,
         test_status='reserved', finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('training_failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
