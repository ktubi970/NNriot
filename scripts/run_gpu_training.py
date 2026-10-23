#!/usr/bin/env python3
"""Verify TensorFlow CUDA execution, then run a training script on that GPU.

Examples:
    python scripts/run_gpu_training.py --check
    python scripts/run_gpu_training.py scripts/train_deepsets_v3.py --epochs 5

Without a script, run train_deepsets_v3.py with its existing defaults.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "train_deepsets_v3.py"


def verify_gpu() -> None:
    """Require a CUDA build, a visible GPU, and a completed GPU operation."""
    if sys.platform == "win32":
        guidance = (
            "Use WSL2 with a Linux environment containing tensorflow[and-cuda]; "
            "current native Windows TensorFlow packages do not enable CUDA."
        )
    else:
        guidance = (
            "Use a CUDA-capable TensorFlow environment (tensorflow[and-cuda]), "
            "check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )

    # Absolute Python invocations do not activate the venv; expose its CUDA tools.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

    try:
        # Keep --help usable even when TensorFlow is not installed.
        import tensorflow as tf
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Cannot import TensorFlow: {exc}. {guidance}") from exc

    if not tf.test.is_built_with_cuda():
        raise RuntimeError(f"TensorFlow {tf.__version__} was built without CUDA. {guidance}")
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError(f"TensorFlow {tf.__version__} sees no GPU. {guidance}")

    try:
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        with tf.device("/GPU:0"):
            matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            product = tf.matmul(matrix, matrix)
        if "/DEVICE:GPU:0" not in product.device.upper():
            raise RuntimeError(f"Matrix multiplication fell back to {product.device}.")
        # Synchronize the operation so a queued CUDA failure cannot pass the check.
        product.numpy()
    except Exception as exc:
        raise RuntimeError(f"CUDA execution check failed: {exc}. {guidance}") from exc

    details = tf.config.experimental.get_device_details(devices[0])
    print(f"TensorFlow {tf.__version__}; GPU: {details.get('device_name', devices[0].name)}", flush=True)
    print(f"GPU matrix multiplication verified on {product.device}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify GPU execution without starting training")
    parser.add_argument("script", nargs="?", help="Training Python script (default: scripts/train_deepsets_v3.py)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed unchanged to the training script")
    args = parser.parse_args(argv)

    script = Path(args.script).resolve() if args.script else DEFAULT_SCRIPT
    if not args.check and not script.is_file():
        parser.error(f"Training script does not exist: {script}")
    try:
        verify_gpu()
    except RuntimeError as exc:
        parser.exit(2, f"GPU training unavailable: {exc}\n")
    if args.check:
        return 0

    original_argv = sys.argv
    original_path = sys.path[:]
    try:
        sys.argv = [str(script), *args.script_args]
        sys.path[:0] = [str(script.parent), str(REPO_ROOT)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
