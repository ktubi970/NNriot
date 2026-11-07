"""Behavior of the GPU-only training entry point."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_gpu_training.py"


def test_help_works_without_importing_tensorflow():
    code = (
        "import runpy, sys; "
        f"sys.argv = [{str(RUNNER)!r}, '--help']; "
        "\ntry:\n    runpy.run_path(sys.argv[0], run_name='__main__')"
        "\nexcept SystemExit as exc:\n    assert exc.code == 0"
        "\nassert 'tensorflow' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--check" in result.stdout


def test_no_visible_gpu_refuses_to_execute_training_script(tmp_path):
    marker = tmp_path / "training-started.txt"
    script = tmp_path / "training.py"
    script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(script)], cwd=REPO_ROOT,
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 2, result.stderr
    assert "GPU training unavailable" in result.stderr
    if sys.platform == "win32":
        assert "WSL2" in result.stderr
    assert not marker.exists()


def test_training_receives_arguments_and_preserves_exit_code(tmp_path, monkeypatch):
    from scripts import run_gpu_training

    # The real hardware check has its own CLI test; only bypass hardware here.
    monkeypatch.setattr(run_gpu_training, "verify_gpu", lambda: None)
    received = tmp_path / "received.json"
    script = tmp_path / "training script.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        f"Path({str(received)!r}).write_text(json.dumps(sys.argv), encoding='utf-8')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    with pytest.raises(SystemExit) as exc:
        run_gpu_training.main([str(script), "--output-dir", "path with spaces", "--check"])
    assert exc.value.code == 7
    assert json.loads(received.read_text(encoding="utf-8")) == [
        str(script), "--output-dir", "path with spaces", "--check",
    ]
    assert sys.argv == original_argv
