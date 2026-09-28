#!/usr/bin/env bash
# Run inside an Ubuntu WSL2 distribution as root. No Linux NVIDIA driver needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$(id -u)" != 0 ]]; then
    echo 'Run this setup as root inside the dedicated WSL2 distribution.' >&2
    exit 1
fi

/usr/lib/wsl/lib/nvidia-smi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv
python3 -m venv /opt/nnriot-venv
/opt/nnriot-venv/bin/python -m pip install --upgrade pip
/opt/nnriot-venv/bin/python -m pip install --no-cache-dir -r requirements-gpu.txt
bash scripts/run_with_cuda.sh /opt/nnriot-venv/bin/python scripts/run_gpu_training.py --check
