#!/usr/bin/env bash
# Expose the CUDA libraries installed by pip before starting the Python loader.
set -euo pipefail
if [[ $# -lt 1 || "$1" != /* ]]; then
    echo 'Usage: bash scripts/run_with_cuda.sh /absolute/path/to/python [args...]' >&2
    exit 2
fi
python_bin="$1"
shift
site_packages="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cuda_libraries=""
for library_dir in "$site_packages"/nvidia/*/lib; do
    if [[ -d "$library_dir" ]]; then
        cuda_libraries="${cuda_libraries:+$cuda_libraries:}$library_dir"
    fi
done
if [[ -n "$cuda_libraries" ]]; then
    export LD_LIBRARY_PATH="$cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# pip provides ptxas in the cuda_nvcc package for TensorFlow/XLA kernels.
export PATH="$(dirname "$python_bin"):$site_packages/nvidia/cuda_nvcc/bin:$PATH"
exec "$python_bin" "$@"
