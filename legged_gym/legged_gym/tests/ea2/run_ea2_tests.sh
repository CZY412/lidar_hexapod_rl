#!/usr/bin/env bash
# Run all envelope_adaptive_2 unit tests with the project conda env.
set -euo pipefail

PY="${EA2_PYTHON:-/home/t3chichi/anaconda3/envs/el4090/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/../.."

# Isaac Gym's gymtorch JIT build needs ninja; CUDA headers/binaries are in /usr/local/cuda.
export PATH="/usr/local/cuda-11.7/bin:/home/t3chichi/anaconda3/envs/el4090/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

exec "${PY}" -m pytest tests/ea2 -q "$@"
