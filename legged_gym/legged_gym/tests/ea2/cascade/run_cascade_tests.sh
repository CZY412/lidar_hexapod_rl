#!/usr/bin/env bash
# Run all envelope_cascade_83 tests with the project conda env.
#
# Layout:
#   1. pytest (contracts + perception, no Isaac sim)
#   2. env integration, main variant          (Isaac, one env per process)
#   3. env integration, --ea2-off variant
#   4. env integration, --birth-condition midpoint variant
#
# Full ea2 regression suite: tests/ea2/run_ea2_tests.sh (includes 1).
set -euo pipefail

PY="${EA2_PYTHON:-/home/t3chichi/anaconda3/envs/el4090/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/../../.."

# Isaac Gym's gymtorch JIT build needs ninja; CUDA headers/binaries are in /usr/local/cuda.
export PATH="/usr/local/cuda-11.7/bin:/home/t3chichi/anaconda3/envs/el4090/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "===== [1/4] pytest: contracts + perception ====="
"${PY}" -m pytest tests/ea2/cascade -q

echo "===== [2/4] env integration: main ====="
"${PY}" tests/ea2/cascade/run_cascade_env_test.py

echo "===== [3/4] env integration: ea2-off fallback ====="
"${PY}" tests/ea2/cascade/run_cascade_env_test.py --ea2-off

echo "===== [4/4] env integration: birth-condition midpoint ====="
"${PY}" tests/ea2/cascade/run_cascade_env_test.py --birth-condition midpoint

echo "===== ALL CASCADE TESTS PASSED ====="
