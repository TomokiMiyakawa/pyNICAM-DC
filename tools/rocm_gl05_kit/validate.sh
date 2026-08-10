#!/bin/bash
# Stage-1 validation: run the gl05rl00 pe1 case on BOTH the numpy CPU reference and
# the jax-ROCm GPU backend (identical IDEAL IC), then compare final PRG_var.
# PASS = jax-ROCm within rtol 1e-9 of numpy (cross-backend libm/reduction floor).
#
#   VENV=/path/venv-rocm bash validate.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:?set VENV to the jax-rocm venv}"

echo "########## reference leg: numpy (CPU) ##########"
BACKEND=numpy VENV="$VENV" bash "$HERE/run_1gpu.sh"
echo "########## candidate leg: jax (ROCm GPU) ##########"
BACKEND=jax   VENV="$VENV" bash "$HERE/run_1gpu.sh"

echo; echo "########## COMPARE (rtol 1e-9) ##########"
source "$VENV/bin/activate"
python "$HERE/validate.py" \
    --ref  "$HERE/run_1gpu_numpy/fin_rank*.npy" \
    --cand "$HERE/run_1gpu_jax/fin_rank*.npy" \
    --rtol 1e-9
