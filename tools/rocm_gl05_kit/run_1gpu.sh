#!/bin/bash
# STAGE 1: single AMD GPU smoke -- prove jax-ROCm runs the pyNICAM-DC dycore.
# gl05rl00 pe1, IDEAL Jablonowski, 8 steps. NO MPI, NO RCCL. This isolates the
# ROCm jax stack from the collective layer. Dumps the final PRG_var for validate.py.
#
#   VENV=/path/venv-rocm bash run_1gpu.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${VENV:?set VENV to the jax-rocm venv}"
BACKEND="${BACKEND:-jax}"   # 'numpy' for the CPU reference leg (validate.py uses it)

source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

RUN="$HERE/run_1gpu_$BACKEND"
rm -rf "$RUN"; mkdir -p "$RUN"
ln -sfn "$REPO/tutorial/case" "$RUN/case"
cp "$HERE/configs/nhm_1gpu.toml" "$RUN/nhm_1gpu.toml"
cp "$HERE/configs/drv_1gpu$([ "$BACKEND" = numpy ] && echo _numpy).toml" "$RUN/drv.toml"
cd "$RUN"

echo "=== $(hostname)  backend=$BACKEND  HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) ==="
[ "$BACKEND" = jax ] && python -c 'import jax; print("jax", jax.__version__, jax.devices())'
export PYNICAM_TIMELOOP_DUMP="$RUN/fin"
python "$REPO/pynicamdc/nhm/driver/driver-dc.py" --driver-setting ./drv.toml 2>&1 | tee run.log
echo "  exit=${PIPESTATUS[0]}  peacefully-done=$(grep -c 'peacefully done' run.log)"
ls -la "$RUN"/fin_rank*.npy 2>/dev/null
