#!/bin/bash
# STAGE 2 (alt / fallback): 8 AMD GPUs via the DEFAULT mpi4jax comm path (device
# -resident alltoall over ROCm-aware MPI) -- this path does NOT use RCCL. Use it to
# (a) get multi-GPU running before the RCCL lib is built, and (b) A/B the RCCL path
# against it (should be bit-identical: same plan, only the wire transport differs).
# Needs mpi4jax in the venv and a ROCm/GPU-aware MPI (UCX with ROCm support).
#
#   VENV=/path/venv-rocm bash run_8gpu_mpi4jax.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${VENV:?set VENV to the jax-rocm venv}"
PE="${PE:-8}"

source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

RUN="$HERE/run_8gpu_mpi4jax"
rm -rf "$RUN"; mkdir -p "$RUN"
ln -sfn "$REPO/pynicamdc/test/case2" "$RUN/case"
cp "$HERE/configs/nhm_8gpu.toml" "$RUN/nhm_8gpu.toml"
cp "$HERE/configs/drv_8gpu.toml" "$RUN/drv.toml"
cd "$RUN"

# default path: alltoall on device via mpi4jax (NCCLFFI OFF).
unset PYNICAM_COMM_NCCLFFI
export PYNICAM_COMM_ALLTOALL=1
export PYNICAM_TIMELOOP_DUMP="$RUN/fin"
# mpi4jax over GPU-aware MPI: keep 0 unless UCX ROCm RMA is confirmed working.
export MPI4JAX_USE_CUDA_MPI=${MPI4JAX_USE_CUDA_MPI:-0}

echo "=== $(hostname)  mpi4jax path  pe=$PE ==="
python -c 'import jax; print("jax", jax.__version__, jax.devices())'
X=(-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH)
for v in $(compgen -v | grep -E '^(PYNICAM_|MPI4JAX_|XLA_PYTHON_|XLA_FLAGS|JAX_PLATFORMS|HSA_)'); do X+=(-x "$v"); done
mpirun "${X[@]}" -np "$PE" "$HERE/bind_rocm.sh" \
    python "$REPO/pynicamdc/nhm/driver/driver-dc.py" --driver-setting ./drv.toml 2>&1 | tee run.log
echo "  exit=${PIPESTATUS[0]}  peacefully-done=$(grep -c 'peacefully done' run.log)/$PE"
ls "$RUN"/fin_rank*.npy 2>/dev/null | wc -l | xargs echo "  dumps:"
