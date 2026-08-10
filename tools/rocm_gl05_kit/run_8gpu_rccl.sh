#!/bin/bash
# STAGE 2 (RCCL): 8 AMD GPUs, gl05rl01 pe8, IDEAL Jablonowski, 8 steps, with the
# halo exchange routed through OUR RCCL communicator (the hipified ncclffi lib).
# This is the run that actually exercises RCCL. Requires:
#   1) tools/ncclffi/build_ncclffi_rocm.sh already built -> libncclffi.so (rocm/)
#   2) mpi4py in the venv (uid bcast bootstrap)
#
#   VENV=/path/venv-rocm bash run_8gpu_rccl.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${VENV:?set VENV to the jax-rocm venv}"
PE="${PE:-8}"
RCCL_LIB="${PYNICAM_NCCLFFI_LIB:-$REPO/tools/ncclffi/rocm/libncclffi.so}"
[ -f "$RCCL_LIB" ] || { echo "xxx RCCL FFI lib missing: $RCCL_LIB (run build_ncclffi_rocm.sh)"; exit 1; }

source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

RUN="$HERE/run_8gpu_rccl"
rm -rf "$RUN"; mkdir -p "$RUN"
ln -sfn "$REPO/pynicamdc/test/case2" "$RUN/case"
cp "$HERE/configs/nhm_8gpu.toml" "$RUN/nhm_8gpu.toml"
cp "$HERE/configs/drv_8gpu.toml" "$RUN/drv.toml"
cd "$RUN"

# RCCL-FFI comm path + ROCm FFI target + our hipified lib.
export PYNICAM_COMM_NCCLFFI=1
export PYNICAM_FFI_PLATFORM=ROCM
export PYNICAM_NCCLFFI_LIB="$RCCL_LIB"
export PYNICAM_TIMELOOP_DUMP="$RUN/fin"
# RCCL over the box interconnect: let it pick; uncomment to pin an IB HCA.
# export NCCL_IB_HCA=... NCCL_SOCKET_IFNAME=...

echo "=== $(hostname)  RCCL path  pe=$PE  lib=$RCCL_LIB ==="
python -c 'import jax; print("jax", jax.__version__, jax.devices())'

# forward the env the ranks need; per-rank GPU via bind_rocm.sh (HIP_VISIBLE_DEVICES)
X=(-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH)
for v in $(compgen -v | grep -E '^(PYNICAM_|XLA_PYTHON_|XLA_FLAGS|JAX_PLATFORMS|NCCL_|HSA_)'); do X+=(-x "$v"); done
mpirun "${X[@]}" -np "$PE" "$HERE/bind_rocm.sh" \
    python "$REPO/pynicamdc/nhm/driver/driver-dc.py" --driver-setting ./drv.toml 2>&1 | tee run.log
echo "  exit=${PIPESTATUS[0]}  peacefully-done=$(grep -c 'peacefully done' run.log)/$PE"
grep -E 'NCCLFFI: comm up|peacefully done' run.log | head
ls "$RUN"/fin_rank*.npy 2>/dev/null | wc -l | xargs echo "  dumps:"
