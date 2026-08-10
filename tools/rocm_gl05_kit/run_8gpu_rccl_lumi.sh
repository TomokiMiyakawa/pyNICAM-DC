#!/bin/bash
# STAGE 2b on LUMI: 8 GCDs (4x MI250X, gfx90a) on ONE node, gl05rl01 pe8, IDEAL
# Jablonowski, halo exchange routed through our RCCL communicator (ncclffi_hip.cpp).
# srun variant of run_8gpu_rccl.sh -- must run inside an allocation (sbatch/salloc).
#
#   source lumi_env.sh && bash run_8gpu_rccl_lumi.sh
#
# BACKEND=numpy runs the same pe8 case on the CPU with plain mpi4py sendrecv: that is
# the reference leg for the correctness A/B (Stage 2a / mpi4jax is deliberately not
# used on LUMI). Cross-backend, so the floor is ~1e-9, not bit-identical.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="${VENV:?source lumi_env.sh first}"
PE="${PE:-8}"
BACKEND="${BACKEND:-jax}"

source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

if [ "$BACKEND" = jax ]; then
    # RUNTAG keeps concurrent diagnostic variants in separate dirs (they would
    # otherwise all rm -rf and rewrite the same run_8gpu_rccl/).
    RUN="$HERE/run_8gpu_rccl${RUNTAG:+_$RUNTAG}"
    RCCL_LIB="${PYNICAM_NCCLFFI_LIB:-$REPO/tools/ncclffi/rocm/libncclffi.so}"
    [ -f "$RCCL_LIB" ] || { echo "xxx RCCL FFI lib missing: $RCCL_LIB (build_ncclffi_rocm.sh OFFLOAD_ARCH=gfx90a)"; exit 1; }
    export PYNICAM_COMM_NCCLFFI=1        # wire = our RCCL comm
    export PYNICAM_COMM_ALLTOALL=1       # (default) the path NCCLFFI swaps into
    export PYNICAM_FFI_PLATFORM=ROCM
    export PYNICAM_NCCLFFI_LIB="$RCCL_LIB"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    DRV=drv_8gpu.toml
    BIND="$HERE/bind_lumi.sh"
else
    RUN="$HERE/run_8cpu_numpy"
    unset PYNICAM_COMM_NCCLFFI
    DRV=drv_8gpu_numpy.toml
    BIND=""                              # CPU reference leg: no GPU to bind
fi

rm -rf "$RUN"; mkdir -p "$RUN"
ln -sfn "$REPO/pynicamdc/test/case2" "$RUN/case"
cp "$HERE/configs/nhm_8gpu.toml" "$RUN/nhm_8gpu.toml"
cp "$HERE/configs/$DRV" "$RUN/drv.toml"
cd "$RUN"

export PYNICAM_TIMELOOP_DUMP="$RUN/fin"

echo "=== $(hostname)  backend=$BACKEND  pe=$PE  HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
[ "$BACKEND" = jax ] && echo "    lib=$PYNICAM_NCCLFFI_LIB"

srun --ntasks="$PE" --ntasks-per-node="$PE" --cpus-per-task=7 --export=ALL \
     $BIND python "$REPO/pynicamdc/nhm/driver/driver-dc.py" \
     --driver-setting ./drv.toml 2>&1 | tee run.log
RC=${PIPESTATUS[0]}

echo "  exit=$RC  peacefully-done=$(grep -c 'peacefully done' run.log)/$PE"
grep -E 'NCCLFFI: comm up' run.log | head -1
ls "$RUN"/fin_rank*.npy 2>/dev/null | wc -l | xargs echo "  dumps:"
exit $RC
