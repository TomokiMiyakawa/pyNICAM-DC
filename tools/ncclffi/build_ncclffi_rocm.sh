#!/bin/bash
# ROCm/RCCL build of the NCCL-FFI halo-exchange lib (ncclffi_hip.cpp), the AMD
# counterpart of build_ncclffi.sh. RCCL is ABI-compatible with NCCL, so the same
# grouped ncclSend/ncclRecv source links against -lrccl. Output name is kept as
# libncclffi.so (a distinct dir) so no Python change is needed; point the model at
# it with PYNICAM_NCCLFFI_LIB=<this>/libncclffi.so.
#
#   VENV=... ROCM_PATH=/opt/rocm OFFLOAD_ARCH=gfx942 bash build_ncclffi_rocm.sh
#
# Compile-only; runs on the AMD box's login/build shell (needs hipcc + rccl-dev).
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:?set VENV to the jax-rocm venv (e.g. /path/venv-rocm)}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
# MI300/MI325 = gfx942, MI250/MI210 = gfx90a. Default matches this kit's target.
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx942}"
OUTDIR="${OUTDIR:-$PWD/rocm}"
mkdir -p "$OUTDIR"

source "$VENV/bin/activate"
HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
command -v "$HIPCC" >/dev/null || { echo "xxx hipcc not found ($HIPCC); load ROCm / set ROCM_PATH"; exit 1; }

FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")

# RCCL header may live at <rocm>/include/rccl/rccl.h or <rocm>/include/rccl.h.
RCCL_INC="$ROCM_PATH/include"
[ -f "$RCCL_INC/rccl/rccl.h" ] || [ -f "$RCCL_INC/rccl.h" ] || {
  echo "xxx rccl.h not found under $RCCL_INC (install rccl-dev / set ROCM_PATH)"; exit 1; }

echo "=== hipcc build: arch=$OFFLOAD_ARCH rocm=$ROCM_PATH ==="
"$HIPCC" -shared -std=c++17 --offload-arch="$OFFLOAD_ARCH" -fPIC \
     -I"$FFI_INC" -I"$RCCL_INC" \
     ncclffi_hip.cpp \
     -L"$ROCM_PATH/lib" -lrccl -Wl,-rpath,"$ROCM_PATH/lib" \
     -o "$OUTDIR/libncclffi.so"

echo "built: $(ls -la "$OUTDIR/libncclffi.so")"
ldd "$OUTDIR/libncclffi.so" | grep -Ei 'rccl|amdhip|not found' || true
echo
echo "point the model at it:  export PYNICAM_NCCLFFI_LIB=$OUTDIR/libncclffi.so"
