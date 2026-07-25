#!/bin/bash
# x86_64 build of the NCCL-FFI lib for Levante's A100 partition.
# Same approach as tools/levante/build_ncclffi_levante.sh (g++, no nvcc needed --
# ncclffi.cu has no device code), but outputs libncclffi_x86.so so the aarch64
# libncclffi.so in the shared clone is not clobbered. Runs on the x86 login node.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
unset PYTHONHOME LD_LIBRARY_PATH || true

VENV=${PYNICAM_VENV:-/scratch/b/b381557/pnc-a100/venv-a100}
cd "$(dirname "$0")/../pyNICAM-DC/tools/ncclffi"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")

g++ --version | head -1
g++ -shared -std=c++17 -O3 -fPIC -x c++ \
    -I"$FFI_INC" -I"$NV/nccl/include" -I"$NV/cuda_runtime/include" \
    -I"$NV/cuda_nvcc/include" -I"$NV/cuda_cccl/include" ncclffi.cu \
    -L"$NV/nccl/lib" -l:libnccl.so.2 \
    -L"$NV/cuda_runtime/lib" -l:libcudart.so.12 \
    -Wl,-rpath,"$NV/nccl/lib" -Wl,-rpath,"$NV/cuda_runtime/lib" \
    -o libncclffi_x86.so
echo "built: $(ls -la libncclffi_x86.so)"
ldd libncclffi_x86.so | grep -E 'nccl|cudart|not found' || true
if ldd libncclffi_x86.so | grep -vE "$NV" | grep -q 'libnccl'; then
  echo "ERROR: linked against a non-venv NCCL" >&2; exit 3
fi
