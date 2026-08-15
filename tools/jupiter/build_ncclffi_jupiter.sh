#!/bin/bash
# JUPITER (JSC) variant of tools/ncclffi/build_ncclffi.sh.
#
# Same reasoning as the Levante variant: ncclffi.cu contains NO device code
# (no __global__/__device__, no <<<>>>), so plain g++ against the pip CUDA
# headers is sound and we do not need an nvcc driver binary.
#
# JUPITER differences vs Levante:
#  - python comes from the EasyBuild Python module, NOT a self-contained
#    miniforge. It links against libpython3.13.so.1.0, so the module
#    environment (LD_LIBRARY_PATH) must STAY loaded -- the Levante script's
#    `unset LD_LIBRARY_PATH` breaks the interpreter here.
#  - the login node is aarch64 GH200 (same arch as the compute nodes), so this
#    can be built on the login node; no job needed.
#
#   bash tools/jupiter/build_ncclffi_jupiter.sh
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$(dirname "$0")/../ncclffi"

FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
NCCL_DIR="$NV/nccl"
CUDART_DIR="$NV/cuda_runtime"

g++ --version | head -1
# libnccl.so.2 / libcudart.so.12 have no .so symlinks in the pip layout -> -l: the
# versioned names. Link against the SAME NCCL that jax loads (the pip one), and
# rpath it so the loader cannot prefer a system/module NCCL instead.
g++ -shared -std=c++17 -O3 -fPIC -x c++ \
    -I"$FFI_INC" -I"$NCCL_DIR/include" -I"$CUDART_DIR/include" \
    -I"$NV/cuda_nvcc/include" -I"$NV/cuda_cccl/include" ncclffi.cu \
    -L"$NCCL_DIR/lib" -l:libnccl.so.2 \
    -L"$CUDART_DIR/lib" -l:libcudart.so.12 \
    -Wl,-rpath,"$NCCL_DIR/lib" -Wl,-rpath,"$CUDART_DIR/lib" \
    -o libncclffi.so
echo "built: $(ls -la libncclffi.so)"
ldd libncclffi.so | grep -E 'nccl|cudart|not found' || true
if ldd libncclffi.so | grep -qE '(/lib64/libnccl|not found)'; then
  echo "ERROR: did not link against the venv's pip NCCL" >&2; exit 3
fi
