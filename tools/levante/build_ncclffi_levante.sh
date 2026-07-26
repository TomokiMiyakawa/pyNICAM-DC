#!/bin/bash
# Levante/GH200 variant of tools/ncclffi/build_ncclffi.sh.
#
# Two differences vs the Miyabi script:
#  1. No `module load nvidia/...` -- Levante's dolpung image ships only the CUDA
#     runtime (/usr/local/cuda-12.9 has lib64 but no bin), and the pip wheel
#     nvidia-cuda-nvcc-cu12 ships only ptxas + nvvm, NOT the nvcc driver binary.
#  2. So we compile with plain g++ instead of nvcc. This is sound because
#     ncclffi.cu contains no device code at all -- no __global__/__device__
#     functions and no <<<>>> launches, just host C++ calling the CUDA runtime
#     API and NCCL. -arch=sm_90 was therefore never doing anything for this TU.
#     (If device kernels are ever added here, this build must switch to a real
#     nvcc -- e.g. `conda install -c conda-forge cuda-nvcc=12.9` into the
#     miniforge prefix -- and restore -arch=sm_90.)
#
# MUST run on a dolpung node (aarch64); the login node is x86_64.
#   srun -p dolpung -A mh1571_gpu -N1 -n1 --gpus=1 -t 0:20:00 \
#        env -i HOME="$HOME" USER="$USER" TERM=dumb PATH=/usr/local/bin:/usr/bin \
#        bash --noprofile --norc tools/levante/build_ncclffi_levante.sh
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
unset PYTHONHOME LD_LIBRARY_PATH || true

VENV=${PYNICAM_VENV:-/work/bb1153/b381557/workclaude/venv-gh200}
cd "$(dirname "$0")/../ncclffi"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
NCCL_DIR="$NV/nccl"
CUDART_DIR="$NV/cuda_runtime"

g++ --version | head -1
# libnccl.so.2 / libcudart.so.12 have no .so symlinks in the pip layout -> -l: the
# versioned names. Link against the SAME NCCL that jax loads.
# The pip CUDA headers are split across three wheels: cuda_runtime has cuda_runtime.h,
# cuda_nvcc has crt/host_config.h, cuda_cccl has <nv/target> (pulled in by cuda_fp16.h
# via nccl.h). nvcc would add all of these implicitly; with g++ we must list them.
g++ -shared -std=c++17 -O3 -fPIC -x c++ \
    -I"$FFI_INC" -I"$NCCL_DIR/include" -I"$CUDART_DIR/include" \
    -I"$NV/cuda_nvcc/include" -I"$NV/cuda_cccl/include" ncclffi.cu \
    -L"$NCCL_DIR/lib" -l:libnccl.so.2 \
    -L"$CUDART_DIR/lib" -l:libcudart.so.12 \
    -Wl,-rpath,"$NCCL_DIR/lib" -Wl,-rpath,"$CUDART_DIR/lib" \
    -o libncclffi.so
echo "built: $(ls -la libncclffi.so)"
# Levante ships a system NCCL at /lib64/libnccl.so.2. Without the -rpath above the
# loader picks THAT one instead of the venv's pip NCCL -- the one jax actually uses.
# This must print the venv paths, not /lib64.
ldd libncclffi.so | grep -E 'nccl|cudart|not found' || true
if ldd libncclffi.so | grep -q '/lib64/libnccl'; then
  echo "ERROR: linked against the system NCCL, not the venv's" >&2; exit 3
fi
