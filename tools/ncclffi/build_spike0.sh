#!/bin/bash
# Build the N0 spike FFI lib. Login-node OK (compile only). Needs nvidia/25.9 for nvcc.
set -euo pipefail
cd "$(dirname "$0")"
VENV=/work/gj37/c24028/workforclaude/venv-gh200
source "$VENV/bin/activate"
module load nvidia/25.9 2>/dev/null || true
FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
# GH200 = sm_90; g++11 host compiler; XLA FFI headers need C++17.
nvcc -shared -std=c++17 -arch=sm_90 -Xcompiler -fPIC \
     -I"$FFI_INC" spike0.cu -o libspike0.so
echo "built: $(ls -la libspike0.so)"
