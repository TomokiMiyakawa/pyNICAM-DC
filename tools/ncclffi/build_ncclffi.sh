#!/bin/bash
# Build the N1 NCCL-FFI lib against the venv's own NCCL (same .so jax loads).
# Login-node OK (compile only). Needs nvidia/25.9 for nvcc.
set -euo pipefail
cd "$(dirname "$0")"
VENV=/work/gj37/c24028/workforclaude/venv-gh200
source "$VENV/bin/activate"
module load nvidia/25.9 2>/dev/null || true
FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
NCCL_DIR=$(python -c "import nvidia, os; print(os.path.join(os.path.dirname(nvidia.__file__), 'nccl'))")
# libnccl.so.2 has no .so symlink in the pip layout -> link the versioned name via -l:.
nvcc -shared -std=c++17 -arch=sm_90 -Xcompiler -fPIC \
     -I"$FFI_INC" -I"$NCCL_DIR/include" ncclffi.cu \
     -L"$NCCL_DIR/lib" -l:libnccl.so.2 -o libncclffi.so
echo "built: $(ls -la libncclffi.so)"
ldd libncclffi.so | grep -E 'nccl|not found' || true
