#!/bin/bash
# Miyabi-G venv build (aarch64 GH200). Usage:
#   bash tools/miyabi/setup_venv.sh /path/to/venv-gh200
# Login node OK (same arch as compute). Staged so an io-stack failure cannot
# take down the GPU-critical stack; each stage is verified before moving on.
# Frozen reference versions: tools/levante/requirements-gh200.txt (jax 0.10.2).
set -uo pipefail
VENV="${1:?usage: setup_venv.sh /path/to/venv}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module load python/3.11.15 nvidia/25.9 cuda/12.6 >/dev/null 2>&1
[ -d "$VENV" ] || python3 -m venv "$VENV"
export MPICC="$(command -v mpicc)"   # HPC-X mpicc from nvidia/25.9
source "$VENV/bin/activate"
pip install -U pip >/dev/null

echo "### STAGE 1: jax[cuda12] ALONE"
pip install "jax[cuda12]" 2>&1 | tail -4
python -c "import jax,jaxlib; print('STAGE1 OK jax',jax.__version__,'jaxlib',jaxlib.__version__)" || { echo "STAGE1 FAIL"; exit 1; }

echo "### STAGE 2: mpi4py (no-binary, against HPC-X mpicc=$MPICC)"
pip install --no-binary=mpi4py mpi4py 2>&1 | tail -4
python -c "import mpi4py; print('STAGE2 OK mpi4py',mpi4py.__version__)" || { echo "STAGE2 FAIL"; exit 1; }

echo "### STAGE 3: mpi4jax==0.9.0.post1 (fallback to latest if the pin fails)"
pip install "mpi4jax==0.9.0.post1" 2>&1 | tail -6 \
  && python -c "import mpi4jax; print('STAGE3 OK mpi4jax',mpi4jax.__version__)" \
  || { echo "## pin failed -> trying unpinned"; pip install -U mpi4jax 2>&1 | tail -6; \
       python -c "import mpi4jax; print('STAGE3 OK mpi4jax(unpinned)',mpi4jax.__version__)" || echo "STAGE3 import FAIL"; }

echo "### STAGE 4: io stack (toml xarray dask numpy) -- has wheels"
pip install toml xarray dask numpy 2>&1 | tail -3
echo "### STAGE 4b: zarr<3 (REQUIRED: model io targets the zarr-2 API)"
pip install "zarr<3" 2>&1 | tail -8 \
  && python -c "import zarr; print('STAGE4 OK zarr',zarr.__version__)" \
  || echo "STAGE4 zarr FAIL (fix separately; GPU stack unaffected)"

echo "### STAGE 5: ncclffi extension (direct-NCCL halo transport; nvcc sm_90)"
# Required for PYNICAM_COMM_NCCLFFI=1 (default-on production transport);
# without it set PYNICAM_COMM_NCCLFFI=0 to fall back to mpi4jax alltoall.
VENV="$VENV" bash "$CODE/tools/ncclffi/build_ncclffi.sh" 2>&1 | tail -2 \
  || echo "STAGE5 ncclffi build FAIL (transport unavailable until built manually)"

echo "### FROZEN VERSIONS"
pip list 2>/dev/null | grep -iE "^jax|jaxlib|jax-cuda|mpi4py|mpi4jax|nvidia-cu|^zarr|xarray|dask|numpy|toml|numcodecs"
echo "### BUILD DONE -> venv: $VENV"
