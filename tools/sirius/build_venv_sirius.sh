#!/bin/bash
# Build the jax-ROCm venv for pyNICAM-DC on Sirius (CCS Tsukuba, AMD MI300A APU, gfx942).
# Run on the LOGIN node (compute nodes reach PyPI only via the 10.20.96.11:8080 proxy).
#
#   bash tools/sirius/build_venv_sirius.sh            # -> $VENV (default below)
#
# jax's "jax[rocm]" extra does not exist for jax>=0.7 and pip only WARNS about an
# unknown extra, leaving a CPU-only jax -- so the ROCm plugin wheels are pinned here.
set -euo pipefail
VENV="${VENV:-/work/PYNICAM/miyakawa/forclaude1/venv-rocm}"
JAXVER="${JAXVER:-0.11.1}"
ROCM_MOD="${ROCM_MOD:-rocm/7.2.0}"
MPI_MOD="${MPI_MOD:-openmpi/5.0.10/rocm7.2.0}"

source /usr/share/Modules/init/bash
module purge
module load "$ROCM_MOD" "$MPI_MOD"
module list 2>&1

python3.12 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

echo "=== core deps ==="
pip install -q "numpy>=1.24" "toml>=0.10.2" "zarr<3" xarray dask pytest

echo "=== jax $JAXVER + ROCm 7 plugin ==="
pip install -q "jax==$JAXVER" "jax-rocm7-plugin==$JAXVER" "jax-rocm7-pjrt==$JAXVER"

# Sirius has no python3.12-devel (no Python.h), so nothing can be compiled against
# CPython. mpi4py>=4 binary wheels dlopen libmpi at run time and work with the
# openmpi module (verified: Open MPI v5.0.10). mpi4jax needs a source build -> skipped.
echo "=== mpi4py (binary wheel; runtime-linked to $MPI_MOD) ==="
pip install -q "mpi4py>=4"
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])'

echo "=== pynicamdc (editable) ==="
pip install -q -e "$(cd "$(dirname "$0")/../.." && pwd)"

pip list 2>/dev/null | grep -Ei '^(jax|numpy|mpi4py|mpi4jax|zarr|xarray|dask|pynicamdc)'
python -c 'import jax; print("jax", jax.__version__, "(login node: no GPU expected)", jax.devices())'
echo "=== done: $VENV ==="
