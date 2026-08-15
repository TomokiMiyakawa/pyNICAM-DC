#!/bin/bash
# Build the pyNICAM-DC venv on JUPITER (JSC).  Login node is fine -- it is aarch64
# with a GH200, same arch as compute (no x86 contamination problem like Levante).
#
#   bash tools/jupiter/setup_venv.sh /e/project1/<proj>/<user>/venv-jupiter
#
# TWO-LAYER BY DESIGN (PORT.md trap #2):
#   modules -> numpy, scipy, mpi4py, pytest      (reachable ONLY via PYTHONPATH)
#   venv    -> zarr<3, numcodecs, xarray, dask + the whole jax stack
# So the venv is created with --system-site-packages, and NOTHING here should
# pip-install numpy/mpi4py: mpi4py must stay the module build that matches
# ParaStationMPI, and pip would silently no-op on numpy anyway.
#
# PUT THE VENV UNDER /e. The legacy /p GPFS is login-node-only; a venv under /p is
# invisible to every batch job (PORT.md trap #1). A venv also cannot be MOVED
# afterwards -- pyvenv.cfg/activate/shebangs bake in absolute paths.
set -euo pipefail

VENV="${1:?usage: setup_venv.sh /e/path/to/venv-jupiter}"
case "$VENV" in
  /e/*) ;;
  *) echo "REFUSING: venv must be under /e (got $VENV) -- see PORT.md trap #1" >&2; exit 2 ;;
esac

module purge >/dev/null 2>&1
module load Stages/2026
module load GCC/14.3.0
module load ParaStationMPI/5.13.0-1
module load mpi4py/4.1.0
module load SciPy-bundle/2025.07

python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# --- io stack (zarr MUST stay <3) ---
pip install "zarr==2.18.7" "numcodecs==0.15.1" "xarray==2026.7.0" "dask==2026.7.1"

# --- jax stack. jax[cuda12] pulls jaxlib + the nvidia-* wheels (~5 GB); the pip
#     NCCL is the copy jax loads and the copy libncclffi.so must link against. ---
pip install "jax[cuda12]==0.10.2"

# --- mpi4jax: nanobind is a BUILD-time dep and --no-build-isolation means it must
#     pre-exist (Levante trap #6). Built against the MODULE mpi4py, not a pip one. ---
export MPICC="$(command -v mpicc)"
pip install "nanobind==2.13.0"
pip install --no-build-isolation "mpi4jax==0.9.0.post1"

echo
echo "=== verify ==="
python - <<'PY'
import jax, jaxlib, mpi4jax, mpi4py, numpy, zarr
print("jax", jax.__version__, "| jaxlib", jaxlib.__version__,
      "| mpi4jax", mpi4jax.__version__, "| mpi4py", mpi4py.__version__,
      "| numpy", numpy.__version__, "| zarr", zarr.__version__)
PY
echo
echo "next: bash tools/jupiter/build_ncclffi_jupiter.sh   (needs this venv active)"
