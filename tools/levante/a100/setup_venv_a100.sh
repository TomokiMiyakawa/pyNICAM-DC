#!/bin/bash
# Build the x86_64 venv for pyNICAM-DC on Levante's A100 partition (`gpu`, a100_80:4).
# Runs directly on the (x86) login node -- unlike the GH200 case, no srun needed.
# Mirrors gh200/setup_venv.sh; differences are marked A100:.
set -euo pipefail

# Clean env (same trap as GH200 #1: the login profile exports a conda build
# toolchain -- x86-compatible here, but -march=nocona junk and conda sysroots
# make builds irreproducible. Re-exec pristine.)
if [ "${PNC_CLEANENV:-0}" != "1" ]; then
  exec /usr/bin/env -i PNC_CLEANENV=1 \
    HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" \
    TERM="${TERM:-dumb}" SHELL=/bin/bash TMPDIR=/scratch/b/b381557/pnc-a100/tmp \
    PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin \
    /bin/bash --noprofile --norc "$0" "$@"
fi

WORK=/work/bb1153/b381557/workclaude
CODE=$WORK/pyNICAM-DC
VENV=/scratch/b/b381557/pnc-a100/venv-a100   # /work/bb1153 is at 100% project quota (2026-07-24) -> scratch
FORGE=/scratch/b/b381557/pnc-a100/miniforge
# A100: stable spack OpenMPI (gcc build), addressed by prefix like /opt/mpi on dolpung
MPI=/sw/spack-levante/openmpi-4.1.2-mnmady

echo "=== host: $(hostname) $(uname -m) ==="

# --- 1. python 3.11 (x86_64) via miniforge --------------------------------
if [ ! -x "$FORGE/bin/conda" ]; then
  echo "--- installing miniforge (x86_64) ---"
  cd /scratch/b/b381557/pnc-a100
  [ -f miniforge.sh ] || curl -fsSL -o miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  rm -rf "$FORGE"
  bash miniforge.sh -b -p "$FORGE"
fi
[ -x "$FORGE/bin/python3.11" ] || "$FORGE/bin/conda" install -y -q -p "$FORGE" python=3.11
"$FORGE/bin/python3.11" --version

# --- 2. venv ---------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
  "$FORGE/bin/python3.11" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip install -q --upgrade pip setuptools wheel

# --- 3. requirements minus mpi4py / mpi4jax --------------------------------
REQ=/scratch/b/b381557/pnc-a100/req-nompi.txt
grep -v -E '^(mpi4py|mpi4jax)==' "$CODE/tools/levante/requirements-gh200.txt" > "$REQ"
pip install -r "$REQ"

# --- 4. mpi4py against spack OpenMPI (source build) ------------------------
echo "--- mpi4py 4.1.2 from source against $MPI ---"
export PATH="$MPI/bin:$PATH"
export MPICC="$MPI/bin/mpicc"
"$MPICC" --version | head -1
pip install --no-binary=mpi4py mpi4py==4.1.2

# --- 5. mpi4jax (nanobind is a build-time dep, trap #6) --------------------
pip install nanobind
pip install --no-build-isolation mpi4jax==0.9.0.post1

# --- 6. smoke test (login node = CPU only; GPU check happens on a gpu node) --
export LD_LIBRARY_PATH="$MPI/lib"
echo "=== smoke (cpu-side) ==="
JAX_PLATFORMS=cpu python - <<'PY'
import jax, mpi4py, numpy
from mpi4py import MPI
print("jax        ", jax.__version__, jax.devices())
print("mpi4py     ", mpi4py.__version__, MPI.Get_library_version().split("\n")[0])
import mpi4jax; print("mpi4jax    ", mpi4jax.__version__)
print("numpy      ", numpy.__version__)
PY
echo "VENV=$VENV"
