#!/bin/bash
# Build the aarch64 venv for pyNICAM-DC on Levante GH200 (dolpung) nodes.
# MUST run on a dolpung node: the login node is x86_64.
#   srun -p dolpung -A mh1571_gpu -N1 -n1 --gpus=1 -t 2:00:00 \
#        bash --noprofile --norc /work/bb1153/b381557/workclaude/gh200/setup_venv.sh
set -euo pipefail

# --- clean env ---------------------------------------------------------------
# The login node is x86_64 and its profile activates an x86 mambaforge, exporting a
# whole conda build toolchain (CC/CXX/CFLAGS=-march=nocona/_CONDA_PYTHON_SYSCONFIGDATA_
# NAME/...). srun inherits all of it onto the aarch64 node, where source builds then
# invoke x86_64-conda-linux-gnu-cc and die with "Exec format error". Unsetting the
# variables one by one is whack-a-mole -- re-exec under a pristine environment instead.
if [ "${PNC_CLEANENV:-0}" != "1" ]; then
  exec /usr/bin/env -i PNC_CLEANENV=1 \
    HOME="$HOME" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" \
    TERM="${TERM:-dumb}" SHELL=/bin/bash TMPDIR="${TMPDIR:-/tmp}" \
    PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin \
    /bin/bash --noprofile --norc "$0" "$@"
fi

WORK=/work/bb1153/b381557/workclaude
CODE=$WORK/pyNICAM-DC
VENV=$WORK/venv-gh200
FORGE=$WORK/gh200/miniforge
MPI=/opt/mpi/openmpi/5.0.6.1.6

echo "=== host: $(hostname) $(uname -m) ==="

# --- 1. python 3.11 (aarch64) via miniforge ---------------------------------
if [ ! -x "$FORGE/bin/conda" ]; then
  echo "--- installing miniforge (aarch64) ---"
  cd "$WORK/gh200"
  [ -f miniforge.sh ] || curl -fsSL -o miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  rm -rf "$FORGE"
  bash miniforge.sh -b -p "$FORGE"
fi
# miniforge ships 3.13; pin the interpreter to 3.11 to match the Miyabi venv.
# -p is mandatory: without it conda would fall back to CONDA_PREFIX.
[ -x "$FORGE/bin/python3.11" ] || "$FORGE/bin/conda" install -y -q -p "$FORGE" python=3.11
"$FORGE/bin/python3.11" --version

# --- 2. venv ----------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
  echo "--- creating venv $VENV ---"
  "$FORGE/bin/python3.11" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V
pip install -q --upgrade pip setuptools wheel

# --- 3. requirements minus mpi4py / mpi4jax ---------------------------------
REQ=$WORK/gh200/req-nompi.txt
grep -v -E '^(mpi4py|mpi4jax)==' "$CODE/tools/levante/requirements-gh200.txt" > "$REQ"
echo "--- pip install requirements (no mpi) ---"
pip install -r "$REQ"

# --- 4. mpi4py against Levante's OpenMPI 5 (source build) -------------------
echo "--- mpi4py 4.1.2 from source against $MPI ---"
export PATH="$MPI/bin:$PATH"
export MPICC="$MPI/bin/mpicc"
"$MPICC" --version | head -1
pip install --no-binary=mpi4py mpi4py==4.1.2

# --- 5. mpi4jax (needs mpi4py + jax already present) ------------------------
# --no-build-isolation means the build sees only what is already in the venv, so
# mpi4jax's build deps must be installed by hand. nanobind is NOT in
# requirements-gh200.txt (it is a build-time-only dep) -- without it the build dies
# with "Building mpi4jax requires mpi4py and nanobind".
echo "--- mpi4jax 0.9.0.post1 (+ nanobind build dep) ---"
pip install nanobind
pip install --no-build-isolation mpi4jax==0.9.0.post1

# --- 6. smoke test ----------------------------------------------------------
# Levante's OpenMPI is built --enable-wrapper-rpath=no --enable-wrapper-runpath=no, so
# nothing records the library location: `import mpi4py.MPI` fails with
# "libmpi.so.40: cannot open shared object file" unless LD_LIBRARY_PATH points at it.
# The job templates must export this too.
export LD_LIBRARY_PATH="$MPI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== smoke ==="
python - <<'PY'
import jax, mpi4py, numpy
from mpi4py import MPI
print("jax        ", jax.__version__)
print("devices    ", jax.devices())
print("mpi4py     ", mpi4py.__version__, MPI.Get_library_version().split("\n")[0])
import mpi4jax; print("mpi4jax    ", mpi4jax.__version__)
print("numpy      ", numpy.__version__)
PY
echo "VENV=$VENV"
