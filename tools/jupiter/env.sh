# JUPITER (JSC) runtime env for pyNICAM-DC. Source this and nothing else:
#   source tools/jupiter/env.sh
# Then `source $PYNICAM_CODE/config/production.env` for the validated GPU gate set.
#
# Unlike the Levante/Miyabi kits this does the FULL setup (modules + venv + paths),
# because on JUPITER the module stack is not optional: numpy/scipy/mpi4py/pytest
# come from EasyBuild, NOT from the venv (see PORT.md trap #2).
#
# FILESYSTEM (PORT.md trap #1): this repo and the venv MUST live under /e -- the
# legacy /p GPFS is mounted on LOGIN NODES ONLY. A batch job whose script, venv,
# run dir or --output path sits under /p dies in ~10 s with FAILED 0:53 and *no
# output file*, because slurmstepd cannot even chdir there.
#
# Override PYNICAM_VENV before sourcing to use a different venv.

_JUPITER_KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNICAM_CODE="${PYNICAM_CODE:-$(cd "$_JUPITER_KIT_DIR/../.." && pwd)}"
export PYNICAM_WORK="${PYNICAM_WORK:-$(cd "$PYNICAM_CODE/.." && pwd)}"
export PYNICAM_VENV="${PYNICAM_VENV:-$PYNICAM_WORK/venv-jupiter}"

case "$PYNICAM_CODE" in
  /p/*) echo "WARNING: PYNICAM_CODE is under /p -- invisible on compute nodes." \
             "Batch jobs will fail with FAILED 0:53 and no log. See PORT.md trap #1." >&2 ;;
esac

module purge >/dev/null 2>&1
module load Stages/2026
module load GCC/14.3.0
module load ParaStationMPI/5.13.0-1     # MPI-settings/CUDA is the default variant
module load mpi4py/4.1.0
module load SciPy-bundle/2025.07        # numpy/scipy -- reachable ONLY via PYTHONPATH

source "$PYNICAM_VENV/bin/activate"

# PREPEND, never assign (PORT.md trap #2): assigning drops the module
# site-packages and numpy/mpi4py/pytest vanish.
export PYTHONPATH="$PYNICAM_CODE${PYTHONPATH:+:$PYTHONPATH}"

# jax dlopen's NCCL/CUDA from the pip wheels; put them on the loader path.
NVLIBS=$(python -c "import nvidia,glob,os;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(glob.glob(b+'/*/lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"

export PYNICAM_NCCLFFI_LIB="$PYNICAM_CODE/tools/ncclffi/libncclffi.so"
export MPI4JAX_NO_WARN_JAX_VERSION=1    # jax 0.10.2 vs mpi4jax's declared 0.10.0 --
                                        # the same pairing as the validated Miyabi/Levante stack.
# NB srun ONLY on JUPITER -- mpirun/mpiexec do not exist.
