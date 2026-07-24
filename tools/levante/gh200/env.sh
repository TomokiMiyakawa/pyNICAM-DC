# Shared environment for pyNICAM-DC on Levante dolpung (GH200) nodes.
#   source /work/bb1153/b381557/workclaude/gh200/env.sh
# Assumes the caller already has a pristine environment (env -i / #SBATCH --export=NONE);
# see setup_venv.sh for why the login node's exported conda toolchain must not survive.
WORK=/work/bb1153/b381557/workclaude
CODE=$WORK/pyNICAM-DC
VENV=$WORK/venv-gh200
MPI=/opt/mpi/openmpi/5.0.6.1.6

export PATH="$MPI/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
# Levante's OpenMPI is built with wrapper rpath/runpath disabled, so every dependency
# has to be on LD_LIBRARY_PATH by hand: libmpi for mpi4py's import, and libprrte/libpmix
# for mpirun itself (which otherwise dies with "libprrte.so.3: cannot open shared
# object file"). The PMIx/PRRTE versions are the ones OpenMPI 5.0.6 was configured with.
export LD_LIBRARY_PATH="$MPI/lib:/opt/prrte/3.0.8/lib:/opt/pmix/5/lib"
unset PYTHONHOME
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi.so"
export MPI4JAX_NO_WARN_JAX_VERSION=1   # jax 0.10.2 vs mpi4jax's declared max 0.10.0
