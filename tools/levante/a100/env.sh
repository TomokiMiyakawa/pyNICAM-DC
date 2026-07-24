# Shared environment for pyNICAM-DC on Levante A100 (`gpu` partition, x86_64).
#   source /work/bb1153/b381557/workclaude/a100/env.sh
# Mirror of gh200/env.sh. Assumes a pristine environment (env -i / --export=NONE).
WORK=/work/bb1153/b381557/workclaude
CODE=$WORK/pyNICAM-DC
VENV=/scratch/b/b381557/pnc-a100/venv-a100
MPI=/sw/spack-levante/openmpi-4.1.2-mnmady

export PATH="$MPI/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH="$MPI/lib"      # spack OpenMPI: no wrapper rpath here either
unset PYTHONHOME
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
# x86 build of the FFI lib -- NOT tools/ncclffi/libncclffi.so (that one is aarch64)
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi_x86.so"
export MPI4JAX_NO_WARN_JAX_VERSION=1
