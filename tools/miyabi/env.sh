# Miyabi-G runtime env for pyNICAM-DC. Source AFTER activating the venv:
#   module load nvidia/25.9
#   source $VENV/bin/activate
#   source tools/miyabi/env.sh
# Sets: LD_LIBRARY_PATH (jax pip CUDA libs), PYTHONPATH (repo), NCCLFFI lib.
# Then `source $CODE/config/production.env` for the validated GPU gate set.
_MIYABI_KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYNICAM_CODE="${PYNICAM_CODE:-$(cd "$_MIYABI_KIT_DIR/../.." && pwd)}"

NVLIBS=$(python -c "import nvidia,glob,os;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(glob.glob(b+'/*/lib'))))" 2>/dev/null || true)
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PYNICAM_CODE"
export PYNICAM_NCCLFFI_LIB="$PYNICAM_CODE/tools/ncclffi/libncclffi.so"
# mpirun hygiene (see PORT.md gotchas 2-4):
#   unset OMPI_MCA_mca_base_env_list; mpirun --mca coll ^hcoll -x ... < /dev/null
