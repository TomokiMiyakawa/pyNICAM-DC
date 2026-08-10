#!/bin/bash
# LUMI (HPE Cray EX, 4x MI250X = 8 GCDs/node, gfx90a) environment for the ROCm kit.
# Source this before every build/run step:  source tools/rocm_gl05_kit/lumi_env.sh
#
# Differences from the PORT.md "rented AMD box" assumptions, all handled here:
#   - arch is gfx90a (MI250X), not gfx942 (MI300X)
#   - launcher is srun (Slurm), not mpirun; local rank is SLURM_LOCALID, not
#     OMPI_COMM_WORLD_LOCAL_RANK  -> bind_lumi.sh
#   - MPI is Cray MPICH; mpi4py must be built with the `cc` wrapper
#   - $HOME is at its 100K inode quota -> pip cache / TMPDIR must live on scratch

module load cray-python/3.11.7 2>/dev/null

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"        # 6.3.4 system install
export OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx90a}"     # MI250X
export PATH="$ROCM_PATH/bin:$PATH"

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KIT
export REPO="$(cd "$KIT/../.." && pwd)"
export VENV="${VENV:-$KIT/venv-rocm}"

# keep every scratch-file writer off $HOME (inode quota is full there)
export SCR="${SCR:-/scratch/project_465000454/klockeda/tomoki/.cache}"
mkdir -p "$SCR/pip" "$SCR/tmp"
export PIP_CACHE_DIR="$SCR/pip"
export TMPDIR="$SCR/tmp"
export XDG_CACHE_HOME="$SCR"
export JAX_COMPILATION_CACHE_DIR="$SCR/jaxcache"
