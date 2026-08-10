#!/usr/bin/env bash
# rank -> GPU binding on AMD/ROCm: give each MPI rank a DISTINCT physical GPU so
# each rank sees ITS gpu as device 0 (mod_ncclffi.py inits device 0 per rank).
# ROCm counterpart of bind.sh (CUDA_VISIBLE_DEVICES). Use as an mpirun wrapper:
#   mpirun -n <N> ./bind_rocm.sh python ...
export HIP_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
# Some stacks also read ROCR_VISIBLE_DEVICES; set both for safety.
export ROCR_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
exec "$@"
