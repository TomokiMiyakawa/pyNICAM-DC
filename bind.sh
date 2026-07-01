#!/usr/bin/env bash
# rank -> GPU binding: give each MPI rank a DISTINCT physical GPU.
# Without it, all local ranks land on GPU 0 (contention). Use as a wrapper:
#   mpiexec -n <N> ./bind.sh python ...
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
exec "$@"
