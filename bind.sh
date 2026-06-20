#!/usr/bin/env bash
# rank -> GPU binding for the H100x4 box (STEP 2 of H100_BRINGUP.md).
# Each MPI rank owns a DISTINCT physical GPU; without this all ranks land on
# GPU 0 = the contention that masked the residency win on the GH200.
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
exec "$@"
