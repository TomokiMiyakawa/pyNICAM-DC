#!/bin/bash
if [ "${OMPI_COMM_WORLD_RANK:-0}" = "0" ] && [ "${SPIKE_NSYS:-0}" != "0" ]; then
  exec nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
       --force-overwrite=true -o ring_r0 python "$@"
else
  exec python "$@"
fi
