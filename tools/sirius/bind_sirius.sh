#!/bin/bash
# Per-rank APU binding on Sirius (manual 7.5.10): each MPI rank sees ONE MI300A as device 0.
# Only ROCR_VISIBLE_DEVICES is set. tools/rocm_gl05_kit/bind_rocm.sh also sets
# HIP_VISIBLE_DEVICES to the same index, but HIP_VISIBLE_DEVICES indexes the devices
# LEFT AFTER the ROCR mask, so local rank >= 1 would end up with no GPU at all.
# With -l place=exclhost all 4 APUs of a node are visible, so local rank 0..3 maps 1:1.
# PNC_RANKS_PER_GPU>1 packs several ranks per APU (e.g. 8 ranks on the 2-vnode debug queue).
lr=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
export ROCR_VISIBLE_DEVICES=$(( lr / ${PNC_RANKS_PER_GPU:-1} ))
unset HIP_VISIBLE_DEVICES
exec "$@"
