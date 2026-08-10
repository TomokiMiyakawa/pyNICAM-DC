#!/usr/bin/env bash
# rank -> GCD binding on LUMI (Slurm + 4x MI250X = 8 GCDs per node).
#
# Two changes vs the kit's bind_rocm.sh, both mandatory here:
#
#  1) local rank comes from SLURM_LOCALID, not OMPI_COMM_WORLD_LOCAL_RANK (srun,
#     not mpirun -- the OMPI var is simply unset, so every rank would take the
#     ":-0}" default and all 8 would land on GCD 0).
#
#  2) set exactly ONE of ROCR_/HIP_VISIBLE_DEVICES. bind_rocm.sh sets BOTH to the
#     same index "for safety", but they compose: ROCR masking is applied first and
#     renumbers what is left, so with ROCR_VISIBLE_DEVICES=3 the rank sees a single
#     device numbered 0 and the subsequent HIP_VISIBLE_DEVICES=3 then indexes past
#     the end -> zero visible devices for every rank except 0.
#
# mod_ncclffi.ncclffi_init(..., 0) inits device 0 of whatever is visible, so a
# 1-device mask per rank is exactly what it wants.
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:?bind_lumi.sh must run under srun}"
unset HIP_VISIBLE_DEVICES
exec "$@"
