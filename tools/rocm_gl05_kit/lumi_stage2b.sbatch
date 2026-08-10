#!/bin/bash
#SBATCH --job-name=pnc_rccl8
#SBATCH --account=project_465000454
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Stage 2b on LUMI: the RCCL halo exchange on 8 GCDs, plus the numpy-CPU pe8
# reference leg for the correctness A/B, plus the comparison. One node, one job.
#
#   sbatch tools/rocm_gl05_kit/lumi_stage2b.sbatch
set -uo pipefail
KIT="${SLURM_SUBMIT_DIR:-$PWD}/tools/rocm_gl05_kit"
[ -f "$KIT/lumi_env.sh" ] || KIT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT/lumi_env.sh"

echo "##################### RCCL leg (jax-ROCm, 8 GCDs) #####################"
BACKEND=jax bash "$KIT/run_8gpu_rccl_lumi.sh"
RC_RCCL=$?

echo
echo "##################### reference leg (numpy CPU, 8 ranks) ##############"
# The reference is backend-only -- it does not depend on anything in the RCCL leg, so
# reuse a complete set of dumps across resubmits (FORCE_REF=1 to recompute).
if [ "${FORCE_REF:-0}" = 0 ] && [ "$(ls "$KIT"/run_8cpu_numpy/fin_rank*.npy 2>/dev/null | wc -l)" = 8 ]; then
    echo "  reusing existing 8 reference dumps in $KIT/run_8cpu_numpy (FORCE_REF=1 to redo)"
    RC_NP=0
else
    BACKEND=numpy bash "$KIT/run_8gpu_rccl_lumi.sh"
    RC_NP=$?
fi

echo
echo "##################### COMPARE (cross-backend floor) ###################"
source "$VENV/bin/activate"
python "$KIT/validate.py" \
    --ref  "$KIT/run_8cpu_numpy/fin_rank*.npy" \
    --cand "$KIT/run_8gpu_rccl/fin_rank*.npy" \
    --rtol 1e-9
RC_CMP=$?

echo
echo "SUMMARY: rccl_leg=$RC_RCCL numpy_leg=$RC_NP compare=$RC_CMP"
