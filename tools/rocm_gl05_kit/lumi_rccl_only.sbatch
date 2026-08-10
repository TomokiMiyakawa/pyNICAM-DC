#!/bin/bash
#SBATCH --job-name=pnc_rccl_dbg
#SBATCH --account=project_465000454
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#
# RCCL leg only, with whatever diagnostic env the submitter exports. Used to bisect
# the "Memory access fault by GPU" seen in the first 8-GCD run. The numpy reference
# dumps are already banked, so this skips straight to the comparison.
#
#   sbatch --export=ALL,NCCL_DEBUG=INFO,NCCL_P2P_DISABLE=1 lumi_rccl_only.sbatch
#
# RUNTAG names the output dir so parallel variants don't stomp each other.
set -uo pipefail
KIT="${SLURM_SUBMIT_DIR:-$PWD}/tools/rocm_gl05_kit"
[ -f "$KIT/lumi_env.sh" ] || KIT="$(cd "$(dirname "$0")" && pwd)"
source "$KIT/lumi_env.sh"

echo "### diagnostic env:"
env | grep -E '^(NCCL_|PYNICAM_|AMD_|HSA_)' | sort | sed 's/^/    /'
echo

bash "$KIT/run_8gpu_rccl_lumi.sh"
RC=$?
echo "rccl_leg=$RC"

if [ "$(ls "$KIT"/run_8gpu_rccl${RUNTAG:+_$RUNTAG}/fin_rank*.npy 2>/dev/null | wc -l)" = 8 ]; then
    echo
    echo "##################### COMPARE vs numpy CPU pe8 #####################"
    source "$VENV/bin/activate"
    python "$KIT/validate.py" --ref "$KIT/run_8cpu_numpy/fin_rank*.npy" \
                              --cand "$KIT/run_8gpu_rccl${RUNTAG:+_$RUNTAG}/fin_rank*.npy" --rtol 1e-9
    echo "--- scale-relative (per-variable, the meaningful verdict) ---"
    python "$KIT/validate_scale.py" --ref "$KIT/run_8cpu_numpy/fin_rank*.npy" \
                                    --cand "$KIT/run_8gpu_rccl${RUNTAG:+_$RUNTAG}/fin_rank*.npy" --tol 1e-9
fi
