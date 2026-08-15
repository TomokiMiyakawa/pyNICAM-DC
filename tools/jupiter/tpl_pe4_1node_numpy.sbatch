#!/bin/bash -l
#SBATCH --job-name=pnc_g5_np
#SBATCH --partition=booster
#SBATCH --account=jureap22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one full Grace socket per rank (4 x 72C per node)
#SBATCH --time=00:20:00
#SBATCH --output=/e/project1/jureap22/miyakawa/workClaude1/sweep/logs/pnc_g5_np.%j.out
# T2-CPU: gl05 rl01 pe04, JW baroclinic wave, numpy backend, IDEAL init.
# Functional check of the multi-rank stack on JUPITER: srun + psmpi + mpi4py.
#
# `#!/bin/bash -l` is REQUIRED: without a login shell Lmod is not initialised in the
# batch script and `module` is undefined.
# --output MUST be an absolute path under /e (PORT.md trap #1): under /p the job dies
# in ~10 s with FAILED 0:53 and no log at all.
# NB mpiexec/mpirun do NOT exist on JUPITER -- srun only.
set -uo pipefail

CODE=/e/project1/jureap22/miyakawa/workClaude1/pyNICAM-DC   # repo clone (must be /e)
ROOT=/e/project1/jureap22/miyakawa/workClaude1/sweep        # unpacked benchmark inputs
source "$CODE/tools/jupiter/env.sh"

G=5; PE=4; LSTEP=12; LAB=jup_np
export OMP_NUM_THREADS=1

python "$ROOT/scripts/make_config.py" "$G" --backend numpy --precision float64 \
       --lstep "$LSTEP" --pe "$PE" --label "$LAB" --init ideal
RD="$ROOT/run/gl0${G}_${LAB}"
cd "$RD" || exit 1
rm -rf testout_tmp.zarr msg.pe* out*

# 'peacefully done' goes to stdout (driver-dc.py), NOT the msg.pe* logs -> tee it.
# Python warnings go to STDERR and never reach stdout.log -- read the JOB log for those.
srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket \
     python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
     --driver-setting ./driversettings.toml < /dev/null | tee stdout.log

echo "done=$(grep -c 'peacefully done' stdout.log 2>/dev/null || true)/$PE"
echo "### TIMER ###"
grep -E "MAIN_Main_Loop|MAIN___Dynamics" msg.pe00000000 2>/dev/null | head
# Reference (2026-07-28, job 1085888): done=4/4, MAIN_Main_Loop 17.357 s (Miyabi 17.143).
# NB overflow/invalid RuntimeWarnings from mod_src_tracer/mod_thrmdyn are BENIGN here:
# IDEAL init has identically zero tracers, so the flux limiter divides by zero
# denominators. Same class appears in the July Miyabi gl09 numpy logs.
