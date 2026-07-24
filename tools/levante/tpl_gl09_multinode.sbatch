#!/bin/bash
#SBATCH --job-name=pnc_g9multi
#SBATCH --partition=dolpung
#SBATCH --account=mh1571_gpu
#SBATCH --nodes=2              # pe8->2, pe20->5, pe40->10 (edit with PE below)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one Grace socket per rank -- CRITICAL for multi-node
                               # (see tpl_pe4_1node.sbatch: 4.4x A/B on the IB legs)
#SBATCH --gpus-per-node=4      # NOT --gpus-per-task=1, which cgroup-isolates each rank
                               # and breaks NCCL's intra-node transports -- see
                               # tpl_pe4_1node.sbatch for the full explanation
#SBATCH --time=00:30:00
#SBATCH --export=NONE          # see tpl_pe4_1node.sbatch (x86 conda toolchain leak)
#SBATCH --output=pnc_g9multi.%j.out
# T3-T4: multi-node gl09 (2-10 nodes, production resident+fused+FFI).
# Adapted from Miyabi tools/ncclffi/vi_v3b_pe4_ab.pbs run_one; filled in for Levante.
set -uo pipefail
G=9; PE=8; LSTEP=18   # pe in {8,20,40}; 4 ranks/node -- keep --nodes=PE/4 in sync
CODE=/work/bb1153/b381557/workclaude/pyNICAM-DC     # pyNICAM-DC clone (main)
VENV=/work/bb1153/b381557/workclaude/venv-gh200
ROOT=/work/bb1153/b381557/workclaude                # unpacked levante_inputs_gl05-09.tar.gz
                                                    # (make_config.py resolves ROOT as its own parent)
# See tpl_pe4_1node.sbatch: --export=NONE forces SLURM_EXPORT_ENV back to ALL, and no
# CUDA module is needed.
export SLURM_EXPORT_ENV=ALL
MPI=/opt/mpi/openmpi/5.0.6.1.6
export PATH="$MPI/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH="$MPI/lib:/opt/prrte/3.0.8/lib:/opt/pmix/5/lib"  # no rpath in
                                    # Levante's OpenMPI -> libmpi/libprrte/libpmix all
                                    # have to be found by hand
unset PYTHONHOME
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi.so"

# --- per-rank single-GPU visibility (see tpl_pe4_1node.sbatch) ---
WRAP="$SLURM_SUBMIT_DIR/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

LAB=lev_g9_pe${PE}
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float32 \
    --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output on
RD="$ROOT/run/gl0${G}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"; rm -rf out* *.zarr msg.pe*
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
# first run: also set PYNICAM_NCCLFFI_CKSUM=1 once and parse the audit (PORT.md T2)

# 'peacefully done' goes to stdout (driver-dc.py), NOT the msg.pe* logs -> tee it.
srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
    --driver-setting ./driversettings.toml < /dev/null | tee stdout.log
echo "done=$(grep -c 'peacefully done' stdout.log 2>/dev/null || true)/$PE"
