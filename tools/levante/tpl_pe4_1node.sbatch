#!/bin/bash
#SBATCH --job-name=pnc_pe4
#SBATCH --partition=dolpung
#SBATCH --account=mh1571_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one full Grace socket per rank (4 x 72C per node).
                               # CRITICAL: without this Slurm allocates 1 core/rank and
                               # NCCL's IB proxy threads + jax dispatch starve -- measured
                               # 4.4x slowdown on inter-node runs (same-job A/B, gl09 pe8:
                               # 0.97 -> 0.22 s/step). Intra-node NVLink hides it (no CPU
                               # proxy), so a 1-node run looks fine and the trap only
                               # springs on multi-node.
#SBATCH --gpus-per-node=4      # MUST be per-node, NOT --gpus-per-task=1. Both end up with
                               # one GPU per rank, but --gpus-per-task=1 cgroup-isolates
                               # each task to its own GPU, so NCCL cannot open a peer's
                               # device and EVERY intra-node transport dies at
                               # ncclP2pImportShareableBuffer / ncclShmImportShareableBuffer
                               # with Cuda failure 101 'invalid device ordinal' (only the
                               # IB path survives). Granting all 4 in the cgroup and masking
                               # with CUDA_VISIBLE_DEVICES (below) keeps 1 rank = 1 GPU and
                               # leaves NVLink P2P usable. Verified with a standalone
                               # NCCL AllReduce/SendRecv reproducer, gh200/nccl_min.cc.
                               # NB cgroup device limits bite at open(), not readdir, so
                               # `ls /dev/nvidia*` shows all 4 under both modes.
#SBATCH --time=00:30:00
#SBATCH --export=NONE          # the x86 login profile exports a full conda build
                               # toolchain (CC/CFLAGS=-march=nocona/_CONDA_PYTHON_
                               # SYSCONFIGDATA_NAME/...) that poisons the aarch64 node
#SBATCH --output=pnc_pe4.%j.out
# T2: 1 node x 4 GPU functional check (gl05 pe4, production resident+fused+FFI).
# Adapted from Miyabi tools/ncclffi/vi_v3b_pe4_ab.pbs run_one; filled in for Levante.
set -uo pipefail
G=5; PE=4; LSTEP=8
CODE=/work/bb1153/b381557/workclaude/pyNICAM-DC     # pyNICAM-DC clone (main)
VENV=/work/bb1153/b381557/workclaude/venv-gh200
ROOT=/work/bb1153/b381557/workclaude                # unpacked levante_inputs_gl05-09.tar.gz
                                                    # (data/ scripts/ config/ sit at this level;
                                                    #  make_config.py resolves ROOT as its own parent)
# --export=NONE (above) also makes srun default to SLURM_EXPORT_ENV=NONE, so the steps
# would NOT inherit this script's PATH/venv and die with "bash: No such file or
# directory". Put it back to ALL -- by then the environment is the clean one we built.
export SLURM_EXPORT_ENV=ALL
# No CUDA module needed: driver CUDA 12.9 is in the image, nvcc/NCCL come from the venv.
MPI=/opt/mpi/openmpi/5.0.6.1.6
export PATH="$MPI/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
# REQUIRED: Levante's OpenMPI is built with wrapper rpath/runpath disabled, so nothing
# is found by itself -- libmpi.so.40 for mpi4py's import, libprrte/libpmix for mpirun.
export LD_LIBRARY_PATH="$MPI/lib:/opt/prrte/3.0.8/lib:/opt/pmix/5/lib"
unset PYTHONHOME
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi.so"

# --- per-rank single-GPU visibility (4 GPU/node -- REQUIRED, see PORT.md) ---
# Must be done inside the task: SLURM_LOCALID only exists there.
WRAP="$SLURM_SUBMIT_DIR/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

LAB=lev_pe4
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float64 \
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
