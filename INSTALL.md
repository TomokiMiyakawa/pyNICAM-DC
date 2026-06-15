# Installing / running pyNICAM-DC

The Python dependency declared by the package itself is just `numpy` (see
`setup.py` / `requirements.txt`). The lists below describe the **full
environment** needed to actually run the model and the optional tooling.

## For a basic run
- python 3.11 or 3.12 (others may work but are not tested)
- mpich
- mpi4py
- toml
- xarray
- dask
- zarr (2.15 recommended; needs a downgrade from 3.xx)

## For quicklook / plotting
- matplotlib
- mpl_toolkits
- cartopy
- scipy
- jupyterlab
- ipykernel

## To run with jax (experimental)
- jax
- jaxlib
- if on macOS: jax-metal (experimental), e.g.
  `jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1`

## GPU development on a rented cloud instance (NVIDIA / GH200)

For developing the jax GPU / mpi4jax path, an **interactive** GPU machine with
outbound internet works far better than a batch/submit supercomputer (Miyabi):
you get a tight edit-run-inspect loop, and the box can reach the network the
dev tooling needs. A good first step is an hourly-rented cloud GPU rather than
buying hardware. Lambda gives the cheapest self-serve **GH200 (aarch64 Grace +
H100)**, which matches Miyabi-G's architecture; fall back to **x86 + H100**
(Vultr, Runpod, etc.) when GH200 stock is unavailable — the GPU math is
identical, only the CPU arch and CUDA wheels differ.

Steps below assume a Lambda GH200 instance (default ssh user `ubuntu`, NVIDIA
driver + CUDA preinstalled via "Lambda Stack"). Attach a persistent filesystem
so work survives instance termination.

```sh
# 1. connect and confirm the GPU
ssh ubuntu@<INSTANCE_IP>
nvidia-smi

# 2. Claude Code (optional, for agent-assisted development)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc && source ~/.bashrc

# 3. system + python env
sudo apt-get update
sudo apt-get install -y python3-venv libopenmpi-dev openmpi-bin
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -U pip wheel

# 4. jax on GPU -- try pip first
pip install -U "jax[cuda12]"
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]

# 5. mpi4jax (or jax.distributed + NCCL for single-node multi-GPU)
pip install mpi4py
pip install mpi4jax --no-build-isolation
python -c "import mpi4jax; print('mpi4jax ok')"

# 6. fetch + smoke-test the package
git clone https://github.com/TomokiMiyakawa/pyNICAM-DC.git
cd pyNICAM-DC && pip install -r requirements.txt pytest flake8
pytest -q                                       # 24 passed / 3 skipped (jax parity runs if jax present)
```

Gotchas, in the order you are likely to hit them:
- **aarch64 (GH200) jax wheels**: if step 4 shows CPU only or fails to import,
  use NVIDIA's official JAX container instead (it supports GH200/aarch64):
  `sudo docker run --gpus all -it --rm -v $PWD:/work -w /work nvcr.io/nvidia/jax:<latest>-py3 bash`.
  On x86 + H100 the pip wheels almost always work, so this branch rarely bites.
- **mpi4jax build**: if it fails to build, `jax.distributed` + NCCL is the
  official, robust alternative for single-node multi-GPU (no MPI needed).
- **GH200 is one GPU per node**: exercise multi-rank mpi4jax by co-locating a
  couple of ranks on the single GPU (`mpirun -np 2 python ...`) for correctness;
  true multi-GPU/multi-node scaling belongs on Miyabi (batch) or a multi-GPU
  cloud cluster.
- **Billing**: instances bill while running -- `Terminate` when done and keep
  work on the persistent filesystem. Prefer per-second/minute-billed providers
  for bursty debugging.

Final large-scale scaling runs go on Miyabi via `pjsub` batch jobs; the cloud
GPU is for the fast develop/debug loop, not production scaling.

## Other notes
- May run faster with micromamba.
- May need Xcode on macOS.

### Example setup (Tomoki's MacBook, 2025/05/23)
```sh
micromamba create -n jax_mpi python=3.11 -c conda-forge
micromamba activate jax_mpi
micromamba install -c conda-forge mpich mpi4py
pip uninstall -y jax jaxlib jax-metal && \
  pip install --upgrade jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1
micromamba install -n jax_mpi -c conda-forge toml matplotlib dask xarray \
  cartopy scipy zarr==2.15
micromamba install -n jax_mpi -c conda-forge jupyterlab ipykernel
python -m ipykernel install --user --name=jax_mpi --display-name "Python (jax_mpi)"
```
