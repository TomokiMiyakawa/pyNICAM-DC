# Per-GPU microbench: HBM triad bandwidth + fp32 matmul TFLOPS.
# Run one process per GPU (CUDA_VISIBLE_DEVICES set by wrapper).
import os, time
import jax, jax.numpy as jnp

rank = int(os.environ.get("SLURM_PROCID", "0"))
dev = jax.devices()[0]

N = 256 * 1024 * 1024  # 1 GiB per fp32 array
a = jnp.ones((N,), jnp.float32)
b = jnp.ones((N,), jnp.float32)
triad = jax.jit(lambda a, b: a + 2.5 * b)
triad(a, b).block_until_ready()
t0 = time.perf_counter()
ITER = 20
for _ in range(ITER):
    c = triad(a, b)
c.block_until_ready()
dt = (time.perf_counter() - t0) / ITER
gbs = 3 * N * 4 / dt / 1e9  # 2 reads + 1 write

M = 8192
x = jnp.ones((M, M), jnp.float32)
mm = jax.jit(lambda x: x @ x)
mm(x).block_until_ready()
t0 = time.perf_counter()
for _ in range(ITER):
    y = mm(x)
y.block_until_ready()
dt = (time.perf_counter() - t0) / ITER
tflops = 2 * M**3 / dt / 1e12

print(f"MICROBENCH rank={rank} dev={dev.device_kind} triad={gbs:.0f} GB/s matmul_fp32={tflops:.1f} TFLOPS", flush=True)
