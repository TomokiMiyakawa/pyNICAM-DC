"""Device<->host transfer-cost microbenchmark for the GH200 coherent-memory premise.

The whole residency / STEP-7 plan assumes device<->host copies are EXPENSIVE
(true on PCIe H100, ~25 GB/s). GH200's Grace-Hopper NVLink-C2C makes the link
~10-20x faster and the memory coherent. If transfers are cheap HERE, residency /
STEP-7 may matter much less on Miyabi (aarch64 GH200). This measures it directly,
single-process / 1 GPU (no inter-rank COMM needed -> valid on one GH200).

Measures, per size, with jax (the model's backend):
  - D2H: np.asarray(device_array)         (the per-kernel to_numpy boundary)
  - H2D: jax.device_put(host_array) + ready

Reports realised GB/s. Async dispatch is forced with block_until_ready / asarray
materialisation, and each direction is warmed up before timing.

Run:  python bench_d2h_coherent.py
"""
import time
import numpy as np


def main():
    import jax
    import jax.numpy as jnp

    dev = jax.devices()[0]
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  device={dev}", flush=True)

    # f64 like the model. Sizes span the per-kernel halo (~MB) to a full
    # resident field (~GB) so we see both latency- and bandwidth-bound regimes.
    sizes_mb = [1, 4, 16, 64, 256, 1024]
    reps = 20

    print(f"{'bytes':>12} {'MB':>7} {'D2H GB/s':>10} {'H2D GB/s':>10} "
          f"{'D2H us':>9} {'H2D us':>9}")
    for mb in sizes_mb:
        n = (mb * 1024 * 1024) // 8           # f64 elements
        host = np.random.default_rng(mb).standard_normal(n).astype(np.float64)
        nbytes = host.nbytes

        # H2D: host -> device
        dlist = []
        d = jax.device_put(host, dev); d.block_until_ready()      # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            dd = jax.device_put(host, dev); dd.block_until_ready()
            dlist.append(dd)
        h2d = (time.perf_counter() - t0) / reps

        # D2H: device -> host. np.asarray() is a blocking copy, BUT jax CACHES the
        # host materialisation of an (immutable) array, so re-asarray of the SAME
        # array is a cache hit (measures ~inf GB/s, not a copy). The model does
        # to_numpy on a DIFFERENT array each step -> no caching. So defeat the cache
        # with distinct device arrays (memory-bounded count), each asarray'd once.
        ndist = max(3, min(reps, int(8e9 // nbytes)))             # cap ~8GB on device
        darrs = [jax.device_put(host + k, dev) for k in range(ndist)]
        for a in darrs:
            a.block_until_ready()
        _ = np.asarray(darrs[0])                                  # warm (one cache fill ok)
        t0 = time.perf_counter()
        for a in darrs:
            h = np.asarray(a)                                     # first materialise => real D2H
        d2h = (time.perf_counter() - t0) / len(darrs)
        del darrs

        gbps = lambda sec: nbytes / sec / 1e9
        print(f"{nbytes:12d} {mb:7d} {gbps(d2h):10.1f} {gbps(h2d):10.1f} "
              f"{d2h*1e6:9.1f} {h2d*1e6:9.1f}", flush=True)
        del dlist

    print("\nReference: PCIe4 x16 ~25 GB/s (H100 box) | GH200 NVLink-C2C ~450 GB/s "
          "theoretical. If D2H/H2D here >> 25 GB/s, host round-trips are cheap on "
          "GH200 -> residency/STEP-7 matters LESS on Miyabi.", flush=True)


if __name__ == "__main__":
    main()
