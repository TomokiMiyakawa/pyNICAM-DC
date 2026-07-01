"""Device<->host transfer-cost benchmark for the GH200 coherent-memory premise,
and the key follow-up: can jax USE the fast C2C link?

Background: the residency / STEP-7 plan assumes device<->host copies are expensive
(true on PCIe H100, ~25 GB/s). GH200 NVLink-C2C is ~10-20x faster + coherent. The
raw hardware IS cheap (see bench_cudamemcpy_raw.py: ~300-340 GB/s, pageable~=pinned).
But jax's DEFAULT path (np.asarray / device_put) exposes only ~11 GB/s (f64) — it
leaves ~95% of the link on the floor.

This bench shows the default path AND the fast path: jax's `pinned_host` memory
kind (SingleDeviceSharding(memory_kind="pinned_host")) hits ~230-290 GB/s f64 and
is BIT-EXACT. So jax CAN exploit C2C -> "route the model's large host round-trips
through pinned_host" is a real, simple lever, complementary to residency (which
removes the small latency-bound halo round-trips that bandwidth can't help).

MUST enable x64 (the model is float64); without it device_put silently downcasts
to f32 and halves the bytes moved, corrupting both correctness and the GB/s.

Run:  JAX_ENABLE_X64=1 python bench_d2h_coherent.py
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.sharding as shd


def main():
    d = jax.devices()[0]
    pin_sh = shd.SingleDeviceSharding(d, memory_kind="pinned_host")
    dev_sh = shd.SingleDeviceSharding(d, memory_kind="device")
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  device={d}  "
          f"x64={jax.config.jax_enable_x64}", flush=True)

    # correctness: device -> pinned_host -> numpy must be bit-exact (pure move)
    hv = np.random.default_rng(1).standard_normal(4096).astype(np.float64)
    xp = jax.device_put(jax.device_put(hv, d), pin_sh); xp.block_until_ready()
    assert np.array_equal(np.asarray(xp), hv), "pinned_host round-trip NOT bit-exact!"
    print("pinned_host round-trip: BIT-EXACT", flush=True)

    reps = 30
    print(f"\n{'MB':>6} {'asarray D2H':>12} {'pinned D2H':>11} {'pin->dev H2D':>13} "
          f"{'(GB/s, f64)':>12}")
    for mb in [1, 4, 16, 64, 256]:
        n = (mb * 1024 * 1024) // 8
        host = np.random.default_rng(mb).standard_normal(n).astype(np.float64)
        nb = n * 8
        g = lambda sec: nb / sec / 1e9

        # distinct device arrays defeat jax's asarray cache (re-asarray of an
        # immutable array is a cache hit, not a copy)
        def make(k):
            a = [jax.device_put(host + i, d) for i in range(k)]
            for x in a:
                x.block_until_ready()
            return a

        # default D2H: np.asarray(device_array) -> slow XLA path to pageable host
        da = make(reps); _ = np.asarray(da[0])
        t0 = time.perf_counter()
        for x in da:
            np.asarray(x)
        t_as = (time.perf_counter() - t0) / len(da); del da

        # fast D2H: device -> pinned_host (C2C), then asarray (pinned is host-mapped)
        da = make(reps)
        def d2h_pin(x):
            xp = jax.device_put(x, pin_sh); xp.block_until_ready(); return np.asarray(xp)
        _ = d2h_pin(da[0])
        t0 = time.perf_counter()
        for x in da:
            d2h_pin(x)
        t_pin = (time.perf_counter() - t0) / len(da); del da

        # fast H2D: pinned_host -> device (host data assumed already pinned)
        hpin = jax.device_put(host, pin_sh); hpin.block_until_ready()
        def h2d_pin():
            x = jax.device_put(hpin, dev_sh); x.block_until_ready()
        h2d_pin()
        t0 = time.perf_counter()
        for _ in range(reps):
            h2d_pin()
        t_h2d = (time.perf_counter() - t0) / reps

        print(f"{mb:6d} {g(t_as):12.1f} {g(t_pin):11.1f} {g(t_h2d):13.1f}", flush=True)

    print("\nraw cudaMemcpy ref (bench_cudamemcpy_raw): H2D ~340, D2H ~295 GB/s | "
          "PCIe4 ~25. Default jax D2H ~11 GB/s; pinned_host ~230 GB/s (>=16MB). "
          "Win is bandwidth-bound (helps >=16MB whole-field moves); MB-scale halos "
          "stay latency-bound (~100us/call) -> residency still needed for those.")


if __name__ == "__main__":
    main()
