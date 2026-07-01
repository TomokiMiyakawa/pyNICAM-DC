"""Raw cudaMemcpy device<->host bandwidth on GH200 (no jax) — the cross-check for
bench_d2h_coherent.py. Calls libcudart directly via ctypes to separate the
hardware C2C bandwidth from jax's transfer overhead.

Measures H2D and D2H for both PINNED (cudaHostAlloc) and PAGEABLE host memory.
cudaMemcpy is synchronous, so wall-clock around it (+ a final sync) is the time.

If PINNED bandwidth >> 25 GB/s (PCIe class) and approaches GH200 NVLink-C2C
(~hundreds GB/s), then the hardware link IS cheap and jax simply isn't exploiting
it -> the residency/STEP-7 conclusion hinges on whether the model's path can be
made to use pinned/coherent transfers. If even raw pinned is ~PCIe-class, the
link itself is the limit.

Run:  python bench_cudamemcpy_raw.py
"""
import ctypes
import time

H2D = 1  # cudaMemcpyHostToDevice
D2H = 2  # cudaMemcpyDeviceToHost


def _chk(rc, where):
    if rc != 0:
        raise RuntimeError(f"CUDA error {rc} at {where}")


def main():
    cu = ctypes.CDLL("libcudart.so.12")
    for fn, args in {
        "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "cudaFree": [ctypes.c_void_p],
        "cudaHostAlloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "cudaFreeHost": [ctypes.c_void_p],
        "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "cudaDeviceSynchronize": [],
    }.items():
        getattr(cu, fn).argtypes = args
        getattr(cu, fn).restype = ctypes.c_int

    sizes_mb = [1, 4, 16, 64, 256, 1024]
    reps = 30
    print(f"raw cudaMemcpy via libcudart  (reps={reps})")
    print(f"{'MB':>6} {'pin H2D':>9} {'pin D2H':>9} {'page H2D':>9} {'page D2H':>9}   GB/s")
    for mb in sizes_mb:
        n = mb * 1024 * 1024
        dptr = ctypes.c_void_p()
        _chk(cu.cudaMalloc(ctypes.byref(dptr), n), "cudaMalloc")
        pin = ctypes.c_void_p()
        _chk(cu.cudaHostAlloc(ctypes.byref(pin), n, 0), "cudaHostAlloc")
        page = (ctypes.c_byte * n)()  # pageable host buffer

        def band(dst, src, kind):
            cu.cudaMemcpy(dst, src, n, kind); cu.cudaDeviceSynchronize()  # warm
            t0 = time.perf_counter()
            for _ in range(reps):
                cu.cudaMemcpy(dst, src, n, kind)
            cu.cudaDeviceSynchronize()
            sec = (time.perf_counter() - t0) / reps
            return n / sec / 1e9

        p_h2d = band(dptr, pin, H2D)
        p_d2h = band(pin, dptr, D2H)
        g_h2d = band(dptr, ctypes.cast(page, ctypes.c_void_p), H2D)
        g_d2h = band(ctypes.cast(page, ctypes.c_void_p), dptr, D2H)
        print(f"{mb:6d} {p_h2d:9.1f} {p_d2h:9.1f} {g_h2d:9.1f} {g_d2h:9.1f}", flush=True)

        cu.cudaFree(dptr); cu.cudaFreeHost(pin)

    print("\npin = pinned (cudaHostAlloc), page = pageable. Compare to jax effective "
          "(~30 GB/s D2H) and PCIe4 ~25 GB/s. High pinned bandwidth => C2C is cheap, "
          "jax just isn't using it.")


if __name__ == "__main__":
    main()
