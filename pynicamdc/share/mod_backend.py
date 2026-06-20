import numpy as np
import os


class _XferProf:
    """Gated D2H transfer profiler for the to_numpy boundary (PYNICAM_XFER_PROF=1).

    Answers the C2 GO/NO-GO question: are the model's host round-trips big enough
    (>=16MB) for the pinned_host fast path to help, or all small/latency-bound?
    Measures only (no value change), so numpy/jax paths stay bit-exact. Dumps a
    per-rank size histogram at exit.
    """
    THRESH = 16 * 1024 * 1024  # the pinned_host break-even from bench_d2h_coherent

    def __init__(self):
        self.n = 0
        self.bytes = 0
        self.n_ge = 0          # calls >= THRESH (pinned would help)
        self.bytes_ge = 0
        self.hist = {}         # log2(MB) bucket -> count
        self.maxb = 0

    def record(self, nbytes):
        self.n += 1
        self.bytes += nbytes
        if nbytes >= self.THRESH:
            self.n_ge += 1
            self.bytes_ge += nbytes
        if nbytes > self.maxb:
            self.maxb = nbytes
        mb = nbytes / 1e6
        b = 0 if mb < 1 else int(mb).bit_length()  # ~log2(MB) bucket
        self.hist[b] = self.hist.get(b, 0) + 1

    def dump(self):
        if self.n == 0:
            return
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        path = os.environ.get("PYNICAM_XFER_PROF_OUT", "xfer_prof") + f".pe{rank}"
        with open(path, "w") as f:
            f.write(f"=== to_numpy D2H profile (rank {rank}) ===\n")
            f.write(f"calls={self.n}  total={self.bytes/1e6:.1f} MB  "
                    f"max_call={self.maxb/1e6:.2f} MB\n")
            f.write(f">={self.THRESH//1024//1024}MB calls: {self.n_ge} "
                    f"({100*self.n_ge/self.n:.1f}% of calls, "
                    f"{100*self.bytes_ge/max(1,self.bytes):.1f}% of bytes)\n")
            f.write("size buckets (MB, count):\n")
            for b in sorted(self.hist):
                lo = 0 if b == 0 else (1 << (b - 1))
                f.write(f"  [{lo:>5}-{(1<<b):>5}) MB : {self.hist[b]}\n")


class Backend:
    
    _instance = None

    def __init__(self):
        self.configured = False
        self.type = None
        self.xp = None
        self.dtype = None
        self.jax = None

    def configure(self, backend_name, precision):
        self.np = np  # always have numpy available
        self.ndtype = np.float32 if precision == "float32" else np.float64

        # gated D2H transfer profiler (measures only -> values unchanged, bit-exact)
        prof = None
        if os.environ.get("PYNICAM_XFER_PROF") == "1":
            import atexit
            prof = _XferProf()
            atexit.register(prof.dump)

        def _to_numpy(x):
            if prof is not None:
                prof.record(getattr(x, "nbytes", np.asarray(x).nbytes))
            return np.asarray(x)

        if backend_name == "numpy":
            self.type = "numpy"
            self.xp = np
            self.dtype = self.ndtype
            self.to_numpy = _to_numpy
            self.jax = None

        elif backend_name == "jax":
            import jax
            jax.config.update("jax_enable_x64", precision == "float64")
            import jax.numpy as jnp

            self.type = "jax"
            self.jax = jax
            self.xp = jnp
            self.dtype = jnp.float32 if precision == "float32" else jnp.float64
            self.to_numpy = _to_numpy

    def maybe_jit(self, fn, *, static_argnames=None):
        """Wrap a pure kernel for the active backend.

        numpy backend : return `fn` unchanged (eager execution).
        jax backend   : return `jax.jit(fn, static_argnames=...)`.

        This keeps the numpy-vs-jax branch in exactly one place so that
        kernel call sites never need an `if bk.type == "jax"` of their own.
        """
        if self.type == "jax":
            return self.jax.jit(fn, static_argnames=static_argnames)
        return fn

    def device_consts(self, owner, key, builder):
        """Return a device-resident dict of read-only constants, cached on `owner`.

        This is the single, uniform mechanism for the per-kernel "build the
        constant/metric dict once, then reuse it" pattern (formerly hand-written
        as `if getattr(self, "_X_dev", None) is None: self._X_dev = {...}` at
        ~12 call sites). Centralising it removes the repeated boilerplate and the
        class of bug where one site silently forgets to cache (the kernel that
        re-`asarray`s its metrics on every call).

        Parameters
        ----------
        owner : object
            The instance to cache on (typically `self`). The cache is stored in
            `owner.__dict__["_dev_cache"]`, so distinct owners never collide.
        key : str
            Unique name for this constant set (e.g. "presgrad", "vimain").
        builder : callable
            Zero-arg callable returning a `{name: value}` dict of RAW host
            values. It is invoked at most once (on first use). `numpy.ndarray`
            values are moved to the active backend via `xp.asarray`; non-array
            values (scalars such as GRD_rscale) pass through unchanged. This
            mirrors the previous explicit code exactly, so the numpy backend
            stays bit-identical (asarray is a no-op there).

        Returns
        -------
        dict
            The cached, device-resident constant dict.
        """
        cache = owner.__dict__.setdefault("_dev_cache", {})
        d = cache.get(key)
        if d is None:
            d = {k: (self.xp.asarray(v) if isinstance(v, np.ndarray) else v)
                 for k, v in builder().items()}
            cache[key] = d
        return d

backend = Backend()
