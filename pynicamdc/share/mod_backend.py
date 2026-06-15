import numpy as np

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

        if backend_name == "numpy":
            self.type = "numpy"
            self.xp = np
            self.dtype = self.ndtype
            self.to_numpy = lambda x: np.asarray(x)
            self.jax = None

        elif backend_name == "jax":
            import jax
            jax.config.update("jax_enable_x64", precision == "float64")
            import jax.numpy as jnp

            self.type = "jax"
            self.jax = jax
            self.xp = jnp
            self.dtype = jnp.float32 if precision == "float32" else jnp.float64
            self.to_numpy = lambda x: np.asarray(x)

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
