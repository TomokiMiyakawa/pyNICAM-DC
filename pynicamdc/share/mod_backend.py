import numpy as np
import os
import time


# RC in-loop audit: a coarse loop-context tag the xfer profiler appends to each
# callsite key, so the residency audit can tell IN-LOOP (per-nl body) transfers from
# the PRE/POST per-step boundary marshal. dynamics_step sets it via backend.set_loop_ctx.
_LOOP_CTX = "setup"


# Unified profiling selector. PYNICAM_PROFILE=<comma-separated tags> replaces the
# individual on/off instrumentation gates (h2d, xfer, xfer_sites, perstep, mem,
# devconst, timeloop_debug, timeloop_timing, commdebug). Pure instrumentation -> no numerics.
# The VALUE params (PYNICAM_XFER_PROF_ATTR_MB / NSYS_STEP / NSYS_STEP_END) stay separate.
_PROFILE = None
def _profile(tag):
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = frozenset(
            t.strip() for t in os.environ.get("PYNICAM_PROFILE", "").split(",") if t.strip())
    return tag in _PROFILE


class _XferProf:
    """Gated D2H transfer profiler for the to_numpy boundary (PYNICAM_PROFILE=xfer).

    Answers the C2 GO/NO-GO question: are the model's host round-trips big enough
    (>=16MB) for the pinned_host fast path to help, or all small/latency-bound?
    Measures only (no value change), so numpy/jax paths stay bit-exact. Dumps a
    per-rank size histogram at exit.
    """
    THRESH = 16 * 1024 * 1024  # the pinned_host break-even from bench_d2h_coherent
    # attribute call sites for transfers >= this. Default 32MB (only the big re-uploads);
    # set PYNICAM_XFER_PROF_ATTR_MB=0 to attribute EVERY transfer (the residency audit --
    # the per-callsite count then distinguishes per-nl recurring host ops from setup).
    ATTR_THRESH = int(os.environ.get("PYNICAM_XFER_PROF_ATTR_MB", "32")) * 1024 * 1024

    def __init__(self, mode="asarray", out_env="PYNICAM_XFER_PROF_OUT", tag="to_numpy D2H"):
        self.mode = mode       # which to_numpy path is active this run (for A/B logs)
        self.out_env = out_env # env var giving the output-file base
        self.tag = tag         # label in the dump header (D2H vs H2D)
        self.n = 0
        self.bytes = 0
        self.secs = 0.0        # total wall time spent in to_numpy transfers
        self.n_ge = 0          # calls >= THRESH (pinned would help)
        self.bytes_ge = 0
        self.secs_ge = 0.0     # time spent on the >=THRESH calls
        self.hist = {}         # log2(MB) bucket -> count
        self.maxb = 0
        # call-site attribution (PYNICAM_PROFILE=xfer_sites): aggregate the consumer
        # file:line:func behind every >=ATTR_THRESH transfer, to map which call sites
        # produce the big H2D re-uploads / D2H drains (the full-residency capstone
        # target list). Off by default (the stack walk adds per-call overhead).
        self.attr_sites = _profile("xfer_sites")
        self.sites = {}        # "file:line func" -> [count, bytes, secs]

    def _callsite(self):
        # first frame outside this module = the actual to_numpy/asarray consumer.
        import sys
        f = sys._getframe(2)  # skip _callsite + record
        while f is not None and f.f_code.co_filename.endswith("mod_backend.py"):
            f = f.f_back
        if f is None:
            return "??"
        base = f.f_code.co_filename.rsplit("/", 1)[-1]
        return f"{base}:{f.f_lineno} {f.f_code.co_name} [{_LOOP_CTX}]"

    def record(self, nbytes, secs=0.0):
        self.n += 1
        self.bytes += nbytes
        self.secs += secs
        if nbytes >= self.THRESH:
            self.n_ge += 1
            self.bytes_ge += nbytes
            self.secs_ge += secs
        if nbytes > self.maxb:
            self.maxb = nbytes
        mb = nbytes / 1e6
        b = 0 if mb < 1 else int(mb).bit_length()  # ~log2(MB) bucket
        self.hist[b] = self.hist.get(b, 0) + 1
        if self.attr_sites and nbytes >= self.ATTR_THRESH:
            key = self._callsite()
            e = self.sites.get(key)
            if e is None:
                self.sites[key] = [1, nbytes, secs]
            else:
                e[0] += 1; e[1] += nbytes; e[2] += secs

    def dump(self):
        if self.n == 0:
            return
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        path = os.environ.get(self.out_env, "xfer_prof") + f".pe{rank}"
        gbps = (self.bytes / self.secs / 1e9) if self.secs > 0 else 0.0
        gbps_ge = (self.bytes_ge / self.secs_ge / 1e9) if self.secs_ge > 0 else 0.0
        with open(path, "w") as f:
            f.write(f"=== {self.tag} profile (rank {rank}, mode={self.mode}) ===\n")
            f.write(f"calls={self.n}  total={self.bytes/1e6:.1f} MB  "
                    f"max_call={self.maxb/1e6:.2f} MB\n")
            f.write(f"xfer time: {self.secs*1e3:.1f} ms total ({gbps:.1f} GB/s eff); "
                    f">=16MB: {self.secs_ge*1e3:.1f} ms ({gbps_ge:.1f} GB/s eff)\n")
            f.write(f">={self.THRESH//1024//1024}MB calls: {self.n_ge} "
                    f"({100*self.n_ge/self.n:.1f}% of calls, "
                    f"{100*self.bytes_ge/max(1,self.bytes):.1f}% of bytes)\n")
            f.write("size buckets (MB, count):\n")
            for b in sorted(self.hist):
                lo = 0 if b == 0 else (1 << (b - 1))
                f.write(f"  [{lo:>5}-{(1<<b):>5}) MB : {self.hist[b]}\n")
            if self.sites:
                f.write(f"top call sites (>= {self.ATTR_THRESH//1024//1024}MB transfers, "
                        f"by total MB):\n")
                for key, (c, b, s) in sorted(self.sites.items(),
                                             key=lambda kv: -kv[1][1]):
                    f.write(f"  {b/1e6:9.1f} MB  {c:5d}x  {s*1e3:8.1f} ms  {key}\n")


class _XpProxy:
    """Thin proxy over xp (jnp) that instruments asarray for H2D profiling.

    Only installed when PYNICAM_PROFILE=h2d; otherwise xp is jnp untouched. Forwards
    every attribute to the real module; intercepts asarray to record host->device
    transfers (counts only numpy-array inputs -> real H2D; device-array asarray is a
    no-op and is skipped). Time is wall-clock around the real asarray.
    """
    def __init__(self, real, prof):
        self._real = real
        self._prof = prof

    def asarray(self, x, *a, **k):
        if isinstance(x, np.ndarray):           # real host->device transfer
            t0 = time.perf_counter()
            r = self._real.asarray(x, *a, **k)
            getattr(r, "block_until_ready", lambda: None)()
            self._prof.record(x.nbytes, time.perf_counter() - t0)
            return r
        return self._real.asarray(x, *a, **k)   # device array -> no-op, skip

    def __getattr__(self, name):
        return getattr(self._real, name)


class Backend:


    def __init__(self):
        self.configured = False
        self.type = None
        self.xp = None
        self.dtype = None
        self.jax = None
        self._resident_master = None
        self.mesh = None   # global device Mesh(('p',)) when PYNICAM_COMM_SHARDING is on

    def resident(self):
        # RESIDENT master switch (the full-NICAM escape hatch). The whole device-resident
        # stack -- what used to be ~67 individual PYNICAM_RESIDENT_*/HDIFF/SINGLE_DRAIN gates
        # -- collapses to this one switch. True = the validated resident stack (device arrays
        # carried across kernels/steps + the within-step fusion that depends on those carries);
        # PYNICAM_RESIDENT=0 forces the jax NON-RESIDENT (host-staged) reference path -- state
        # drained to host and re-uploaded around each kernel, so fusion falls away too -- to
        # A/B a new-physics bug against its residency interaction (numpy is a different backend
        # and can't isolate that). numpy always gets False here (type != "jax"), keeping its
        # non-resident else branch. Default ON. This is the fold target for the 129 positive
        # `!= "0"` gate sites; the ~8 inverted `== "0"` sites are "keep-host" sub-options with
        # per-gate semantics -- fold each individually, not via a blanket negation.
        if self._resident_master is None:
            self._resident_master = os.environ.get("PYNICAM_RESIDENT", "1") != "0"
        return self.type == "jax" and self._resident_master

    def profile(self, tag):
        # PYNICAM_PROFILE membership test (see _profile). For driver/dynamics use via bk.
        return _profile(tag)

    def _init_distributed(self, jax):
        # Join the multi-process JAX runtime so cross-rank collectives (the future ragged
        # halo exchange) lower to NCCL, and build the global 1-D device mesh. Runs ONCE at
        # setup, before any jax device op. The coordinator (rank0 host:free-port) is shared
        # over the model's mpi4py MPI_COMM_WORLD -- jax uses its own gRPC coordinator, which
        # coexists with mpi4py (proven in the spikes). Phase A wires this but the transport
        # stays mpi4jax.alltoall, so runs must be bit-identical to today (validation V6).
        from mpi4py import MPI
        import socket
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank(); size = comm.Get_size()
        if size == 1:
            return  # single process: no distributed runtime / mesh needed
        if rank == 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0)); coord = f"{socket.gethostname()}:{s.getsockname()[1]}"; s.close()
        else:
            coord = None
        coord = comm.bcast(coord, root=0)
        jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
        from jax.sharding import Mesh
        self.mesh = Mesh(np.array(jax.devices()), ("p",))

    def configure(self, backend_name, precision):
        self.np = np  # always have numpy available
        # ndtype is the WORKING PRECISION as a CALLABLE numpy scalar TYPE (np.float32 /
        # np.float64) -- the canonical `rdtype` threaded through the model. CONTRACT:
        #   rdtype(x)         -> cast a scalar to working precision
        #   dtype=rdtype      -> array creation in working precision (numpy AND jax accept it)
        #   np.dtype(rdtype)  -> the dtype INSTANCE, when one is actually needed (e.g. dict keys)
        # Derive it from an array via `arr.dtype.type` (the callable), NEVER bare `arr.dtype`
        # (a non-callable dtype instance -> `rdtype(x)` would raise). Keep this contract so the
        # 1600+ `rdtype(...)` cast sites and the shared kernels stay uniform across backends.
        self.ndtype = np.float32 if precision == "float32" else np.float64

        # C2: pinned_host fast path for large D2H (DEFAULT ON; set PYNICAM_PINNED_D2H=0
        # to disable). jax's default np.asarray(device) tops out ~11 GB/s f64 on GH200;
        # moving device->pinned_host (NVLink-C2C) then asarray is BIT-EXACT (pure data
        # movement) and ~10x for >=16MB transfers. MEASURED 2026-06-25 gl08 full-resident:
        # D2H 12.2->117.5 GB/s (144.7 for >=16MB), Dynamics 5.80->5.13 (-11.5%), gl07
        # 12-step bit-exact 0.00e+00 vs off + 1.15e-11 vs gold. numpy backend never takes
        # this path -> stays bit-exact.
        use_pinned = os.environ.get("PYNICAM_PINNED_D2H", "1") != "0"
        pin_thresh = int(os.environ.get("PYNICAM_PINNED_D2H_MB", "16")) * 1024 * 1024
        self._pin_sh = None  # SingleDeviceSharding(pinned_host), built lazily

        # gated D2H transfer profiler (measures only -> values unchanged, bit-exact)
        prof = None
        if _profile("xfer"):
            import atexit
            prof = _XferProf(mode=("pinned" if use_pinned else "asarray"))
            atexit.register(prof.dump)

        def _to_numpy(x):
            nb = getattr(x, "nbytes", None)
            t0 = time.perf_counter() if prof is not None else 0.0
            if (use_pinned and self.type == "jax"
                    and nb is not None and nb >= pin_thresh):
                if self._pin_sh is None:
                    import jax.sharding as shd
                    # local_devices()[0] (not devices()[0]) so this stays THIS rank's GPU
                    # after jax.distributed.initialize makes devices() the global set.
                    self._pin_sh = shd.SingleDeviceSharding(
                        self.jax.local_devices()[0], memory_kind="pinned_host")
                xp = self.jax.device_put(x, self._pin_sh)
                xp.block_until_ready()
                r = np.asarray(xp)
            else:
                r = np.asarray(x)
            if prof is not None:
                prof.record(nb if nb is not None else r.nbytes,
                            time.perf_counter() - t0)
            return r

        if backend_name == "numpy":
            self.type = "numpy"
            self.xp = np
            self.dtype = self.ndtype
            self.to_numpy = _to_numpy
            self.jax = None

        elif backend_name == "jax":
            import jax
            jax.config.update("jax_enable_x64", precision == "float64")
            # Phase A (COMM sharding migration): join the multi-process JAX runtime and
            # build the global device mesh BEFORE any jax device op (jax.devices/device_put).
            # Gated PYNICAM_COMM_SHARDING (default off -> no behavior change; the transport
            # still uses mpi4jax.alltoall). Coordinator is distributed over the model's own
            # mpi4py world. See comm-replace-plan_v1.txt Phase A / memory pynicam-comm-architecture.
            if os.environ.get("PYNICAM_COMM_SHARDING", "0") != "0":
                self._init_distributed(jax)
                # Option-1 whole-step shard_map: device_consts must NOT memoize run-constants on
                # self._dev_cache. Built inside the per-step shard_map (or the warm-up nl-scan)
                # they are manual-axis / scan-scoped tracers; stashing them on a persistent object
                # leaks across traces (jax UnexpectedTracerError). Build FRESH every call for the
                # whole run instead -- bit-exact (same values), XLA CSEs the duplicates so there is
                # no runtime cost, only a little extra trace/compile work. Also read by the guarded
                # self._X_d const caches (mod_numfilter/mod_vi). plan v3 §10b.
                self._devconst_bypass = True
            import jax.numpy as jnp

            self.type = "jax"
            self.jax = jax
            self.xp = jnp
            self.dtype = jnp.float32 if precision == "float32" else jnp.float64
            self.to_numpy = _to_numpy

            # gated H2D profiler: wrap xp so xp.asarray(host) is measured (default off)
            if _profile("h2d"):
                import atexit
                h2d_prof = _XferProf(mode="asarray", out_env="PYNICAM_H2D_PROF_OUT",
                                     tag="xp.asarray H2D")
                atexit.register(h2d_prof.dump)
                self.xp = _XpProxy(jnp, h2d_prof)

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

    def set_at(self, a, idx, val):
        """Backend-agnostic element assignment: the value of `a` with `a[idx]`
        set to `val`.

        numpy backend : in-place `a[idx] = val`, then return `a` (same object).
        jax backend   : functional `return a.at[idx].set(val)` (jax arrays are
                        immutable, so this returns a NEW array).

        This is the abstraction over the one primitive numpy and jax spell
        differently -- `a[idx] = x` vs `a.at[idx].set(x)` -- so a single
        xp-clean kernel can serve both backends without a per-site
        `if bk.type == "jax"`. Always use the return value and rebind:

            a = bk.set_at(a, idx, val)

        `idx` is any index expression `a[idx]` accepts (int, tuple, slice(...),
        boolean/integer array). Because the numpy branch mutates in place and
        the jax branch does not, callers must not rely on aliases of `a` seeing
        the update -- rebind the name, as above.
        """
        if self.type == "jax":
            return a.at[idx].set(val)
        a[idx] = val
        return a

    def add_at(self, a, idx, val):
        """Backend-agnostic `a[idx] += val` (accumulate). numpy: in-place then
        return; jax: `a.at[idx].add(val)`. Same rebind contract as set_at."""
        if self.type == "jax":
            return a.at[idx].add(val)
        a[idx] += val
        return a

    def set_loop_ctx(self, c):
        """Set the coarse loop-context tag for the xfer profiler (in-loop audit). Cheap
        global assignment; no effect unless PYNICAM_XFER_PROF_SITES is on."""
        global _LOOP_CTX
        _LOOP_CTX = c

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
        # Option-1 shard_map (PYNICAM_COMM_SHARDING): while a fused _step_core runs INSIDE a
        # jax.shard_map, building a const here yields a manual-axis TRACER; caching it on `owner`
        # (a persistent object) is a trace side effect whose value ESCAPES to the next chunk's
        # trace -> UnexpectedTracerError. So bypass the cache in that window: build fresh each
        # call (the values are pure constants, so XLA CSEs the duplicates at compile time -> no
        # runtime cost) and never write to owner.__dict__. Gated by _devconst_bypass (set only
        # around the shard_map loop in Dyn._run_chunk_shardmap).
        if _profile("devconst"):
            print(f"DC_CALL key={key} bypass={getattr(self, '_devconst_bypass', False)} warm={getattr(self,'_in_warm',False)}", flush=True)
        if getattr(self, "_devconst_bypass", False):
            return {k: (self.xp.asarray(v) if isinstance(v, np.ndarray) else v)
                    for k, v in builder().items()}
        cache = owner.__dict__.setdefault("_dev_cache", {})
        d = cache.get(key)
        if d is None:
            d = {k: (self.xp.asarray(v) if isinstance(v, np.ndarray) else v)
                 for k, v in builder().items()}
            cache[key] = d
            # RC-71 (b): confirm device_consts is build-once (setup), not per-nl. On a
            # cache MISS (the only place the geometry asarrays run), print the key. If a
            # key prints exactly once over a full run it is cached setup; if it prints
            # per-nl it is a live in-loop leak. Gated, print-only -> bit-exact.
            if _profile("devconst"):
                _n = sum(1 for v in d.values() if hasattr(v, "shape"))
                try:
                    import jax as _jx
                    _tr = any(isinstance(v, _jx.core.Tracer) for v in d.values() if hasattr(v, "shape"))
                except Exception:
                    _tr = "?"
                print(f"DEVCONST_BUILD key={key} arrays={_n} tracer={_tr} warm={getattr(self,'_in_warm',False)}", flush=True)
        return d

backend = Backend()
