#!/usr/bin/env python
"""DECISIVE attribution of the S1 hang (diag4 G3 / spike#5 S1).

Established so far:
  - spike#3  scan(ragged, TRIVIAL all-zero offsets)        -> PASS
  - spike#5 S0  per-step shard_map(compute+pole+ragged, COMPUTED offsets, NO scan) -> PASS
  - diag4 G3 / spike#5 S1  scan(ragged, COMPUTED offsets)  -> HANG (rc=124)
So the hang is specific to scan + COMPUTED-offset ragged. This test isolates that ONE
variable head-to-head and separates COMPILE from EXECUTE so a hang is localized:

  A: scan(1-array ragged, TRIVIAL offsets) + pvary   = reproduce spike#3      -> expect OK
  B: scan(1-array ragged, COMPUTED offsets) + pvary  = the diag4 suspect      -> ?

Each case: AOT .lower().compile()  [prints COMPILED]  then execute+block  [prints RAN].
=> "COMPILED" but no "RAN"  => collective/runtime deadlock.
   no "COMPILED"            => compile hang.
Generous timeout in the .pbs (600s) so a slow compile is NOT misread as a hang.
"""
import os, socket, importlib
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank(); size = comm.Get_size(); host = socket.gethostname()
if rank == 0:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0))
    coord = f"{host}:{s.getsockname()[1]}"; s.close()
else:
    coord = None
coord = comm.bcast(coord, root=0); comm.Barrier()
import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
import numpy as np, jax.numpy as jnp, jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

_pvary = None
for _p in ("jax.lax", "jax._src.shard_map", "jax._src.core"):
    try:
        _m = importlib.import_module(_p)
        if hasattr(_m, "pvary"): _pvary = getattr(_m, "pvary"); break
    except Exception: pass
assert _pvary is not None, "pvary not found"

N = size; M = 8; KSTEP = int(os.environ.get("SUB_K", "6"))
mesh = Mesh(np.array(jax.devices()), ("p",))
shA = NamedSharding(mesh, P("p", None))
gA = jax.make_array_from_process_local_data(shA, (np.arange(M, dtype=np.float32) + rank*100).reshape(1, M), (N, M))
Sr = jnp.asarray(np.array([[M if j == (i+1) % N else 0 for j in range(N)] for i in range(N)], np.int32))

def _ragged_trivial(op):                       # spike#3 style: ring, all-zero offsets
    r = lax.axis_index("p"); idx = jnp.arange(N)
    ss = jnp.where(idx == (r + 1) % N, M, 0).astype(jnp.int32)
    rs = jnp.where(idx == (r - 1) % N, M, 0).astype(jnp.int32)
    io = jnp.zeros(N, jnp.int32); oo = jnp.zeros(N, jnp.int32)
    return _pvary(lax.ragged_all_to_all(op, jnp.zeros(M, op.dtype), io, ss, oo, rs, axis_name="p"), "p")

def _ragged_computed(op):                       # diag4 style: prefix-sum offsets
    r = lax.axis_index("p"); idx = jnp.arange(N)
    ss = Sr[r, :]; rs = Sr[:, r]
    io = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(ss)[:-1].astype(jnp.int32)])
    oo = jnp.sum(jnp.where((idx < r)[:, None], Sr, 0), axis=0).astype(jnp.int32)
    return _pvary(lax.ragged_all_to_all(op, jnp.zeros(M, op.dtype), io, ss, oo, rs, axis_name="p"), "p")

def make_scan(ragfn):
    def f(a):
        def b(c, _): return (ragfn(c[0]),), c[0][0]
        (fa,), _ = lax.scan(b, (a[0],), xs=None, length=KSTEP)
        return fa[None]
    return jax.jit(jax.shard_map(f, mesh=mesh, in_specs=(P("p", None),), out_specs=P("p", None)))

def run(name, ragfn):
    comm.Barrier()
    if rank == 0: print(f"[d5] START {name}", flush=True)
    st = 1; msg = "ok"; stage = "lower"
    try:
        J = make_scan(ragfn)
        compiled = J.lower(gA).compile()          # AOT compile (blocking) -- NO collective yet
        comm.Barrier()
        if rank == 0: print(f"[d5]   COMPILED {name}", flush=True)
        stage = "execute"
        out = compiled(gA); out.block_until_ready()   # collective runs here
        if rank == 0: print(f"[d5]   RAN {name} sample={float(np.asarray(out.addressable_shards[0].data).reshape(M)[0])}", flush=True)
    except Exception as e:
        st = 0; msg = f"[{stage}] {type(e).__name__}: {str(e)[:180]}"
    ok = comm.allreduce(st, op=MPI.MIN)
    if rank == 0: print(f"[d5] END   {name}: {'OK' if ok else 'FAIL'} ({msg})", flush=True)
    comm.Barrier(); return ok

if rank == 0: print(f"[d5] N={N} M={M} KSTEP={KSTEP} jax={jax.__version__} pvary=yes", flush=True)
run("A trivial-offset scan (spike#3 ctrl)", _ragged_trivial)
run("B computed-offset scan (diag4 suspect)", _ragged_computed)
if rank == 0: print("[d5] DONE", flush=True)
comm.Barrier()
