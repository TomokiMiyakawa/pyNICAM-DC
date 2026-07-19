#!/usr/bin/env python
"""DECISIVE, all-valid-executable test. Prior diags conflated (a) trace-time TYPE errors
(missing pvary -> {V:p} mismatch) with (b) genuine runtime HANGS. Here EVERY case is valid
and executable (pvary applied, pytrees matched), so a hang is a real hang and an OK is a real
pass. Tiny arrays, END-prints, short timeout.

  G3: 1-array ragged+pvary scan                         = spike #3 control, expect OK
  G2: 2-array scan, arr1 STENCIL (no collective) + arr2 passthrough   (the D1/E1 essence)
  G1: 2-array scan, arr1 RAGGED+pvary        + arr2 passthrough   <- THE Option-1 case
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

N = size; M = 8; M2 = 4; KSTEP = int(os.environ.get("SUB_K", "4"))
mesh = Mesh(np.array(jax.devices()), ("p",))
shA = NamedSharding(mesh, P("p", None)); shB = NamedSharding(mesh, P("p", None))
gA = jax.make_array_from_process_local_data(shA, (np.arange(M, dtype=np.float32) + rank*100).reshape(1, M), (N, M))
gB = jax.make_array_from_process_local_data(shB, (np.ones((1, M2), np.float32) * (rank+1)), (N, M2))
Sr = jnp.asarray(np.array([[M if j == (i+1) % N else 0 for j in range(N)] for i in range(N)], np.int32))

def _ragged(op):
    r = lax.axis_index("p"); idx = jnp.arange(N)
    ss = Sr[r, :]; rs = Sr[:, r]
    io = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(ss)[:-1].astype(jnp.int32)])
    oo = jnp.sum(jnp.where((idx < r)[:, None], Sr, 0), axis=0).astype(jnp.int32)
    return _pvary(lax.ragged_all_to_all(op, jnp.zeros(M, op.dtype), io, ss, oo, rs, axis_name="p"), "p")

def g3(a):
    def b(c, _): return (_ragged(c[0]),), c[0][0]
    (fa,), _ = lax.scan(b, (a[0],), xs=None, length=KSTEP); return fa[None]
def g2(a, bb):
    def b(c, _):
        v = c[0] + 0.1 * (jnp.roll(c[0], 1, 0) - c[0])          # stencil, no collective
        return (v, c[1]), v[0]
    (fa, fbb), _ = lax.scan(b, (a[0], bb[0]), xs=None, length=KSTEP); return fa[None], fbb[None]
def g1(a, bb):
    def b(c, _): return (_ragged(c[0]), c[1]), c[0][0]          # ragged+pvary arr1, arr2 passthrough
    (fa, fbb), _ = lax.scan(b, (a[0], bb[0]), xs=None, length=KSTEP); return fa[None], fbb[None]

def SM(fn, ins, outs):
    return jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=ins, out_specs=outs))
s1 = (P("p", None),); s2 = (P("p", None), P("p", None))
G3 = SM(g3, s1, P("p", None))
G2 = SM(g2, s2, (P("p", None), P("p", None)))
G1 = SM(g1, s2, (P("p", None), P("p", None)))

def trial(name, fn, args):
    comm.Barrier()
    if rank == 0: print(f"[d4] START {name}", flush=True)
    st = 1; msg = "ok"
    try:
        out = fn(*args)
        for a in (out if isinstance(out, tuple) else (out,)):
            if a is not None: a.block_until_ready()
    except Exception as e:
        st = 0; msg = f"{type(e).__name__}: {str(e)[:150]}"
    ok = comm.allreduce(st, op=MPI.MIN)
    if rank == 0: print(f"[d4] END   {name}: {'OK' if ok else 'FAIL'} ({msg})", flush=True)
    comm.Barrier(); return ok

if rank == 0: print(f"[d4] jax={jax.__version__} KSTEP={KSTEP} pvary=yes", flush=True)
trial("G3 1-array ragged+pvary scan (ctrl)", G3, (gA,))
trial("G2 2-array STENCIL scan (D1 essence)", G2, (gA, gB))
trial("G1 2-array RAGGED+pvary scan (Opt-1)", G1, (gA, gB))
if rank == 0: print("[d4] DONE", flush=True)
comm.Barrier()
