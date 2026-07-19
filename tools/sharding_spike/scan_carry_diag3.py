#!/usr/bin/env python
"""MINIMAL clean repro: does a MULTI-array carry in lax.scan under shard_map hang, and does
check_rep matter? Tiny arrays, CORRECT pytrees, END-prints, short timeout. Controls for the
harness bugs that made diag/diag2 inconclusive (bare-vs-tuple pytree, silent check_rep
fallback). Uses ragged so it matches the real COMM case.

  probe: is check_rep actually accepted by jax.shard_map here? (print it -- don't guess)
  F1: shard_map(scan(1-array ragged))              -> = spike #3 control, expect OK
  F3: shard_map(2-array ragged, NO scan)           -> = spike #5 S0 shape, expect OK
  F2a: shard_map(scan(2-array: ragged arr1 + arr2 passthrough)), check_rep=True
  F2b: same, check_rep=False   <- the decisive pair (F2a vs F2b)
"""
import os, socket
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
import inspect

N = size; M = 8; M2 = 4; KSTEP = int(os.environ.get("SUB_K", "4"))
mesh = Mesh(np.array(jax.devices()), ("p",))
shA = NamedSharding(mesh, P("p", None)); shB = NamedSharding(mesh, P("p", None))
gA = jax.make_array_from_process_local_data(shA, (np.arange(M, dtype=np.float32) + rank*100).reshape(1, M), (N, M))
gB = jax.make_array_from_process_local_data(shB, (np.ones((1, M2), np.float32) * (rank+1)), (N, M2))

_CR = "check_rep" in inspect.signature(jax.shard_map).parameters
if rank == 0: print(f"[d3] jax={jax.__version__} shard_map.check_rep_param={_CR} KSTEP={KSTEP}", flush=True)

Sr = jnp.asarray(np.array([[M if j == (i+1) % N else 0 for j in range(N)] for i in range(N)], np.int32))
def _ragged(op):
    r = lax.axis_index("p"); idx = jnp.arange(N)
    ss = Sr[r, :]; rs = Sr[:, r]
    io = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(ss)[:-1].astype(jnp.int32)])
    oo = jnp.sum(jnp.where((idx < r)[:, None], Sr, 0), axis=0).astype(jnp.int32)
    return lax.ragged_all_to_all(op, jnp.zeros(M, op.dtype), io, ss, oo, rs, axis_name="p")

def one(a):
    def b(c, _): return (_ragged(c[0]),), c[0][0]
    (fa,), _ = lax.scan(b, (a[0],), xs=None, length=KSTEP); return fa[None]
def two_noscan(a, bb):
    return _ragged(a[0])[None], bb              # ragged once, pass bb through (no scan)
def two_scan(a, bb):
    def body(c, _): return (_ragged(c[0]), c[1]), c[0][0]
    (fa, fbb), _ = lax.scan(body, (a[0], bb[0]), xs=None, length=KSTEP)
    return fa[None], fbb[None]

def SM(fn, ins, outs, cr=None):
    kw = {} if (cr is None or not _CR) else {"check_rep": cr}
    return jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=ins, out_specs=outs, **kw))

s1 = (P("p", None),); s2 = (P("p", None), P("p", None))
F1  = SM(one,        s1, P("p", None))
F3  = SM(two_noscan, s2, (P("p", None), P("p", None)))
F2a = SM(two_scan,   s2, (P("p", None), P("p", None)), cr=True)
F2b = SM(two_scan,   s2, (P("p", None), P("p", None)), cr=False)

def trial(name, fn, args):
    comm.Barrier()
    if rank == 0: print(f"[d3] START {name}", flush=True)
    st = 1; msg = "ok"
    try:
        out = fn(*args)
        for a in (out if isinstance(out, tuple) else (out,)):
            if a is not None: a.block_until_ready()
    except Exception as e:
        st = 0; msg = f"{type(e).__name__}: {str(e)[:150]}"
    ok = comm.allreduce(st, op=MPI.MIN)
    if rank == 0: print(f"[d3] END   {name}: {'OK' if ok else 'FAIL'} ({msg})", flush=True)
    comm.Barrier(); return ok

trial("F1  scan(1-array ragged)          ", F1,  (gA,))
trial("F3  2-array ragged, NO scan       ", F3,  (gA, gB))
trial("F2b scan(2-array), check_rep=False", F2b, (gA, gB))
trial("F2a scan(2-array), check_rep=True ", F2a, (gA, gB))   # last: may hang
if rank == 0: print("[d3] DONE", flush=True)
comm.Barrier()
