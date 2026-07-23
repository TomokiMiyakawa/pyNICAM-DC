#!/usr/bin/env python
"""Isolate WHY D1 (shard_map(scan(stencil+pole)), 2-array carry, NO collective) HANGS while
spike #3 (shard_map(scan(ragged)), 1-array carry) passed. Prime suspect: shard_map's default
check_rep=True inserting a replication-check collective that deadlocks on the multi-array /
unchanged-pole out_spec. Also isolate single- vs multi-array carry. SHORT timeout; each
variant prints END before the next, so a hang localizes the exact cause.

  E2: shard_map(scan(1-array stencil)),          check_rep default  -> control, expect OK
  E1: shard_map(scan(2-array stencil+pole)),     check_rep=False     -> candidate FIX
  E3: shard_map(scan(2-array stencil+pole)),     check_rep default   -> = D1, expect HANG (last)
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
from functools import partial

N = size; I = J = 18; K = 40; L = 1; V = 6; IPL = 6; LPL = 1
KSTEP = int(os.environ.get("SUB_K", "4"))
mesh = Mesh(np.array(jax.devices()), ("p",))
sh6 = NamedSharding(mesh, P("p", None, None, None, None, None))
sh5 = NamedSharding(mesh, P("p", None, None, None, None))
locv  = (np.arange(I*J*K*L*V, dtype=np.float32).reshape(1, I, J, K, L, V) * 1e-4 + rank).astype(np.float32)
locpl = (np.ones((1, IPL, K, LPL, V), np.float32) * (rank + 1)).astype(np.float32)
gvar    = jax.make_array_from_process_local_data(sh6, locv,  (N, I, J, K, L, V))
gvar_pl = jax.make_array_from_process_local_data(sh5, locpl, (N, IPL, K, LPL, V))

def _stencil(v):
    v = v + 0.1 * (jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0) - 2.0 * v)
    v = v + 0.1 * (jnp.roll(v, 1, 1) + jnp.roll(v, -1, 1) - 2.0 * v)
    return v

def _one(var6):                          # 1-array carry
    def b(c, _): return _stencil(c[0])[None], None
    fv, _ = lax.scan(b, (var6,), xs=None, length=KSTEP); return fv
def _two(var6, varpl5):                   # 2-array carry (var + pole)
    def b(c, _):
        v = _stencil(c[0][0])
        v = v.at[0, :, :, 0, :].add(0.01 * jnp.mean(c[1][0], axis=0)[:, 0, :][None, :, :])
        return (v[None], c[1]), v[0, 0, 0, 0, 0]
    (fv, fpl), _ = lax.scan(b, (var6, varpl5), xs=None, length=KSTEP); return fv, fpl

sm1 = dict(mesh=mesh, in_specs=(P("p", None, None, None, None, None),),
           out_specs=P("p", None, None, None, None, None))
sm2 = dict(mesh=mesh, in_specs=(P("p", None, None, None, None, None), P("p", None, None, None, None)),
           out_specs=(P("p", None, None, None, None, None), P("p", None, None, None, None)))

def build(fn, sm, check_rep):
    try:
        return jax.jit(jax.shard_map(fn, check_rep=check_rep, **sm))
    except TypeError:                     # older/newer signature without check_rep kw
        return jax.jit(jax.shard_map(fn, **sm))

E2 = build(_one, sm1, True)
E1 = build(_two, sm2, False)
E3 = build(_two, sm2, True)

def trial(name, fn, args):
    comm.Barrier()
    if rank == 0: print(f"[d2] START {name}", flush=True)
    st = 1; msg = "ok"
    try:
        out = fn(*args)
        for a in (out if isinstance(out, tuple) else (out,)):
            if a is not None: a.block_until_ready()
    except Exception as e:
        st = 0; msg = f"{type(e).__name__}: {str(e)[:180]}"
    ok = comm.allreduce(st, op=MPI.MIN)
    if rank == 0: print(f"[d2] END   {name}: {'OK' if ok else 'FAIL'} ({msg})", flush=True)
    comm.Barrier(); return ok

if rank == 0: print(f"[d2] N={N} KSTEP={KSTEP} jax={jax.__version__}", flush=True)
trial("E2 scan(1-array), check_rep=True ", E2, (gvar,))
trial("E1 scan(2-array), check_rep=False", E1, (gvar, gvar_pl))
trial("E3 scan(2-array), check_rep=True ", E3, (gvar, gvar_pl))   # last: may hang
if rank == 0: print("[d2] DONE", flush=True)
comm.Barrier()
