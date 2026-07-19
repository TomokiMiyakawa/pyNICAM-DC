#!/usr/bin/env python
"""Isolate the spike #5 S1 hang: the fused scan form (shard_map(scan(compute+pole+ragged)))
hung (rc=124), while spike #3 proved ragged-in-scan-in-shard_map works. Bisect the cause
with progress markers BEFORE/AFTER each variant so a compile/collective hang is localized
(not masked). KSTEP small; each variant barriers + prints on rank0.

  D1: scan(stencil + pole, NO ragged)        -> is it the 2-array/pole carry or the scan?
  D2: scan(ragged only, single 1-array carry) -> ~spike #3, expected OK (control)
  D3: scan(FULL body: stencil+pole+ragged)    -> the S1 that hung
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
import importlib
_pvary = None
for _p in ("jax.lax", "jax._src.shard_map", "jax._src.core"):
    try:
        _m = importlib.import_module(_p)
        if hasattr(_m, "pvary"): _pvary = getattr(_m, "pvary"); break
    except Exception: pass

N = size; I = J = 18; K = 40; L = 1; V = 6; IPL = 6; LPL = 1
KSTEP = int(os.environ.get("SUB_K", "3"))
mesh = Mesh(np.array(jax.devices()), ("p",))
sh6 = NamedSharding(mesh, P("p", None, None, None, None, None))
sh5 = NamedSharding(mesh, P("p", None, None, None, None))
FACE = J * V
S = jnp.asarray(np.array([[FACE if j == (i + 1) % N else 0 for j in range(N)] for i in range(N)], np.int32))
MAXROWS = FACE

locv  = (np.arange(I*J*K*L*V, dtype=np.float32).reshape(1, I, J, K, L, V) * 1e-4 + rank).astype(np.float32)
locpl = (np.ones((1, IPL, K, LPL, V), np.float32) * (rank + 1)).astype(np.float32)
gvar    = jax.make_array_from_process_local_data(sh6, locv,  (N, I, J, K, L, V))
gvar_pl = jax.make_array_from_process_local_data(sh5, locpl, (N, IPL, K, LPL, V))
gface   = jax.make_array_from_process_local_data(NamedSharding(mesh, P("p", None)),
                                                 (np.arange(FACE, dtype=np.float32) + rank).reshape(1, FACE), (N, FACE))

def _ragged(operand):
    r = lax.axis_index("p"); idx = jnp.arange(N)
    ss = S[r, :]; rs = S[:, r]
    io = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(ss)[:-1].astype(jnp.int32)])
    oo = jnp.sum(jnp.where((idx < r)[:, None], S, 0), axis=0).astype(jnp.int32)
    res = lax.ragged_all_to_all(operand, jnp.zeros(MAXROWS, operand.dtype), io, ss, oo, rs, axis_name="p")
    return _pvary(res, "p") if _pvary else res

def _stencil_pole(var6, varpl5):
    v = var6[0]; pl = varpl5[0]
    v = v + 0.1 * (jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0) - 2.0 * v)
    v = v + 0.1 * (jnp.roll(v, 1, 1) + jnp.roll(v, -1, 1) - 2.0 * v)
    v = v.at[0, :, :, 0, :].add(0.01 * jnp.mean(pl, axis=0)[:, 0, :][None, :, :])
    return v[None], varpl5

def _full(var6, varpl5):
    v6, pl = _stencil_pole(var6, varpl5)
    v = v6[0]
    recvd = _ragged(v[0, :, 0, 0, :].reshape(FACE))
    v = v.at[I - 1, :, 0, 0, :].set(recvd[:FACE].reshape(J, V))
    return jnp.tanh(v)[None], pl

sm6_5 = dict(mesh=mesh, in_specs=(P("p",None,None,None,None,None), P("p",None,None,None,None)),
             out_specs=(P("p",None,None,None,None,None), P("p",None,None,None,None)))

def D1(v6, pl):
    def b(c, _): nv, npl = _stencil_pole(c[0], c[1]); return (nv, npl), nv[0,0,0,0,0,0]
    (fv, fpl), _ = lax.scan(b, (v6, pl), xs=None, length=KSTEP); return fv, fpl
def D3(v6, pl):
    def b(c, _): nv, npl = _full(c[0], c[1]); return (nv, npl), nv[0,0,0,0,0,0]
    (fv, fpl), _ = lax.scan(b, (v6, pl), xs=None, length=KSTEP); return fv, fpl
def D2(face2):
    def b(c, _): nc = _ragged(c[0]); return (nc,), nc[0]
    (fc,), _ = lax.scan(b, (face2[0],), xs=None, length=KSTEP); return fc[None]

J1 = jax.jit(jax.shard_map(D1, **sm6_5))
J3 = jax.jit(jax.shard_map(D3, **sm6_5))
J2 = jax.jit(jax.shard_map(D2, mesh=mesh, in_specs=(P("p", None),), out_specs=P("p", None)))

def trydiag(name, fn, args):
    comm.Barrier()
    if rank == 0: print(f"[diag] START {name} ...", flush=True)
    status = 1; msg = "ok"
    try:
        out = fn(*args); jax.block_until_ready(out)
    except Exception as e:
        status = 0; msg = f"{type(e).__name__}: {str(e)[:200]}"
    allok = comm.allreduce(status, op=MPI.MIN)     # symmetric on ALL ranks (no desync)
    if rank == 0:
        print(f"[diag] END   {name}: {'OK' if allok else 'FAIL'}  (rank0 {msg})", flush=True)
    comm.Barrier(); return allok

if rank == 0: print(f"[diag] N={N} KSTEP={KSTEP} jax={jax.__version__} pvary={'yes' if _pvary else 'no'}", flush=True)
trydiag("D1 scan(stencil+pole, NO ragged)", J1, (gvar, gvar_pl))
trydiag("D2 scan(ragged only, 1-array)   ", J2, (gface,))
trydiag("D3 scan(FULL stencil+pole+ragged)", J3, (gvar, gvar_pl))
if rank == 0: print("[diag] DONE", flush=True)
comm.Barrier()
