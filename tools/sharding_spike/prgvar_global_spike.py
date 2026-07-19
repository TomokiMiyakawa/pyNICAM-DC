#!/usr/bin/env python
"""inc2 validation: local <-> GLOBAL-SHARDED prgvar bridge (Option-1 / plan v2 Phase C2).

Reproduces the ADM prgvar shapes (PRG_var = (i,j,k,l,vmax); pole = (g_pl,k,l,vmax)) with
rank-distinct values, then exercises the EXACT primitive the Dyn helpers use
(_prgvar_to_global_sharded / _prgvar_from_global_sharded):

  1. promote local (i,j,k,l,v) -> global-sharded (nproc,i,j,k,l,v) on mesh 'p'
  2. extract this rank's block back -> assert BIT-EXACT round-trip (identity)
  3. assert global shape + that addressable_shards[0].data[0] == the original local block
  4. run a trivial jax.shard_map over the global array and assert each shard sees ITS local
     block (leading process dim squeezed) -- the property inc3's step wrapper relies on

Success = all asserts pass on every rank -> the inc2 bridge is correct; inc3 can wrap the
step in shard_map on these global handles. Tiny K/vmax; N = process count.
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
import numpy as np, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

N = size
I = J = 18; K = 40; L = 1; VMAX = 11; GPL = 6          # gl05-ish region + pole shapes
mesh = Mesh(np.array(jax.devices()), ("p",))

# rank-distinct local blocks (so a mis-placed shard is caught, not masked by equal values)
loc    = (np.arange(I*J*K*L*VMAX, dtype=np.float64).reshape(I, J, K, L, VMAX) * 1e-3 + rank*1000.0)
loc_pl = (np.arange(GPL*K*L*VMAX, dtype=np.float64).reshape(GPL, K, L, VMAX) * 1e-3 + rank*1000.0 + 7.0)
loc    = jnp.asarray(loc); loc_pl = jnp.asarray(loc_pl)

def to_global(x):                                       # == Dyn._prgvar_to_global_sharded
    sh = NamedSharding(mesh, P('p', *([None] * x.ndim)))
    return jax.make_array_from_process_local_data(sh, x[None], (N,) + x.shape)
def from_global(g):                                     # == Dyn._prgvar_from_global_sharded
    return g.addressable_shards[0].data[0]

g    = to_global(loc)
g_pl = to_global(loc_pl)

ok = 1; why = "ok"
def check(cond, name):
    global ok, why
    if not cond and ok:
        ok = 0; why = f"FAIL:{name}"

# (1)+(2)+(3) round-trip + shape + placement
check(tuple(g.shape) == (N, I, J, K, L, VMAX), "g.shape")
check(tuple(g_pl.shape) == (N, GPL, K, L, VMAX), "g_pl.shape")
back    = np.asarray(from_global(g))
back_pl = np.asarray(from_global(g_pl))
check(np.array_equal(back,    np.asarray(loc)),    "roundtrip_var")
check(np.array_equal(back_pl, np.asarray(loc_pl)), "roundtrip_pl")

# (4) shard_map sees the LOCAL block (leading proc dim squeezed) -- inc3's wrapper property
def _body(gv, gvpl):
    v  = gv[0]                                           # (i,j,k,l,v) local shard
    pl = gvpl[0]
    return (v + 1.0)[None], (pl + 1.0)[None]
_sm = jax.jit(jax.shard_map(
    _body, mesh=mesh,
    in_specs=(P('p', None, None, None, None, None), P('p', None, None, None, None)),
    out_specs=(P('p', None, None, None, None, None), P('p', None, None, None, None))))
o, o_pl = _sm(g, g_pl)
so    = np.asarray(from_global(o))
so_pl = np.asarray(from_global(o_pl))
check(np.allclose(so,    np.asarray(loc)    + 1.0), "shardmap_var")
check(np.allclose(so_pl, np.asarray(loc_pl) + 1.0), "shardmap_pl")

allok = comm.allreduce(ok, op=MPI.MIN)
whys = comm.gather(why, root=0)
if rank == 0:
    bad = [w for w in whys if w != "ok"]
    print(f"[inc2] N={N} shapes var{tuple(loc.shape)}->g{tuple(g.shape)} pole{tuple(loc_pl.shape)}->g{tuple(g_pl.shape)}", flush=True)
    print(f"[inc2] RESULT: {'ALL-PASS' if allok else 'FAIL'}  {'' if allok else bad}", flush=True)
    print("[inc2] DONE", flush=True)
comm.Barrier()
