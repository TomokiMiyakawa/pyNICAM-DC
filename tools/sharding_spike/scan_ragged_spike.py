#!/usr/bin/env python
"""De-risk spike #3 (R1): does shard_map + ragged_all_to_all compose INSIDE jax.lax.scan?

phaseC_jitcheck already proved jit(shard_map(gather->ragged->scatter)) compiles (L2 PASS).
The OPEN question for production (FUSE_TIMELOOP, driver-dc.py:417) is whether the ragged
collective survives being nested inside the whole-step lax.scan -- i.e. a COLLECTIVE inside
a scan inside shard_map, then jit'd. If it compiles + runs correct, Phase G (comm/compute
overlap) is reachable and Phase C2 can target the fused form directly. If it does NOT nest,
the exchange must stay OUTSIDE the scan and the overlap route changes -- we want to know
BEFORE building the exchange body (comm-replace-plan_v2 sec 2 spike#3 / Phase C0).

Nesting levels tested (learn exactly where, if anywhere, it breaks):
  S0: shard_map(ragged) once                       (baseline, = ragged_spike)
  S1: shard_map( lax.scan_K( ragged ) )            <-- THE R1 question: collective in scan
  S2: jax.jit(S1)                                  <-- the fused production shape
Ring semantics: each rank sends its whole row to r+1; after K scan steps rank r holds the
row that originated on rank (r-K) mod N -> a cheap exact correctness oracle.
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
if rank == 0: print(f"[scan] N={size} coordinator={coord}", flush=True)

import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

N = size
M = 8                                    # feature width of the exchanged row
K = int(os.environ.get("SCAN_K", "6"))   # number of fused scan steps
mesh = Mesh(np.array(jax.devices()), ("p",))
shard = NamedSharding(mesh, P("p", None))

# global (N, M): row-block r = [r*100 + 0..M-1]; local shard on rank r = (1, M)
local0 = (np.arange(M, dtype=np.float32) + rank * 100.0).reshape(1, M)
g = jax.make_array_from_process_local_data(shard, local0, (N, M))

# ONE ring exchange (gather -> ragged -> scatter), reused as the scan body.
def _ring_body(operand2d):
    gi = jnp.arange(M)[::-1]
    x = operand2d[:, gi].reshape(M)                        # gather + flatten -> send buf (M,)
    r = lax.axis_index("p")
    idx = jnp.arange(N)
    send_sizes     = jnp.where(idx == (r + 1) % N, M, 0).astype(jnp.int32)
    input_offsets  = jnp.zeros(N, jnp.int32)
    output_offsets = jnp.zeros(N, jnp.int32)
    recv_sizes     = jnp.where(idx == (r - 1) % N, M, 0).astype(jnp.int32)
    out = jnp.zeros(M, jnp.float32)
    res = lax.ragged_all_to_all(x, out, input_offsets, send_sizes,
                                output_offsets, recv_sizes, axis_name="p")
    si = jnp.arange(M)[::-1]
    return jnp.zeros((1, M), jnp.float32).at[:, si].set(res)   # scatter (un-reverse)

# S0: one exchange under shard_map (baseline)
S0 = partial(jax.shard_map, mesh=mesh, in_specs=P("p", None), out_specs=P("p", None))(_ring_body)

# A ragged_all_to_all OUTPUT loses the 'p'-varying (vma) annotation the scan carry needs:
# input carry is float32[..]{V:p} but the collective returns plain float32[..], so scan's
# carry-type check rejects it. Re-mark the carry as varying over 'p' with jax.lax.pvary.
import importlib
_pvary = None
for _p in ("jax.lax", "jax._src.shard_map", "jax._src.core"):
    try:
        _m = importlib.import_module(_p)
        if hasattr(_m, "pvary"):
            _pvary = getattr(_m, "pvary"); break
    except Exception:
        pass
if rank == 0:
    print(f"[scan] jax={jax.__version__}  pvary={'yes' if _pvary else 'MISSING'}", flush=True)

# S1: K exchanges via lax.scan, INSIDE one shard_map region  <-- R1
def _scan_body(operand2d):
    def step(carry, _):
        nxt = _ring_body(carry)
        if _pvary is not None:
            nxt = _pvary(nxt, "p")                         # restore {V:p} so carry types match
        return nxt, nxt[0, 0]                              # stack first elem as a trace
    final, ys = lax.scan(step, operand2d, xs=None, length=K)
    return final
S1 = partial(jax.shard_map, mesh=mesh, in_specs=P("p", None), out_specs=P("p", None))(_scan_body)

# S2: jit the fused shape
S2 = jax.jit(S1)

def check(fn, name, steps):
    try:
        res = fn(g)
        got = np.asarray(res.addressable_shards[0].data).reshape(M)
        exp = np.arange(M, dtype=np.float32) + ((rank - steps) % N) * 100.0
        ok = np.allclose(got, exp)
        allok = comm.allreduce(1 if ok else 0, op=MPI.LAND)
        if rank == 0:
            print(f"[scan] {name}: {'ALL-PASS' if allok else 'FAIL'}  "
                  f"(rank0 got[0]={got[0]} exp[0]={exp[0]})", flush=True)
        return allok
    except Exception as e:
        if rank == 0:
            print(f"[scan] {name}: EXCEPTION {type(e).__name__}: {str(e)[:400]}", flush=True)
        comm.Barrier()
        return 0

r0 = check(S0, "S0 shard_map(ragged) x1", 1);                       comm.Barrier()
r1 = check(S1, f"S1 shard_map(scan_{K}(ragged))  <-- R1", K);       comm.Barrier()
r2 = check(S2, f"S2 jit(shard_map(scan_{K}(ragged)))  <-- fused", K); comm.Barrier()

if rank == 0:
    print(f"[scan] RESULT S0={'ok' if r0 else 'FAIL'} S1(scan)={'ok' if r1 else 'FAIL'} "
          f"S2(jit-fused)={'ok' if r2 else 'FAIL'}  (K={K}, N={N})", flush=True)
    verdict = "SCAN-NEST OK -> Phase G reachable" if (r1 and r2) else \
              "SCAN-NEST FAILS -> keep exchange OUTSIDE the scan (see spike#3 outcomes)"
    print(f"[scan] VERDICT: {verdict}", flush=True)
    print("[scan] DONE", flush=True)
comm.Barrier()
