#!/usr/bin/env python3
"""
nsys_comm_split.py -- device-side comm/compute split from nsys traces.

WHY THIS EXISTS
Host-side rap timers cannot attribute time in this code: JAX dispatches asynchronously,
so Python timers measure ENQUEUE cost. At gl11 a steady step was 0.4834 s with the named
timers accounting for 1.9% of it (job 1098292). Only device-side timing can split comm
from compute. driver-dc.py wraps a chosen step range in cudaProfilerStart/Stop
(PYNICAM_NSYS_STEP / _END, live on the FUSED path too), so nsys captures the real
production config.

TWO TRAPS THAT MADE THE FIRST ATTEMPT (job 1098494) UNREADABLE
1. `nsys stats --report cuda_gpu_kern_sum` showed ONLY NCCL kernels plus a 0.09 ms
   `wrapped_compare`, i.e. "comm = 99.9% of GPU time". False. XLA runs the fused step body
   inside CUDA GRAPHS, and nsys's default --cuda-graph-trace=graph records a graph as one
   opaque activity, so its inner kernels never reach the kernel table. The compute was in
   CUPTI_ACTIVITY_KIND_GRAPH_TRACE all along (402 graph executions/rank). This tool reads
   that table, so it is correct with either granularity; pass --cuda-graph-trace=node when
   profiling if you also want the compute broken down per kernel.
2. ONE NCCL instance dominated: 667 ms when the next longest was 1.7 ms. cudaProfilerStart
   is not synchronised across ranks, so the first exchange inside the window absorbs the
   whole rank skew. Counting it makes comm look ~2x its real size. --skew-trim drops the
   single largest NCCL instance per rank (default); better still, capture two chunks and
   point --after at the second so the skew lands outside the analysed region.

WHAT NCCL TIME IS, AND IS NOT
NCCL device kernels spin-wait on the GPU until the peer's data arrives, so their duration
is transfer + WAITING FOR THE PEER. Never read it as bandwidth cost. A rung whose NCCL
share grows may be bandwidth-bound OR straggler-bound; the per-rank spread distinguishes
them (uniform => bandwidth, skewed => stragglers).

OVERLAP CHECK
compute and NCCL durations are summed independently, so compute+nccl can exceed the wall
span if they truly overlap. When it comes out well BELOW span, comm is NOT being hidden
behind compute -- that is itself a finding, and the residual is genuine idle.

Usage
-----
    python nsys_comm_split.py <trace_dir> [--pattern 'nsys_rank*.nsys-rep']
                              [--no-skew-trim] [--after MS] [--top 10]
"""

import argparse
import glob
import os
import re
import sqlite3
import subprocess

import numpy as np


def ensure_sqlite(rep):
    sq = rep.rsplit(".nsys-rep", 1)[0] + ".sqlite"
    if not os.path.exists(sq):
        r = subprocess.run(["nsys", "export", "--type", "sqlite",
                            "--force-overwrite", "true", "--output", sq, rep],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(sq):
            return None
    return sq


def analyse(sq, skew_trim=True, after_ms=0.0):
    con = sqlite3.connect(sq)

    def q(sql):
        try:
            return con.execute(sql).fetchall()
        except sqlite3.Error:
            return []

    span = q("""SELECT MIN(start), MAX(end) FROM (
                  SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL
                  UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE)""")
    if not span or span[0][0] is None:
        con.close()
        return None
    t0, t1 = span[0]
    cut = t0 + after_ms * 1e6

    nccl = [(s, e) for s, e in q(
        """SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds i ON k.demangledName=i.id
           WHERE i.value LIKE '%nccl%'""") if s >= cut]
    other = [(s, e) for s, e in q(
        """SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds i ON k.demangledName=i.id
           WHERE i.value NOT LIKE '%nccl%'""") if s >= cut]
    graph = [(s, e) for s, e in q(
        "SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE") if s >= cut]
    con.close()
    if not nccl:
        return None

    nd = np.array([e - s for s, e in nccl], dtype=float)
    skew = float(nd.max())
    nd_used = np.sort(nd)[:-1] if (skew_trim and nd.size > 1) else nd
    return dict(
        span=float(t1 - cut),
        nccl=float(nd_used.sum()), nccl_raw=float(nd.sum()), skew=skew,
        nccl_med=float(np.median(nd_used)), ninst=int(nd.size),
        compute=float(sum(e - s for s, e in graph)), ngraph=len(graph),
        other=float(sum(e - s for s, e in other)),
    )


def rank_of(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description="Device-side comm/compute split from nsys")
    ap.add_argument("trace_dir")
    ap.add_argument("--pattern", default="*.nsys-rep")
    ap.add_argument("--no-skew-trim", action="store_true",
                    help="keep the window-open skew instance (inflates comm ~2x)")
    ap.add_argument("--after", type=float, default=0.0, metavar="MS",
                    help="ignore activity in the first MS of the window")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    reps = sorted(glob.glob(os.path.join(a.trace_dir, a.pattern)), key=rank_of)
    if not reps:
        raise SystemExit(f"no traces matching {a.pattern!r} in {a.trace_dir}")

    per = {}
    for rep in reps:
        sq = ensure_sqlite(rep)
        if not sq:
            print(f"  !! could not export sqlite for {os.path.basename(rep)}")
            continue
        r = analyse(sq, skew_trim=not a.no_skew_trim, after_ms=a.after)
        if r is None:
            print(f"  !! rank {rank_of(rep)}: no GPU activity in the window "
                  f"(check PYNICAM_NSYS_STEP lands on a chunk boundary)")
            continue
        per[rank_of(rep)] = r
    if not per:
        raise SystemExit("no usable traces")

    ranks = sorted(per)
    ms = lambda v: v / 1e6  # noqa: E731
    mean = lambda f: float(np.mean([per[r][f] for r in ranks]))  # noqa: E731

    span, comp, nc, oth = mean("span"), mean("compute"), mean("nccl"), mean("other")
    acc = comp + nc + oth
    print(f"traces: {len(ranks)}   NCCL instances/rank {mean('ninst'):.0f}   "
          f"graph execs/rank {mean('ngraph'):.0f}")
    print(f"\n{'':<32}{'mean ms':>10} {'share':>8}")
    print("-" * 52)
    for lbl, v in (("wall span", span), ("compute (CUDA graphs)", comp),
                   ("NCCL (transfer + peer wait)", nc),
                   ("other kernels", oth), ("accounted", acc),
                   ("residual / idle", span - acc)):
        print(f"{lbl:<32}{ms(v):10.1f} {100.0 * v / span:7.1f}%")
    print("-" * 52)
    if not a.no_skew_trim:
        print(f"window-open skew dropped per rank: {ms(mean('skew')):.1f} ms "
              f"({100.0 * mean('skew') / span:.1f}% of span) -- "
              f"cudaProfilerStart is not rank-synchronised")

    sh = np.array([100.0 * per[r]["nccl"]
                   / (per[r]["compute"] + per[r]["nccl"] + per[r]["other"])
                   for r in ranks])
    print(f"\nNCCL share of accounted device time: mean {sh.mean():.1f}%  "
          f"min {sh.min():.1f}%  max {sh.max():.1f}%  (spread {sh.max() / sh.min():.1f}x)")
    if sh.max() / sh.min() > 2.0:
        print("  spread > 2x => STRAGGLER-shaped: some ranks wait far longer than others.")
    else:
        print("  spread tight => BANDWIDTH-shaped: all ranks pay a similar comm cost.")
    if acc < 0.9 * span:
        print(f"\ncompute+NCCL is only {100.0 * acc / span:.0f}% of the wall window, so comm is")
        print("NOT hidden behind compute -- the exchange is largely EXPOSED, and the")
        print(f"residual {100.0 * (span - acc) / span:.0f}% is genuine GPU idle.")

    order = sorted(ranks, key=lambda r: -per[r]["nccl"])
    print(f"\nhighest-NCCL {min(a.top, len(order))} ranks:")
    print(f"  {'rank':>6} {'nccl ms':>9} {'compute ms':>11} {'inst':>6}")
    for r in order[:a.top]:
        print(f"  {r:6d} {ms(per[r]['nccl']):9.1f} {ms(per[r]['compute']):11.1f} "
              f"{per[r]['ninst']:6d}")

    print("\nFor weak scaling, compare these shares against another rung. NCCL share")
    print("growing with a TIGHT rank spread => bandwidth/degree. NCCL share growing with a")
    print("WIDE spread, or idle growing => synchronisation. Both flat => the kernels.")
    print("nsys perturbs absolute time (gl11: 478 ms/step vs 360 unperturbed); read shares.")


if __name__ == "__main__":
    main()
