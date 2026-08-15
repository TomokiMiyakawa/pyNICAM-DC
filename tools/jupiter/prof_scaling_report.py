#!/usr/bin/env python3
"""
prof_scaling_report.py -- find what limits weak scaling, from a run's per-rank timers.

The ladder's per-step numbers say gl09 -> gl11 -> gl13 gets slower but not WHY. This
tool answers that from data the code already produces, by joining three things nobody
had joined before:

  1. per-rank rap timers      msg.pe%08d  "*** ID=nnn : NAME  T=  ttt N=nnn"
  2. where each rank ran      the placement file (rank -> node)
  3. what each rank OWNS      the mnginfo (territory perimeter / degree / connectivity)

and reporting the ACROSS-RANK DISTRIBUTION of each timer plus how it correlates with (2)
and (3). Mean s/step cannot distinguish "the network gets slower" from "a few ranks are
slow and everyone waits"; this can.

The decisive split is COMM_barrier vs COMM_data_transfer:

    COMM_barrier        time spent waiting at the pre-exchange barrier
                        = LOAD IMBALANCE / straggler wait
    COMM_data_transfer  time spent in the exchange itself
                        = NETWORK (volume, degree, hop count)

so the run being analysed must keep the barrier ON, i.e. must NOT set
PYNICAM_COMM_NO_BARRIER=1. With the barrier off the two are conflated and the barrier
row will read ~0 -- the tool warns when it sees that.

It must also be a PLAIN run (PYNICAM_FUSE_TIMELOOP=0). Under FUSE_TIMELOOP the 40 fused
steps run inside a lax.scan and the Python-level timers only ever see the warmup steps:
a fused gl12 run reports MAIN___Dynamics N=3 and MAIN_COMM_data_transfer N=22 for a
43-step run. The tool checks N and says so rather than reporting nonsense.

Usage
-----
    python prof_scaling_report.py <run_dir> \
        [--placement sweep/logs/<rung>_placement.txt] \
        [--mnginfo  kit/mnginfo/rl05-prc001024.toml] \
        [--timers COMM_data_transfer,COMM_barrier,_Dynamics,_Atmos] \
        [--top 10]

Compare two rungs by running it on each and reading the FRACTION column: that is the
number that must stay flat for weak scaling to hold.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

_RAP = re.compile(r"^\*\*\* ID=(\d+)\s*:\s*(\S+)\s+T=\s*([0-9.eE+-]+)\s+N=(\d+)")
_DEL = re.compile(r"^\*\*\* ID=(\d+)\s*:\s*(\S+)\s+dT=\s*([0-9.eE+-]+)\s+dN=(\d+)")
_STP = re.compile(r"^\*\*\* Per-step Time Report \[step (\d+)\]")
_MSG = re.compile(r"^msg\.pe(\d{8})$")

DEFAULT_TIMERS = ("_Atmos", "___Dynamics", "COMM_data_transfer", "COMM_barrier")

# ---------------------------------------------------------------------------
# READ THIS BEFORE TRUSTING ANY NUMBER THIS TOOL PRINTS.
#
# The rap timers are HOST-side (prc.PRC_MPItime() around Python calls) and the JAX
# backend dispatches ASYNCHRONOUSLY. Once a step is compiled, the Python call only
# ENQUEUES device work and returns in microseconds; the GPU and the network run later,
# and the wall time surfaces in whichever timer happens to contain the first blocking
# sync. Measured at gl11 pe64 (job 1098292), a steady step:
#
#     _Atmos 0.539   ___Dynamics 0.539
#       COMM_barrier 0.000   COMM_data_transfer 0.001
#       Tracer_Advection 0.001   Pre_Post 0.002       <- children sum to ~0.004
#
# 0.535 s of a 0.539 s step is in NO leaf timer. COMM_barrier reads 0 because every rank
# races ahead enqueueing and reaches the barrier immediately -- the real waiting happens
# at a later device sync, not there.
#
# => These timers CANNOT attribute comm vs compute in this code. Not in FUSE_TIMELOOP
#    mode (where they only see warmup), and not in PLAIN mode either. The `unaccounted`
#    row below quantifies the blindness; when it dominates, the breakdown is meaningless
#    and the tool says so.
#
# Attribution needs DEVICE-side timing: nsys over a steady step range (PYNICAM_NSYS_STEP
# / PYNICAM_NSYS_STEP_END, already in driver-dc.py), CUDA events inside the step, or a
# sync-gated profiling mode. Alternatively perturb the halo and difference the wall time
# -- that is the only method that has actually produced a number so far (gl12 job
# 1092608: -56.5% halo volume bought -4.83% s/step).
# ---------------------------------------------------------------------------


def parse_run(run_dir):
    """rank -> (cumulative {name: (T,N)}, per-step [ {name: (dT,dN)}, ... ])."""
    per_rank = {}
    for name in sorted(os.listdir(run_dir)):
        m = _MSG.match(name)
        if not m:
            continue
        rank = int(m.group(1))
        cum, steps, cur = {}, [], None
        with open(os.path.join(run_dir, name), errors="replace") as fh:
            for line in fh:
                if _STP.match(line):
                    cur = {}
                    steps.append(cur)
                    continue
                d = _DEL.match(line)
                if d:
                    if cur is not None:
                        cur[d.group(2)] = (float(d.group(3)), int(d.group(4)))
                    continue
                r = _RAP.match(line)
                if r:
                    cum[r.group(2)] = (float(r.group(3)), int(r.group(4)))
        per_rank[rank] = (cum, steps)
    if not per_rank:
        raise SystemExit(f"no msg.pe######## files in {run_dir} "
                         f"(needs io_log_allnode = true)")
    return per_rank


def steady_steps(steps, base="MAIN__Atmos"):
    """Indices of steps that are NOT compile-dominated (delta < 3x median)."""
    vals = [(i, s[base][0]) for i, s in enumerate(steps) if base in s]
    if not vals:
        return []
    med = float(np.median([v for _, v in vals]))
    return [i for i, v in vals if v < 3.0 * med]


def parse_placement(path):
    """rank -> node, from lines 'rank <id> <host> gpu=<n>'."""
    rank2node = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 3 and f[0] == "rank":
                rank2node[int(f[1])] = f[2].split(".")[0]
    return rank2node


def territory_metrics(mnginfo):
    """rank -> (perimeter, degree, n_components), via mkmnginfo_compact's own measure()."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "../../pynicamdc/prep/mnginfo"))
    import toml
    from mkmnginfo_compact import load_neighbours

    data = toml.load(mnginfo)
    nb = load_neighbours(data)
    nprc = data["PROC_INFO"]["NUM_OF_PROC"]
    part = [[int(x) for x in data["RGN_MNG_INFO"][f"{p:06}"]["MNG_RGNID"]]
            for p in range(nprc)]
    owner = [-1] * len(nb)
    for p, regs in enumerate(part):
        for g in regs:
            owner[g] = p

    out = {}
    for p, regs in enumerate(part):
        mine = set(regs)
        ext, peers = 0, set()
        for g in regs:
            for m in nb[g]:
                if m not in mine:
                    ext += 1
                    peers.add(owner[m])
        seen, comps = set(), 0
        for g in regs:
            if g in seen:
                continue
            comps += 1
            stack = [g]
            seen.add(g)
            while stack:
                for m in nb[stack.pop()]:
                    if m in mine and m not in seen:
                        seen.add(m)
                        stack.append(m)
        out[p] = (ext, len(peers), comps)
    return out


def pick(per_rank, want):
    """Resolve a bare timer name to the actual prefixed key present in the logs."""
    keys = set()
    for cum, _steps in per_rank.values():
        keys.update(cum)
    hits = [k for k in sorted(keys) if k.endswith(want) or k == want]
    # prefer MAIN_ over INIT_: the time loop is what scales, init is one-off
    main = [k for k in hits if k.startswith("MAIN")]
    return (main or hits or [None])[0]


def dist(vals):
    a = np.asarray(vals, dtype=float)
    return dict(mean=a.mean(), min=a.min(), max=a.max(),
                p50=np.percentile(a, 50), p95=np.percentile(a, 95),
                p99=np.percentile(a, 99), spread=100.0 * (a.max() - a.mean())
                / a.mean() if a.mean() else 0.0)


def main():
    ap = argparse.ArgumentParser(
        description="Per-rank timer distribution + straggler attribution for a run")
    ap.add_argument("run_dir")
    ap.add_argument("--placement", help="rank->node map written by the sbatch")
    ap.add_argument("--mnginfo", help="mnginfo toml for territory attribution")
    ap.add_argument("--timers", default=",".join(DEFAULT_TIMERS))
    ap.add_argument("--top", type=int, default=10, help="how many slow ranks to list")
    a = ap.parse_args()

    per_rank = parse_run(a.run_dir)
    ranks = sorted(per_rank)
    print(f"run: {a.run_dir}")
    print(f"ranks reporting: {len(ranks)}  (rank ids {ranks[0]}..{ranks[-1]})")

    wanted = [w.strip() for w in a.timers.split(",") if w.strip()]
    resolved = {w: pick(per_rank, w) for w in wanted}
    missing = [w for w, k in resolved.items() if k is None]
    for w in missing:
        print(f"  !! timer {w!r} not present in any msg.pe file")

    # --- is this run even instrumentable? ------------------------------------
    dyn = resolved.get("___Dynamics")
    if dyn:
        ns = [per_rank[r][0][dyn][1] for r in ranks if dyn in per_rank[r][0]]
        if ns and max(ns) <= 5:
            print(f"\n  !! {dyn} fired only N={max(ns)} times. This is a FUSED run "
                  f"(PYNICAM_FUSE_TIMELOOP=1): the steady steps ran inside a lax.scan and "
                  f"the Python timers never saw them. Re-run with FUSE_TIMELOOP=0 for a "
                  f"breakdown; the numbers below describe the WARMUP steps only.")

    base = resolved.get("_Atmos")
    have_steps = base is not None and any(per_rank[r][1] for r in ranks)
    if not have_steps:
        print("\n  !! no per-step blocks found (needs PYNICAM_PROFILE=perstep). Only "
              "cumulative totals are available, and those are dominated by the "
              "JIT-compile steps -- at gl11 the first step was 78% of the whole run.")

    # --- steady-state per-step means (the only defensible numbers) -----------
    STEADY = {}          # timer -> {rank: mean steady dT}
    nsteady = 0
    if have_steps:
        for r in ranks:
            steps = per_rank[r][1]
            idx = steady_steps(steps, base)
            if not idx:
                continue
            nsteady = max(nsteady, len(idx))
            for w in wanted:
                k = resolved[w]
                if k is None:
                    continue
                vals = [steps[i][k][0] for i in idx if k in steps[i]]
                if vals:
                    STEADY.setdefault(w, {})[r] = float(np.mean(vals))

    src = STEADY if STEADY else None
    label = (f"STEADY per-step mean (compile steps excluded, {nsteady} steps/rank)"
             if src else "CUMULATIVE totals (compile-contaminated -- see warning above)")
    if src is None:
        src = {}
        for w in wanted:
            k = resolved[w]
            if k is None:
                continue
            src[w] = {r: per_rank[r][0][k][0] for r in ranks if k in per_rank[r][0]}

    base_tot = np.mean(list(src[base_w].values())) if (base_w := "_Atmos") in src else None

    print(f"\n{label}")
    print(f"{'timer':<26} {'N':>5} {'mean':>9} {'p50':>9} {'p95':>9} "
          f"{'max':>9} {'max-mean':>9} {'frac':>7}")
    print("-" * 92)
    series = {}
    for w in wanted:
        if w not in src or not src[w]:
            continue
        k = resolved[w]
        nrep = max((per_rank[r][0][k][1] for r in ranks if k in per_rank[r][0]), default=0)
        d = dist(list(src[w].values()))
        series[w] = src[w]
        frac = f"{100.0 * d['mean'] / base_tot:6.1f}%" if base_tot else "     --"
        print(f"{w:<26} {nrep:>5} {d['mean']:9.4f} {d['p50']:9.4f} {d['p95']:9.4f} "
              f"{d['max']:9.4f} {d['spread']:8.1f}% {frac:>7}")

    # --- the blindness check: does anything actually account for the step? ---
    if base_tot and STEADY:
        kids = sum(np.mean(list(src[w].values())) for w in src
                   if w not in ("_Atmos", "___Dynamics") and src[w])
        unacc = base_tot - kids
        print(f"{'unaccounted':<26} {'':>5} {unacc:9.4f} {'':>9} {'':>9} {'':>9} "
              f"{'':>9} {100.0 * unacc / base_tot:6.1f}%")
        print("-" * 92)
        if unacc > 0.5 * base_tot:
            print("!! HOST TIMERS ARE BLIND HERE. The named timers account for only "
                  f"{100.0 * kids / base_tot:.1f}% of a step.")
            print("   JAX dispatches asynchronously: these Python-side timers measure "
                  "ENQUEUE cost, not")
            print("   device execution, so comm-vs-compute CANNOT be attributed from "
                  "them. COMM_barrier")
            print("   reads ~0 for the same reason -- ranks race ahead enqueueing and "
                  "never wait there.")
            print("   Use device-side timing (PYNICAM_NSYS_STEP over a steady range, "
                  "CUDA events, or a")
            print("   sync-gated profile mode), or perturb the halo and difference the "
                  "wall time instead.")
            print("   Everything below is reported for completeness but does NOT support "
                  "an attribution.")
            return
    print("-" * 92)
    print("frac = share of _Atmos. For weak scaling to hold the COMM_* fractions must")
    print("stay FLAT as ranks increase: COMM_data_transfer growing => network is the")
    print("limiter; COMM_barrier growing => load imbalance is.")

    # --- straggler attribution ----------------------------------------------
    key = "COMM_barrier" if "COMM_barrier" in series and \
          sum(series["COMM_barrier"].values()) > 0 else None
    key = key or ("COMM_data_transfer" if "COMM_data_transfer" in series else None)
    if key is None:
        return
    s = series[key]
    order = sorted(s, key=lambda r: -s[r])
    print(f"\nslowest {a.top} ranks by {key}:")
    rank2node = parse_placement(a.placement) if a.placement else {}
    terr = territory_metrics(a.mnginfo) if a.mnginfo else {}
    hdr = f"  {'rank':>6} {'T':>9}"
    if rank2node:
        hdr += f" {'node':>14}"
    if terr:
        hdr += f" {'perim':>6} {'deg':>4} {'parts':>6}"
    print(hdr)
    for r in order[:a.top]:
        line = f"  {r:6d} {s[r]:9.3f}"
        if rank2node:
            line += f" {rank2node.get(r, '?'):>14}"
        if terr:
            p, dg, c = terr.get(r, (0, 0, 0))
            line += f" {p:6d} {dg:4d} {c:6d}"
        print(line)

    # --- the two attribution questions, answered numerically ----------------
    if terr:
        split = [s[r] for r in s if terr.get(r, (0, 0, 1))[2] > 1]
        whole = [s[r] for r in s if terr.get(r, (0, 0, 1))[2] == 1]
        if split and whole:
            ds = 100.0 * (np.mean(split) / np.mean(whole) - 1.0)
            print(f"\nterritory connectivity vs {key}:")
            print(f"  disconnected ranks ({len(split):4d}): mean {np.mean(split):.3f}")
            print(f"  connected    ranks ({len(whole):4d}): mean {np.mean(whole):.3f}")
            print(f"  ==> disconnected ranks are {ds:+.1f}% slower")
            print("      A large positive number means the mnginfo SHAPE is the limiter"
                  " and Hilbert should fix it.")
        else:
            print(f"\nterritory connectivity vs {key}: all ranks are "
                  f"{'connected' if whole else 'disconnected'}; nothing to compare.")

    if rank2node:
        by_node = defaultdict(list)
        for r, v in s.items():
            by_node[rank2node.get(r, "?")].append(v)
        nm = {n: float(np.mean(v)) for n, v in by_node.items()}
        hi = sorted(nm, key=lambda n: -nm[n])
        allv = np.array(list(nm.values()))
        print(f"\nnode-level spread of {key} ({len(nm)} nodes):")
        print(f"  slowest node {hi[0]} {nm[hi[0]]:.3f}   "
              f"fastest {hi[-1]} {nm[hi[-1]]:.3f}   "
              f"ratio {nm[hi[0]] / nm[hi[-1]] if nm[hi[-1]] else float('nan'):.2f}x")
        print(f"  across-node CoV {100.0 * allv.std() / allv.mean():.1f}%")
        print("      A high ratio with slow ranks clustered on a few nodes points at"
              " PLACEMENT/fabric, not decomposition.")


if __name__ == "__main__":
    main()
