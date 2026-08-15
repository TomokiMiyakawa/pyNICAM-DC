#!/usr/bin/env python3
"""
pick_leaf_arms.py -- carve an allocation into arms that differ ONLY in network locality.

WHY
The ladder's per-step cost looks like a staircase in network locality rather than a smooth
weak-scaling decay:

    gl09  1 node                  313.3 ms
    gl11  16 nodes, 1 leaf switch 360.2 ms   (+46.9)
    gl12  64 nodes, many leaves   388.7 ms   (+28.5)
    gl13  256 nodes, many l2      389.2 ms   (+0.5)   <- 4x the nodes, no cost

A JUPITER leaf switch (jpbi-<rack>-l1-<nn>) holds exactly 16 nodes = 64 GPUs, verified for
367 of 368 leaves. gl11 pe64 is therefore the largest rung that FITS IN ONE LEAF, and every
rung above it pays one flat penalty. But node count and leaf count moved together across
those rungs, so the two cannot be separated from that data.

This tool removes the confound: it picks subsets of ONE allocation that all have the same
node count and rank count, and differ only in how many switches they span.

    arm  leaves  l2  meaning
    A       1     1  fits one leaf, like gl11's measured placement
    B      >1     1  leaf boundary crossed, l2 boundary NOT
    C      >1    >1  both crossed, like gl12/gl13

If A -> B costs ~28 ms and B -> C costs ~0, the entire ladder is a leaf-boundary effect and
there is no weak-scaling degradation to chase above 64 GPUs.

TOPOLOGY RULE (verified against `scontrol show topology`)
    jpbo-<rack>-<nn>:  nn 01..16 -> jpbi-<rack>-l1-01
                       nn 17..32 -> jpbi-<rack>-l1-02
                       nn 33..48 -> jpbi-<rack>-l1-03
    l2 groups racks in fives: racks 001-005 -> jpbi-001-l2-01, 006-010 -> jpbi-006-l2-01, ...
One leaf out of 368 deviates (a rack whose l1-02 is [17-28]); it still maps to l1-02, so the
grouping stays correct.

Usage
-----
    # inside a job
    python pick_leaf_arms.py --nodes-per-arm 16
    # offline, to check what a hypothetical allocation would give
    python pick_leaf_arms.py --nodelist 'jpbo-010-[01-48],jpbo-011-[01-16]' --nodes-per-arm 16

Prints one line per arm:  ARM <name> <leaves> <l2s> <compressed nodelist>
and exits non-zero if arm A (the reference) cannot be formed.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

_HOST = re.compile(r"^jpbo-(\d+)-(\d+)$")


def expand(nodelist):
    out = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"scontrol show hostnames failed: {out.stderr.strip()}")
    return [h for h in out.stdout.split() if h]


def compress(hosts):
    out = subprocess.run(["scontrol", "show", "hostlistsorted", ",".join(hosts)],
                         capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    return ",".join(hosts)


def leaf_of(host):
    m = _HOST.match(host)
    if not m:
        return None
    rack, nn = m.group(1), int(m.group(2))
    idx = "01" if nn <= 16 else ("02" if nn <= 32 else "03")
    return f"jpbi-{rack}-l1-{idx}"


def l2_of(host):
    m = _HOST.match(host)
    if not m:
        return None
    rack = int(m.group(1))
    base = ((rack - 1) // 5) * 5 + 1        # 1..5 -> 1, 6..10 -> 6, ...
    return f"jpbi-{base:03d}-l2-01"


def summarise(hosts):
    return len({leaf_of(h) for h in hosts}), len({l2_of(h) for h in hosts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodelist", default=os.environ.get("SLURM_JOB_NODELIST", ""))
    ap.add_argument("--nodes-per-arm", type=int, default=16)
    a = ap.parse_args()
    if not a.nodelist:
        raise SystemExit("no --nodelist and SLURM_JOB_NODELIST unset")

    hosts = sorted(expand(a.nodelist))
    n = a.nodes_per_arm
    by_leaf = defaultdict(list)
    for h in hosts:
        by_leaf[leaf_of(h)].append(h)
    for v in by_leaf.values():
        v.sort()
    by_l2 = defaultdict(list)
    for lf, v in by_leaf.items():
        by_l2[l2_of(v[0])].append(lf)

    print(f"# allocation: {len(hosts)} nodes, {len(by_leaf)} leaves, {len(by_l2)} l2", flush=True)
    for lf in sorted(by_leaf):
        print(f"#   {lf}: {len(by_leaf[lf])} nodes", flush=True)

    arms = {}

    # ARM A -- one leaf, the gl11 reference placement.
    full = [lf for lf in sorted(by_leaf) if len(by_leaf[lf]) >= n]
    if not full:
        big = max(by_leaf, key=lambda lf: len(by_leaf[lf]))
        print(f"# FATAL: no leaf holds {n} nodes (largest is {big} with "
              f"{len(by_leaf[big])}). Ask for more nodes, or lower --nodes-per-arm.",
              flush=True)
        return 2
    arms["A"] = by_leaf[full[0]][:n]

    # ARM B -- >1 leaf, all inside ONE l2. Spread as evenly as possible.
    for l2, leaves in sorted(by_l2.items()):
        if len(leaves) < 2:
            continue
        pick, i = [], 0
        leaves = sorted(leaves, key=lambda lf: -len(by_leaf[lf]))
        while len(pick) < n:
            added = False
            for lf in leaves:
                if i < len(by_leaf[lf]) and len(pick) < n:
                    pick.append(by_leaf[lf][i])
                    added = True
            if not added:
                break
            i += 1
        if len(pick) == n and len({leaf_of(h) for h in pick}) > 1:
            arms["B"] = pick
            break

    # ARM C -- >1 l2, i.e. traffic must reach the head switch. Interleave over the LEAVES
    # inside each l2 as well, so C ends up with a leaf count close to B's: otherwise B->C
    # would change leaf count and l2 count together and isolate neither.
    if len(by_l2) >= 2:
        def l2_pool(g):
            """round-robin over this l2's leaves, so the pool spreads across leaves"""
            leaves = sorted(by_l2[g], key=lambda lf: -len(by_leaf[lf]))
            out, k = [], 0
            while True:
                added = False
                for lf in leaves:
                    if k < len(by_leaf[lf]):
                        out.append(by_leaf[lf][k])
                        added = True
                if not added:
                    return out
                k += 1

        pick, i = [], 0
        order = sorted(by_l2, key=lambda g: -sum(len(by_leaf[lf]) for lf in by_l2[g]))
        pools = [l2_pool(g) for g in order]
        while len(pick) < n:
            added = False
            for p in pools:
                if i < len(p) and len(pick) < n:
                    pick.append(p[i])
                    added = True
            if not added:
                break
            i += 1
        if len(pick) == n and len({l2_of(h) for h in pick}) > 1:
            arms["C"] = pick

    for name in ("A", "B", "C"):
        if name not in arms:
            print(f"# ARM {name} not formable from this allocation -- skipped", flush=True)
            continue
        lv, l2 = summarise(arms[name])
        print(f"ARM {name} {lv} {l2} {compress(arms[name])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
