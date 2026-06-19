#!/usr/bin/env python3
"""
Parse the PROF "Computational Time Report" from each resolution's rank-0 log
(run/gl0g/msg.pe00000000) and print a CSV summary across glevels.

PROF lines look like:
    *** ID=006 : MAIN_Main_Loop                    T=     4.583 N=1

Usage:
  python scripts/collect_timers.py 5 6 7 8 9
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# PROF entries to report (exact timer-name match). These are the high-level
# phases most useful for an f90-vs-pyNICAM comparison; add more names from the
# "Computational Time Report" in run/gl0N/msg.pe00000000 if you want finer detail.
WANT = [
    "MAIN_Main_Loop",
    "MAIN_Main_Loop_step1",      # first iteration only (warmup / jax JIT) -> excluded from per-step
    "MAIN___Dynamics",
    "MAIN____Large_step",
    "MAIN____Small_step",
    "MAIN____Tracer_Advection",
    "MAIN_COMM_data_transfer",
    "MAIN_COMM_barrier",
]
# timer whose N count equals the number of large steps actually run
NSTEP_TIMER = "MAIN__Atmos"
LINE = re.compile(r"ID=\d+\s*:\s*(?P<name>\S+)\s+T=\s*(?P<t>[-\d.eE+]+)\s+N=\s*(?P<n>\d+)")


def parse(msgfile):
    out = {}
    if not os.path.exists(msgfile):
        return out
    with open(msgfile) as f:
        for ln in f:
            m = LINE.search(ln)
            if m:
                out[m.group("name")] = (float(m.group("t")), int(m.group("n")))
    return out


def main(glevels, backend="numpy", label=None):
    # per_step_excl1 = (Main_Loop - Main_Loop_step1) / (nstep - 1): steady-state
    # wall time per large step, excluding the first (warmup / jax JIT) iteration.
    # `label` is the run-dir/CSV suffix (default = backend); `backend` is the
    # reported column (so the hybrid run is label=jax_be, backend=jax_be).
    label = label or backend
    cols = ["glevel", "backend", "gall_1d", "dtl", "nstep"] + WANT + ["per_step_excl1"]
    print(",".join(cols))
    for g in glevels:
        g = int(g)
        gp = f"{g:02d}"
        gall_1d = 2 ** (g - 1) + 2
        dtl = 1200.0 / (2 ** (g - 5))
        t = parse(os.path.join(ROOT, "run", f"gl{gp}_{label}", "msg.pe00000000"))
        nstep = t[NSTEP_TIMER][1] if NSTEP_TIMER in t else None
        row = [gp, label, str(gall_1d), f"{dtl:g}", str(nstep) if nstep else "NA"]
        for w in WANT:
            row.append(f"{t[w][0]:.3f}" if w in t else "NA")
        if "MAIN_Main_Loop" in t and "MAIN_Main_Loop_step1" in t and nstep and nstep > 1:
            per = (t["MAIN_Main_Loop"][0] - t["MAIN_Main_Loop_step1"][0]) / (nstep - 1)
            row.append(f"{per:.3f}")
        else:
            row.append("NA")
        print(",".join(row))


if __name__ == "__main__":
    args = sys.argv[1:]
    backend, label = "numpy", None
    for opt in ("--backend", "--label"):
        if opt in args:
            i = args.index(opt)
            val = args[i + 1]
            del args[i:i + 2]
            if opt == "--backend":
                backend = val
            else:
                label = val
    main(args or [5, 6, 7, 8, 9], backend=backend, label=label)
