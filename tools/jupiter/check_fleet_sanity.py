#!/usr/bin/env python3
"""Full-fleet physical sanity sweep of a PYNICAM_TIMELOOP_DUMP set.

Extends tutorial/check_validation.py (interior finite / RHOG>0 / RHOGE>0) with
derived-quantity checks that catch a *plausible but wrong* state, which
positivity alone does not:
  - |V| = |RHOGV|/RHOG  -- Jablonowski baroclinic-wave jet peaks ~35-45 m/s;
    43 steps x dtl=9.375 s = 403 s of model time, so the IC should barely have
    evolved. A runaway shows up here long before it shows up as a NaN.
  - tracer (var 6) -- IDEAL init starts with identically zero tracers.
"""
import sys, glob, os
import numpy as np

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "out_rank*.npy")),
               key=lambda f: int(f.split("rank")[-1].split(".")[0]))
print(f"ranks found: {len(files)}", flush=True)

bad = []
g_rhog = [np.inf, -np.inf]
g_rhoge = [np.inf, -np.inf]
g_vmax = 0.0
g_trc = [np.inf, -np.inf]

for n, f in enumerate(files):
    a = np.load(f, mmap_mode="r")
    kmax = a.shape[2] - 2
    it = np.asarray(a[1:-1, 1:-1, 1:kmax + 1, :, :])   # strip halo + ghost levels
    rhog, vx, vy, vz, ge = it[..., 0], it[..., 1], it[..., 2], it[..., 3], it[..., 5]
    trc = it[..., 6]
    problems = []
    if not np.isfinite(it).all(): problems.append("non-finite")
    if not (rhog > 0).all():      problems.append("RHOG<=0")
    if not (ge > 0).all():        problems.append("RHOGE<=0")
    v = np.sqrt(vx * vx + vy * vy + vz * vz) / rhog
    vmax = float(v.max())
    if not np.isfinite(vmax) or vmax > 200.0: problems.append(f"|V|max={vmax:.1f}")
    if problems: bad.append((n, problems))
    g_rhog = [min(g_rhog[0], float(rhog.min())), max(g_rhog[1], float(rhog.max()))]
    g_rhoge = [min(g_rhoge[0], float(ge.min())), max(g_rhoge[1], float(ge.max()))]
    g_vmax = max(g_vmax, vmax)
    g_trc = [min(g_trc[0], float(trc.min())), max(g_trc[1], float(trc.max()))]
    del a, it
    if (n + 1) % 32 == 0:
        print(f"  ...{n+1}/{len(files)} ranks, |V|max so far {g_vmax:.2f} m/s", flush=True)

print(f"\n=== {len(files)} ranks, interior only ===")
print(f"  RHOG   range : {g_rhog[0]:.4e} .. {g_rhog[1]:.4e}")
print(f"  RHOGE  range : {g_rhoge[0]:.4e} .. {g_rhoge[1]:.4e}")
print(f"  |V|max       : {g_vmax:.3f} m/s   (JW jet ~35-45 m/s expected)")
print(f"  tracer range : {g_trc[0]:.4e} .. {g_trc[1]:.4e}   (IDEAL init = 0)")
print(f"  ranks with problems: {len(bad)}")
for n, p in bad[:10]:
    print(f"    rank{n}: {', '.join(p)}")
print("=== ALL RANKS PHYSICALLY SANE ===" if not bad else "=== PROBLEMS FOUND ===")
