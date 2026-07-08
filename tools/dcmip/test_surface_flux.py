"""Sanity checks for _sst_tsurf and _surface_flux (simple_physics step 3).

The surface-flux output is further modified by the (not-yet-ported) PBL
diffusion, so BIT-EXACT validation lands at config D once _pbl_diffusion is in.
Until then, verify invariants that would trip on an index/branch/precedence
error:

  _sst_tsurf:
    test=0        -> exactly SST_TC (302.15) everywhere
    test=1        -> lat-symmetric, plausible SST range
    use_HS MITC1  -> lat-symmetric Gaussian, peak at equator, in [271,300]
  _surface_flux (lowest level only):
    only level -1 changes; all other levels untouched
    u,v: exact drag ratio u_new == u_old/(1+Cd*wind*dtime/za), |u_new|<=|u_old|
    t  : convex combination of t_old and Tsurf (bounded between them)
    q  : convex combination of q_old and qsats

Run: python3 tools/dcmip/test_surface_flux.py
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from tools.dcmip.refdata import load_ref                                    # noqa: E402
from pynicamdc.nhm.forcing.simple_physics import (                         # noqa: E402
    _constants, _hydrostatic_za, _sst_tsurf, _large_scale_precip,
    _pbl_coeffs, _surface_flux,
)


def _check(cond, msg, fails):
    print(f"      [{'ok ' if cond else 'FAIL'}] {msg}")
    if not cond:
        fails.append(msg)


def between(x, a, b):
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    eps = 1e-9 * (np.abs(hi) + 1.0)
    return np.all((x >= lo - eps) & (x <= hi + eps))


def run_level(path):
    ref = load_ref(path)
    pver = ref.meta["pver"]; dtime = ref.meta["dtime"]
    S = ref.shared
    fails = []
    print(f"=== {os.path.basename(path)} (pver={pver}) ===")

    # --- _sst_tsurf branches ---
    C0 = _constants(use_HS=False)
    lat = S["lat"]
    print("  -- _sst_tsurf --")
    ts0 = _sst_tsurf(lat, 0, False, 1, C0)
    _check(np.allclose(ts0, C0["SST_TC"], rtol=0, atol=0), "test0 == SST_TC", fails)
    ts1 = _sst_tsurf(lat, 1, False, 1, C0)
    _check(between(ts1, 240.0, 320.0), f"test1 SST range [{ts1.min():.1f},{ts1.max():.1f}]", fails)
    # lat symmetry: cols are -60,-30,0,30,60 (evenly spread) -> ends symmetric
    _check(np.isclose(ts1[0], ts1[-1], rtol=1e-12), "test1 SST lat-symmetric", fails)
    CH = _constants(use_HS=True)
    tsH = _sst_tsurf(lat, 0, True, 1, CH)
    _check(between(tsH, 271.0, 300.0), f"MITC1 SST range [{tsH.min():.1f},{tsH.max():.1f}]", fails)
    _check(np.isclose(tsH[0], tsH[-1], rtol=1e-12), "MITC1 SST lat-symmetric", fails)
    _check(np.argmax(tsH) == np.argmin(np.abs(lat)), "MITC1 SST peaks at equator", fails)

    # --- _surface_flux (use config-A-like: precip on, RJ2012 PBL, test0) ---
    print("  -- _surface_flux --")
    C = _constants(use_HS=False)
    za, zi = _hydrostatic_za(S["t_in"].copy(), S["q_in"].copy(), S["ps"].copy(), S["pint"].copy(), C)
    Tsurf = _sst_tsurf(lat, 0, False, 1, C)
    t, q, _ = _large_scale_precip(S["t_in"].copy(), S["q_in"].copy(), S["pmid"], S["pdel"], dtime, C)
    u, v = S["u_in"].copy(), S["v_in"].copy()
    wind, Cd, Km, Ke = _pbl_coeffs(u, v, t, q, S["pint"], za, zi, False, C)

    u_in, v_in, t_in, q_in = u.copy(), v.copy(), t.copy(), q.copy()
    u, v, t, q = _surface_flux(u, v, t, q, S["ps"], Tsurf, wind, Cd, za, dtime, C)

    # only bottom level changed
    _check(np.array_equal(u[:, :-1], u_in[:, :-1]) and np.array_equal(v[:, :-1], v_in[:, :-1])
           and np.array_equal(t[:, :-1], t_in[:, :-1]) and np.array_equal(q[:, :-1], q_in[:, :-1]),
           "only lowest level modified", fails)
    # exact drag ratio
    denom_m = 1.0 + Cd * wind * dtime / za
    _check(np.allclose(u[:, -1], u_in[:, -1] / denom_m, rtol=1e-14, atol=0)
           and np.allclose(v[:, -1], v_in[:, -1] / denom_m, rtol=1e-14, atol=0),
           "u,v == old/(1+Cd*wind*dt/za)", fails)
    _check(np.all(np.abs(u[:, -1]) <= np.abs(u_in[:, -1]) + 1e-12), "|u| reduced by drag", fails)
    # t,q relax toward surface values (bounded between old and target)
    qsats = C["epsilo"]*C["e0"]/S["ps"]*np.exp(-C["latvap"]/C["rh2o"]*((1.0/Tsurf)-1.0/C["T0"]))
    _check(between(t[:, -1], t_in[:, -1], Tsurf), "t bounded between t_old and Tsurf", fails)
    _check(between(q[:, -1], q_in[:, -1], qsats), "q bounded between q_old and qsats", fails)
    return fails


def main():
    files = sorted(glob.glob(os.path.join(HERE, "ref_simple_physics_z*.txt")),
                   key=lambda p: int(p.split("_z")[-1].split(".")[0]))
    if not files:
        print("ERROR: no ref_simple_physics_z*.txt. Run build_and_run.sh first.")
        return 2
    all_fails = []
    for path in files:
        all_fails += run_level(path)
    print("\nRESULT:", "PASS" if not all_fails else f"FAIL ({len(all_fails)} checks)")
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
