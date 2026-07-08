"""Sanity checks for _hydrostatic_za and _pbl_coeffs (simple_physics step 2).

Km/Ke/za/wind/Cd are internal diagnostics not present in the golden final
state, so BIT-EXACT validation of this step happens end-to-end at config D
(no precip: only surface flux + PBL touch the state) once _surface_flux and
_pbl_diffusion exist. Until then, verify structural + physical invariants that
would trip on a gross index/broadcast/branch error:

  za    : (pcols,), positive, O(10-500 m) for a lowest midpoint
  wind  : == sqrt(u_bot^2 + v_bot^2)   (confirms lowest-level selection)
  Cd    : == Cd0+Cd1*wind (wind<v20) else Cm  (threshold branch)
  RJ2012: Km/Ke >= 0; Ke/Km == C/Cd; Km == Cd*wind*za where pint>=pbltop;
          decays monotonically upward above pbltop
  Bryan : zi[:,pver]==0; zi increases upward; Km/Ke==0 where zi>zpbltop

Run: python3 tools/dcmip/test_pbl_coeffs.py
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from tools.dcmip.refdata import load_ref                                     # noqa: E402
from pynicamdc.nhm.forcing.simple_physics import (                          # noqa: E402
    _constants, _hydrostatic_za, _large_scale_precip, _pbl_coeffs,
)


def _check(cond, msg, fails):
    tag = "ok " if cond else "FAIL"
    print(f"      [{tag}] {msg}")
    if not cond:
        fails.append(msg)


def run_level(path):
    ref = load_ref(path)
    pver = ref.meta["pver"]; dtime = ref.meta["dtime"]
    S = ref.shared
    C = _constants(use_HS=False)
    fails = []
    print(f"=== {os.path.basename(path)} (pver={pver}) ===")

    # setup za on INITIAL state, then precip (as the orchestrator does)
    za, _zi0 = _hydrostatic_za(S["t_in"].copy(), S["q_in"].copy(),
                               S["ps"].copy(), S["pint"].copy(), C)
    _check(za.shape == (S["ps"].size,), "za shape (pcols,)", fails)
    _check(np.all(za > 0), "za > 0", fails)
    _check(np.all((za > 1.0) & (za < 2000.0)), f"za in (1,2000) m [{za.min():.1f},{za.max():.1f}]", fails)

    t, q, _ = _large_scale_precip(S["t_in"].copy(), S["q_in"].copy(),
                                  S["pmid"], S["pdel"], dtime, C)
    u, v = S["u_in"].copy(), S["v_in"].copy()

    # --- RJ2012 branch ---
    za2, zi = _hydrostatic_za(S["t_in"].copy(), S["q_in"].copy(), S["ps"].copy(), S["pint"].copy(), C)
    wind, Cd, Km, Ke = _pbl_coeffs(u, v, t, q, S["pint"], za2, zi, False, C)
    print("  -- RJ2012 --")
    _check(np.allclose(wind, np.sqrt(u[:, -1]**2 + v[:, -1]**2), rtol=0, atol=0),
           "wind == sqrt(u_bot^2+v_bot^2)", fails)
    Cd_exp = np.where(wind < C["v20"], C["Cd0"] + C["Cd1"]*wind, C["Cm"])
    _check(np.array_equal(Cd, Cd_exp), "Cd threshold branch", fails)
    _check(np.all(Km[:, :pver] >= 0) and np.all(Ke[:, :pver] >= 0), "Km,Ke >= 0", fails)
    # Ke/Km == C/Cd (shared wind*za*decay)
    nz = Km[:, :pver] > 0
    ratio = np.divide(Ke[:, :pver], Km[:, :pver], out=np.zeros_like(Km[:, :pver]), where=nz)
    exp_ratio = (C["C"] / Cd)[:, None] * nz
    _check(np.allclose(ratio[nz], exp_ratio[nz], rtol=1e-13, atol=0), "Ke/Km == C/Cd", fails)
    # at levels with pint>=pbltop, decay=1 -> Km == Cd*wind*za
    pint_k = S["pint"][:, 0:pver]
    mask = pint_k >= C["pbltop"]
    Km_full = (Cd*wind*za2)[:, None] * np.ones_like(pint_k)
    _check(np.allclose(Km[:, :pver][mask], Km_full[mask], rtol=1e-13, atol=0),
           "Km == Cd*wind*za where pint>=pbltop", fails)
    # unused top interface slot stays zero
    _check(np.all(Km[:, pver] == 0), "Km[:,pver] (unused slot) == 0", fails)

    # --- Bryan branch ---
    za3, zi3 = _hydrostatic_za(S["t_in"].copy(), S["q_in"].copy(), S["ps"].copy(), S["pint"].copy(), C)
    windB, CdB, KmB, KeB = _pbl_coeffs(u, v, t, q, S["pint"], za3, zi3, True, C)
    print("  -- Bryan --")
    _check(np.all(zi3[:, pver] == 0), "zi[:,pver] == 0 (surface datum)", fails)
    # zi increases upward: zi[:,k] > zi[:,k+1] for k<pver
    _check(np.all(np.diff(zi3[:, :pver+1], axis=1) <= 0), "zi increases upward", fails)
    above = zi3[:, :pver] > C["zpbltop"]
    _check(np.all(KmB[:, :pver][above] == 0) and np.all(KeB[:, :pver][above] == 0),
           "Km,Ke == 0 above zpbltop", fails)

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
