"""Validate the numpy simple_physics port against the Fortran reference dump.

Runs pynicamdc.nhm.forcing.simple_physics.simple_physics for every config in
ref_simple_physics.txt and compares (t,q,u,v,precl) elementwise. Reports the
worst absolute/relative error per field so the tridiagonal solve and the
theta<->T transforms can be diagnosed if they drift.

Run (after building the reference):
    tools/dcmip/build_and_run.sh
    python3 tools/dcmip/test_simple_physics.py

Exits nonzero if any field exceeds RTOL/ATOL. Skips with a clear message while
simple_physics is still a NotImplementedError scaffold.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from tools.dcmip.refdata import load_ref  # noqa: E402

RTOL = 1e-11        # double-precision, allow ARM/x86 libm reduction-order slack
ATOL = 1e-12

# config name -> simple_physics flags (must mirror simple_physics_ref.f90)
CONFIGS = {
    "A_rj_noBryan_test0": dict(test=0, RJ2012_precip=True,  TC_PBL_mod=False, use_HS=False, MITC_TYPE=1),
    "B_rj_Bryan_test0":   dict(test=0, RJ2012_precip=True,  TC_PBL_mod=True,  use_HS=False, MITC_TYPE=1),
    "C_rj_noBryan_test1": dict(test=1, RJ2012_precip=True,  TC_PBL_mod=False, use_HS=False, MITC_TYPE=1),
    "D_noprecip_test0":   dict(test=0, RJ2012_precip=False, TC_PBL_mod=False, use_HS=False, MITC_TYPE=1),
    "E_useHS_mitc1":      dict(test=0, RJ2012_precip=True,  TC_PBL_mod=False, use_HS=True,  MITC_TYPE=1),
}


def worst(name, got, ref):
    aerr = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), ATOL)
    rerr = aerr / denom
    k = np.unravel_index(np.argmax(rerr), rerr.shape)
    ok = np.all(aerr <= ATOL + RTOL * np.abs(ref))
    flag = "ok " if ok else "FAIL"
    print(f"    [{flag}] {name:6s} max|abs|={aerr.max():.3e} "
          f"max|rel|={rerr.max():.3e} @ {k}")
    return ok


def _ref_files():
    """All ref_simple_physics_z*.txt (fall back to the plain name)."""
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "ref_simple_physics_z*.txt")),
                   key=lambda p: int(p.split("_z")[-1].split(".")[0]))
    if not files:
        p = os.path.join(HERE, "ref_simple_physics.txt")
        if os.path.exists(p):
            files = [p]
    return files


def check_ref(ref, simple_physics):
    """Run every config for one reference (one level count); return all_ok."""
    pcols, pver, dtime = ref.meta["pcols"], ref.meta["pver"], ref.meta["dtime"]
    S = ref.shared
    all_ok = True
    for name, flags in CONFIGS.items():
        print(f"  CONFIG {name}")
        t = S["t_in"].copy(); q = S["q_in"].copy()
        u = S["u_in"].copy(); v = S["v_in"].copy()
        try:
            t, q, u, v, precl = simple_physics(
                pcols, pver, dtime, S["lat"].copy(),
                t, q, u, v,
                S["pmid"].copy(), S["pint"].copy(),
                S["pdel"].copy(), S["rpdel"].copy(), S["ps"].copy(),
                **flags,
            )
        except NotImplementedError as e:
            print(f"    SKIP (scaffold not implemented): {e}")
            return False
        R = ref.configs[name]
        all_ok &= worst("t", t, R["t_out"])
        all_ok &= worst("q", q, R["q_out"])
        all_ok &= worst("u", u, R["u_out"])
        all_ok &= worst("v", v, R["v_out"])
        all_ok &= worst("precl", precl, R["precl"])
    return all_ok


def main():
    files = _ref_files()
    if not files:
        print("ERROR: no ref_simple_physics_z*.txt. Run build_and_run.sh first.")
        return 2
    try:
        from pynicamdc.nhm.forcing.simple_physics import simple_physics
    except Exception as e:
        print(f"cannot import simple_physics: {e}")
        return 2

    all_ok = True
    for path in files:
        ref = load_ref(path)
        print(f"=== {os.path.basename(path)} (pver={ref.meta['pver']}) ===")
        all_ok &= check_ref(ref, simple_physics)

    print("\nRESULT:", "PASS" if all_ok else "FAIL / incomplete")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
