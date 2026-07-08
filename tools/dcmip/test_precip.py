"""Unit test for _large_scale_precip (simple_physics step 1).

precl is FINAL after the precip step (surface flux / PBL never touch it), so it
can be validated exactly against the Fortran golden even before the later steps
exist. Also checks the latent-heat energy budget internally:
    sum_k cpair * dT * pdel/gravit  ==  latvap * rhow * precl
(condensational heating balances the precipitated mass), which pins down the
tmp/qsat formula independently of the golden.

Run (after tools/dcmip/build_and_run.sh):
    python3 tools/dcmip/test_precip.py
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from tools.dcmip.refdata import load_ref                              # noqa: E402
from pynicamdc.nhm.forcing.simple_physics import _constants, _large_scale_precip  # noqa: E402

RTOL = 1e-11
ATOL = 1e-14


def worst(got, ref):
    aerr = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-30)
    return aerr.max(), (aerr / denom).max()


def main():
    files = sorted(glob.glob(os.path.join(HERE, "ref_simple_physics_z*.txt")),
                   key=lambda p: int(p.split("_z")[-1].split(".")[0]))
    if not files:
        print("ERROR: no ref_simple_physics_z*.txt. Run build_and_run.sh first.")
        return 2

    C = _constants(use_HS=False)  # precip step is independent of use_HS
    all_ok = True
    for path in files:
        ref = load_ref(path)
        pver = ref.meta["pver"]; dtime = ref.meta["dtime"]
        S = ref.shared
        t, q, precl = _large_scale_precip(
            S["t_in"].copy(), S["q_in"].copy(), S["pmid"], S["pdel"], dtime, C)

        # config A/B/C/E all share the same precip (precl identical); use A's golden
        ref_precl = ref.configs["A_rj_noBryan_test0"]["precl"]
        amax, rmax = worst(precl, ref_precl)
        ok_precl = np.all(np.abs(precl - ref_precl) <= ATOL + RTOL * np.abs(ref_precl))

        # internal energy budget: column heating over dtime == latent heat of
        # the precipitated mass.  precl is a RATE [m/s], so multiply by dtime:
        #   sum_k cpair*dT*pdel/gravit  ==  latvap * rhow * precl * dtime
        dT = t - S["t_in"]
        heat = np.sum(C["cpair"] * dT * S["pdel"] / C["gravit"], axis=1)
        latent = C["latvap"] * C["rhow"] * precl * dtime
        eb_max = np.abs(heat - latent).max()
        ok_eb = np.all(np.abs(heat - latent) <= 1e-6 * np.maximum(np.abs(latent), 1e-30))

        f1 = "ok " if ok_precl else "FAIL"
        f2 = "ok " if ok_eb else "FAIL"
        print(f"z{pver:<3d} precl[{f1}] max|abs|={amax:.2e} max|rel|={rmax:.2e}   "
              f"energy[{f2}] max|heat-latent|={eb_max:.2e}")
        all_ok &= ok_precl and ok_eb

    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
