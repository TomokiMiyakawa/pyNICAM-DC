"""Validate the numpy AF_dcmip glue against the Fortran golden (ref_af_dcmip_z*.txt).

The golden driver (af_dcmip_ref.f90) is a verbatim copy of the AF_dcmip
SimpleMicrophys transform calling the UNMODIFIED simple_physics_v6.f90, so this
is a true bit-exact check of the glue: level flip, vh<->uv projection, pint
construction, and fvx/fvy/fvz/fe/fq tendency assembly (RAIN_TYPE DRY & WARM).

Run (after tools/dcmip/build_af_dcmip_ref.sh):
    python3 tools/dcmip/test_af_dcmip.py
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import pynicamdc.nhm.forcing.mod_af_dcmip as _afmod       # noqa: E402
_afmod.prf = None    # PROF not set up in this standalone test -> no-op the rap timers
from pynicamdc.nhm.forcing.mod_af_dcmip import AfDcmip   # noqa: E402

RTOL = 1e-11
ATOL = 1e-14


def load(path):
    meta, arrays = {}, {}
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i, n = 0, len(lines)
    cur = None
    configs = {}
    while i < n:
        ln = lines[i].strip(); i += 1
        if not ln or ln.startswith("#"):
            continue
        tok = ln.split()
        if tok[0] == "META":
            if tok[1] == "ijdim":
                meta.update(ijdim=int(tok[4]), vlayer=int(tok[5]), kdim=int(tok[6]))
            elif tok[1] == "kmin":
                meta.update(kmin=int(tok[3]), kmax=int(tok[4]))
            elif tok[1] == "ntrc":
                meta.update(ntrc=int(tok[3]), I_QV=int(tok[4]))
            elif tok[1] == "dt":
                meta["dt"] = float(tok[2])
            elif tok[1] == "Rdry":
                meta.update(Rdry=float(tok[5]), CPdry=float(tok[6]),
                            CVdry=float(tok[7]), PRE00=float(tok[8]))
            elif tok[1] == "CVW_QV":
                meta.update(CVW_QV=float(tok[4]), CVW_QC=float(tok[5]), CVW_QR=float(tok[6]))
        elif tok[0] == "CONFIG":
            cur = tok[1]; configs[cur] = {}
        elif tok[0] == "ARRAY":
            name, cnt = tok[1], int(tok[2])
            vals = np.array([float(lines[i + j]) for j in range(cnt)], dtype=np.float64)
            i += cnt
            (configs[cur] if cur else arrays)[name] = vals
    return meta, arrays, configs


def shape_in(meta, arrays):
    """Reshape flat input arrays to (ijdim,kdim[,ntrc]) / (ijdim,)."""
    ij, kd, nt = meta["ijdim"], meta["kdim"], meta["ntrc"]
    out = {}
    for k, v in arrays.items():
        if v.size == ij:
            out[k] = v
        elif v.size == ij * kd:
            out[k] = v.reshape(ij, kd)
        elif v.size == ij * kd * nt:
            out[k] = v.reshape(ij, kd, nt)
        else:
            out[k] = v
    return out


def worst(name, got, ref, fails):
    ref = ref.reshape(got.shape)
    aerr = np.abs(got - ref)
    ok = np.all(aerr <= ATOL + RTOL * np.abs(ref))
    # field-scaled relative error (near-zero entries don't blow up the metric)
    scale = max(np.abs(ref).max(), 1e-300)
    print(f"    [{'ok ' if ok else 'FAIL'}] {name:7s} max|abs|={aerr.max():.3e} "
          f"max|abs|/max|ref|={aerr.max()/scale:.3e}")
    if not ok:
        fails.append(name)


def run_level(path):
    meta, A, configs = load(path)
    S = shape_in(meta, A)
    fails = []
    print(f"=== {os.path.basename(path)} (vlayer={meta['vlayer']}) ===")

    # 0-based kmin/kmax; vlayer levels
    kmin = meta["kmin"] - 1
    kmax = meta["kmax"] - 1
    CVW = {meta["I_QV"]: meta["CVW_QV"], meta["I_QV"] + 1: meta["CVW_QC"],
           meta["I_QV"] + 2: meta["CVW_QR"]}

    for rain in ("DRY", "WARM"):
        print(f"  RAIN_TYPE={rain}")
        af = AfDcmip()
        af.AF_dcmip_init({"SET_RJ2012": True})   # SimpleMicrophys + LSC + latSST
        af.SM_PBL_Bryan = False
        cfg = dict(kmin=kmin, kmax=kmax, vlayer=meta["vlayer"],
                   I_QV=meta["I_QV"], I_QC=meta["I_QV"] + 1, I_QR=meta["I_QV"] + 2,
                   CVW=CVW, CVdry=meta["CVdry"], RAIN_TYPE=rain)
        fvx, fvy, fvz, fe, fq, precip = af.AF_dcmip(
            S["lat"], S["lon"], S["alt"], S["alth"], S["rho"], S["pre"], S["tem"],
            S["vx"], S["vy"], S["vz"], S["q"], S["ein"], S["pre_sfc"],
            S["ix"], S["iy"], S["iz"], S["jx"], S["jy"], S["jz"], meta["dt"], cfg)
        R = configs[rain]
        worst("fvx", fvx, R["fvx"], fails)
        worst("fvy", fvy, R["fvy"], fails)
        worst("fvz", fvz, R["fvz"], fails)
        worst("fe", fe, R["fe"], fails)
        worst("fq", fq, R["fq"], fails)
        worst("precip", precip, R["precip"], fails)

    # --- pure-Kessler branch (USE_Kessler, no SimpleMicrophys). The kessler.f90
    # f32 locals put the floor at ~1e-7 (validated to machine precision in
    # test_kessler.py); here a looser tolerance isolates the GLUE (wet<->dry
    # conversion, cv, fq/fe assembly) -- a glue bug would be O(1). ---
    if "KESSLER" in configs:
        print("  KESSLER (pure, USE_SimpleMicrophys=False)")
        af = AfDcmip()
        af.USE_Kessler = True
        af.USE_SimpleMicrophys = False
        cfg = dict(kmin=kmin, kmax=kmax, vlayer=meta["vlayer"],
                   I_QV=meta["I_QV"], I_QC=meta["I_QV"] + 1, I_QR=meta["I_QV"] + 2,
                   CVW=CVW, CVdry=meta["CVdry"], RAIN_TYPE="WARM",
                   PRE00=meta["PRE00"], Rdry=meta["Rdry"], CPdry=meta["CPdry"])
        fvx, fvy, fvz, fe, fq, precip = af.AF_dcmip(
            S["lat"], S["lon"], S["alt"], S["alth"], S["rho"], S["pre"], S["tem"],
            S["vx"], S["vy"], S["vz"], S["q"], S["ein"], S["pre_sfc"],
            S["ix"], S["iy"], S["iz"], S["jx"], S["jy"], S["jz"], meta["dt"], cfg)
        R = configs["KESSLER"]
        kfails = []
        for nm, got in (("fvx", fvx), ("fvy", fvy), ("fvz", fvz),
                        ("fe", fe), ("fq", fq), ("precip", precip)):
            ref = R[nm].reshape(got.shape)
            rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300)
            ok = rel < 1e-6
            print(f"    [{'ok ' if ok else 'FAIL'}] {nm:7s} max|abs|/max|ref|={rel:.3e}")
            if not ok:
                kfails.append(nm)
        fails += kfails
    return fails


def main():
    files = sorted(glob.glob(os.path.join(HERE, "ref_af_dcmip_z*.txt")),
                   key=lambda p: int(p.split("_z")[-1].split(".")[0]))
    if not files:
        print("ERROR: no ref_af_dcmip_z*.txt. Run build_af_dcmip_ref.sh first.")
        return 2
    all_fails = []
    for path in files:
        all_fails += run_level(path)
    print("\nRESULT:", "PASS" if not all_fails else f"FAIL ({len(all_fails)})")
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
