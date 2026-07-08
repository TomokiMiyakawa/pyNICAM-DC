"""Validate Frc.forcing_step (part A: compute+apply) against the Fortran golden.

forcing_step_ref.f90 is a verbatim copy of nicamdc forcing_step L253-390 core
(ein, pre_srf, AF_dcmip transform via the UNMODIFIED simple_physics_v6.f90,
tendency apply). Here the flat (ij,k) golden data is placed into pyNICAM 4D
arrays (i=ij, j=1, k=kdim, l=1); we call forcing_step and compare the updated
PROG (rhog,rhogvx,rhogvy,rhogvz,rhoge), PROGq, and precip.

Run (after building the golden):
    tools/dcmip/build_forcing_step_ref.sh
    python3 tools/dcmip/test_forcing_step.py
"""
import os
import sys
import glob
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

RTOL = 1e-11
ATOL = 1e-12

# minimal mpi4py stub so the pyNICAM share modules import without MPI
if "mpi4py" not in sys.modules:
    _m = types.ModuleType("mpi4py")
    _m.MPI = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(
        Get_rank=lambda: 0, Get_size=lambda: 1))
    sys.modules["mpi4py"] = _m
    sys.modules["mpi4py.MPI"] = _m.MPI

import pynicamdc.share.mod_adm as adm_mod              # noqa: E402
from pynicamdc.share.mod_gmtr import Gmtr              # noqa: E402
import pynicamdc.share.mod_prof as prof_mod            # noqa: E402
prof_mod.prf.PROF_rapstart = lambda *a, **k: None       # no-op profiling for the unit test
prof_mod.prf.PROF_rapend = lambda *a, **k: None
from pynicamdc.nhm.forcing.mod_forcing import frc      # noqa: E402
from pynicamdc.nhm.forcing.mod_af_dcmip import afdcmip  # noqa: E402


def load(path):
    meta, arr, cfgs, cur = {}, {}, {}, None
    with open(path) as f:
        L = [x.rstrip("\n") for x in f]
    i = 0
    while i < len(L):
        s = L[i].strip(); i += 1
        if not s or s.startswith("#"):
            continue
        t = s.split()
        if t[0] == "META":
            if t[1] == "ijdim":
                meta.update(ijdim=int(t[4]), vlayer=int(t[5]), kdim=int(t[6]))
            elif t[1] == "kmin":
                meta.update(kmin=int(t[3]), kmax=int(t[4]))
            elif t[1] == "ntrc":
                meta.update(ntrc=int(t[3]), I_QV=int(t[4]))
            elif t[1] == "dt":
                meta["dt"] = float(t[2])
            elif t[1] == "Rdry":
                meta.update(CVdry=float(t[7]))
            elif t[1] == "CVW_QV":
                meta.update(CVW=[float(t[4]), float(t[5]), float(t[6])])
            elif t[1] == "GRAV":
                meta["GRAV"] = float(t[2])
        elif t[0] == "CONFIG":
            cur = t[1]; cfgs[cur] = {}
        elif t[0] == "ARRAY":
            n = int(t[2])
            v = np.array([float(L[i + j]) for j in range(n)]); i += n
            (cfgs[cur] if cur else arr)[t[1]] = v
    return meta, arr, cfgs


def main():
    files = sorted(glob.glob(os.path.join(HERE, "ref_forcing_step_z*.txt")),
                   key=lambda p: int(p.split("_z")[-1].split(".")[0]))
    if not files:
        print("ERROR: no ref_forcing_step_z*.txt. Build forcing_step_ref.f90 first.")
        return 2

    IDX = frc  # for the I_RHOG.. constants
    fails = []
    for path in files:
        meta, A, cfgs = load(path)
        ij, kd, nt = meta["ijdim"], meta["kdim"], meta["ntrc"]
        vlayer = meta["vlayer"]
        print(f"=== {os.path.basename(path)} (vlayer={vlayer}) ===")

        # configure adm/gmtr/grd/vmtr/cnst/rcnf stand-ins for the (ij,1,kd,1) layout
        adm = adm_mod.adm
        adm.ADM_gall_1d = ij; adm.ADM_K0 = 0; adm.ADM_have_pl = False
        adm.ADM_kall = kd; adm.ADM_lall = 1; adm.ADM_vlayer = vlayer
        adm.ADM_kmin = meta["kmin"] - 1; adm.ADM_kmax = meta["kmax"] - 1

        def C(name):     # (ij*kd,) -> (ij,1,kd,1)
            return A[name].reshape(ij, kd)[:, None, :, None].copy()

        def CT(name):    # (ij*kd*nt,) -> (ij,1,kd,1,nt)
            return A[name].reshape(ij, kd, nt)[:, None, :, None, :].copy()

        def H(name):     # (ij,) -> (ij,1,1)
            return A[name].reshape(ij)[:, None, None].copy()

        gmtr = Gmtr()
        gp = np.zeros((ij, 1, 1, 1, gmtr.GMTR_p_nmax))
        for nm, idx in (("ix", gmtr.GMTR_p_IX), ("iy", gmtr.GMTR_p_IY), ("iz", gmtr.GMTR_p_IZ),
                        ("jx", gmtr.GMTR_p_JX), ("jy", gmtr.GMTR_p_JY), ("jz", gmtr.GMTR_p_JZ)):
            gp[:, 0, 0, 0, idx] = A[nm]
        gmtr.GMTR_p = gp

        # grd stand-in: GRD_vz[...,Z/ZH], GRD_zs[...,ZSFC], GRD_LAT/LON (i,j,1,l)
        grd = types.SimpleNamespace(GRD_Z=0, GRD_ZH=1, GRD_ZSFC=0)
        grd.GRD_vz = np.stack([A["alt"].reshape(ij, kd), A["alth"].reshape(ij, kd)], axis=-1)[:, None, :, None, :]
        grd.GRD_zs = A["z_srf"].reshape(ij, 1, 1, 1, 1)
        grd.GRD_LAT = A["lat"].reshape(ij, 1, 1, 1)
        grd.GRD_LON = A["lon"].reshape(ij, 1, 1, 1)

        vmtr = types.SimpleNamespace(VMTR_GSGAM2=C("gsgam2"), VMTR_GSGAM2H=C("gsgam2h"))
        cnst = types.SimpleNamespace(CONST_GRAV=meta["GRAV"], CONST_CVdry=meta["CVdry"])

        afdcmip.AF_dcmip_init({"SET_RJ2012": True}); afdcmip.SM_PBL_Bryan = False

        for rain in ("DRY", "WARM"):
            print(f"  RAIN_TYPE={rain}")
            nqe = 1 if rain == "DRY" else 3
            rcnf = types.SimpleNamespace(
                I_QV=meta["I_QV"], I_QC=meta["I_QV"] + 1, I_QR=meta["I_QV"] + 2,
                CVW=meta["CVW"], RAIN_TYPE=rain,
                NQW_STR=meta["I_QV"], NQW_END=meta["I_QV"] + (nqe - 1))

            # assemble PROG (i,j,k,l,6) and PROGq (i,j,k,l,nt)
            PROG = np.zeros((ij, 1, kd, 1, 6))
            PROG[..., IDX.I_RHOG] = C("rhog")
            PROG[..., IDX.I_RHOGVX] = C("rhogvx")
            PROG[..., IDX.I_RHOGVY] = C("rhogvy")
            PROG[..., IDX.I_RHOGVZ] = C("rhogvz")
            PROG[..., IDX.I_RHOGW] = C("rhogw")
            PROG[..., IDX.I_RHOGE] = C("rhoge")
            PROGq = CT("rhogq")[:, :, :, :, :]

            precip = frc.forcing_step(
                PROG, PROGq, C("rho"), C("pre"), C("tem"),
                C("vx"), C("vy"), C("vz"), CT("q"),
                vmtr, gmtr, grd, cnst, rcnf, meta["dt"], np.float64)

            R = cfgs[rain]

            def chk(name, got, ref):
                ref = ref.reshape(got.shape)
                ae = np.abs(got - ref)
                ok = np.all(ae <= ATOL + RTOL * np.abs(ref))
                sc = max(np.abs(ref).max(), 1e-300)
                print(f"    [{'ok ' if ok else 'FAIL'}] {name:7s} "
                      f"max|abs|={ae.max():.3e} max|abs|/max|ref|={ae.max()/sc:.3e}")
                if not ok:
                    fails.append(f"{path}:{rain}:{name}")

            chk("rhog",   PROG[..., IDX.I_RHOG].reshape(ij, kd), R["rhog"])
            chk("rhogvx", PROG[..., IDX.I_RHOGVX].reshape(ij, kd), R["rhogvx"])
            chk("rhogvy", PROG[..., IDX.I_RHOGVY].reshape(ij, kd), R["rhogvy"])
            chk("rhogvz", PROG[..., IDX.I_RHOGVZ].reshape(ij, kd), R["rhogvz"])
            chk("rhoge",  PROG[..., IDX.I_RHOGE].reshape(ij, kd), R["rhoge"])
            chk("rhogq",  PROGq.reshape(ij, kd, nt), R["rhogq"])
            chk("precip", precip.reshape(ij), R["precip"])

    print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
