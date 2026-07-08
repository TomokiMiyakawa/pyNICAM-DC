import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.bndcnd import (
    BndCfg,
    compute_bndcnd_thermo_reg, compute_bndcnd_thermo_pl,
    compute_bndcnd_rhovxvyvz_reg, compute_bndcnd_rhovxvyvz_pl,
    compute_bndcnd_rhow_reg, compute_bndcnd_rhow_pl,
)
#from mod_prof import prf

class Bndc:
    
    _instance = None

    is_top_tem   = False
    is_top_epl   = False
    is_btm_tem   = False
    is_btm_epl   = False
    is_top_rigid = False
    is_top_free  = False
    is_btm_rigid = False
    is_btm_free  = False

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Backend-switchable kernel staging (numpy <-> jax) for the COMM-free
    # boundary-condition kernels in kernels/bndcnd.py.
    # ------------------------------------------------------------------
    def _bnd_flags(self):
        return dict(
            is_top_tem=self.is_top_tem, is_top_epl=self.is_top_epl,
            is_btm_tem=self.is_btm_tem, is_btm_epl=self.is_btm_epl,
            is_top_rigid=self.is_top_rigid, is_top_free=self.is_top_free,
            is_btm_rigid=self.is_btm_rigid, is_btm_free=self.is_btm_free,
        )

    def _bnd_kernels_get(self):
        if getattr(self, "_bnd_kernels", None) is None:
            self._bnd_kernels = {
                "thermo_reg": bk.maybe_jit(compute_bndcnd_thermo_reg, static_argnames=("cfg", "xp")),
                "thermo_pl":  bk.maybe_jit(compute_bndcnd_thermo_pl,  static_argnames=("cfg", "xp")),
                "rhov_reg":   bk.maybe_jit(compute_bndcnd_rhovxvyvz_reg, static_argnames=("cfg", "xp")),
                "rhov_pl":    bk.maybe_jit(compute_bndcnd_rhovxvyvz_pl,  static_argnames=("cfg", "xp")),
                "rhow_reg":   bk.maybe_jit(compute_bndcnd_rhow_reg, static_argnames=("cfg", "xp")),
                "rhow_pl":    bk.maybe_jit(compute_bndcnd_rhow_pl,  static_argnames=("cfg", "xp")),
            }
        return self._bnd_kernels

    def _bnd_cfg_thermo(self, kmin, kmax, cnst):
        cfg = getattr(self, "_bnd_cfg_t", None)
        if cfg is None or cfg.kmin != kmin or cfg.kmax != kmax:
            cfg = BndCfg(kmin=kmin, kmax=kmax, have_pl=False,
                         GRAV=float(cnst.CONST_GRAV), Rdry=float(cnst.CONST_Rdry),
                         **self._bnd_flags())
            self._bnd_cfg_t = cfg
        return cfg

    def _bnd_cfg_mom(self, kmin, kmax):
        # GRAV / Rdry are unused by the momentum kernels; placeholders keep the
        # cfg identical between rhovxvyvz and rhow (shared jit cache).
        cfg = getattr(self, "_bnd_cfg_m", None)
        if cfg is None or cfg.kmin != kmin or cfg.kmax != kmax:
            cfg = BndCfg(kmin=kmin, kmax=kmax, have_pl=False,
                         GRAV=0.0, Rdry=0.0, **self._bnd_flags())
            self._bnd_cfg_m = cfg
        return cfg

    def BNDCND_setup(self, fname_in, rdtype):

        # Set default boundary types
        BND_TYPE_T_TOP    = 'TEM'
        BND_TYPE_T_BOTTOM = 'TEM'
        BND_TYPE_M_TOP    = 'FREE'
        BND_TYPE_M_BOTTOM = 'RIGID'

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[bndcnd]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'bndcndparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** bndcndparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['bndcndparam']
            #self.GRD_grid_type = cnfs['GRD_grid_type']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        if BND_TYPE_T_TOP == 'TEM':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, top   ) : equal to uppermost atmosphere', file=log_file)
            self.is_top_tem = True

        elif BND_TYPE_T_TOP == 'EPL':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, top   ) : lagrange extrapolation', file=log_file)
            self.is_top_epl = True
            
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_T_TOP. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)



        if BND_TYPE_T_BOTTOM == 'TEM':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, bottom) : equal to lowermost atmosphere', file=log_file)
            self.is_btm_tem = True

        elif BND_TYPE_T_BOTTOM == 'EPL':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, bottom) : lagrange extrapolation', file=log_file)
            self.is_btm_epl = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_T_BOTTOM. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)


        if BND_TYPE_M_TOP == 'RIGID':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    top   ) : rigid', file=log_file)
            self.is_top_rigid = True

        elif BND_TYPE_M_TOP == 'FREE':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    top   ) : free', file=log_file)
            self.is_top_free = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_M_TOP. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)


        if BND_TYPE_M_BOTTOM == 'RIGID':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    bottom) : rigid', file=log_file)
            self.is_btm_rigid = True

        elif BND_TYPE_M_BOTTOM == 'FREE':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    bottom) : free', file=log_file)
            self.is_btm_free = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_M_BOTTOM. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)

        return 
        

    def BNDCND_all(self, msc):
    
        adm  = msc.adm
        rcnf = msc.rcnf
        cnst = msc.cnst

        I_RHOG = rcnf.I_RHOG
        I_RHOGVX = rcnf.I_RHOGVX
        I_RHOGVY = rcnf.I_RHOGVY
        I_RHOGVZ = rcnf.I_RHOGVZ
        I_RHOGW = rcnf.I_RHOGW
        I_RHOGE = rcnf.I_RHOGE

        I_pre = rcnf.I_pre
        I_tem = rcnf.I_tem
        I_vx = rcnf.I_vx
        I_vy = rcnf.I_vy
        I_vz = rcnf.I_vz
        I_w  = rcnf.I_w

        rho  = msc.dyn.rho
        ein  = msc.dyn.ein

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        kmaxp1 = kmax + 1
        kminm1 = kmin - 1
        CVdry = cnst.CONST_CVdry

        vx  = msc.dyn.DIAG[:, :, :, :, I_vx]    
        vy  = msc.dyn.DIAG[:, :, :, :, I_vy]     
        vz  = msc.dyn.DIAG[:, :, :, :, I_vz]     
        w   = msc.dyn.DIAG[:, :, :, :, I_w]      
        tem = msc.dyn.DIAG[:, :, :, :, I_tem] 
        pre = msc.dyn.DIAG[:, :, :, :, I_pre]

        rhog   = msc.dyn.PROG[:, :, :, :, I_RHOG]
        rhogvx = msc.dyn.PROG[:, :, :, :, I_RHOGVX]
        rhogvy = msc.dyn.PROG[:, :, :, :, I_RHOGVY]
        rhogvz = msc.dyn.PROG[:, :, :, :, I_RHOGVZ]
        rhogw  = msc.dyn.PROG[:, :, :, :, I_RHOGW]
        rhoge  = msc.dyn.PROG[:, :, :, :, I_RHOGE]

        gsqrtgam2  = msc.vmtr.VMTR_GSGAM2
        phi        = msc.vmtr.VMTR_PHI
        c2wfact    = msc.vmtr.VMTR_C2Wfact
        c2wfact_Gz = msc.vmtr.VMTR_C2WfactGz
        cnst       = msc.cnst
        rdtype     = msc.bk.ndtype


        #--- Thermodynamical variables ( rho, ein, tem, pre, rhog, rhoge ), q = 0 at boundary
        self.BNDCND_thermo(
            kmin, kmax,
            tem, rho, pre, phi, 
            cnst, rdtype
        )

        rhog[:, :, kmaxp1, :] = rho[:, :, kmaxp1, :] * gsqrtgam2[:, :, kmaxp1, :]
        rhog[:, :, kminm1, :] = rho[:, :, kminm1, :] * gsqrtgam2[:, :, kminm1, :]
        ein[:, :, kmaxp1, :] = CVdry * tem[:, :, kmaxp1, :]
        ein[:, :, kminm1, :] = CVdry * tem[:, :, kminm1, :]
        rhoge[:, :, kmaxp1, :] = rhog[:, :, kmaxp1, :] * ein[:, :, kmaxp1, :]
        rhoge[:, :, kminm1, :] = rhog[:, :, kminm1, :] * ein[:, :, kminm1, :]



        #--- Momentum ( rhogvx, rhogvy, rhogvz, vx, vy, vz )
        self.BNDCND_rhovxvyvz(
            kmin, kmax,
            rhog, rhogvx, rhogvy, rhogvz,
            cnst, rdtype,
        )
        

        vx[:, :, kmaxp1, :] = rhogvx[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vx[:, :, kminm1, :] = rhogvx[:, :, kminm1, :] / rhog[:, :, kminm1, :]
        vy[:, :, kmaxp1, :] = rhogvy[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vy[:, :, kminm1, :] = rhogvy[:, :, kminm1, :] / rhog[:, :, kminm1, :]
        vz[:, :, kmaxp1, :] = rhogvz[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vz[:, :, kminm1, :] = rhogvz[:, :, kminm1, :] / rhog[:, :, kminm1, :]


        #--- Momentum ( rhogw, w ) ok
        self.BNDCND_rhow(
            kmin, kmax,
            rhogvx, rhogvy, rhogvz, rhogw, c2wfact_Gz,
            rdtype,
        )


        w[:, :, kmaxp1, :] = rhogw[:, :, kmaxp1, :] / (
            c2wfact[:, :, kmaxp1, :, 0] * rhog[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 1] * rhog[:, :, kmax, :]
        )

        w[:, :, kmin, :] = rhogw[:, :, kmin, :] / (
            c2wfact[:, :, kmin, :, 0] * rhog[:, :, kmin,   :] +
            c2wfact[:, :, kmin, :, 1] * rhog[:, :, kminm1, :]
        )

        w[:, :, kminm1, :] = rdtype(0.0)


        return
    

    def BNDCND_all_resident(self, msc, DIAG_d, PROG_d, rho_d, ein_d):
        # Device-resident BNDCND_all (REGULAR path): same sequence as BNDCND_all
        # (the L235-288 numpy block) but on jax arrays via functional .at[].set()
        # and no internal asarray/to_numpy drains -- for the Pre_Post resident
        # chain. JAX-only. Returns updated (DIAG_d, PROG_d, rho_d, ein_d).
        xp = bk.xp
        adm, rcnf, cnst, vmtr = msc.adm, msc.rcnf, msc.cnst, msc.vmtr
        rdtype = bk.ndtype

        I_RHOG, I_RHOGVX, I_RHOGVY = rcnf.I_RHOG, rcnf.I_RHOGVX, rcnf.I_RHOGVY
        I_RHOGVZ, I_RHOGW, I_RHOGE = rcnf.I_RHOGVZ, rcnf.I_RHOGW, rcnf.I_RHOGE
        I_pre, I_tem = rcnf.I_pre, rcnf.I_tem
        I_vx, I_vy, I_vz, I_w = rcnf.I_vx, rcnf.I_vy, rcnf.I_vz, rcnf.I_w
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        kmaxp1, kminm1 = kmax + 1, kmin - 1
        CVdry = cnst.CONST_CVdry

        # RES-CAPSTONE: cache the loop-invariant vertical-metric constants device-
        # resident once (the standard device_consts pattern) instead of re-uploading
        # them via asarray every nl. These are read-only geometry (same object every
        # call); the call-site profiler attributed the single biggest H2D leak here
        # (VMTR_C2WfactGz alone = 340MB re-uploaded x3 nl x step). Bit-identical:
        # asarray-once vs asarray-every-call yields the same device values.
        _bgeom = bk.device_consts(self, "bndcnd_geom", lambda: {
            "gsgam2":  vmtr.VMTR_GSGAM2,
            "phi":     vmtr.VMTR_PHI,
            "c2wfact": vmtr.VMTR_C2Wfact,
            "c2wGz":   vmtr.VMTR_C2WfactGz,
        })
        gsgam2  = _bgeom["gsgam2"]
        phi     = _bgeom["phi"]
        c2wfact = _bgeom["c2wfact"]
        c2wGz   = _bgeom["c2wGz"]

        cfg_t = self._bnd_cfg_thermo(kmin, kmax, cnst)
        cfg_m = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()

        # --- thermo: tem/pre/rho boundary rows (kernel reads interior, undrained) ---
        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_reg"](
            DIAG_d[:, :, :, :, I_tem], rho_d, DIAG_d[:, :, :, :, I_pre], phi,
            cfg=cfg_t, xp=xp)
        DIAG_d = DIAG_d.at[:, :, kmaxp1, :, I_tem].set(tem_t).at[:, :, kminm1, :, I_tem].set(tem_b)
        DIAG_d = DIAG_d.at[:, :, kmaxp1, :, I_pre].set(pre_t).at[:, :, kminm1, :, I_pre].set(pre_b)
        rho_d  = rho_d.at[:, :, kmaxp1, :].set(rho_t).at[:, :, kminm1, :].set(rho_b)

        # --- rhog / ein / rhoge at boundary rows (L242-247) ---
        PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOG].set(rho_d[:, :, kmaxp1, :] * gsgam2[:, :, kmaxp1, :])
        PROG_d = PROG_d.at[:, :, kminm1, :, I_RHOG].set(rho_d[:, :, kminm1, :] * gsgam2[:, :, kminm1, :])
        ein_d  = ein_d.at[:, :, kmaxp1, :].set(CVdry * DIAG_d[:, :, kmaxp1, :, I_tem])
        ein_d  = ein_d.at[:, :, kminm1, :].set(CVdry * DIAG_d[:, :, kminm1, :, I_tem])
        PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOGE].set(PROG_d[:, :, kmaxp1, :, I_RHOG] * ein_d[:, :, kmaxp1, :])
        PROG_d = PROG_d.at[:, :, kminm1, :, I_RHOGE].set(PROG_d[:, :, kminm1, :, I_RHOG] * ein_d[:, :, kminm1, :])

        # --- horizontal momentum boundary rows + vx/vy/vz (L252-264) ---
        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_reg"](
            PROG_d[:, :, :, :, I_RHOG], PROG_d[:, :, :, :, I_RHOGVX],
            PROG_d[:, :, :, :, I_RHOGVY], PROG_d[:, :, :, :, I_RHOGVZ], cfg=cfg_m, xp=xp)
        PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOGVX].set(vx_t).at[:, :, kminm1, :, I_RHOGVX].set(vx_b)
        PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOGVY].set(vy_t).at[:, :, kminm1, :, I_RHOGVY].set(vy_b)
        PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOGVZ].set(vz_t).at[:, :, kminm1, :, I_RHOGVZ].set(vz_b)
        for kk in (kmaxp1, kminm1):
            DIAG_d = DIAG_d.at[:, :, kk, :, I_vx].set(PROG_d[:, :, kk, :, I_RHOGVX] / PROG_d[:, :, kk, :, I_RHOG])
            DIAG_d = DIAG_d.at[:, :, kk, :, I_vy].set(PROG_d[:, :, kk, :, I_RHOGVY] / PROG_d[:, :, kk, :, I_RHOG])
            DIAG_d = DIAG_d.at[:, :, kk, :, I_vz].set(PROG_d[:, :, kk, :, I_RHOGVZ] / PROG_d[:, :, kk, :, I_RHOG])

        # --- vertical momentum boundary rows + w (L268-285) ---
        rw_t, rw_b = ker["rhow_reg"](
            PROG_d[:, :, :, :, I_RHOGVX], PROG_d[:, :, :, :, I_RHOGVY],
            PROG_d[:, :, :, :, I_RHOGVZ], c2wGz, cfg=cfg_m, xp=xp)
        if rw_t is not None:
            PROG_d = PROG_d.at[:, :, kmaxp1, :, I_RHOGW].set(rw_t)
        if rw_b is not None:
            PROG_d = PROG_d.at[:, :, kmin, :, I_RHOGW].set(rw_b)
        PROG_d = PROG_d.at[:, :, kminm1, :, I_RHOGW].set(rdtype(0.0))

        rhogw = PROG_d[:, :, :, :, I_RHOGW]
        rhog  = PROG_d[:, :, :, :, I_RHOG]
        w_top = rhogw[:, :, kmaxp1, :] / (
            c2wfact[:, :, kmaxp1, :, 0] * rhog[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 1] * rhog[:, :, kmax, :])
        w_kmin = rhogw[:, :, kmin, :] / (
            c2wfact[:, :, kmin, :, 0] * rhog[:, :, kmin, :] +
            c2wfact[:, :, kmin, :, 1] * rhog[:, :, kminm1, :])
        DIAG_d = DIAG_d.at[:, :, kmaxp1, :, I_w].set(w_top)
        DIAG_d = DIAG_d.at[:, :, kmin, :, I_w].set(w_kmin)
        DIAG_d = DIAG_d.at[:, :, kminm1, :, I_w].set(rdtype(0.0))

        return DIAG_d, PROG_d, rho_d, ein_d

    def BNDCND_all_pl_resident(self, msc, DIAG_pl_d, PROG_pl_d, rho_pl_d, ein_pl_d):
        # Device-resident BNDCND_all for the POLE (_pl) path: line-by-line mirror
        # of BNDCND_all_resident above, but on pole-shaped jax arrays. The pole
        # drops the second spatial axis, so k moves from axis 2 -> axis 1:
        #   regular DIAG_d[:, :, k, :, I]  -> pole DIAG_pl_d[:, k, :, I]
        #   regular rho_d[:, :, k, :]      -> pole rho_pl_d[:, k, :]
        # The _pl boundary kernels (thermo_pl/rhov_pl/rhow_pl) are byte-for-byte
        # the _reg logic with that same axis shift and return IDENTICAL tuples, so
        # the unpacking is unchanged. Geometry uses the plain VMTR_*_pl constants
        # (the plmask tweak belongs only to the diagnostics rho division, not here
        # -- the host BNDCND_all_pl is likewise fed plain VMTR_GSGAM2_pl). JAX-only.
        # Returns updated (DIAG_pl_d, PROG_pl_d, rho_pl_d, ein_pl_d).
        xp = bk.xp
        adm, rcnf, cnst, vmtr = msc.adm, msc.rcnf, msc.cnst, msc.vmtr
        rdtype = bk.ndtype

        I_RHOG, I_RHOGVX, I_RHOGVY = rcnf.I_RHOG, rcnf.I_RHOGVX, rcnf.I_RHOGVY
        I_RHOGVZ, I_RHOGW, I_RHOGE = rcnf.I_RHOGVZ, rcnf.I_RHOGW, rcnf.I_RHOGE
        I_pre, I_tem = rcnf.I_pre, rcnf.I_tem
        I_vx, I_vy, I_vz, I_w = rcnf.I_vx, rcnf.I_vy, rcnf.I_vz, rcnf.I_w
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        kmaxp1, kminm1 = kmax + 1, kmin - 1
        CVdry = cnst.CONST_CVdry

        _bgeom = bk.device_consts(self, "bndcnd_geom_pl", lambda: {
            "gsgam2":  vmtr.VMTR_GSGAM2_pl,
            "phi":     vmtr.VMTR_PHI_pl,
            "c2wfact": vmtr.VMTR_C2Wfact_pl,
            "c2wGz":   vmtr.VMTR_C2WfactGz_pl,
        })
        gsgam2  = _bgeom["gsgam2"]
        phi     = _bgeom["phi"]
        c2wfact = _bgeom["c2wfact"]
        c2wGz   = _bgeom["c2wGz"]

        cfg_t = self._bnd_cfg_thermo(kmin, kmax, cnst)
        cfg_m = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()

        # --- thermo: tem/pre/rho boundary rows (kernel reads interior, undrained) ---
        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_pl"](
            DIAG_pl_d[:, :, :, I_tem], rho_pl_d, DIAG_pl_d[:, :, :, I_pre], phi,
            cfg=cfg_t, xp=xp)
        DIAG_pl_d = DIAG_pl_d.at[:, kmaxp1, :, I_tem].set(tem_t).at[:, kminm1, :, I_tem].set(tem_b)
        DIAG_pl_d = DIAG_pl_d.at[:, kmaxp1, :, I_pre].set(pre_t).at[:, kminm1, :, I_pre].set(pre_b)
        rho_pl_d  = rho_pl_d.at[:, kmaxp1, :].set(rho_t).at[:, kminm1, :].set(rho_b)

        # --- rhog / ein / rhoge at boundary rows ---
        PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOG].set(rho_pl_d[:, kmaxp1, :] * gsgam2[:, kmaxp1, :])
        PROG_pl_d = PROG_pl_d.at[:, kminm1, :, I_RHOG].set(rho_pl_d[:, kminm1, :] * gsgam2[:, kminm1, :])
        ein_pl_d  = ein_pl_d.at[:, kmaxp1, :].set(CVdry * DIAG_pl_d[:, kmaxp1, :, I_tem])
        ein_pl_d  = ein_pl_d.at[:, kminm1, :].set(CVdry * DIAG_pl_d[:, kminm1, :, I_tem])
        PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOGE].set(PROG_pl_d[:, kmaxp1, :, I_RHOG] * ein_pl_d[:, kmaxp1, :])
        PROG_pl_d = PROG_pl_d.at[:, kminm1, :, I_RHOGE].set(PROG_pl_d[:, kminm1, :, I_RHOG] * ein_pl_d[:, kminm1, :])

        # --- horizontal momentum boundary rows + vx/vy/vz ---
        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_pl"](
            PROG_pl_d[:, :, :, I_RHOG], PROG_pl_d[:, :, :, I_RHOGVX],
            PROG_pl_d[:, :, :, I_RHOGVY], PROG_pl_d[:, :, :, I_RHOGVZ], cfg=cfg_m, xp=xp)
        PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOGVX].set(vx_t).at[:, kminm1, :, I_RHOGVX].set(vx_b)
        PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOGVY].set(vy_t).at[:, kminm1, :, I_RHOGVY].set(vy_b)
        PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOGVZ].set(vz_t).at[:, kminm1, :, I_RHOGVZ].set(vz_b)
        for kk in (kmaxp1, kminm1):
            DIAG_pl_d = DIAG_pl_d.at[:, kk, :, I_vx].set(PROG_pl_d[:, kk, :, I_RHOGVX] / PROG_pl_d[:, kk, :, I_RHOG])
            DIAG_pl_d = DIAG_pl_d.at[:, kk, :, I_vy].set(PROG_pl_d[:, kk, :, I_RHOGVY] / PROG_pl_d[:, kk, :, I_RHOG])
            DIAG_pl_d = DIAG_pl_d.at[:, kk, :, I_vz].set(PROG_pl_d[:, kk, :, I_RHOGVZ] / PROG_pl_d[:, kk, :, I_RHOG])

        # --- vertical momentum boundary rows + w ---
        rw_t, rw_b = ker["rhow_pl"](
            PROG_pl_d[:, :, :, I_RHOGVX], PROG_pl_d[:, :, :, I_RHOGVY],
            PROG_pl_d[:, :, :, I_RHOGVZ], c2wGz, cfg=cfg_m, xp=xp)
        if rw_t is not None:
            PROG_pl_d = PROG_pl_d.at[:, kmaxp1, :, I_RHOGW].set(rw_t)
        if rw_b is not None:
            PROG_pl_d = PROG_pl_d.at[:, kmin, :, I_RHOGW].set(rw_b)
        PROG_pl_d = PROG_pl_d.at[:, kminm1, :, I_RHOGW].set(rdtype(0.0))

        rhogw = PROG_pl_d[:, :, :, I_RHOGW]
        rhog  = PROG_pl_d[:, :, :, I_RHOG]
        w_top = rhogw[:, kmaxp1, :] / (
            c2wfact[:, kmaxp1, :, 0] * rhog[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 1] * rhog[:, kmax, :])
        w_kmin = rhogw[:, kmin, :] / (
            c2wfact[:, kmin, :, 0] * rhog[:, kmin, :] +
            c2wfact[:, kmin, :, 1] * rhog[:, kminm1, :])
        DIAG_pl_d = DIAG_pl_d.at[:, kmaxp1, :, I_w].set(w_top)
        DIAG_pl_d = DIAG_pl_d.at[:, kmin, :, I_w].set(w_kmin)
        DIAG_pl_d = DIAG_pl_d.at[:, kminm1, :, I_w].set(rdtype(0.0))

        return DIAG_pl_d, PROG_pl_d, rho_pl_d, ein_pl_d

    def BNDCND_all_pl(
        self,
        kmin,
        kmax,
        idim, 
        kdim, 
        ldim, 
        rho,       # (idim, kdim, ldim)  density
        vx,        # (idim, kdim, ldim)  horizontal wind (x)
        vy,        # (idim, kdim, ldim)  horizontal wind (y) 
        vz,        # (idim, kdim, ldim)  horizontal wind (z)
        w,         # (idim, kdim, ldim)  vertical wind           ####
        ein,       # (idim, kdim, ldim)  internal energy
        tem,       # (idim, kdim, ldim)  temperature
        pre,       # (idim, kdim, ldim)  pressure
        rhog,
        rhogvx,
        rhogvy,
        rhogvz,
        rhogw,                                                         ####
        rhoge,
        gsqrtgam2,  
        phi,       # (idim, kdim, ldim)  geopotential
        c2wfact,    
        c2wfact_Gz,
        cnst,
        rdtype,
    ):

        #kmin = adm.ADM_kmin
        #kmax = adm.ADM_kmax
        kmaxp1 = kmax + 1
        kminm1 = kmin - 1
        CVdry = cnst.CONST_CVdry


        # with open(std.fname_log, 'a') as log_file:
        #     print("ZERO0", file=log_file)
        #     print(tem[16,0,kmaxp1,0], file=log_file)
        #     print(rho[16,0,kmaxp1,0], gsqrtgam2[16,0,kmaxp1,0], file=log_file)
        #     print(pre[16,0,kmaxp1,0], file=log_file)
        #     print(phi[16,0,kmaxp1,0], file=log_file)
        #     print(phi[16,0,kmax,0], file=log_file)    
            #print(phi[16,0,3,0], file=log_file)    
            #print(phi[16,0,0,0], file=log_file)    
            #print(phi[10,10,3,0], file=log_file)    
            #print(rho[16,0,kmax,0],gsqrtgam2[16,0,kmax,0], file=log_file)
            #print(rho[17,0,kmaxp1,0],gsqrtgam2[17,0,kmaxp1,0], file=log_file)   
            #print(rho[17,0,kmax,0],gsqrtgam2[17,0,kmax,0], file=log_file)

        #--- Thermodynamical variables ( rho, ein, tem, pre, rhog, rhoge ), q = 0 at boundary
        self.BNDCND_thermo_pl(
            kmin, kmax,
            tem, rho, pre, phi, 
            cnst, rdtype
        )

        rhog[:, kmaxp1, :] = rho[:, kmaxp1, :] * gsqrtgam2[:, kmaxp1, :]
        rhog[:, kminm1, :] = rho[:, kminm1, :] * gsqrtgam2[:, kminm1, :]
        ein[:, kmaxp1, :] = CVdry * tem[:, kmaxp1, :]
        ein[:, kminm1, :] = CVdry * tem[:, kminm1, :]
        rhoge[:, kmaxp1, :] = rhog[:, kmaxp1, :] * ein[:, kmaxp1, :]
        rhoge[:, kminm1, :] = rhog[:, kminm1, :] * ein[:, kminm1, :]

        # with open(std.fname_log, 'a') as log_file:
        #     print("ZERO1", file=log_file)
        #     print(rho[16,0,kmaxp1,0],gsqrtgam2[16,0,kmaxp1,0], file=log_file)
        #     print(rho[16,0,kmax,0],gsqrtgam2[16,0,kmax,0], file=log_file)
        #     print(rho[17,0,kmaxp1,0],gsqrtgam2[17,0,kmaxp1,0], file=log_file)   
        #     print(rho[17,0,kmax,0],gsqrtgam2[17,0,kmax,0], file=log_file)


        #--- Momentum ( rhogvx, rhogvy, rhogvz, vx, vy, vz )
        self.BNDCND_rhovxvyvz_pl(
            kmin, kmax,
            rhog, rhogvx, rhogvy, rhogvz,
            cnst, rdtype,
        )
        

        vx[:, kmaxp1, :] = rhogvx[:, kmaxp1, :] / rhog[:, kmaxp1, :]
        vx[:, kminm1, :] = rhogvx[:, kminm1, :] / rhog[:, kminm1, :]
        vy[:, kmaxp1, :] = rhogvy[:, kmaxp1, :] / rhog[:, kmaxp1, :]
        vy[:, kminm1, :] = rhogvy[:, kminm1, :] / rhog[:, kminm1, :]
        vz[:, kmaxp1, :] = rhogvz[:, kmaxp1, :] / rhog[:, kmaxp1, :]
        vz[:, kminm1, :] = rhogvz[:, kminm1, :] / rhog[:, kminm1, :]


        #--- Momentum ( rhogw, w ) 
        self.BNDCND_rhow_pl(
            kmin, kmax,
            rhogvx, rhogvy, rhogvz, rhogw, c2wfact_Gz,
            rdtype,
        )



        w[:, kmaxp1, :] = rhogw[:, kmaxp1, :] / (
            c2wfact[:, kmaxp1, :, 0] * rhog[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 1] * rhog[:, kmax, :]
        )

        w[:, kmin, :] = rhogw[:, kmin, :] / (
            c2wfact[:, kmin, :, 0] * rhog[:, kmin,   :] +
            c2wfact[:, kmin, :, 1] * rhog[:, kminm1, :]
        )

        w[:, kminm1, :] = rdtype(0.0)

        return
    

    def BNDCND_thermo(
        self,
        kmin, kmax,
        tem, rho, pre, phi,
        cnst, rdtype
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_thermo(kmin, kmax, cnst)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_reg"](
            xp.asarray(tem), xp.asarray(rho), xp.asarray(pre), xp.asarray(phi),
            cfg=cfg, xp=xp,
        )
        tem[:, :, kmaxp1, :] = bk.to_numpy(tem_t)
        tem[:, :, kminm1, :] = bk.to_numpy(tem_b)
        pre[:, :, kmaxp1, :] = bk.to_numpy(pre_t)
        pre[:, :, kminm1, :] = bk.to_numpy(pre_b)
        rho[:, :, kmaxp1, :] = bk.to_numpy(rho_t)
        rho[:, :, kminm1, :] = bk.to_numpy(rho_b)

        return


    def BNDCND_thermo_pl(
        self,
        kmin, kmax,
        tem, rho, pre, phi,
        cnst, rdtype
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_thermo(kmin, kmax, cnst)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_pl"](
            xp.asarray(tem), xp.asarray(rho), xp.asarray(pre), xp.asarray(phi),
            cfg=cfg, xp=xp,
        )
        tem[:, kmaxp1, :] = bk.to_numpy(tem_t)
        tem[:, kminm1, :] = bk.to_numpy(tem_b)
        pre[:, kmaxp1, :] = bk.to_numpy(pre_t)
        pre[:, kminm1, :] = bk.to_numpy(pre_b)
        rho[:, kmaxp1, :] = bk.to_numpy(rho_t)
        rho[:, kminm1, :] = bk.to_numpy(rho_b)

        return

    
    def BNDCND_rhovxvyvz(
        self,
        kmin, kmax,
        rhog, rhogvx, rhogvy, rhogvz,
        cnst, rdtype,
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_reg"](
            xp.asarray(rhog), xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            cfg=cfg, xp=xp,
        )
        rhogvx[:, :, kmaxp1, :] = bk.to_numpy(vx_t)
        rhogvy[:, :, kmaxp1, :] = bk.to_numpy(vy_t)
        rhogvz[:, :, kmaxp1, :] = bk.to_numpy(vz_t)
        rhogvx[:, :, kminm1, :] = bk.to_numpy(vx_b)
        rhogvy[:, :, kminm1, :] = bk.to_numpy(vy_b)
        rhogvz[:, :, kminm1, :] = bk.to_numpy(vz_b)

        return

    def BNDCND_rhovxvyvz_pl(
        self,
        kmin, kmax,
        rhog, rhogvx, rhogvy, rhogvz,
        cnst, rdtype,
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_pl"](
            xp.asarray(rhog), xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            cfg=cfg, xp=xp,
        )
        rhogvx[:, kmaxp1, :] = bk.to_numpy(vx_t)
        rhogvy[:, kmaxp1, :] = bk.to_numpy(vy_t)
        rhogvz[:, kmaxp1, :] = bk.to_numpy(vz_t)
        rhogvx[:, kminm1, :] = bk.to_numpy(vx_b)
        rhogvy[:, kminm1, :] = bk.to_numpy(vy_b)
        rhogvz[:, kminm1, :] = bk.to_numpy(vz_b)

        return


    def BNDCND_rhow(
        self,
        kmin, kmax,
        rhogvx, rhogvy, rhogvz, rhogw, c2wfact,
        rdtype,
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        rw_t, rw_b = ker["rhow_reg"](
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            xp.asarray(c2wfact), cfg=cfg, xp=xp,
        )
        if rw_t is not None:
            rhogw[:, :, kmaxp1, :] = bk.to_numpy(rw_t)
        if rw_b is not None:
            rhogw[:, :, kmin, :] = bk.to_numpy(rw_b)
        rhogw[:, :, kminm1, :] = rdtype(0.0)

        return

    def BNDCND_rhow_pl(
        self,
        kmin, kmax,
        rhogvx, rhogvy, rhogvz, rhogw, c2wfact,
        rdtype,
    ):

        xp = bk.xp
        cfg = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()
        kmaxp1, kminm1 = kmax + 1, kmin - 1

        rw_t, rw_b = ker["rhow_pl"](
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            xp.asarray(c2wfact), cfg=cfg, xp=xp,
        )
        if rw_t is not None:
            rhogw[:, kmaxp1, :] = bk.to_numpy(rw_t)
        if rw_b is not None:
            rhogw[:, kmin, :] = bk.to_numpy(rw_b)
        rhogw[:, kminm1, :] = rdtype(0.0)

        return

    def BNDCND_pre_sfc(self, kmin, rho, pre, z, z_srf, cnst, rdtype):
        """Surface density/pressure by Lagrange extrapolation to the surface.

        SCAFFOLD (DCMIP port) -- ported from full NICAM mod_bndcnd.f90
        (BNDCND_pre_sfc_DP). NICAM-DC's dry core never needed this; the DCMIP
        physics glue (AF_dcmip) requires pre_sfc as input.

        Fortran interface:
          BNDCND_pre_sfc(ijdim, rho, pre, z, z_srf, rho_srf, pre_srf)
            rho,pre [IN] (gall,kall) ; z [IN] (gall,kall) geopotential height
            z_srf   [IN] (gall,) surface height
          rho_srf = quadratic Lagrange interp of rho over kmin..kmin+2 to z_srf
          pre_srf = pre(kmin) + hydrostatic correction (GRAV * rho_srf * dz)
        Returns: rho_srf, pre_srf  (both shape (gall,) or (gall_1d,gall_1d))
        """
        raise NotImplementedError(
            "BNDCND_pre_sfc: scaffold only. Port Lagrange z-extrap of rho + "
            "hydrostatic pre_srf from full NICAM mod_bndcnd.f90."
        )
