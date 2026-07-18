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
        

    def _bndcnd_all_core(self, msc, DIAG, PROG, rho, ein):
        # Backend-agnostic BNDCND_all (REGULAR path): set the top/bottom boundary rows
        # of the thermodynamic + momentum fields. ONE source for both backends -- via
        # bk.set_at, which is in-place `a[idx]=x` on numpy and functional `a.at[idx].set(x)`
        # on jax. Returns the updated (DIAG, PROG, rho, ein): on numpy the same objects
        # mutated in place; on jax new arrays. (Replaces the old numpy/jax twin methods --
        # the in-place BNDCND_all and the .at[]-based BNDCND_all_resident.)
        xp = bk.xp
        adm, rcnf, cnst, vmtr = msc.adm, msc.rcnf, msc.cnst, msc.vmtr
        rdtype = bk.ndtype
        sa = bk.set_at

        I_RHOG, I_RHOGVX, I_RHOGVY = rcnf.I_RHOG, rcnf.I_RHOGVX, rcnf.I_RHOGVY
        I_RHOGVZ, I_RHOGW, I_RHOGE = rcnf.I_RHOGVZ, rcnf.I_RHOGW, rcnf.I_RHOGE
        I_pre, I_tem = rcnf.I_pre, rcnf.I_tem
        I_vx, I_vy, I_vz, I_w = rcnf.I_vx, rcnf.I_vy, rcnf.I_vz, rcnf.I_w
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        kmaxp1, kminm1 = kmax + 1, kmin - 1
        CVdry = cnst.CONST_CVdry

        # Loop-invariant vertical-metric constants, staged (device-resident) once.
        _bgeom = bk.device_consts(self, "bndcnd_geom", lambda: {
            "gsgam2":  vmtr.VMTR_GSGAM2,
            "phi":     vmtr.VMTR_PHI,
            "c2wfact": vmtr.VMTR_C2Wfact,
            "c2wGz":   vmtr.VMTR_C2WfactGz,
        })
        gsgam2, phi, c2wfact, c2wGz = _bgeom["gsgam2"], _bgeom["phi"], _bgeom["c2wfact"], _bgeom["c2wGz"]

        cfg_t = self._bnd_cfg_thermo(kmin, kmax, cnst)
        cfg_m = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()

        # --- thermo: tem/pre/rho boundary rows (kernel reads interior, unchanged) ---
        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_reg"](
            DIAG[:, :, :, :, I_tem], rho, DIAG[:, :, :, :, I_pre], phi, cfg=cfg_t, xp=xp)
        DIAG = sa(sa(DIAG, np.s_[:, :, kmaxp1, :, I_tem], tem_t), np.s_[:, :, kminm1, :, I_tem], tem_b)
        DIAG = sa(sa(DIAG, np.s_[:, :, kmaxp1, :, I_pre], pre_t), np.s_[:, :, kminm1, :, I_pre], pre_b)
        rho  = sa(sa(rho,  np.s_[:, :, kmaxp1, :],        rho_t), np.s_[:, :, kminm1, :],        rho_b)

        # --- rhog / ein / rhoge at boundary rows ---
        PROG = sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOG], rho[:, :, kmaxp1, :] * gsgam2[:, :, kmaxp1, :])
        PROG = sa(PROG, np.s_[:, :, kminm1, :, I_RHOG], rho[:, :, kminm1, :] * gsgam2[:, :, kminm1, :])
        ein  = sa(ein,  np.s_[:, :, kmaxp1, :], CVdry * DIAG[:, :, kmaxp1, :, I_tem])
        ein  = sa(ein,  np.s_[:, :, kminm1, :], CVdry * DIAG[:, :, kminm1, :, I_tem])
        PROG = sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOGE], PROG[:, :, kmaxp1, :, I_RHOG] * ein[:, :, kmaxp1, :])
        PROG = sa(PROG, np.s_[:, :, kminm1, :, I_RHOGE], PROG[:, :, kminm1, :, I_RHOG] * ein[:, :, kminm1, :])

        # --- horizontal momentum boundary rows + vx/vy/vz ---
        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_reg"](
            PROG[:, :, :, :, I_RHOG], PROG[:, :, :, :, I_RHOGVX],
            PROG[:, :, :, :, I_RHOGVY], PROG[:, :, :, :, I_RHOGVZ], cfg=cfg_m, xp=xp)
        PROG = sa(sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOGVX], vx_t), np.s_[:, :, kminm1, :, I_RHOGVX], vx_b)
        PROG = sa(sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOGVY], vy_t), np.s_[:, :, kminm1, :, I_RHOGVY], vy_b)
        PROG = sa(sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOGVZ], vz_t), np.s_[:, :, kminm1, :, I_RHOGVZ], vz_b)
        for kk in (kmaxp1, kminm1):
            DIAG = sa(DIAG, np.s_[:, :, kk, :, I_vx], PROG[:, :, kk, :, I_RHOGVX] / PROG[:, :, kk, :, I_RHOG])
            DIAG = sa(DIAG, np.s_[:, :, kk, :, I_vy], PROG[:, :, kk, :, I_RHOGVY] / PROG[:, :, kk, :, I_RHOG])
            DIAG = sa(DIAG, np.s_[:, :, kk, :, I_vz], PROG[:, :, kk, :, I_RHOGVZ] / PROG[:, :, kk, :, I_RHOG])

        # --- vertical momentum boundary rows + w ---
        rw_t, rw_b = ker["rhow_reg"](
            PROG[:, :, :, :, I_RHOGVX], PROG[:, :, :, :, I_RHOGVY],
            PROG[:, :, :, :, I_RHOGVZ], c2wGz, cfg=cfg_m, xp=xp)
        if rw_t is not None:
            PROG = sa(PROG, np.s_[:, :, kmaxp1, :, I_RHOGW], rw_t)
        if rw_b is not None:
            PROG = sa(PROG, np.s_[:, :, kmin, :, I_RHOGW], rw_b)
        PROG = sa(PROG, np.s_[:, :, kminm1, :, I_RHOGW], rdtype(0.0))

        rhogw = PROG[:, :, :, :, I_RHOGW]
        rhog  = PROG[:, :, :, :, I_RHOG]
        w_top = rhogw[:, :, kmaxp1, :] / (
            c2wfact[:, :, kmaxp1, :, 0] * rhog[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 1] * rhog[:, :, kmax, :])
        w_kmin = rhogw[:, :, kmin, :] / (
            c2wfact[:, :, kmin, :, 0] * rhog[:, :, kmin, :] +
            c2wfact[:, :, kmin, :, 1] * rhog[:, :, kminm1, :])
        DIAG = sa(DIAG, np.s_[:, :, kmaxp1, :, I_w], w_top)
        DIAG = sa(DIAG, np.s_[:, :, kmin,   :, I_w], w_kmin)
        DIAG = sa(DIAG, np.s_[:, :, kminm1, :, I_w], rdtype(0.0))

        return DIAG, PROG, rho, ein

    def BNDCND_all(self, msc):
        # numpy call convention: mutate msc.dyn.DIAG/PROG/rho/ein IN PLACE (bk.set_at is
        # in-place on numpy) -- the returned handles are the same objects, so callers that
        # read msc.dyn afterwards need no change.
        self._bndcnd_all_core(msc, msc.dyn.DIAG, msc.dyn.PROG, msc.dyn.rho, msc.dyn.ein)
        return

    def BNDCND_all_resident(self, msc, DIAG_d, PROG_d, rho_d, ein_d):
        # jax (functional) call convention: return the updated device handles.
        return self._bndcnd_all_core(msc, DIAG_d, PROG_d, rho_d, ein_d)

    def _bndcnd_all_pl_core(self, msc, DIAG, PROG, rho, ein):
        # Backend-agnostic BNDCND_all for the POLE (_pl) path: the axis-shifted twin of
        # _bndcnd_all_core (pole drops the 2nd spatial axis, so k moves axis 2 -> axis 1:
        # regular DIAG[:, :, k, :, I] -> pole DIAG[:, k, :, I]). ONE source for both
        # backends via bk.set_at. Returns the updated (DIAG, PROG, rho, ein). (Replaces the
        # numpy/jax twins BNDCND_all_pl and BNDCND_all_pl_resident.)
        xp = bk.xp
        adm, rcnf, cnst, vmtr = msc.adm, msc.rcnf, msc.cnst, msc.vmtr
        rdtype = bk.ndtype
        sa = bk.set_at

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
        gsgam2, phi, c2wfact, c2wGz = _bgeom["gsgam2"], _bgeom["phi"], _bgeom["c2wfact"], _bgeom["c2wGz"]

        cfg_t = self._bnd_cfg_thermo(kmin, kmax, cnst)
        cfg_m = self._bnd_cfg_mom(kmin, kmax)
        ker = self._bnd_kernels_get()

        # --- thermo: tem/pre/rho boundary rows ---
        tem_t, tem_b, pre_t, pre_b, rho_t, rho_b = ker["thermo_pl"](
            DIAG[:, :, :, I_tem], rho, DIAG[:, :, :, I_pre], phi, cfg=cfg_t, xp=xp)
        DIAG = sa(sa(DIAG, np.s_[:, kmaxp1, :, I_tem], tem_t), np.s_[:, kminm1, :, I_tem], tem_b)
        DIAG = sa(sa(DIAG, np.s_[:, kmaxp1, :, I_pre], pre_t), np.s_[:, kminm1, :, I_pre], pre_b)
        rho  = sa(sa(rho,  np.s_[:, kmaxp1, :],        rho_t), np.s_[:, kminm1, :],        rho_b)

        # --- rhog / ein / rhoge at boundary rows ---
        PROG = sa(PROG, np.s_[:, kmaxp1, :, I_RHOG], rho[:, kmaxp1, :] * gsgam2[:, kmaxp1, :])
        PROG = sa(PROG, np.s_[:, kminm1, :, I_RHOG], rho[:, kminm1, :] * gsgam2[:, kminm1, :])
        ein  = sa(ein,  np.s_[:, kmaxp1, :], CVdry * DIAG[:, kmaxp1, :, I_tem])
        ein  = sa(ein,  np.s_[:, kminm1, :], CVdry * DIAG[:, kminm1, :, I_tem])
        PROG = sa(PROG, np.s_[:, kmaxp1, :, I_RHOGE], PROG[:, kmaxp1, :, I_RHOG] * ein[:, kmaxp1, :])
        PROG = sa(PROG, np.s_[:, kminm1, :, I_RHOGE], PROG[:, kminm1, :, I_RHOG] * ein[:, kminm1, :])

        # --- horizontal momentum boundary rows + vx/vy/vz ---
        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = ker["rhov_pl"](
            PROG[:, :, :, I_RHOG], PROG[:, :, :, I_RHOGVX],
            PROG[:, :, :, I_RHOGVY], PROG[:, :, :, I_RHOGVZ], cfg=cfg_m, xp=xp)
        PROG = sa(sa(PROG, np.s_[:, kmaxp1, :, I_RHOGVX], vx_t), np.s_[:, kminm1, :, I_RHOGVX], vx_b)
        PROG = sa(sa(PROG, np.s_[:, kmaxp1, :, I_RHOGVY], vy_t), np.s_[:, kminm1, :, I_RHOGVY], vy_b)
        PROG = sa(sa(PROG, np.s_[:, kmaxp1, :, I_RHOGVZ], vz_t), np.s_[:, kminm1, :, I_RHOGVZ], vz_b)
        for kk in (kmaxp1, kminm1):
            DIAG = sa(DIAG, np.s_[:, kk, :, I_vx], PROG[:, kk, :, I_RHOGVX] / PROG[:, kk, :, I_RHOG])
            DIAG = sa(DIAG, np.s_[:, kk, :, I_vy], PROG[:, kk, :, I_RHOGVY] / PROG[:, kk, :, I_RHOG])
            DIAG = sa(DIAG, np.s_[:, kk, :, I_vz], PROG[:, kk, :, I_RHOGVZ] / PROG[:, kk, :, I_RHOG])

        # --- vertical momentum boundary rows + w ---
        rw_t, rw_b = ker["rhow_pl"](
            PROG[:, :, :, I_RHOGVX], PROG[:, :, :, I_RHOGVY],
            PROG[:, :, :, I_RHOGVZ], c2wGz, cfg=cfg_m, xp=xp)
        if rw_t is not None:
            PROG = sa(PROG, np.s_[:, kmaxp1, :, I_RHOGW], rw_t)
        if rw_b is not None:
            PROG = sa(PROG, np.s_[:, kmin, :, I_RHOGW], rw_b)
        PROG = sa(PROG, np.s_[:, kminm1, :, I_RHOGW], rdtype(0.0))

        rhogw = PROG[:, :, :, I_RHOGW]
        rhog  = PROG[:, :, :, I_RHOG]
        w_top = rhogw[:, kmaxp1, :] / (
            c2wfact[:, kmaxp1, :, 0] * rhog[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 1] * rhog[:, kmax, :])
        w_kmin = rhogw[:, kmin, :] / (
            c2wfact[:, kmin, :, 0] * rhog[:, kmin, :] +
            c2wfact[:, kmin, :, 1] * rhog[:, kminm1, :])
        DIAG = sa(DIAG, np.s_[:, kmaxp1, :, I_w], w_top)
        DIAG = sa(DIAG, np.s_[:, kmin,   :, I_w], w_kmin)
        DIAG = sa(DIAG, np.s_[:, kminm1, :, I_w], rdtype(0.0))

        return DIAG, PROG, rho, ein

    def BNDCND_all_pl(self, msc, DIAG_pl, PROG_pl, rho_pl, ein_pl):
        # numpy call convention: mutate the pole arrays IN PLACE (bk.set_at is in-place on
        # numpy). Returns the same objects, so callers reading them afterwards need no change.
        self._bndcnd_all_pl_core(msc, DIAG_pl, PROG_pl, rho_pl, ein_pl)
        return

    def BNDCND_all_pl_resident(self, msc, DIAG_pl_d, PROG_pl_d, rho_pl_d, ein_pl_d):
        # jax (functional) call convention: return the updated device handles.
        return self._bndcnd_all_pl_core(msc, DIAG_pl_d, PROG_pl_d, rho_pl_d, ein_pl_d)
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
        """Surface density/pressure by extrapolation (full NICAM mod_bndcnd.f90
        BNDCND_pre_sfc_DP). Needed by the DCMIP glue (AF_dcmip pre_sfc input);
        NICAM-DC's dry core never had it.

          rho_srf = quadratic Lagrange extrapolation of rho over the lowest 3
                    full levels (kmin,kmin+1,kmin+2) to the surface height z_srf.
          pre_srf = pre(kmin) + 0.5*(rho_srf+rho(kmin))*GRAV*(z(kmin)-z_srf)
                    (hydrostatic balance).

        Args (pyNICAM 4D layout (i,j,k,l)):
          kmin  : lowest full-level python index
          rho,pre,z : (i,j,kall,l)  density, pressure, geopotential height
          z_srf : (i,j,l) surface height
        Returns rho_srf, pre_srf : (i,j,l).
        """
        GRAV = cnst.CONST_GRAV

        z_k1 = z[:, :, kmin,     :]
        z_k2 = z[:, :, kmin + 1, :]
        z_k3 = z[:, :, kmin + 2, :]
        z_ks = z_srf
        r1 = rho[:, :, kmin,     :]
        r2 = rho[:, :, kmin + 1, :]
        r3 = rho[:, :, kmin + 2, :]

        # quadratic Lagrange interpolation p(z_ks) over nodes (z_k1,z_k2,z_k3)
        rho_srf = (((z_ks - z_k2) * (z_ks - z_k3)) / ((z_k1 - z_k2) * (z_k1 - z_k3)) * r1
                   + ((z_ks - z_k1) * (z_ks - z_k3)) / ((z_k2 - z_k1) * (z_k2 - z_k3)) * r2
                   + ((z_ks - z_k1) * (z_ks - z_k2)) / ((z_k3 - z_k1) * (z_k3 - z_k2)) * r3)

        pre_srf = pre[:, :, kmin, :] + rdtype(0.5) * (rho_srf + r1) * GRAV * (z_k1 - z_ks)

        return rho_srf, pre_srf
