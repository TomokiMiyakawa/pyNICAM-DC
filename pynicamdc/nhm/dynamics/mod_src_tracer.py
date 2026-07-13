import toml
import os
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_ppmask import ppm
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.tracervertflux import (
    TracerVertFluxCfg, compute_tracer_vert_flux,
)
from pynicamdc.nhm.dynamics.kernels.tracervertadv import (
    TracerVertAdvCfg, compute_vert_qh, compute_vert_qh_pl,
    compute_vert_update, compute_vert_update_pl,
)
from pynicamdc.nhm.dynamics.kernels.horizontalremap import (
    RemapCfg, compute_horizontal_remap, RemapCfgPl, compute_horizontal_remap_pl,
)
from pynicamdc.nhm.dynamics.kernels.horizontallimiter import (
    HLimiterCfg, compute_horizontal_limiter_qout, compute_horizontal_limiter_apply,
    HLimiterCfgPl, compute_horizontal_limiter_qout_pl, compute_horizontal_limiter_apply_pl,
)
from pynicamdc.nhm.dynamics.kernels.verticallimiter import (
    VLimiterCfg, compute_vertical_limiter,
)
from pynicamdc.nhm.dynamics.kernels.horizontalflux import (
    HorizFluxCfg, compute_horizontal_flux,
)

class Srctr:
    
    _instance = None

    def __init__(self,cnst,rdtype):
        pass


    def _vertadv_setup(self, grd):
        """Lazily build + cache the per-tracer vertical-advection kernels and
        their geometry consts (gated PYNICAM_FUSE_VTRACERADV). Returns
        (enabled, kernels-dict, cfg, consts-dict). jax-only; numpy path keeps
        the original Python (l,k) loops untouched."""
        enabled = (bk.type == "jax"
                   and getattr(self, "use_fuse_vtraceradv",
                               os.environ.get("PYNICAM_FUSE_VTRACERADV", "1") != "0"))
        if not enabled:
            return False, None, None, None
        if getattr(self, "_vta_kernels", None) is None:
            self._vta_cfg = TracerVertAdvCfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax, have_pl=adm.ADM_have_pl,
            )
            self._vta_kernels = {
                "qh":  bk.maybe_jit(compute_vert_qh,        static_argnames=("cfg", "xp")),
                "qhp": bk.maybe_jit(compute_vert_qh_pl,     static_argnames=("cfg", "xp")),
                "up":  bk.maybe_jit(compute_vert_update,    static_argnames=("cfg", "xp")),
                "upp": bk.maybe_jit(compute_vert_update_pl, static_argnames=("cfg", "xp")),
            }
        d = bk.device_consts(self, "vertadv", lambda: {
            "afact": grd.GRD_afact, "bfact": grd.GRD_bfact, "rdgz": grd.GRD_rdgz,
        })
        return True, self._vta_kernels, self._vta_cfg, d


    def src_tracer_advection(self,
       vmax,                         # [IN] number of tracers   
       rhogq,       rhogq_pl,        # [INOUT] rhogq   ( G^1/2 x gam2 )     
       rhog_in,     rhog_in_pl,      # [IN] rho(old)( G^1/2 x gam2 )
       rhog_mean,   rhog_mean_pl,    # [IN] rho     ( G^1/2 x gam2 )
       rhogvx_mean, rhogvx_mean_pl,  # [IN] rho*Vx  ( G^1/2 x gam2 )
       rhogvy_mean, rhogvy_mean_pl,  # [IN] rho*Vy  ( G^1/2 x gam2 )
       rhogvz_mean, rhogvz_mean_pl,  # [IN] rho*Vz  ( G^1/2 x gam2 )
       rhogw_mean,  rhogw_mean_pl,   # [IN] rho*w  ( G^1/2 x gam2 )
       frhog,       frhog_pl,        # [IN] hyperviscosity tendency for rhog
       dt,                           # [IN] delta t
       thuburn_lim,                  # [IN] switch of thuburn limiter [add] 20130613 R.Yoshida   
       thuburn_lim_v, thuburn_lim_h, # [IN] switch of thuburn limiter, optional
       cnst, comm, grd, gmtr, oprt, vmtr, rdtype,
       rhog_in_d=None,               # U1 (RES-CAPSTONE-19): device PROG00[I_RHOG] snapshot
       rhog_in_pl_d=None,            # RC-82: device POLE PROG00[I_RHOG] snapshot (skips asarray(rhog_in_pl) @TVF/vert-adv)
                                     #   (== xp.asarray(rhog_in)); skips the in-tracer H2D
       rhogq_d=None,                 # RC-74: device rhogq input (== _PROGq_carry_d, the
                                     #   nl-invariant device PROGq); skips asarray(rhogq) @332
       rhogq_pl_d=None,              # RES-TRACER-3: device POLE rhogq input (== _PROGq_pl_
                                     #   carry_d); skips asarray(rhogq_pl) @~372 (pole analog)
       skip_drain=False,             # U5-D.2: caller does PROGq update+marshal on device
                                     #   from the returned _rhogq_d -> skip the host drain
       frhog_d=None,                 # RES-CAPSTONE-36: device f_TEND[I_RHOG] (the hdiff
                                     #   stash _ftrho); == asarray(frhog), skips 4 H2D
       rhog_mean_d=None,             # RES-CAPSTONE-35: device PROG_mean slices (already
       rhogvx_mean_d=None,           #   on-device COMM'd in vi); == asarray(rho/..._mean),
       rhogvy_mean_d=None,           #   skips the mean mass flux H2D uploads (TVF @229,
       rhogvz_mean_d=None,           #   rhogvx_d @497, flux rho @1307). Regular only;
       rhogw_mean_d=None,            #   pole _mean stays host (Track B).
       rhog_mean_pl_d=None,          # RC-81: device POLE mean flux slices (pole analog of
       rhogvx_mean_pl_d=None,        #   rhog_mean_d; == asarray(rho/..._mean_pl), skips the
       rhogvy_mean_pl_d=None,        #   pole TVF @255/256 + scaled flux @580 + horizontal_flux
       rhogvz_mean_pl_d=None,        #   @1534 H2D uploads. None -> host fallback (bit-exact).
       rhogw_mean_pl_d=None,
       frhog_pl_d=None,              # RES-CAPSTONE-39 (Track B): device f_TEND_pl[I_RHOG];
                                     #   == asarray(frhog_pl), no-ops the pole TVF upload @241.
       skip_drain_pl=False,          # RES-CAPSTONE-44 (Track B unit 5): caller does the pole
                                     #   PROGq_pl update + marshal on device -> skip the pole
                                     #   rhogq drain and return the device handle.
    ):

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0

        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR

        rhog     = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhog_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvx   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvx_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvy   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvy_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvz   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvz_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        q        = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        q_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        d        = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        d_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        q_h      = np.full(adm.ADM_shape, cnst.CONST_UNDEF)          # q at layer face
        q_h_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        flx_v    = np.full(adm.ADM_shape, cnst.CONST_UNDEF)          # mass flux
        flx_v_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        ck       = np.full(adm.ADM_shape +(2,), cnst.CONST_UNDEF)    # Courant number
        ck_pl    = np.full(adm.ADM_shape_pl +(2,), cnst.CONST_UNDEF)

        q_a      = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # q at cell face
        q_a_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        flx_h    = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # mass flux
        flx_h_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        ch       = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # Courant number
        ch_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        cmask    = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # upwind direction mask
        cmask_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        grd_xc   = np.full(adm.ADM_shape + (AJ - AI + 1, ZDIR - XDIR +1,), cnst.CONST_UNDEF)                   # mass centroid position
        grd_xc_pl= np.full(adm.ADM_shape_pl + (ZDIR - XDIR +1,), cnst.CONST_UNDEF)

        EPS = cnst.CONST_EPS

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        b1 = rdtype(0.0)
        b2 = rdtype(1.0)
        b3 = rdtype(1.0) - (b1+b2)

        apply_limiter_v = np.full(vmax, thuburn_lim, dtype=bool)
        apply_limiter_h = np.full(vmax, thuburn_lim, dtype=bool)

        if thuburn_lim_v is not None:
            apply_limiter_v[:] = thuburn_lim_v[:]

        if thuburn_lim_h is not None:
            apply_limiter_h[:] = thuburn_lim_h[:]

        #---------------------------------------------------------------------------
        # Vertical Advection (fractioanl step) : 1st
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vertical_adv',2)

        # RES-TP-3a: single-drain rhogq across the tracer's three phases (vertical-1
        # -> horizontal -> vertical-2). Instead of draining rhogq to host + re-
        # asarray'ing it between each phase, carry one device handle (_rhogq_carry_d)
        # straight through: rhogq is uploaded once at vertical-1 entry and drained once
        # at vertical-2 exit. Requires EVERY phase on its device path (the vert-adv
        # kernels + RESIDENT_TRACER_V for vertical-1/2, and the full resident hadv chain
        # + RESIDENT_TRACER_HADV for the horizontal phase) so no host rhogq reader sees
        # a stale array; the use_* getattr overrides are never set, so these env checks
        # match the runtime phase flags exactly. Gate PYNICAM_RESIDENT_TRACER_DRAIN1
        # (default on); per-phase drain fallback otherwise. Bit-identical: threading the
        # device handle replaces a to_numpy()+asarray() pure-copy round-trip.
        _drain1 = (
            (bk.type == "jax")
            and os.environ.get("PYNICAM_RESIDENT_TRACER_DRAIN1", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_VTRACERADV", "1") != "0"
            and os.environ.get("PYNICAM_RESIDENT_TRACER_V", "1") != "0"
            and os.environ.get("PYNICAM_RESIDENT_HADV", "1") != "0"
            and os.environ.get("PYNICAM_HADV_QA_RESIDENT", "1") != "0"
            and os.environ.get("PYNICAM_HADV_UPD_DEVICE", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_FLUX", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_REMAP", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_HLIMITER", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_OPRTGRADIENT", "1") != "0"
            and os.environ.get("PYNICAM_RESIDENT_TRACER_HADV", "1") != "0"
        )
        _rhogq_carry_d = None   # device rhogq carried across phases when _drain1

        # U5-C.4 (RES-CAPSTONE-26): when the full device tracer path is active, the host
        # flx_v drain @~216, rhog drain @~219, and host d compute @~447 are all DEAD
        # (poison job 2261585/2261921 proved it under HADVD+CKD+RHOG). Skip them. Computed
        # here (env-mirror) because the actual _resident_* flags are built later in phases
        # 2/3, after these sites; _drain1 already encodes jax + RESIDENT_TRACER_V + HADV +
        # the fuses, so we only add the HADVD/CKD/RHOG/VLIM conjuncts. CKD covers the
        # phase-3 flx_v ck/d, RHOG the rhog denom, HADVD the host d (-> device hlimiter).
        _hostfree = (_drain1
                     and bk.resident()
                     and bk.resident()
                     and bk.resident()
                     and os.environ.get("PYNICAM_RESIDENT_TRACER_VLIM", "1") != "0")
        # RC-41 (Track B unit 3): device pole vert-adv gate. Built here (env-mirror, like
        # _hostfree) so it's available before the TVF pole drains below. _drain1 already
        # encodes jax + RESIDENT_TRACER_V + the fuses; add VLIM + FUSE_VLIMITER + VPOLE.
        # RC-41r: under _vpole the phase-1 host pole limiter is skipped, so the TVF ck_pl/
        # d_pl drains (its only readers; phase-3 recomputes its own ck_pl) are removable.
        _vpole = (_drain1 and adm.ADM_have_pl
                  and os.environ.get("PYNICAM_RESIDENT_TRACER_VLIM", "1") != "0"
                  and os.environ.get("PYNICAM_FUSE_VLIMITER", "1") != "0"
                  and bk.resident())
        # TRACER-JIT Stage 1: skip the DEAD pole vert-adv host drains under _vpole. POISON-
        # CONFIRMED dead (env_check/tracer_pole_poison.sh): q_h_pl@394 (phase-1), rhogq_pl@531
        # (phase-1 bulk -> phase-2 reads the carried _rhogq_pl_d@780), q_pl@1228 + q_h_pl@1229
        # (phase-3 terminal). The device _qhp_d/_qp_d/_rhogq_pl_d feed every live consumer.
        # NOTE: q_pl@393 is KEPT (LIVE -- feeds the @1842 pole gradient; dies when 1842 ports).
        _vpole_nodrain = (_vpole
                  and bk.resident())
        # RES-TRACER-2: when the @1842 pole gradient runs on device (PYNICAM_RESIDENT_TRACER_
        # GRAD_PL), the phase-1 q_pl@393 drain's sole reader is gone -> q_pl@393 is dead too
        # (poison qpl1 PASS under GRAD_PL=1, job 2285881). Skip that drain under this flag.
        _grad_pl_on = (_vpole
                  and bk.resident())
        # U5-C.6 (RES-CAPSTONE-28): build rhogvx/vy/vz (= rho*_mean * VMTR_RGAM) on DEVICE
        # and thread into horizontal_flux (its asarray(rhovx) no-ops) -> the host compute
        # @~477 becomes unread (poison job 2262091 pinned rhogvx as the last live host
        # input). _drain1 already requires the fused flux kernel; just add the toggle.
        _resident_rhogv = (_drain1
                           and bk.resident())

        # ---- flx_v / ck / d / rhog via backend-switchable kernel ----
        # (replaces the in-line flx_v/ck/d computation + pole Python loops AND
        #  the later rhog k-loop; rhog depends only on flx_v/rhog_in/frhog, so
        #  computing it here is value-identical to the original ordering.)
        xp = bk.xp
        if getattr(self, "_tvf_kernel", None) is None:
            self._tvf_cfg = TracerVertFluxCfg(
                kmin=kmin, kmax=kmax, have_pl=adm.ADM_have_pl,
            )
            self._tvf_kernel = bk.maybe_jit(
                compute_tracer_vert_flux, static_argnames=("cfg", "xp"),
            )
        _tvf = bk.device_consts(self, "tracervertflux", lambda: {
            "C2WfactGz":    vmtr.VMTR_C2WfactGz,
            "RGAMH":        vmtr.VMTR_RGAMH,
            "RGSQRTH":      vmtr.VMTR_RGSQRTH,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "RGSQRTH_pl":   vmtr.VMTR_RGSQRTH_pl,
            "rdgz":         grd.GRD_rdgz,
        })
        _fv, _ck, _d, _rg, _fvp, _ckp, _dp, _rgp = self._tvf_kernel(
            (rhogvx_mean_d if rhogvx_mean_d is not None else xp.asarray(rhogvx_mean)),   # RES-CAPSTONE-35
            (rhogvy_mean_d if rhogvy_mean_d is not None else xp.asarray(rhogvy_mean)),
            (rhogvz_mean_d if rhogvz_mean_d is not None else xp.asarray(rhogvz_mean)),
            (rhogw_mean_d  if rhogw_mean_d  is not None else xp.asarray(rhogw_mean)),
            (rhogvx_mean_pl_d if rhogvx_mean_pl_d is not None else xp.asarray(rhogvx_mean_pl)),   # RC-81
            (rhogvy_mean_pl_d if rhogvy_mean_pl_d is not None else xp.asarray(rhogvy_mean_pl)),
            (rhogvz_mean_pl_d if rhogvz_mean_pl_d is not None else xp.asarray(rhogvz_mean_pl)),
            (rhogw_mean_pl_d  if rhogw_mean_pl_d  is not None else xp.asarray(rhogw_mean_pl)),
            (rhog_in_d if rhog_in_d is not None else xp.asarray(rhog_in)),
            (rhog_in_pl_d if rhog_in_pl_d is not None else xp.asarray(rhog_in_pl)),   # RC-82
            (frhog_d if frhog_d is not None else xp.asarray(frhog)),
            (frhog_pl_d if frhog_pl_d is not None else xp.asarray(frhog_pl)),   # RES-CAPSTONE-39
            _tvf["C2WfactGz"], _tvf["RGAMH"], _tvf["RGSQRTH"],
            _tvf["C2WfactGz_pl"], _tvf["RGAMH_pl"], _tvf["RGSQRTH_pl"],
            _tvf["rdgz"], dt, b1, cfg=self._tvf_cfg, xp=xp,
        )
        if not _hostfree: flx_v[:, :, :, :] = bk.to_numpy(_fv)   # U5-C.4: dead under CKD
        if not _hostfree: ck[:, :, :, :, :] = bk.to_numpy(_ck)   # U5-C.5: phase-1 vlim uses device _ck
        if not _hostfree: d[:, :, :, :]     = bk.to_numpy(_d)    # U5-C.5: phase-1 vlim uses device _d
        if not _hostfree: rhog[:, :, :, :]  = bk.to_numpy(_rg)   # U5-C.4: dead under RHOG+HADVD
        # U5-core-B (RES-CAPSTONE-22): capture the phase-1 device rhog handle BEFORE
        # the phase-1 iq-loop clobbers `_rg` (@~346 reuses it for the rhogq update).
        # Used (under RESIDENT_TRACER_RHOG) to thread device rhog through the horizontal
        # ch/cmask (@~513), the device rhog update (@~758), and the vert-adv-2 denom
        # (@~868), skipping their asarray(rhog) re-uploads. Just a name binding (no copy).
        _rhog_phase1_d = _rg
        # U5-C.2 POISON instrument (bisection): NaN-fill the host arrays that U5-C.2
        # intends to stop maintaining (flx_v @~216, rhog @~219, d @~447) AFTER their
        # device handles are captured (_fv, _rhog_phase1_d above -- unaffected). If gl07
        # still PASSES vs gold, no resident-path host reader remains -> that producer is
        # safe to remove (device handles feed the downstream kernels).
        if adm.ADM_have_pl:
            # RC-65: under _vpole the device pole vert-adv path (RC-41..43) feeds every
            # downstream consumer from the device handles (_flx_v_pl_d/_rhog_phase1_pl_d
            # below, used as `_x if _vpole`), so host flx_v_pl/ck_pl/d_pl/rhog_pl are ALL
            # unread -> skip their drains (poison-confirmed dead: tvfpl PASS job 2267422).
            # ck_pl/d_pl were already skipped under _vpole (RC-41r); flx_v_pl/rhog_pl join.
            if not _vpole:
                flx_v_pl[:, :, :]  = bk.to_numpy(_fvp)
                ck_pl[:, :, :, :]  = bk.to_numpy(_ckp)
                d_pl[:, :, :]      = bk.to_numpy(_dp)
                rhog_pl[:, :, :]   = bk.to_numpy(_rgp)
            # Unit 4c: device pole phase-1 rhog handle (pole analog of _rhog_phase1_d
            # @271) -> the horizontal ch_pl/cmask_pl/q_pl/d_pl denominators read it on
            # device instead of re-uploading asarray(rhog_pl). Just a name binding.
            _rhog_phase1_pl_d = _rgp
            # Track B POLE-POISON (RC-37 classify): NaN the TVF pole outputs after the drain;
            # PASS vs gold => host flx_v_pl/ck_pl/d_pl/rhog_pl unread (device _fvp.. threadable).




        # (the old per-l flx_v/ck/d Python loops and the pole loops are now in
        #  kernels/tracervertflux.py; see compute_tracer_vert_flux above)

        # backend-switchable per-tracer vertical advection (gated; jax-only).
        # denominator = rhog_in for this 1st fractional step.
        _vta_on, _vtak, _vtacfg, _vtad = self._vertadv_setup(grd)

        # RES-TP-1: keep rhogq + the loop-invariant denominator/flux device-resident
        # across the per-iq vertical advection loop, instead of re-uploading
        # asarray(rhogq[...,iq]) (x2) + asarray(rhog_in)/asarray(flx_v) every iq.
        # Reuse the tvf device flux (_fv); rhogq is held device and drained once
        # after the loop. The vertical limiter stays host (q_h round-trip -> TP-1b).
        # Bit-identical: device handle == asarray(host). Requires the vert-adv kernels
        # (_vta_on); asarray fallback otherwise. (1st step: denominator = rhog_in.)
        _resident_tracer_v = _vta_on and (bk.type == "jax") and \
            os.environ.get("PYNICAM_RESIDENT_TRACER_V", "1") != "0"
        # RES-TP-1b: keep q_h (and q) device-resident across qh -> vertical limiter
        # -> update, so q_h never round-trips to host (removes to_numpy(_q_h)/_q,
        # the limiter's asarray-in/to_numpy-out, and asarray(q_h) at the update).
        # Requires the fused vertical limiter (the host per-l limiter path needs host
        # q_h/q/d/ck). Gate PYNICAM_RESIDENT_TRACER_VLIM (default on under TRACER_V).
        _fuse_vlim_on = (bk.type == "jax") and \
            os.environ.get("PYNICAM_FUSE_VLIMITER", "1") != "0"
        _resident_vlim = _resident_tracer_v and _fuse_vlim_on and \
            os.environ.get("PYNICAM_RESIDENT_TRACER_VLIM", "1") != "0"
        if _resident_tracer_v:
            # RC-74: device rhogq input (caller's _PROGq_carry_d, nl-invariant device
            # PROGq == asarray(rhogq)); skips the per-step asarray(rhogq) H2D @here.
            _rhogq_d = rhogq_d if rhogq_d is not None else xp.asarray(rhogq)
            _flx_v_d = _fv
            _rhog_den_d = rhog_in_d if rhog_in_d is not None else xp.asarray(rhog_in)
            if _resident_vlim:
                # tvf already produced device ck/d (drained to host above); reuse the
                # device handles directly as the limiter denominator/coeff inputs.
                _ck_d = _ck
                _d_d  = _d
        # RC-41 (Track B unit 3): carry the POLE vertical advection on device through
        # qhp -> reshape-reuse limiter -> upp (the pole analog of RES-TP-1/1b). The TVF
        # device pole handles (_fvp/_ckp/_dp) feed it; _vpole computed early (above).
        # RC-41r: the host pole limiter is skipped under _vpole (skip_pole below), so the
        # host qhp pole drain is unread there too; rhogq_pl is device-determined.
        _rhogq_pl_d = None
        if _vpole:
            # RES-TRACER-3: device POLE rhogq input (== _PROGq_pl_carry_d) overrides the
            # host asarray(rhogq_pl) upload (the last tracer pole input host op). Bit-exact:
            # rhogq_pl_d == device(rhogq_pl). Pole analog of RC-74 rhogq_d.
            _rhogq_pl_d    = rhogq_pl_d if rhogq_pl_d is not None else xp.asarray(rhogq_pl)
            _flx_v_pl_d    = _fvp
            _rhog_den_pl_d = rhog_in_pl_d if rhog_in_pl_d is not None else xp.asarray(rhog_in_pl)   # RC-82
            _ck_pl_d = _ckp
            _d_pl_d  = _dp

        #--- vertical advection: 2nd-order centered difference
        for iq in range (vmax):

            if _vta_on:
                _q, _q_h = _vtak["qh"](
                    (_rhogq_d[:, :, :, :, iq] if _resident_tracer_v
                     else xp.asarray(rhogq[:, :, :, :, iq])),
                    (_rhog_den_d if _resident_tracer_v else xp.asarray(rhog_in)),
                    _vtad["afact"], _vtad["bfact"], cfg=_vtacfg, xp=xp,
                )
                if _resident_vlim:
                    # RES-TP-1b: hold q_h (and q) on device for the limiter + update.
                    _q_h_d = _q_h
                    _q_d   = _q
                else:
                    q[:, :, :, :] = bk.to_numpy(_q)
                    q_h[:, :, :, :] = bk.to_numpy(_q_h)
                if adm.ADM_have_pl:
                    _qp, _qhp = _vtak["qhp"](
                        (_rhogq_pl_d[:, :, :, iq] if _vpole else xp.asarray(rhogq_pl[:, :, :, iq])),
                        (_rhog_den_pl_d if _vpole else xp.asarray(rhog_in_pl)),
                        _vtad["afact"], _vtad["bfact"], cfg=_vtacfg, xp=xp,
                    )
                    if _vpole:
                        _qhp_d = _qhp; _qp_d = _qp   # RC-41: device pole q_h/q for the device limiter + upp
                    # writable copy: q_pl is slice-assigned later (horizontal adv),
                    # but bk.to_numpy returns a read-only (jax-derived) array.
                    if not _grad_pl_on:                       # qpl1 dead once @1842 grad is device
                        q_pl = np.array(bk.to_numpy(_qp))     # else: feeds the @1842 pole gradient
                    if not _vpole_nodrain:                    # qhpl1 dead (device _qhp_d feeds limiter/upp)
                        q_h_pl[:, :, :] = bk.to_numpy(_qhp)
            else:
                for l in range(lall):
                    for k in range(kall):
                        q[:, :, k, l] = rhogq[:, :, k, l, iq] / rhog_in[:, :, k, l]

                    for k in range(kmin, kmax + 2):  # +2 to include kmax+1
                        q_h[:, :, k, l] = (
                            grd.GRD_afact[k] * q[:, :, k, l] +
                            grd.GRD_bfact[k] * q[:, :, k - 1, l]
                        )

                    q_h[:, :, kmin - 1, l] = rdtype(0.0)

                # end loop l

                if adm.ADM_have_pl:
                    # Compute q_pl across all g and l at once
                    q_pl = rhogq_pl[:, :, :, iq] / rhog_in_pl

                    # Compute q_h_pl for k in [kmin, kmax+1]
                    for k in range(kmin, kmax + 2):
                        q_h_pl[:, k, :] = (
                            grd.GRD_afact[k] * q_pl[:, k, :] +
                            grd.GRD_bfact[k] * q_pl[:, k - 1, :]
                        )

                    # Boundary condition
                    q_h_pl[:, kmin - 1, :] = rdtype(0.0)
                #endif

            # with open(std.fname_log, 'a') as log_file: 
            #     print("q_h before vlimiter, 6531", iq, q_h[6,5,3,1],file=log_file)
            if apply_limiter_v[iq]:
                if _resident_vlim:
                    # RES-TP-1b: device q_h in, device q_h out (pole q_h_pl stays host).
                    _q_h_d = self.vertical_limiter_thuburn(
                        _q_h_d, q_h_pl,   # [INOUT]
                        _q_d  , q_pl  ,   # [IN]
                        _d_d  , d_pl  ,   # [IN]
                        _ck_d , ck_pl ,   # [IN]
                        cnst, rdtype, resident=True,
                        skip_pole=_vpole,   # RC-41r: pole limited on device below
                        )
                else:
                    self.vertical_limiter_thuburn(
                        q_h,   q_h_pl,    # [INOUT]
                        q  ,   q_pl  ,    # [IN]
                        d  ,   d_pl  ,    # [IN]
                        ck , ck_pl ,   # [IN]
                        cnst, rdtype,
                        )
                if _vpole:
                    # RC-41: device POLE limiter via reshape (g,k,l)->(g,1,k,l), reusing
                    # the per-column compute_vertical_limiter (set up by the resident call
                    # above). Bit-exact vs the host pole section: both use ck_pl[k,1] (the
                    # RC-40-fixed form -- no kmin override). Only kmin+1..kmax modified.
                    _qhp_d = self._vlim_kernel(
                        _qhp_d[:, None, :, :], _qp_d[:, None, :, :],
                        _d_pl_d[:, None, :, :], _ck_pl_d[:, None, :, :, :],
                        cfg=self._vlim_cfg, xp=xp)[:, 0, :, :]
            # with open(std.fname_log, 'a') as log_file:
            #     print("q_h after vlimiter, 6531", iq, q_h[6,5,3,1],file=log_file)

            # --- update rhogq

            if _vta_on:
                _rg = _vtak["up"](
                    (_rhogq_d[:, :, :, :, iq] if _resident_tracer_v
                     else xp.asarray(rhogq[:, :, :, :, iq])),
                    (_flx_v_d if _resident_tracer_v else xp.asarray(flx_v)),
                    (_q_h_d if _resident_vlim else xp.asarray(q_h)),
                    _vtad["rdgz"], cfg=_vtacfg, xp=xp,
                )
                if _resident_tracer_v:
                    _rhogq_d = _rhogq_d.at[:, :, :, :, iq].set(_rg)
                else:
                    rhogq[:, :, :, :, iq] = bk.to_numpy(_rg)
                if adm.ADM_have_pl:
                    _rgp = _vtak["upp"](
                        (_rhogq_pl_d[:, :, :, iq] if _vpole else xp.asarray(rhogq_pl[:, :, :, iq])),
                        (_flx_v_pl_d if _vpole else xp.asarray(flx_v_pl)),
                        (_qhp_d if _vpole else xp.asarray(q_h_pl)),
                        _vtad["rdgz"], cfg=_vtacfg, xp=xp,
                    )
                    if _vpole:
                        _rhogq_pl_d = _rhogq_pl_d.at[:, :, :, iq].set(_rgp)   # RC-41: carry device pole rhogq
                    else:
                        rhogq_pl[:, :, :, iq] = bk.to_numpy(_rgp)
            else:
                for l in range(lall):
                    # Zero out boundaries at kmin and kmax+1
                    q_h[:, :, kmin, l] = rdtype(0.0)
                    q_h[:, :, kmax + 1, l] = rdtype(0.0)

                    # Update rhogq with flux divergence
                    for k in range(kmin, kmax + 1):
                        rhogq[:, :, k, l, iq] -= (
                            flx_v[:, :, k + 1, l] * q_h[:, :, k + 1, l]
                            - flx_v[:, :, k,     l] * q_h[:, :, k,     l]
                        ) * grd.GRD_rdgz[k]

                    # Zero out boundaries at kmin-1 and kmax+1
                    rhogq[:, :, kmin - 1, l, iq] = rdtype(0.0)
                    rhogq[:, :, kmax + 1, l, iq] = rdtype(0.0)


                if adm.ADM_have_pl:
                    # Set q_h_pl boundaries
                    q_h_pl[:, kmin,  :] = rdtype(0.0)
                    q_h_pl[:, kmax+1, :] = rdtype(0.0)

                    for k in range(kmin, kmax + 1):
                        rhogq_pl[:, k, :, iq] -= (
                            flx_v_pl[:, k + 1, :] * q_h_pl[:, k + 1, :] -
                            flx_v_pl[:, k    , :] * q_h_pl[:, k    , :]
                        ) * grd.GRD_rdgz[k]

                    # Set rhogq_pl boundaries
                    rhogq_pl[:, kmin - 1, :, iq] = rdtype(0.0)
                    rhogq_pl[:, kmax + 1, :, iq] = rdtype(0.0)
                #endif

        # end loop iq

        # RES-TP-1: drain the device-resident rhogq back to host once (the horizontal
        # phase below reads host rhogq). Bit-identical to the per-iq to_numpy path.
        # RES-TP-3a: under _drain1, skip the drain and carry the device handle into the
        # horizontal phase (host rhogq stays stale but no phase reads it under _drain1).
        if _resident_tracer_v:
            if _drain1:
                _rhogq_carry_d = _rhogq_d
            else:
                rhogq[:, :, :, :, :] = bk.to_numpy(_rhogq_d)
        if _vpole:
            # RC-41: drain the device pole rhogq once (the horizontal phase reads host
            # rhogq_pl). The host per-iq upp write was skipped; this is the only writer.
            # TRACER-JIT: rqpl1 dead -- phase-2 hadv reads the carried _rhogq_pl_d@780 under
            # _resident_hadv_pl, not host rhogq_pl. Skip the drain.
            if not _vpole_nodrain:
                rhogq_pl[:, :, :, :] = bk.to_numpy(_rhogq_pl_d)

        #with open(std.fname_log, 'a') as log_file:
        #     print("STA1:rhogq[0,0,6,1,:]  ", rhogq[0, 0, 6, 1, :], file=log_file)    # 0, 0 is off at step 1 (after step 0))
        #     print("     rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)
        #     print("     rhogq[1,1,6,1,:]  ", rhogq[1, 1, 6, 1, :], file=log_file)
        #     print("     rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)
        #     print("     rhogq[1,1,5,1,:]  ", rhogq[1, 1, 5, 1, :], file=log_file)
        #     print("     rhogq[1,1,8,1,:]  ", rhogq[1, 1, 8, 1, :], file=log_file)

        #    print("STB1:rhogq [6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[0,10,0,:]  ", rhogq_pl[0, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[1,10,0,:]  ", rhogq_pl[1, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[2,10,0,:]  ", rhogq_pl[2, 10, 0, :], file=log_file)

        #    print("STC1:rhogq [6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)
        #    print("STD1:rhogq [6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)
        # if adm.ADM_have_pl:
        #     print("rhogq_pl.shape", rhogq_pl.shape)
        #     print(rhogq_pl[0,3,0,0])

        #--- update rhog : already computed above by compute_tracer_vert_flux
        # (rhog / rhog_pl depend only on flx_v, rhog_in and frhog, so the kernel
        #  produced them together with flx_v/ck/d; no separate k-loop needed.)

        prf.PROF_rapend('____vertical_adv',2)
        #---------------------------------------------------------------------------
        # Horizontal advection by MIURA scheme
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____horizontal_adv',2)


        #for l in range(lall):
        #    for k in range(kall):
        if not _hostfree:   # U5-C.4: host d dead under HADVD (device _d_hadv_d feeds the hlimiter)
            d[:, :, :, :] = b2 * frhog[:, :, :, :] / rhog[:, :, :, :] * dt

        #for l in range(lall):
        #    for k in range(kall):
        _rhogvx_d = _rhogvy_d = _rhogvz_d = None
        if not _resident_rhogv:
            rhogvx[:, :, :, :] = rhogvx_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]
            rhogvy[:, :, :, :] = rhogvy_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]
            rhogvz[:, :, :, :] = rhogvz_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]
        else:
            # U5-C.6: device rhogvx/vy/vz (VMTR_RGAM is loop-invariant geometry -> cached).
            _rgam_d = bk.device_consts(self, "tracer_rgam", lambda: {"r": vmtr.VMTR_RGAM})["r"]
            _rhogvx_d = (rhogvx_mean_d if rhogvx_mean_d is not None else bk.xp.asarray(rhogvx_mean)) * _rgam_d   # RES-CAPSTONE-35
            _rhogvy_d = (rhogvy_mean_d if rhogvy_mean_d is not None else bk.xp.asarray(rhogvy_mean)) * _rgam_d
            _rhogvz_d = (rhogvz_mean_d if rhogvz_mean_d is not None else bk.xp.asarray(rhogvz_mean)) * _rgam_d


        _rhogvx_pl_d = _rhogvy_pl_d = _rhogvz_pl_d = None
        if adm.ADM_have_pl:
            # (4c-6: host d_pl moved into the courant block below, gated -- it is dead
            # under the device courant, which builds _d_pl_d_hadv for the limiter.)
            if rhogvx_mean_pl_d is not None:
                # RC-81: device pole scaled flux (pole analog of _rhogvx_d @572; RGAM_pl is
                # loop-invariant geometry -> cached). Fed to horizontal_flux device pole args;
                # host rhogvx_pl left stale (the device path doesn't read it). Bit-exact.
                _rgam_pl_d = bk.device_consts(self, "tracer_rgam_pl", lambda: {"r": vmtr.VMTR_RGAM_pl})["r"]
                _rhogvx_pl_d = rhogvx_mean_pl_d * _rgam_pl_d
                _rhogvy_pl_d = rhogvy_mean_pl_d * _rgam_pl_d
                _rhogvz_pl_d = rhogvz_mean_pl_d * _rgam_pl_d
            else:
                rhogvx_pl[:, :, :] = rhogvx_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
                rhogvy_pl[:, :, :] = rhogvy_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
                rhogvz_pl[:, :, :] = rhogvz_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]

        # Stage-4a resident horizontal_adv: requires the flux/remap/limiter jax
        # kernels (device ch/cmask/grd_xc are fed to them), else no-op (numpy).
        _resident_hadv = (
            (bk.type == "jax")
            and os.environ.get("PYNICAM_RESIDENT_HADV", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_FLUX", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_REMAP", "1") != "0"
            and os.environ.get("PYNICAM_FUSE_HLIMITER", "1") != "0"
        )
        self._hadv_resident = _resident_hadv
        # Stage-4b: keep q_a on device remap->limiter (on-device Qout COMM); needs 4a.
        _resident_hadv_qa = _resident_hadv and os.environ.get("PYNICAM_HADV_QA_RESIDENT", "1") != "0"
        self._hadv_qa_resident = _resident_hadv_qa
        # Stage-4c: rhogq update on device (no q_a drain); needs 4b.
        _resident_hadv_upd = _resident_hadv_qa and os.environ.get("PYNICAM_HADV_UPD_DEVICE", "1") != "0"
        # RES-TP-2: device-resident q across the horizontal phase. Compute q on
        # device from the rhogq slice, run the gradient resident (no host q needed),
        # and feed device q to the remap + limiter kernels (their asarray(q) is a
        # no-op on a device array). Removes the per-iq host q=rhogq/rhog divide and
        # the 3 asarray(q) uploads (gradient + remap + limiter). Requires the fused
        # gradient (the host q consumer) on top of the resident hadv chain (4a gives
        # _rhog_d + the fused flux/remap/hlimiter kernels). Gate
        # PYNICAM_RESIDENT_TRACER_HADV (default on); host fallback otherwise.
        _resident_hadv_q = (
            _resident_hadv
            and os.environ.get("PYNICAM_FUSE_OPRTGRADIENT", "1") != "0"
            and os.environ.get("PYNICAM_RESIDENT_TRACER_HADV", "1") != "0"
        )
        # RES-TP-2b: keep the resident gradq on device through its halo exchange via
        # the on-device COMM (auto-routed when a jax array is passed), instead of
        # draining it + re-uploading the ~3-component field in the remap kernel.
        # Needs the device gradq (resident q). Gate PYNICAM_RESIDENT_TRACER_HADV_COMM.
        _resident_hadv_qcomm = _resident_hadv_q and \
            os.environ.get("PYNICAM_RESIDENT_TRACER_HADV_COMM", "1") != "0"
        # Unit 4c-1: device POLE horizontal courant (ch_pl/cmask_pl/d_pl) + per-iq
        # q_pl, built from the phase-1 device pole rhog/rhogq (_rhog_phase1_pl_d,
        # _rhogq_pl_d) + the device pole flux (self._flx_h_pl_d), threaded into the 4a
        # remap + 4b limiter kernels so their asarray(q_pl/cmask_pl/ch_pl/d_pl/grd_xc_pl)
        # uploads no-op. Needs the resident vert-adv pole path (_vpole, source of the
        # device pole rhog/rhogq) + the kernels. Gate PYNICAM_RESIDENT_HADV_PL (default
        # OFF); host ch_pl/cmask_pl/q_pl/d_pl stay valid (still computed) for now.
        _resident_hadv_pl = (_resident_hadv and _vpole and adm.ADM_have_pl
                             and bk.resident())
        # Unit 4c-2: device POLE flux apply (rhogq/rhog centre updates) -> carry the
        # device pole rhogq/rhog into phase-3 (skip its asarray re-uploads). Separate
        # gate because the apply REORDERS the 5-neighbour sum (host subtracts the terms
        # sequentially; device sums-then-subtracts) -> machine-eps, not bit-exact (like
        # unit B). Requires the device courant (4c-1). Gate PYNICAM_RESIDENT_HADV_APPLY_PL.
        _resident_hadv_apply_pl = (_resident_hadv_pl
                                   and bk.resident())
        # Unit 4c-3b: the device pole q_a (remap/limiter output, stashed on self by
        # those methods) is used directly by the flux apply -> no asarray(q_a_pl).
        # Mirrors the limiter's _qa_resident_pl gate. Gate PYNICAM_RESIDENT_HADV_QA_PL.
        _resident_hadv_qa_pl = (_resident_hadv_apply_pl
                                and getattr(self, "_hadv_qa_resident", False)
                                and bk.resident()
                                and bk.resident()
                                and bk.resident())
        # 4c-6: single consistent flag threaded into horizontal_flux/remap/limiter so
        # their now-dead pole drains (flx_h_pl/grd_xc_pl/Qin_pl/Qout_pl/q_a_pl) are
        # skipped under EXACTLY the same condition the device consumers use -- no
        # half-on gate combo can read a stale host array.
        self._hadv_qa_pl_active = _resident_hadv_qa_pl

        # U5-core-B (RES-CAPSTONE-22): thread device rhog (phase-1 handle + an on-device
        # update) through the horizontal ch/cmask (@~513), the device rhog update
        # (@~758), and the vert-adv-2 denominator (@~868), instead of re-uploading
        # asarray(rhog) at each. Needs the device flx_h (_resident_hadv) for the device
        # rhog update + the resident vert path + the rhogq carry. Host rhog is kept fully
        # valid (the host update @~758 still runs) -- this only swaps the DEVICE consumers
        # onto a device handle, so any host rhog reader (e.g. d @~441) is unaffected.
        # Default OFF. Bit-identical to machine-eps (device f64 == host f64).
        _resident_rhog = (_resident_hadv and _resident_tracer_v and _drain1
                          and bk.resident())
        _rhog_carry_d = None   # device-updated rhog (built at the @~758 rhog update)
        # U5-C (RES-CAPSTONE-23): hoisted here (same value as the phase-3 def) so the
        # @~758 HOST rhog update can be skipped when BOTH device paths cover phase 3.
        _resident_ckd = (_resident_tracer_v and _resident_vlim and _drain1
                         and bk.resident())
        # U5-C.3 (RES-CAPSTONE-25): build the horizontal-phase d (= b2*frhog/rhog*dt) on
        # DEVICE from _rhog_phase1_d + device frhog and feed it to the fused hlimiter
        # (its asarray(d) @~2330 then no-ops, exactly like ch/cmask/q). Makes the host d
        # @~447 UNREAD -> removable (poison job 2261585 pinned host d as the last reader).
        _resident_hadvd = (_resident_hadv and _resident_rhog
                           and bk.resident())
        _d_hadv_d = None

        self.horizontal_flux(
            flx_h, flx_h_pl,            # [OUT]
            grd_xc, grd_xc_pl,          # [OUT]   grd_xc for AIJ and AJ broken?
            rhog_mean, rhog_mean_pl,    # [IN]
            rhogvx, rhogvx_pl,          # [IN]
            rhogvy, rhogvy_pl,          # [IN]
            rhogvz, rhogvz_pl,          # [IN]
            dt,                         # [IN]
            cnst, grd, gmtr, rdtype,
            rhovx_d=_rhogvx_d, rhovy_d=_rhogvy_d, rhovz_d=_rhogvz_d,   # U5-C.6 device rho*v
            rho_d=rhog_mean_d,   # RES-CAPSTONE-35: device rho (= PROG_mean[I_RHOG]); flux asarray(rho) no-ops
            # RC-81: device POLE rho/rho*v (skips asarray(rho_pl/rhovx_pl..) @horizontal_flux:1534)
            rho_pl_d=rhog_mean_pl_d, rhovx_pl_d=_rhogvx_pl_d, rhovy_pl_d=_rhogvy_pl_d, rhovz_pl_d=_rhogvz_pl_d,
        )


        #--- Courant number             
        # for l in range(lall):
        #     for k in range(kall):
        if _resident_hadv:
            # Stage-4a: ch/cmask on device from the device flx_h; grd_xc kept on
            # device. remap/limiter then consume ch/cmask/grd_xc with no host
            # upload. IEEE float64 divide is correctly rounded -> ch (hence the
            # hard cmask step) is bit-identical to the numpy path.
            xp = bk.xp
            # U5-core-B: thread the captured phase-1 device rhog instead of re-uploading.
            _rhog_d = _rhog_phase1_d if _resident_rhog else xp.asarray(rhog)
            ch = self._flx_h_d / _rhog_d[:, :, :, :, None]
            cmask = rdtype(0.5) - xp.copysign(rdtype(0.5), ch - EPS)
            grd_xc = self._grd_xc_d
            if _resident_hadvd:
                # U5-C.3: device hadv d for the fused hlimiter (phase-1 rhog == host
                # rhog @~447; device f64 == host f64). Threaded at the limiter call.
                _d_hadv_d = b2 * (frhog_d if frhog_d is not None else xp.asarray(frhog)) / _rhog_phase1_d * dt
        else:
            ch[:, :, :, :, :] = flx_h[:, :, :, :, :] / rhog[:, :, :, :, None]
            cmask[:, :, :, :, :] = rdtype(0.5) - np.copysign(rdtype(0.5), ch[:, :, :, :, :] - EPS)
                #cmask[:, :, k, l, :] = rdtype(0.5) - np.sign(rdtype(0.5) - ch[:, :, k, l, :] + EPS)


        if adm.ADM_have_pl and not _resident_hadv_pl:   # 4c-5: host pole ch/cmask dead (device courant feeds the kernels)
            g = adm.ADM_gslf_pl  # scalar index

            ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                flx_h_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] / rhog_pl[g, :, :]
            )

            # cmask_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
            #     rdtype(0.5) - np.sign(rdtype(0.5) - ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] + EPS)
            # )
            cmask_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                rdtype(0.5) - np.copysign(rdtype(0.5), ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] - EPS)
            )

        # Unit 4c-1: device pole courant (ch/cmask) + d, from the device pole flux +
        # phase-1 device pole rhog. ch_pl denom is the CENTRE rhog (matches host @~675);
        # d_pl is per-g (matches host @~565). Bit-identical to the host arithmetic.
        _ch_pl_d = _cmask_pl_d = _d_pl_d_hadv = None
        if _resident_hadv_pl:
            _xpp = bk.xp
            _g_pl = adm.ADM_gslf_pl
            _rhog_ctr_pl_d = _rhog_phase1_pl_d[_g_pl]                 # (kall, lall_pl)
            _ch_pl_d = self._flx_h_pl_d / _rhog_ctr_pl_d[None]       # (gall_pl,k,l)
            _cmask_pl_d = rdtype(0.5) - _xpp.copysign(rdtype(0.5), _ch_pl_d - EPS)
            _frhog_pl_dev = frhog_pl_d if frhog_pl_d is not None else _xpp.asarray(frhog_pl)
            _d_pl_d_hadv = b2 * _frhog_pl_dev / _rhog_phase1_pl_d * dt
        elif adm.ADM_have_pl:
            d_pl[:, :, :] = b2 * frhog_pl[:, :, :] / rhog_pl[:, :, :] * dt   # 4c-6: host d_pl (device-off path)

        for iq in range (vmax):
            _q_pl_d = None

            # for l in range(lall):
            #     for k in range(kall):
            if _resident_hadv_q:
                # RES-TP-2: device q from the device rhogq slice (the slice is reused
                # by the on-device update below). IEEE f64 divide is correctly rounded
                # -> bit-identical to the host q=rhogq/rhog.
                # RES-TP-3a: under _drain1 the slice comes from the carried device rhogq
                # (vertical-1 output), not a host re-upload.
                _rhogq_iq_d = (_rhogq_carry_d[:, :, :, :, iq] if _drain1
                               else xp.asarray(rhogq[:, :, :, :, iq]))
                _q_d = _rhogq_iq_d / _rhog_d
            else:
                q[:, :, :, :] = rhogq[:, :, :, :, iq] / rhog[:, :, :, :]

            if adm.ADM_have_pl:
                if _resident_hadv_pl:
                    # device pole q = device rhogq slice / phase-1 device pole rhog
                    # (4c-5: host q_pl dead -- the kernels read this device q)
                    _q_pl_d = _rhogq_pl_d[:, :, :, iq] / _rhog_phase1_pl_d
                else:
                    q_pl[:, :, :] = rhogq_pl[:, :, :, iq] / rhog_pl[:, :, :]

            #with open(std.fname_log, 'a') as log_file:
                #print("STC1.3:q_a[6,5,3,1,:]  ", q_a[6, 5, 3, 1, :], file=log_file)

                #print(f"STE1.2:rhogq[16,:,24,1,iq={iq}]", q_a[16,:,24,1,iq]  , file=log_file)
                #print("STE1.2:rhog[16,:,24,1]", rhog[16,:,24,1]  , file=log_file)
                #print("STE1.2:q[16,:,24,1]", q[16,:,24,1]  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,0]", cmask[16,:,24,1,0]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,0] + EPS", ch[16,:,24,1,0]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,1]", cmask[16,:,24,1,1]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,1] + EPS", ch[16,:,24,1,1]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,2]", cmask[16,:,24,1,2]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,2] + EPS", ch[16,:,24,1,2]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,3]", cmask[16,:,24,1,3]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,3] + EPS", ch[16,:,24,1,3]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,4]", cmask[16,:,24,1,4]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,4] + EPS", ch[16,:,24,1,4]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,5]", cmask[16,:,24,1,5]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,5] + EPS", ch[16,:,24,1,5]+EPS  , file=log_file)

                # print("STE1.2:grd_xc[16,:,24,1,0,0]", grd_xc[16,:,24,1,0,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,0,1]", grd_xc[16,:,24,1,0,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,0,2]", grd_xc[16,:,24,1,0,2]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,0]", grd_xc[16,:,24,1,1,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,1]", grd_xc[16,:,24,1,1,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,2]", grd_xc[16,:,24,1,1,2]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,0]", grd_xc[16,:,24,1,2,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,1]", grd_xc[16,:,24,1,2,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,2]", grd_xc[16,:,24,1,2,2]  , file=log_file)

            # calculate q at cell face, upwind side
            self.horizontal_remap(
                q_a, q_a_pl,            # [OUT]
                (_q_d if _resident_hadv_q else q),   q_pl,    # [IN]
                cmask, cmask_pl,        # [IN]
                grd_xc, grd_xc_pl,      # [IN]
                cnst, comm, grd, oprt, rdtype,
                resident_q=_resident_hadv_q,
                resident_comm=_resident_hadv_qcomm,
                # Unit 4c-1: device pole inputs (None -> asarray fallback in the kernel)
                q_pl_d=(_q_pl_d if _resident_hadv_pl else None),
                cmask_pl_d=(_cmask_pl_d if _resident_hadv_pl else None),
                grd_xc_pl_d=(self._grd_xc_pl_d if _resident_hadv_pl else None),
                qa_resident_pl=_resident_hadv_qa_pl,   # 4c-6: skip the dead q_a_pl drain
            )

            #with open(std.fname_log, 'a') as log_file:
                #print("STC1.3:q_a[6,5,3,1,:]  ", q_a[6, 5, 3, 1, :], file=log_file)
            #    print("STE1.3:q_a[16,:,24,1,1]", q_a[16,:,24,1,1]  , file=log_file)
                #print("STD1.3:q_a[6,5,2,1,:]  ", q_a[6, 5, 2, 1, :], file=log_file)
            #     print("STA1.3 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,6,1,:]  ",   q_a[1, 1, 6, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 6, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.

            # if adm.ADM_have_pl:
            #     print("q_a_pl")
            #     print(q_a_pl[:,3,0])

            # apply flux limiter
            if apply_limiter_h[iq]:
                self.horizontal_limiter_thuburn(
                    q_a, q_a_pl,            # [INOUT]    #  1 1 6 1 and 1 1 7 1 in SP get undefs out of here
                    (_q_d if _resident_hadv_q else q),   q_pl,    # [IN]  (device q: asarray(q) in the fused kernel is a no-op)
                    (_d_hadv_d if _resident_hadvd else d),   d_pl,              # [IN]  (U5-C.3: device d -> hlimiter asarray(d) no-op)
                    ch,  ch_pl,             # [IN]
                    cmask, cmask_pl,        # [IN]
                    cnst, comm, rdtype,
                    # Unit 4c-1: device pole inputs (None -> asarray fallback in the kernel)
                    q_pl_d=(_q_pl_d if _resident_hadv_pl else None),
                    d_pl_d=(_d_pl_d_hadv if _resident_hadv_pl else None),
                    ch_pl_d=(_ch_pl_d if _resident_hadv_pl else None),
                    cmask_pl_d=(_cmask_pl_d if _resident_hadv_pl else None),
                    qa_resident_pl=_resident_hadv_qa_pl,   # 4c-6: skip the dead Qin/Qout/q_a drains
                )
            # endif

            # with open(std.fname_log, 'a') as log_file:
            # #     print("STA1.4 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0, 1, 2 are undef
            # #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            # #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 4 is undef
            # #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.
            #     print("STA1.4 :  q_a[1,1,6,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     #print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.



            #--- update rhogq        

            # for l in range(lall):
            #     for k in range(kall):
            # rhogq[:, :, :, :, iq] -= (
            #     flx_h[:, :, :, :, 0] * q_a[:, :, :, :, 0] +
            #     flx_h[:, :, :, :, 1] * q_a[:, :, :, :, 1] +
            #     flx_h[:, :, :, :, 2] * q_a[:, :, :, :, 2] +
            #     flx_h[:, :, :, :, 3] * q_a[:, :, :, :, 3] +
            #     flx_h[:, :, :, :, 4] * q_a[:, :, :, :, 4] +
            #     flx_h[:, :, :, :, 5] * q_a[:, :, :, :, 5]
            # )

            # Prepare slices for i=2:iall-1, j=2:jall-1
            isl = slice(1, iall-1)
            jsl = slice(1, jall-1)

            if _resident_hadv_upd:
                # Stage-4c: rhogq update on device, reading the resident q_a/flx_h
                # (no q_a drain). Only the rhogq iq-slice round-trips. The 6-term
                # product-sum is FMA-fusable -> machine-precision (not bit-exact).
                _xpL = bk.xp
                _qad = self._q_a_d; _fhd = self._flx_h_d
                _upd = (_fhd[isl, jsl, :, :, 0] * _qad[isl, jsl, :, :, 0]
                      + _fhd[isl, jsl, :, :, 1] * _qad[isl, jsl, :, :, 1]
                      + _fhd[isl, jsl, :, :, 2] * _qad[isl, jsl, :, :, 2]
                      + _fhd[isl, jsl, :, :, 3] * _qad[isl, jsl, :, :, 3]
                      + _fhd[isl, jsl, :, :, 4] * _qad[isl, jsl, :, :, 4]
                      + _fhd[isl, jsl, :, :, 5] * _qad[isl, jsl, :, :, 5])
                _new_iq = (_rhogq_iq_d[isl, jsl, :, :] if _resident_hadv_q
                           else _xpL.asarray(rhogq[isl, jsl, :, :, iq])) - _upd
                if _drain1:
                    # RES-TP-3a: update the carried device rhogq in place (interior
                    # only; boundaries retain the vertical-1 output, same as host).
                    _rhogq_carry_d = _rhogq_carry_d.at[isl, jsl, :, :, iq].set(_new_iq)
                else:
                    rhogq[isl, jsl, :, :, iq] = bk.to_numpy(_new_iq)
            else:
                if _resident_hadv_qa:
                    # Stage-4b: single drain of the resident q_a for the numpy update.
                    q_a[:, :, :, :, :] = bk.to_numpy(self._q_a_d)

                # Fully vectorized calculation
                rhogq[isl, jsl, :, :, iq] -= (
                    flx_h[isl, jsl, :, :, 0] * q_a[isl, jsl, :, :, 0] +
                    flx_h[isl, jsl, :, :, 1] * q_a[isl, jsl, :, :, 1] +
                    flx_h[isl, jsl, :, :, 2] * q_a[isl, jsl, :, :, 2] +
                    flx_h[isl, jsl, :, :, 3] * q_a[isl, jsl, :, :, 3] +
                    flx_h[isl, jsl, :, :, 4] * q_a[isl, jsl, :, :, 4] +
                    flx_h[isl, jsl, :, :, 5] * q_a[isl, jsl, :, :, 5]
                )



            #with open(std.fname_log, 'a') as log_file:
            #     print(f"iq=  {iq} ",file=log_file)
            #     print("STA1.5 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  e+23
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  e+23
            #     print("        flx_h[0,0,7,1,:]  ", flx_h[0, 0, 7, 1, :], file=log_file)  
            #     print("          q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0, 1, 2 are undef
            #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)
            #     print("        flx_h[1,1,7,1,:]  ", flx_h[1, 1, 7, 1, :], file=log_file)  
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 4 is undef
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)

            #    print("STB1.5 :rhogq[6,5,10,0,:] ", rhogq[6, 5, 10, 0, :], file=log_file)  #you  e+23
            #     print("        flx_h[6,5,10,0,:] ", flx_h[6, 5, 10, 0, :], file=log_file)  
            #     print("          q_a[6,5,10,0,:] ",   q_a[6, 5, 10, 0, :], file=log_file)  # 0, 1, 2 are undef
            #     print("            q[6,5,10,0]   ",   q  [6, 5, 10, 0]   , file=log_file)
            #     print("          q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[0,0,7,1]    ",     q[0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",     q[1, 1, 7, 1]   , file=log_file)  # 0.
                # print("STC1.5 :rhogq[6,5,3,1,:] ", rhogq[6, 5, 3, 1, :], file=log_file) 
                # print("STD1.5 :rhogq[6,5,2,1,:] ", rhogq[6, 5, 2, 1, :], file=log_file) 
                # print("STE1.5 :rhogq[16,:,24,1,1]", rhogq[16,:,24,1,1] , file=log_file)

            if adm.ADM_have_pl:
                g = adm.ADM_gslf_pl

                if not _resident_hadv_apply_pl:   # 4c-4: host pole rhogq apply dead under the device apply
                    for l in range(lall_pl):
                        for k in range(kall):
                            for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   # 1 to 5  range(1,6)
                                rhogq_pl[g, k, l, iq] -= flx_h_pl[v, k, l] * q_a_pl[v, k, l]

                if _resident_hadv_apply_pl:
                    # Unit 4c-2: device pole rhogq flux apply (centre g) -- mirror the
                    # host loop on device, updating the carried device pole rhogq so
                    # phase-3 reads it without re-uploading asarray(rhogq_pl). Keeps host
                    # rhogq_pl valid. Bit-exact (_flx_h_pl_d==flx_h_pl, q_a_pl host).
                    _xpa = bk.xp
                    _vsl = slice(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1)
                    # 4c-3b: use the device q_a (limiter output if it ran, else remap
                    # output) instead of asarray(q_a_pl). Same values (host q_a_pl is the
                    # drain of these device handles), so still bit-exact.
                    if _resident_hadv_qa_pl:
                        _qa_pl_dev = (self._qa_pl_lim_d if apply_limiter_h[iq]
                                      else self._qa_pl_remap_d)
                    else:
                        _qa_pl_dev = _xpa.asarray(q_a_pl)
                    _fhsum_q = _xpa.sum(self._flx_h_pl_d[_vsl, :, :] * _qa_pl_dev[_vsl, :, :], axis=0)
                    _rhogq_pl_d = _rhogq_pl_d.at[g, :, :, iq].add(-_fhsum_q)



            # with open(std.fname_log, 'a') as log_file:
            # #     print("STA2:rhogq[0,0,6,1,:]  ", rhogq[0, 0, 6, 1, :], file=log_file)
            # #     print("     rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  e+23
            # #     print("     rhogq[1,1,6,1,:]  ", rhogq[1, 1, 6, 1, :], file=log_file)
            # #     print("     rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  e+23
            # #     print("     rhogq[1,1,5,1,:]  ", rhogq[1, 1, 5, 1, :], file=log_file)
            # #     print("     rhogq[1,1,8,1,:]  ", rhogq[1, 1, 8, 1, :], file=log_file)
            #     print("STA2.0 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 6, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.


            # if adm.ADM_have_pl:
            #     print("rhogq_pl.shape", rhogq_pl.shape)
            #     print(rhogq_pl[0,3,0,0])
            #     print("flx_h_pl")
            #     print(flx_h_pl[:,3,0])
            #     print("q_a_pl")
            #     print(q_a_pl[:,3,0])
            #endif

        #end iq LOOP

        # with open(std.fname_log, 'a') as log_file:
        #     print("STA2.1 : rhog[0,0,7,1]  ",  rhog[0, 0, 7, 1], file=log_file)  
        #     print("         rhog[1,1,7,1]  ",  rhog[1, 1, 7, 1], file=log_file)  


        #--- update rhog

        isl = slice(1, iall-1)
        jsl = slice(1, jall-1)

        # U5-C (RES-CAPSTONE-23): skip the HOST rhog mass update when BOTH device paths
        # cover phase 3 -- RHOG provides the device-updated rhog denom (@~868) and CKD
        # gates off the phase-3 host d @~856 (the only other post-update host rhog
        # reader). The pre-update host rhog reader (d @~447) is unaffected. Removes the
        # genuine host mass-update arithmetic. Keep host when either path is off.
        if not (_resident_rhog and _resident_ckd):
            rhog[isl, jsl, :, :] -= (
                flx_h[isl, jsl, :, :, 0] + flx_h[isl, jsl, :, :, 1] +
                flx_h[isl, jsl, :, :, 2] + flx_h[isl, jsl, :, :, 3] +
                flx_h[isl, jsl, :, :, 4] + flx_h[isl, jsl, :, :, 5]
            ) - (b2 * frhog[isl, jsl, :, :] * dt)

        if _resident_rhog:
            # U5-core-B: device rhog update mirroring the host update above -- from the
            # device flx_h (self._flx_h_d) + device frhog + the phase-1 device rhog.
            # Interior [isl,jsl] only; the halo stays at the phase-1 value (as on host).
            _fhd = self._flx_h_d
            _fhsum = (_fhd[isl, jsl, :, :, 0] + _fhd[isl, jsl, :, :, 1] +
                      _fhd[isl, jsl, :, :, 2] + _fhd[isl, jsl, :, :, 3] +
                      _fhd[isl, jsl, :, :, 4] + _fhd[isl, jsl, :, :, 5])
            _frhog_d = frhog_d if frhog_d is not None else xp.asarray(frhog)
            _rhog_carry_d = _rhog_phase1_d.at[isl, jsl, :, :].add(
                -_fhsum + b2 * _frhog_d[isl, jsl, :, :] * dt)


        # for l in range(lall):
        #     for k in range(kall):
        #         rhog[:, :, k, l] -= (
        #             flx_h[:, :, k, l, 0] +
        #             flx_h[:, :, k, l, 1] +
        #             flx_h[:, :, k, l, 2] +
        #             flx_h[:, :, k, l, 3] +
        #             flx_h[:, :, k, l, 4] +
        #             flx_h[:, :, k, l, 5]
        #         )
        #         rhog[:, :, k, l] += b2 * frhog[:, :, k, l] * dt

        # with open(std.fname_log, 'a') as log_file:
        #     print("STA2.2 : rhog[0,0,7,1]  ",  rhog[0, 0, 7, 1], file=log_file)  
        #     print("         rhog[1,1,7,1]  ",  rhog[1, 1, 7, 1], file=log_file)  
        #     print("        frhog[0,0,7,1]  ", frhog[0, 0, 7, 1], file=log_file)  
        #     print("        frhog[1,1,7,1]  ", frhog[1, 1, 7, 1], file=log_file)  
        #     print("        rhogq[0,0,7,1]  ", rhogq[0, 0, 7, 1], file=log_file)  
        #     print("        rhogq[1,1,7,1]  ", rhogq[1, 1, 7, 1], file=log_file) 

        if adm.ADM_have_pl and not _resident_hadv_apply_pl:   # 4c-4: host pole rhog apply dead under the device apply
            g = adm.ADM_gslf_pl  # Constant index for pole surface

            for l in range(lall_pl):
                for k in range(kall):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        rhog_pl[g, k, l] -= flx_h_pl[v, k, l]

                    rhog_pl[g, k, l] += b2 * frhog_pl[g, k, l] * dt

        # Unit 4c-2: device pole rhog flux apply (centre g) -- mirror the host update
        # on device into _rhog_carry_pl_d so phase-3 reads it without re-uploading
        # asarray(rhog_pl). Keeps host rhog_pl valid. Bit-exact.
        _rhog_carry_pl_d = None
        if _resident_hadv_apply_pl:
            _xpa = bk.xp
            _g_pl = adm.ADM_gslf_pl
            _vsl = slice(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1)
            _fhsum_r = _xpa.sum(self._flx_h_pl_d[_vsl, :, :], axis=0)               # (k,l)
            _frhog_pl_dev = frhog_pl_d if frhog_pl_d is not None else _xpa.asarray(frhog_pl)
            _rhog_carry_pl_d = _rhog_phase1_pl_d.at[_g_pl, :, :].add(
                -_fhsum_r + b2 * _frhog_pl_dev[_g_pl, :, :] * dt)

        prf.PROF_rapend('____horizontal_adv',2)
        #---------------------------------------------------------------------------
        # Vertical Advection (fractioanl step) : 2nd
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vertical_adv',2)

        # for l in range(lall):
        #     d[:, :, :, l] = b3 * frhog[:, :, :, l] / rhog[:, :, :, l] * dt

        #     for k in range(kmin, kmax + 1):
        #         ck[:, :, k, l, 0] = -flx_v[:, :, k,   l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]
        #         ck[:, :, k, l, 1] =  flx_v[:, :, k+1, l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]

        #     ck[:, :, kmin - 1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmin - 1, l, 1] = rdtype(0.0)
        #     ck[:, :, kmax + 1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmax + 1, l, 1] = rdtype(0.0)

        # U5-core-A (RES-CAPSTONE-21): the vertical-adv-2 ck/d courant/coeff state was
        # recomputed on HOST here (from host flx_v drained @~216 + host rhog) then
        # re-uploaded via asarray(ck)/asarray(d) @~860. Under RESIDENT_TRACER_CKD,
        # compute ck/d ON DEVICE in the _resident_vlim block below directly from the
        # phase-1 device flx_v handle (_fv, still in scope) + the device rhog
        # (_rhog_den_d), skipping this host recompute AND the re-upload. Requires the
        # resident vert-adv-2 + vlim device path + the rhogq carry (_drain1). Default OFF.
        # Bit-identical to machine-eps (device f64 == host f64 on the kernel-read rows).
        # _resident_ckd is hoisted to the phase-2 gate block (so the @~758 host rhog
        # update can also key off it); same value, in scope here.
        if not _resident_ckd:
            d[:, :, :, :] = b3 * frhog[:, :, :, :] / rhog[:, :, :, :] * dt

            # Prepare k slice
            k_slice = slice(kmin, kmax + 1)

            # Main ck calculation, fully vectorized over (i, j, k, l)
            ck[:, :, k_slice, :, 0] = -flx_v[:, :, kmin:kmax+1, :] / rhog[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1, np.newaxis]
            ck[:, :, k_slice, :, 1] =  flx_v[:, :, kmin+1:kmax+2, :] / rhog[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1, np.newaxis]

            # Boundary conditions for kmin-1 and kmax+1
            ck[:, :, kmin-1, :, 0] = 0.0
            ck[:, :, kmin-1, :, 1] = 0.0
            ck[:, :, kmax+1, :, 0] = 0.0
            ck[:, :, kmax+1, :, 1] = 0.0


        if adm.ADM_have_pl and not _vpole:   # RC-43 (3b): pole ck/d computed on device below
            d_pl = b3 * frhog_pl / rhog_pl * dt  # fully vectorized over g, k, l

            for k in range(kmin, kmax + 1):
                ck_pl[:, k, :, 0] = -flx_v_pl[:, k,   :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]
                ck_pl[:, k, :, 1] =  flx_v_pl[:, k+1, :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]

            ck_pl[:, kmin - 1, :, 0] = rdtype(0.0)
            ck_pl[:, kmin - 1, :, 1] = rdtype(0.0)
            ck_pl[:, kmax + 1, :, 0] = rdtype(0.0)
            ck_pl[:, kmax + 1, :, 1] = rdtype(0.0)


        # backend-switchable per-tracer vertical advection (gated; jax-only).
        # denominator = rhog (updated by step 1) for this 2nd fractional step.
        _vta_on, _vtak, _vtacfg, _vtad = self._vertadv_setup(grd)

        # RES-TP-1 (2nd fractional step): re-init the device-resident rhogq (post
        # horizontal phase, fresh from host) + the loop-invariant denominator (rhog,
        # updated by step 1) and flux (flx_v reused from step 1). Drained once after
        # the loop. Bit-identical: device handle == asarray(host).
        if _resident_tracer_v:
            # RES-TP-3a: under _drain1, continue from the carried device rhogq (the
            # horizontal-phase output) instead of re-uploading from host.
            _rhogq_d = _rhogq_carry_d if _drain1 else xp.asarray(rhogq)
            # U5-core-A: thread the phase-1 device flx_v (_fv) instead of re-uploading.
            _flx_v_d = _fv if _resident_ckd else xp.asarray(flx_v)
            # U5-core-B: use the device-updated rhog (from the @~758 device update)
            # instead of re-uploading the host-updated rhog.
            _rhog_den_d = _rhog_carry_d if _resident_rhog else xp.asarray(rhog)
            if _resident_vlim:
                if _resident_ckd:
                    # U5-core-A: compute step-2 ck/d ON DEVICE from device flx_v (_fv) +
                    # device rhog (_rhog_den_d), instead of the host recompute @~814 +
                    # asarray re-upload. _ck_d zeros outside [kmin,kmax]; the kmin-1/
                    # kmax+1 boundary rows the kernel reads are 0 (matches host @~824-827).
                    _frhog_d = frhog_d if frhog_d is not None else xp.asarray(frhog)
                    _rdgz_d = bk.device_consts(self, "tracer_v2_rdgz",
                                               lambda: {"r": grd.GRD_rdgz})["r"]
                    _d_d = b3 * _frhog_d / _rhog_den_d * dt
                    _rg2 = _rdgz_d[kmin:kmax+1][None, None, :, None]
                    _ck0 = -_flx_v_d[:, :, kmin:kmax+1, :] / _rhog_den_d[:, :, kmin:kmax+1, :] * _rg2
                    _ck1 =  _flx_v_d[:, :, kmin+1:kmax+2, :] / _rhog_den_d[:, :, kmin:kmax+1, :] * _rg2
                    _ck_d = xp.zeros(adm.ADM_shape + (2,), dtype=_rhog_den_d.dtype)
                    _ck_d = _ck_d.at[:, :, kmin:kmax+1, :, 0].set(_ck0)
                    _ck_d = _ck_d.at[:, :, kmin:kmax+1, :, 1].set(_ck1)
                else:
                    # RES-TP-1b: ck/d are host-computed in this 2nd step (no tvf kernel);
                    # upload them once and reuse the device handles across the per-iq loop.
                    _ck_d = xp.asarray(ck)
                    _d_d  = xp.asarray(d)
        if _vpole:
            # RC-43 (Track B 3b): phase-3 device pole vert-adv. Same as phase-1 but the
            # q-denominator is rhog_pl (updated by step 1), and ck/d are recomputed on
            # device from rhog_pl + the phase-1 device flx_v (_fvp) -- the pole analog of
            # the regular device ck/d above. b3 weights the d term. Replaces the host
            # ck_pl/d_pl recompute (skipped above) + the per-iq host pole compute.
            # Unit 4c-2: reuse the horizontal-phase device pole rhogq/rhog carries
            # (updated by the device flux apply) instead of re-uploading the host
            # arrays. Bit-identical (device carry == asarray(host) after the apply).
            _rhogq_pl_d    = _rhogq_pl_d if _resident_hadv_apply_pl else xp.asarray(rhogq_pl)
            _rhog_den_pl_d = _rhog_carry_pl_d if (_resident_hadv_apply_pl and _rhog_carry_pl_d is not None) else xp.asarray(rhog_pl)
            _flx_v_pl_d    = _fvp
            _frhog_pl_dev  = frhog_pl_d if frhog_pl_d is not None else xp.asarray(frhog_pl)
            _rdgz_pl_d = bk.device_consts(self, "tracer_v2_rdgz", lambda: {"r": grd.GRD_rdgz})["r"]
            _d_pl_d = b3 * _frhog_pl_dev / _rhog_den_pl_d * dt
            _rg2p = _rdgz_pl_d[kmin:kmax+1][None, :, None]
            _ck0p = -_flx_v_pl_d[:, kmin:kmax+1, :]   / _rhog_den_pl_d[:, kmin:kmax+1, :] * _rg2p
            _ck1p =  _flx_v_pl_d[:, kmin+1:kmax+2, :] / _rhog_den_pl_d[:, kmin:kmax+1, :] * _rg2p
            _ck_pl_d = xp.zeros(adm.ADM_shape_pl + (2,), dtype=_rhog_den_pl_d.dtype)
            _ck_pl_d = _ck_pl_d.at[:, kmin:kmax+1, :, 0].set(_ck0p)
            _ck_pl_d = _ck_pl_d.at[:, kmin:kmax+1, :, 1].set(_ck1p)

        #--- vertical advection: 2nd-order centered difference
        for iq in range(vmax):

            if _vta_on:
                _q, _q_h = _vtak["qh"](
                    (_rhogq_d[:, :, :, :, iq] if _resident_tracer_v
                     else xp.asarray(rhogq[:, :, :, :, iq])),
                    (_rhog_den_d if _resident_tracer_v else xp.asarray(rhog)),
                    _vtad["afact"], _vtad["bfact"], cfg=_vtacfg, xp=xp,
                )
                if _resident_vlim:
                    # RES-TP-1b: hold q_h (and q) on device for the limiter + update.
                    _q_h_d = _q_h
                    _q_d   = _q
                else:
                    q[:, :, :, :] = bk.to_numpy(_q)
                    q_h[:, :, :, :] = bk.to_numpy(_q_h)
                if adm.ADM_have_pl:
                    _qp, _qhp = _vtak["qhp"](
                        (_rhogq_pl_d[:, :, :, iq] if _vpole else xp.asarray(rhogq_pl[:, :, :, iq])),
                        (_rhog_den_pl_d if _vpole else xp.asarray(rhog_pl)),
                        _vtad["afact"], _vtad["bfact"], cfg=_vtacfg, xp=xp,
                    )
                    if _vpole:
                        _qhp_d = _qhp; _qp_d = _qp   # RC-43: device pole q_h/q for the device limiter + upp
                    # writable copy: q_pl is slice-assigned later (horizontal adv),
                    # but bk.to_numpy returns a read-only (jax-derived) array.
                    # TRACER-JIT: qpl3+qhpl3 DEAD (poison-confirmed) -- phase-3 is the terminal
                    # Strang half-step (no later horizontal reader); device _qp_d/_qhp_d feed
                    # the limiter/upp under _vpole. Skip both drains under _vpole_nodrain.
                    if not _vpole_nodrain:
                        q_pl = np.array(bk.to_numpy(_qp))
                        q_h_pl[:, :, :] = bk.to_numpy(_qhp)
            else:
                for l in range(lall):
                    # q = rhogq / rhog
                    q[:, :, :, l] = rhogq[:, :, :, l, iq] / rhog[:, :, :, l]

                    # q_h = a * q + b * q[-1]
                    for k in range(kmin, kmax + 2):
                        q_h[:, :, k, l] = (
                            grd.GRD_afact[k] * q[:, :, k,   l] +
                            grd.GRD_bfact[k] * q[:, :, k-1, l]
                        )

                    # Set boundary
                    q_h[:, :, kmin - 1, l] = rdtype(0.0)
                # end loop l

                if adm.ADM_have_pl:
                    # q_pl = rhogq_pl / rhog_pl (element-wise division)
                    q_pl = rhogq_pl[:, :, :, iq] / rhog_pl

                    # q_h_pl = a * q_pl + b * q_pl (shifted k-1)
                    q_h_pl[:, kmin:kmax+2, :] = (
                        grd.GRD_afact[kmin:kmax+2][None, :, None] * q_pl[:, kmin:kmax+2, :] +
                        grd.GRD_bfact[kmin:kmax+2][None, :, None] * q_pl[:, kmin-1:kmax+1, :]
                    )

                    # Boundary at kmin-1
                    q_h_pl[:, kmin-1, :] = rdtype(0.0)
                # endif


            # with open(std.fname_log, 'a') as log_file:
            #     print(f"iq=  {iq} ",file=log_file)
            # #     print("STA2.5 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  bad
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  good
            #     print("          q_h[0,0,7,1]    ",   q_h[0, 0, 7, 1]   , file=log_file)  
            #     print("            q[0,0,7,1]    ",     q[0, 0, 7, 1]   , file=log_file)
            #     print("            d[0,0,7,1]    ",     d[0, 0, 7, 1]   , file=log_file)  
            #     print("           ck[0,0,7,1,:]  ",    ck[0, 0, 7, 1, :], file=log_file)    #you bad
            #     print("          q_h[1,1,7,1]  ",     q_h[1, 1, 7, 1]   , file=log_file)    
            #     print("            q[1,1,7,1]  ",       q[1, 1, 7, 1]   , file=log_file)  
            #     print("            d[1,1,7,1]    ",     d[1, 1, 7, 1]   , file=log_file)
            #     print("           ck[1,1,7,1,:]  ",    ck[1, 1, 7, 1, :], file=log_file)    #you good

            #    print("STB2.5 :rhogq[6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)  #you  e+23
            #     print("          q_h[6,5,10,0]  ",     q_h[6, 5, 10, 0]   , file=log_file)  
            #     print("            q[6,5,10,0]  ",       q[6, 5, 10, 0]   , file=log_file)  # 0, 1, 2 are undef
                # print("            d[6,5,10,0]    ",     d[6, 5, 10, 0]   , file=log_file)
                # print("           ck[6,5,10,0,:]  ",    ck[6, 5, 10, 0, :], file=log_file)

                # print("STC2.5 :rhogq[6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)  #you  e+23
                # print("STD2.5 :rhogq[6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)  #you  e+23
                # print("STD2.5 :rhogq[6,5,1,1,:]  ", rhogq[6, 5, 1, 1, :], file=log_file)  #you  e+23
                # print("          q_h[6,5,3,1]  ",     q_h[6, 5, 3, 1]   , file=log_file)
                # print("            q[6,5,3,1]  ",       q[6, 5, 3, 1]   , file=log_file)  # 0, 1, 2 are undef
                # print("            d[6,5,3,1]    ",     d[6, 5, 3, 1]   , file=log_file)
                # print("           ck[6,5,3,1,:]  ",    ck[6, 5, 3, 1, :], file=log_file)    #you good
                # print("          q_h[6,5,2,1]  ",     q_h[6, 5, 2, 1]   , file=log_file)
                # print("            q[6,5,2,1]  ",       q[6, 5, 2, 1]   , file=log_file)  # 0, 1, 2 are undef
                # print("            d[6,5,2,1]    ",     d[6, 5, 2, 1]   , file=log_file)
                # print("           ck[6,5,2,1,:]  ",    ck[6, 5, 2, 1, :], file=log_file)    #you good
                # print("          q_h[6,5,1,1]  ",     q_h[6, 5, 1, 1]   , file=log_file)
                # print("            q[6,5,1,1]  ",       q[6, 5, 1, 1]   , file=log_file)  # 0, 1, 2 are undef
                # print("            d[6,5,1,1]    ",     d[6, 5, 1, 1]   , file=log_file)
                # print("           ck[6,5,1,1,:]  ",    ck[6, 5, 1, 1, :], file=log_file)    #you good

            if apply_limiter_v[iq]:
                if _resident_vlim:
                    # RES-TP-1b: device q_h in, device q_h out (pole q_h_pl stays host).
                    _q_h_d = self.vertical_limiter_thuburn(
                        _q_h_d, q_h_pl,  # [INOUT]
                        _q_d  , q_pl  ,  # [IN]
                        _d_d  , d_pl  ,  # [IN]
                        _ck_d , ck_pl ,  # [IN]
                        cnst, rdtype, resident=True,
                        skip_pole=_vpole,   # RC-43: pole limited on device below
                    )
                else:
                    self.vertical_limiter_thuburn(
                        q_h,   q_h_pl,  # [INOUT]
                        q  ,   q_pl  ,  # [IN]
                        d  ,   d_pl  ,  # [IN]
                        ck ,   ck_pl ,  # [IN]
                        cnst, rdtype,
                    )
                if _vpole:
                    # RC-43: device POLE limiter via reshape (g,k,l)->(g,1,k,l), reusing
                    # the per-column compute_vertical_limiter (RC-40-fixed; bit-exact vs
                    # the host pole section). Only kmin+1..kmax modified.
                    _qhp_d = self._vlim_kernel(
                        _qhp_d[:, None, :, :], _qp_d[:, None, :, :],
                        _d_pl_d[:, None, :, :], _ck_pl_d[:, None, :, :, :],
                        cfg=self._vlim_cfg, xp=xp)[:, 0, :, :]
            # endif

            #--- update rhogq

            if _vta_on:
                _rg = _vtak["up"](
                    (_rhogq_d[:, :, :, :, iq] if _resident_tracer_v
                     else xp.asarray(rhogq[:, :, :, :, iq])),
                    (_flx_v_d if _resident_tracer_v else xp.asarray(flx_v)),
                    (_q_h_d if _resident_vlim else xp.asarray(q_h)),
                    _vtad["rdgz"], cfg=_vtacfg, xp=xp,
                )
                if _resident_tracer_v:
                    _rhogq_d = _rhogq_d.at[:, :, :, :, iq].set(_rg)
                else:
                    rhogq[:, :, :, :, iq] = bk.to_numpy(_rg)
                if adm.ADM_have_pl:
                    _rgp = _vtak["upp"](
                        (_rhogq_pl_d[:, :, :, iq] if _vpole else xp.asarray(rhogq_pl[:, :, :, iq])),
                        (_flx_v_pl_d if _vpole else xp.asarray(flx_v_pl)),
                        (_qhp_d if _vpole else xp.asarray(q_h_pl)),
                        _vtad["rdgz"], cfg=_vtacfg, xp=xp,
                    )
                    if _vpole:
                        _rhogq_pl_d = _rhogq_pl_d.at[:, :, :, iq].set(_rgp)   # RC-43: carry device pole rhogq
                    else:
                        rhogq_pl[:, :, :, iq] = bk.to_numpy(_rgp)
            else:
                for l in range(lall):
                    q_h[:, :, kmin, l] = rdtype(0.0)
                    q_h[:, :, kmax+1, l] = rdtype(0.0)

                    for k in range(kmin, kmax+1):
                        rhogq[:, :, k, l, iq] -= (
                            flx_v[:, :, k+1, l] * q_h[:, :, k+1, l] -
                            flx_v[:, :, k,   l] * q_h[:, :, k,   l]
                        ) * grd.GRD_rdgz[k]

                    rhogq[:, :, kmin-1, l, iq] = rdtype(0.0)
                    rhogq[:, :, kmax+1, l, iq] = rdtype(0.0)



                if adm.ADM_have_pl:
                    q_h_pl[:, kmin,   :] = rdtype(0.0)
                    q_h_pl[:, kmax+1, :] = rdtype(0.0)

                    for k in range(kmin, kmax+1):
                        rhogq_pl[:, k, :, iq] -= (
                            flx_v_pl[:, k+1, :] * q_h_pl[:, k+1, :] -
                            flx_v_pl[:, k,   :] * q_h_pl[:, k,   :]
                        ) * grd.GRD_rdgz[k]

                    rhogq_pl[:, kmin-1, :, iq] = rdtype(0.0)
                    rhogq_pl[:, kmax+1, :, iq] = rdtype(0.0)

            # with open(std.fname_log, 'a') as log_file:
               
            #     print("STA2.6 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  
            #     print("        flx_v[0,0,8,1]  ", flx_v[0, 0, 8, 1], file=log_file) 
            #     print("        flx_v[0,0,7,1]  ", flx_v[0, 0, 7, 1], file=log_file)  
            #     print("        flx_v[1,1,8,1]  ", flx_v[1, 1, 8, 1], file=log_file) 
            #     print("        flx_v[1,1,7,1]  ", flx_v[1, 1, 7, 1], file=log_file)  
            #     print("          q_h[0,0,8,1]  ",   q_h[0, 0, 8, 1], file=log_file) 
            #     print("          q_h[0,0,7,1]  ",   q_h[0, 0, 7, 1], file=log_file)  
            #     print("          q_h[1,1,8,1]  ",   q_h[1, 1, 8, 1], file=log_file) 
            #     print("          q_h[1,1,7,1]  ",   q_h[1, 1, 7, 1], file=log_file)  
            #     print("       grd.GRD_rdgz[7]  ",   grd.GRD_rdgz[7], file=log_file)  

            #print("STB2.6 :rhogq[6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)  
            #print("          q_h[6,5,10,0]  ",     q_h[6, 5, 10, 0]   , file=log_file)  
                # print("STD2.6 :rhogq[6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)  
                # print("STD2.6 :rhogq[6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)  
                # print("STD2.6 :rhogq[6,5,1,1,:]  ", rhogq[6, 5, 1, 1, :], file=log_file)  

                # print("        flx_v[6,5,3,1]  ", flx_v[6, 5, 3, 1], file=log_file) 
                # print("        flx_v[6,5,2,1]  ", flx_v[6, 5, 2, 1], file=log_file)  
                # print("        flx_v[6,5,1,1]  ", flx_v[6, 5, 1, 1], file=log_file)  
                # print("          q_h[6,5,3,1]  ",   q_h[6, 5, 3, 1], file=log_file) 
                # print("          q_h[6,5,2,1]  ",   q_h[6, 5, 2, 1], file=log_file)  
                # print("          q_h[6,5,1,1]  ",   q_h[6, 5, 1, 1], file=log_file)  

            #--- tiny negative fixer

            if _resident_tracer_v:
                # device clip of tiny negatives (k in [kmin,kmax]; boundaries are
                # exactly 0 so unaffected either way). Bit-identical to the host mask.
                _kc = slice(kmin, kmax + 1)
                _rqi = _rhogq_d[:, :, _kc, :, iq]
                _rqi = xp.where(
                    (_rqi > -rdtype(1.0e-10)) & (_rqi < rdtype(0.0)),
                    rdtype(0.0), _rqi)
                _rhogq_d = _rhogq_d.at[:, :, _kc, :, iq].set(_rqi)
            else:
                for l in range(lall):
                    for k in range(kmin, kmax + 1):
                        mask = (rhogq[:, :, k, l, iq] > -rdtype(1.0e-10)) & (rhogq[:, :, k, l, iq] < rdtype(0.0))
                        rhogq[:, :, k, l, iq][mask] = rdtype(0.0)

            if _vpole:
                # RC-43: device pole tiny-negative clip (k in [kmin,kmax]; boundaries are
                # exactly 0 so unaffected). Bit-identical to the host mask below.
                _kcp = slice(kmin, kmax + 1)
                _rqip = _rhogq_pl_d[:, _kcp, :, iq]
                _rqip = xp.where((_rqip > -rdtype(1.0e-10)) & (_rqip < rdtype(0.0)), rdtype(0.0), _rqip)
                _rhogq_pl_d = _rhogq_pl_d.at[:, _kcp, :, iq].set(_rqip)
            else:
                mask_pl = (rhogq_pl[..., iq] > -rdtype(1.0e-10)) & (rhogq_pl[..., iq] < rdtype(0.0))
                rhogq_pl[..., iq][mask_pl] = rdtype(0.0)

        # end loop iq

        # RES-TP-1: drain the device-resident rhogq back to host once (caller reads
        # host rhogq after the tracer). Bit-identical to the per-iq to_numpy path.
        if _resident_tracer_v and not skip_drain:   # U5-D.2: caller drains _rhogq_d at the marshal
            rhogq[:, :, :, :, :] = bk.to_numpy(_rhogq_d)
        if _vpole and not skip_drain_pl:
            # RC-43: drain the device pole rhogq (the caller's pole PROGq_pl update reads
            # host rhogq_pl). RC-44: under skip_drain_pl the caller does the PROGq_pl update
            # + marshal on device from the returned _rhogq_pl_d, so skip this drain.
            rhogq_pl[:, :, :, :] = bk.to_numpy(_rhogq_pl_d)

        prf.PROF_rapend('____vertical_adv',2)

        # U5-D (RES-CAPSTONE-29): expose the final device rhogq so the caller can do the
        # PROGq update + the prgv marshal ON DEVICE (moves the per-ndyn host rhogq path to
        # a single device->host drain at the step-end marshal). None on the host path.
        # RES-CAPSTONE-44: also return the device POLE rhogq (only under skip_drain_pl +
        # _vpole, when the caller marshals PROGq_pl on device).
        if skip_drain_pl and _vpole:
            return (_rhogq_d if _resident_tracer_v else None), _rhogq_pl_d
        return (_rhogq_d if _resident_tracer_v else None)

    #> Prepare horizontal advection term: mass flux, horizon
    def horizontal_flux(self,
       flx_h,  flx_h_pl,      # [OUT]    # horizontal mass flux
       grd_xc, grd_xc_pl,     # [OUT]    # mass centroid position   
       rho,    rho_pl,        # [IN]     # rho at cell center
       rhovx,  rhovx_pl,      # [IN]
       rhovy,  rhovy_pl,      # [IN]
       rhovz,  rhovz_pl,      # [IN]
       dt,
       cnst, grd, gmtr, rdtype,
       rhovx_d=None, rhovy_d=None, rhovz_d=None,   # U5-C.6: device rho*v (asarray no-op)
       rho_d=None,                                 # RES-CAPSTONE-35: device rho (asarray no-op)
       rho_pl_d=None, rhovx_pl_d=None, rhovy_pl_d=None, rhovz_pl_d=None,   # RC-81: device POLE rho/rho*v
    ):
    
        prf.PROF_rapstart('____horizontal_adv_flux',2)

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        kall = adm.ADM_kall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        EPS = cnst.CONST_EPS

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ 
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0
 
        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR
    
        P_RAREA = gmtr.GMTR_p_RAREA
        T_RAREA = gmtr.GMTR_t_RAREA 
        W1      = gmtr.GMTR_t_W1  
        W2      = gmtr.GMTR_t_W2    
        W3      = gmtr.GMTR_t_W3    
        HNX     = gmtr.GMTR_a_HNX   
        HNY     = gmtr.GMTR_a_HNY   
        HNZ     = gmtr.GMTR_a_HNZ   
    
        rhot_TI  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)  # rho at cell vertex
        rhot_TJ  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)  # rho at cell vertex
        rhovxt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovxt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovyt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovyt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovzt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovzt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)

        rhot_pl  = np.full((gall_pl, kall), cnst.CONST_UNDEF)
        rhovxt_pl= np.full((gall_pl, kall), cnst.CONST_UNDEF)
        rhovyt_pl= np.full((gall_pl, kall), cnst.CONST_UNDEF)
        rhovzt_pl= np.full((gall_pl, kall), cnst.CONST_UNDEF)


        # (A) fused jit-able kernel for flx_h / grd_xc (regular + pole), gated
        # PYNICAM_FUSE_FLUX (default off). kernels/horizontalflux.py (in-branch,
        # validated by proto/test_horizontalflux_kernel.py). Returns all outputs,
        # so the numpy regular + pole loops below are bypassed via early return.
        _fused_flux = (bk.type == "jax") and getattr(
            self, "use_fuse_flux", os.environ.get("PYNICAM_FUSE_FLUX", "1") != "0")
        if _fused_flux:
            xp = bk.xp
            if getattr(self, "_flux_kernel", None) is None:
                self._flux_kernel = bk.maybe_jit(
                    compute_horizontal_flux, static_argnames=("cfg", "xp"))
            _cfg = HorizFluxCfg(
                iall=iall, jall=jall, K0=K0, TI=TI, TJ=TJ, AI=AI, AIJ=AIJ, AJ=AJ,
                W1=W1, W2=W2, W3=W3, HNX=HNX, HNY=HNY, HNZ=HNZ, P_RAREA=P_RAREA,
                XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR, have_pl=adm.ADM_have_pl,
                gslf_pl=adm.ADM_gslf_pl, gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl, EPS=float(EPS))
            # RES-CAPSTONE-34: the flux kernel's geometry inputs (GMTR_t/a/p, GRD_xr,
            # pntmask) are loop-invariant -> cache them ONCE on device via device_consts
            # instead of asarray-uploading them every tracer call (the last un-cached
            # geometry consts on the regular path; vi/numfilter/src already cache theirs).
            # Value-identical -> bit-exact. Gate PYNICAM_RESIDENT_FLUXGEOM (default OFF).
            _fluxgeom = bk.resident()
            if _fluxgeom:
                _fg = bk.device_consts(self, "tracer_flux_geom", lambda: {
                    "Tt": gmtr.GMTR_t, "Ta": gmtr.GMTR_a, "Tp": gmtr.GMTR_p,
                    "xr": grd.GRD_xr, "pm": ppm.pntmask,
                    "Tt_pl": gmtr.GMTR_t_pl, "Ta_pl": gmtr.GMTR_a_pl,
                    "Tp_pl": gmtr.GMTR_p_pl, "xr_pl": grd.GRD_xr_pl})
            _fh, _gxc, _fhp, _gxcp = self._flux_kernel(
                (rho_d if rho_d is not None else xp.asarray(rho)),   # RES-CAPSTONE-35
                (rhovx_d if rhovx_d is not None else xp.asarray(rhovx)),
                (rhovy_d if rhovy_d is not None else xp.asarray(rhovy)),
                (rhovz_d if rhovz_d is not None else xp.asarray(rhovz)),
                (rho_pl_d   if rho_pl_d   is not None else xp.asarray(rho_pl)),     # RC-81: device POLE rho/rho*v
                (rhovx_pl_d if rhovx_pl_d is not None else xp.asarray(rhovx_pl)),
                (rhovy_pl_d if rhovy_pl_d is not None else xp.asarray(rhovy_pl)),
                (rhovz_pl_d if rhovz_pl_d is not None else xp.asarray(rhovz_pl)),
                (_fg["Tt"] if _fluxgeom else xp.asarray(gmtr.GMTR_t)),
                (_fg["Ta"] if _fluxgeom else xp.asarray(gmtr.GMTR_a)),
                (_fg["Tp"] if _fluxgeom else xp.asarray(gmtr.GMTR_p)),
                (_fg["xr"] if _fluxgeom else xp.asarray(grd.GRD_xr)),
                (_fg["pm"] if _fluxgeom else xp.asarray(ppm.pntmask)),
                (_fg["Tt_pl"] if _fluxgeom else xp.asarray(gmtr.GMTR_t_pl)),
                (_fg["Ta_pl"] if _fluxgeom else xp.asarray(gmtr.GMTR_a_pl)),
                (_fg["Tp_pl"] if _fluxgeom else xp.asarray(gmtr.GMTR_p_pl)),
                (_fg["xr_pl"] if _fluxgeom else xp.asarray(grd.GRD_xr_pl)),
                dt, cfg=_cfg, xp=xp)
            if getattr(self, "_hadv_resident", False):
                # Stage-4a: keep flux outputs on device. flx_h is still drained
                # (the numpy rhogq/rhog updates consume it); grd_xc stays
                # device-only and is fed to remap without a host round-trip.
                self._flx_h_d = _fh; self._grd_xc_d = _gxc
                self._flx_h_pl_d = _fhp; self._grd_xc_pl_d = _gxcp
                # RC-73: host regular flx_h is DEAD on the resident path -- the flux-apply
                # uses the device self._flx_h_d (_resident_hadv_upd @~873); the host
                # flx_h*q_a apply (@~893) is the dead else. flxh poison-confirmed unread
                # (job 2270499: NaN host flx_h -> gold 1.15e-11 = identical to base). Skip
                # the 351MB/step D2H under PYNICAM_RESIDENT_HADV_FLXH_SKIP (default OFF).
                if not bk.resident():
                    flx_h[:, :, :, :, :] = bk.to_numpy(_fh)
                if adm.ADM_have_pl and not getattr(self, "_hadv_qa_pl_active", False):
                    # 4c-6: host flx_h_pl/grd_xc_pl are DEAD under the full pole device
                    # path (device courant + device flux apply use self._flx_h_pl_d;
                    # remap uses self._grd_xc_pl_d; the host flux apply was removed in
                    # 4c-4). Drain only when the device path is not fully active.
                    flx_h_pl[:, :, :]     = bk.to_numpy(_fhp)
                    grd_xc_pl[:, :, :, :] = bk.to_numpy(_gxcp)
                    # Track B POLE-POISON (RC-37 classify): NaN the horizontal-flux pole
                    # outputs; PASS => host flx_h_pl/grd_xc_pl unread (device _fhp/_gxcp
                    # threadable into the pole remap/limiter once those are ported).
                prf.PROF_rapend('____horizontal_adv_flux', 2)
                return
            flx_h[:, :, :, :, :]      = bk.to_numpy(_fh)
            grd_xc[:, :, :, :, :, :]  = bk.to_numpy(_gxc)
            if adm.ADM_have_pl:
                flx_h_pl[:, :, :]     = bk.to_numpy(_fhp)
                grd_xc_pl[:, :, :, :] = bk.to_numpy(_gxcp)
            prf.PROF_rapend('____horizontal_adv_flux', 2)
            return

        # Vectorised over k (the geometry at K0 is k-independent; broadcast it
        # over the k-axis with [:, :, None]). Bit-identical to the original
        # per-k loop; just removes the Python k iteration.
        for l in range(lall):

            isl = slice(0, iall - 1)
            jsl = slice(0, jall - 1)

            isl_p = slice(1, iall)
            jsl_p = slice(1, jall)

            gt = gmtr.GMTR_t  # alias

            # First part: (i,j), (i+1,j)
            rhot_TI[isl, jsl, :]   = rho[isl, jsl, :, l]   * gt[isl, jsl, K0, l, TI, W1][:, :, None] + rho[isl_p, jsl, :, l]   * gt[isl, jsl, K0, l, TI, W2][:, :, None]
            rhovxt_TI[isl, jsl, :] = rhovx[isl, jsl, :, l] * gt[isl, jsl, K0, l, TI, W1][:, :, None] + rhovx[isl_p, jsl, :, l] * gt[isl, jsl, K0, l, TI, W2][:, :, None]
            rhovyt_TI[isl, jsl, :] = rhovy[isl, jsl, :, l] * gt[isl, jsl, K0, l, TI, W1][:, :, None] + rhovy[isl_p, jsl, :, l] * gt[isl, jsl, K0, l, TI, W2][:, :, None]
            rhovzt_TI[isl, jsl, :] = rhovz[isl, jsl, :, l] * gt[isl, jsl, K0, l, TI, W1][:, :, None] + rhovz[isl_p, jsl, :, l] * gt[isl, jsl, K0, l, TI, W2][:, :, None]

            rhot_TJ[isl, jsl, :]   = rho[isl, jsl, :, l]   * gt[isl, jsl, K0, l, TJ, W1][:, :, None]
            rhovxt_TJ[isl, jsl, :] = rhovx[isl, jsl, :, l] * gt[isl, jsl, K0, l, TJ, W1][:, :, None]
            rhovyt_TJ[isl, jsl, :] = rhovy[isl, jsl, :, l] * gt[isl, jsl, K0, l, TJ, W1][:, :, None]
            rhovzt_TJ[isl, jsl, :] = rhovz[isl, jsl, :, l] * gt[isl, jsl, K0, l, TJ, W1][:, :, None]

            # Second part: (i+1,j+1), (i,j+1)
            rhot_TI[isl, jsl, :]   += rho[isl_p, jsl_p, :, l]   * gt[isl, jsl, K0, l, TI, W3][:, :, None]
            rhovxt_TI[isl, jsl, :] += rhovx[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TI, W3][:, :, None]
            rhovyt_TI[isl, jsl, :] += rhovy[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TI, W3][:, :, None]
            rhovzt_TI[isl, jsl, :] += rhovz[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TI, W3][:, :, None]

            rhot_TJ[isl, jsl, :]   += rho[isl_p, jsl_p, :, l]   * gt[isl, jsl, K0, l, TJ, W2][:, :, None] + rho[isl, jsl_p, :, l]   * gt[isl, jsl, K0, l, TJ, W3][:, :, None]
            rhovxt_TJ[isl, jsl, :] += rhovx[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W2][:, :, None] + rhovx[isl, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W3][:, :, None]
            rhovyt_TJ[isl, jsl, :] += rhovy[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W2][:, :, None] + rhovy[isl, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W3][:, :, None]
            rhovzt_TJ[isl, jsl, :] += rhovz[isl_p, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W2][:, :, None] + rhovz[isl, jsl_p, :, l] * gt[isl, jsl, K0, l, TJ, W3][:, :, None]

            # singular-pole point fix (scalars over k)
            rhot_TI[0, 0, :]   = rhot_TI[0, 0, :]   * ppm.pntmask[K0, l, 0] + rhot_TJ[1, 0, :]   * ppm.pntmask[K0, l, 1]
            rhovxt_TI[0, 0, :] = rhovxt_TI[0, 0, :] * ppm.pntmask[K0, l, 0] + rhovxt_TJ[1, 0, :] * ppm.pntmask[K0, l, 1]
            rhovyt_TI[0, 0, :] = rhovyt_TI[0, 0, :] * ppm.pntmask[K0, l, 0] + rhovyt_TJ[1, 0, :] * ppm.pntmask[K0, l, 1]
            rhovzt_TI[0, 0, :] = rhovzt_TI[0, 0, :] * ppm.pntmask[K0, l, 0] + rhovzt_TJ[1, 0, :] * ppm.pntmask[K0, l, 1]

            flx_h[:, :, :, l, :]     = rdtype(0.0)
            grd_xc[:, :, :, l, :, :] = rdtype(0.0)

            # --- AI edge ---
            isl = slice(0, iall - 1)
            jsl = slice(1, jall - 1)
            jslm1 = slice(0, jall - 2)

            rrhoa2 = rdtype(1.0) / np.maximum(rhot_TJ[isl, jslm1, :] + rhot_TI[isl, jsl, :], EPS)
            rhovxt2 = rhovxt_TJ[isl, jslm1, :] + rhovxt_TI[isl, jsl, :]
            rhovyt2 = rhovyt_TJ[isl, jslm1, :] + rhovyt_TI[isl, jsl, :]
            rhovzt2 = rhovzt_TJ[isl, jslm1, :] + rhovzt_TI[isl, jsl, :]

            flux = rdtype(0.5) * (
                rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNX][:, :, None] +
                rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNY][:, :, None] +
                rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNZ][:, :, None]
            )

            flx_h[isl, jsl, :, l, 0]  =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
            flx_h[isl.start+1:isl.stop+1, jsl, :, l, 3] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl, K0, l, P_RAREA][:, :, None] * dt

            grd_xc[isl, jsl, :, l, AI, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, XDIR][:, :, None] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AI, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, YDIR][:, :, None] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AI, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, ZDIR][:, :, None] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)

            # --- AIJ edge ---
            isl = slice(0, iall - 1)
            jsl = slice(0, jall - 1)

            rrhoa2 = rdtype(1.0) / np.maximum(rhot_TI[isl, jsl, :] + rhot_TJ[isl, jsl, :], EPS)
            rhovxt2 = rhovxt_TI[isl, jsl, :] + rhovxt_TJ[isl, jsl, :]
            rhovyt2 = rhovyt_TI[isl, jsl, :] + rhovyt_TJ[isl, jsl, :]
            rhovzt2 = rhovzt_TI[isl, jsl, :] + rhovzt_TJ[isl, jsl, :]

            flux = rdtype(0.5) * (
                rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNX][:, :, None] +
                rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNY][:, :, None] +
                rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNZ][:, :, None]
            )

            flx_h[isl, jsl, :, l, 1] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
            flx_h[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, :, l, 4] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, K0, l, P_RAREA][:, :, None] * dt

            grd_xc[isl, jsl, :, l, AIJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, XDIR][:, :, None] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AIJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, YDIR][:, :, None] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AIJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, ZDIR][:, :, None] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)

            # --- AJ edge ---
            isl = slice(1, iall - 1)
            jsl = slice(0, jall - 1)

            rrhoa2 = rdtype(1.0) / np.maximum(rhot_TJ[isl, jsl, :] + rhot_TI[isl.start - 1:isl.stop - 1, jsl, :], EPS)
            rhovxt2 = rhovxt_TJ[isl, jsl, :] + rhovxt_TI[isl.start - 1:isl.stop - 1, jsl, :]
            rhovyt2 = rhovyt_TJ[isl, jsl, :] + rhovyt_TI[isl.start - 1:isl.stop - 1, jsl, :]
            rhovzt2 = rhovzt_TJ[isl, jsl, :] + rhovzt_TI[isl.start - 1:isl.stop - 1, jsl, :]

            flux = rdtype(0.5) * (
                rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNX][:, :, None] +
                rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNY][:, :, None] +
                rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNZ][:, :, None]
            )

            flx_h[isl, jsl, :, l, 2] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
            flx_h[isl, jsl.start + 1:jsl.stop + 1, :, l, 5] = -flux * gmtr.GMTR_p[isl, jsl.start + 1:jsl.stop + 1, K0, l, P_RAREA][:, :, None] * dt

            grd_xc[isl, jsl, :, l, AJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, XDIR][:, :, None] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, YDIR][:, :, None] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
            grd_xc[isl, jsl, :, l, AJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, ZDIR][:, :, None] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)

            flx_h[1, 1, :, l, 5] *= rdtype(ppm.pntmask[K0, l, 0])
        # end loop l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            # Vectorised over k (geometry at K0 is k-independent; the temporaries
            # carry a k-axis). Bit-identical to the original per-k loop.
            for l in range(lall_pl):
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    ijp1 = v + 1
                    if ijp1 == adm.ADM_gmax_pl + 1:
                        ijp1 = adm.ADM_gmin_pl

                    rhot_pl[v, :]   = rho_pl[n, :, l]   * gmtr.GMTR_t_pl[ij, K0, l, W1] + rho_pl[ij, :, l]   * gmtr.GMTR_t_pl[ij, K0, l, W2] + rho_pl[ijp1, :, l]   * gmtr.GMTR_t_pl[ij, K0, l, W3]
                    rhovxt_pl[v, :] = rhovx_pl[n, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + rhovx_pl[ij, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + rhovx_pl[ijp1, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                    rhovyt_pl[v, :] = rhovy_pl[n, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + rhovy_pl[ij, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + rhovy_pl[ijp1, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                    rhovzt_pl[v, :] = rhovz_pl[n, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + rhovz_pl[ij, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + rhovz_pl[ijp1, :, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                # end loop v

                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    ijm1 = v - 1
                    if ijm1 == adm.ADM_gmin_pl - 1:
                        ijm1 = adm.ADM_gmax_pl

                    rrhoa2  = rdtype(1.0) / np.maximum(rhot_pl[ijm1, :] + rhot_pl[ij, :], EPS)
                    rhovxt2 = rhovxt_pl[ijm1, :] + rhovxt_pl[ij, :]
                    rhovyt2 = rhovyt_pl[ijm1, :] + rhovyt_pl[ij, :]
                    rhovzt2 = rhovzt_pl[ijm1, :] + rhovzt_pl[ij, :]

                    flux = rdtype(0.5) * (
                        rhovxt2 * gmtr.GMTR_a_pl[ij, K0, l, HNX] +
                        rhovyt2 * gmtr.GMTR_a_pl[ij, K0, l, HNY] +
                        rhovzt2 * gmtr.GMTR_a_pl[ij, K0, l, HNZ]
                    )

                    flx_h_pl[v, :, l] = flux * gmtr.GMTR_p_pl[n, K0, l, P_RAREA] * dt

                    grd_xc_pl[v, :, l, XDIR] = grd.GRD_xr_pl[v, K0, l, XDIR] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
                    grd_xc_pl[v, :, l, YDIR] = grd.GRD_xr_pl[v, K0, l, YDIR] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
                    grd_xc_pl[v, :, l, ZDIR] = grd.GRD_xr_pl[v, K0, l, ZDIR] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)
                # end loop v
            # end loop l
        # endif

        prf.PROF_rapend  ('____horizontal_adv_flux',2)

        return

    def horizontal_remap(self,
        q_a,    q_a_pl,       # [OUT]    # q at cell face
        q,      q_pl,         # [IN]     # q at cell center
        cmask,  cmask_pl,     # [IN]     # upwind direction mask
        grd_xc, grd_xc_pl,    # [IN]     # position of the mass centroid
        cnst, comm, grd, oprt, rdtype,
        resident_q=False,     # RES-TP-2: q is a device array (gradient runs resident)
        resident_comm=False,  # RES-TP-2b: on-device gradq halo exchange (no drain/re-upload)
        q_pl_d=None, cmask_pl_d=None, grd_xc_pl_d=None,  # Unit 4c-1: device POLE inputs
        qa_resident_pl=False,                            # 4c-6: device q_a carried -> skip the host drain
    ):
        
        prf.PROF_rapstart('____horizontal_adv_remap',2)
        
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        nxyz = adm.ADM_nxyz

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0

        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR

        # nstart1 = suf(ADM_gmin-1,ADM_gmin-1)
        # nstart2 = suf(ADM_gmin  ,ADM_gmin-1)
        # nstart3 = suf(ADM_gmin  ,ADM_gmin  )
        # nstart4 = suf(ADM_gmin-1,ADM_gmin  )
        # nend    = suf(ADM_gmax  ,ADM_gmax  )

        gradq = np.full(adm.ADM_shape + (nxyz,), cnst.CONST_UNDEF)  # grad(q)
        gradq_pl = np.full(adm.ADM_shape_pl + (nxyz,), cnst.CONST_UNDEF)
    
        q_ap1 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am1 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap2 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am2 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap3 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am3 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap4 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am4 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap5 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am5 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap6 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am6 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)

        _gradq_kernel_d = None   # RES-TP-2b: device gradq to feed the remap kernel
        _gradq_pl_d = None       # unit 4a: device POLE gradq (post-COMM) to feed the pole remap
        # RES-TRACER-2: run the POLE gradient on device too (thread q_pl_d as the device
        # pole scl -> kills the host q_pl read inside OPRT_gradient @~3627, the sole live
        # reader of the phase-1 q_pl drain) and keep gradq_pl on device through the on-
        # device COMM -> the pole remap kernel @~2046 already reads _gradq_pl_d. Requires
        # the device pole q (q_pl_d) + on-device COMM. q_pl_d is the validated device twin
        # of q_pl (the remap kernel uses it bit-exactly, RC-48). Gate default OFF.
        _resident_grad_pl = (bk.type == "jax" and resident_q and resident_comm
                             and q_pl_d is not None
                             and bk.resident())
        if resident_q:
            # RES-TP-2: q is on device -> run the gradient resident (device scl in,
            # device regular grad out; pole drained to host gradq_pl). asarray(q) in
            # the fused remap kernel below is a no-op on the device q.
            _gradq_grad = oprt.OPRT_gradient(
                gradq, gradq_pl, q, q_pl,
                oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl, grd, rdtype, resident=True,
                scl_pl_d=(q_pl_d if _resident_grad_pl else None),
                resident_pl=_resident_grad_pl,
            )
            if _resident_grad_pl:
                _gradq_d, _gradq_pl_pre_d = _gradq_grad
            else:
                _gradq_d = _gradq_grad
            if resident_comm:
                # RES-TP-2b: on-device halo exchange. Passing the device grad auto-
                # routes COMM_data_transfer to the on-device path (mod_comm.py:1623),
                # which returns the exchanged device handles -- no drain + re-upload
                # of the ~3-component gradq. Only the pole is drained back to host for
                # the host pole path below. Bit-identical: the on-device COMM uses the
                # same cached index maps as the numpy path.
                # RES-TRACER-2: under _resident_grad_pl the pole grad enters COMM on
                # device (no @3634 drain) and stays on device (no host gradq_pl drain);
                # the pole remap reads _gradq_pl_d directly.
                _gradq_pl_comm_in = _gradq_pl_pre_d if _resident_grad_pl else gradq_pl
                _gradq_d, _gradq_pl_d = comm.COMM_data_transfer(_gradq_d, _gradq_pl_comm_in)
                if not _resident_grad_pl and adm.ADM_have_pl:
                    # non-pole ranks: gradq_pl is a local dead UNDEF buffer (the pole
                    # remap consumer is under adm.ADM_have_pl) -> skip the dead drain.
                    # Bit-exact (dead), and it un-breaks the FUSE_TRACER jit trace on the
                    # non-pole ranks (to_numpy of the COMM-returned device tracer).
                    gradq_pl[...] = bk.to_numpy(_gradq_pl_d)
                _gradq_kernel_d = _gradq_d
            else:
                # RES-TP-2a: drain the regular grad for the host halo exchange; the
                # remap kernel re-uploads gradq.
                gradq[:, :, :, :, :] = bk.to_numpy(_gradq_d)
                comm.COMM_data_transfer( gradq, gradq_pl)
        else:
            oprt.OPRT_gradient(
                gradq, gradq_pl,
                q, q_pl,
                oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl,
                grd, rdtype,
            )
            comm.COMM_data_transfer( gradq, gradq_pl)

        isl = slice(0, iall - 1)
        jsl = slice(0, jall - 1)
        # (A) fused jit-able kernel for the regular-grid q_a (harvested from branch
        # tracer-remap-fuse). Gated PYNICAM_FUSE_REMAP (default off); the pole (_pl)
        # branch stays on the host path below. When on, the per-l numpy loop is skipped.
        _fused_remap = (bk.type == "jax") and getattr(
            self, "use_fuse_remap", os.environ.get("PYNICAM_FUSE_REMAP", "1") != "0")
        if _fused_remap:
            xp = bk.xp
            if getattr(self, "_remap_kernel", None) is None:
                self._remap_kernel = bk.maybe_jit(
                    compute_horizontal_remap, static_argnames=("cfg", "xp"))
            _rc = bk.device_consts(self, "remap_gx",
                                   lambda: {"gx": xp.asarray(grd.GRD_x[:, :, K0, :, :])})
            _cfg = RemapCfg(AI=AI, AIJ=AIJ, AJ=AJ, XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR)
            _qa = self._remap_kernel(
                xp.asarray(q),
                (_gradq_kernel_d if _gradq_kernel_d is not None else xp.asarray(gradq)),
                xp.asarray(grd_xc),
                xp.asarray(cmask), _rc["gx"], cfg=_cfg, xp=xp)
            if getattr(self, "_hadv_qa_resident", False):
                # Stage-4b: keep q_a on device into the limiter (no drain). Only the
                # interior is meaningful; the limiter's interior output depends only
                # on q_a's interior, so the kernel's ring values are irrelevant.
                self._q_a_d = _qa
            else:
                q_a[isl, jsl, :, :, :] = bk.to_numpy(_qa)[isl, jsl, :, :, :]

        # interpolated Q at cell arc

        q_ap1.fill(rdtype(0.0))
        q_am1.fill(rdtype(0.0))
        q_ap2.fill(rdtype(0.0))
        q_am2.fill(rdtype(0.0))
        q_ap3.fill(rdtype(0.0))
        q_am3.fill(rdtype(0.0))
        q_ap4.fill(rdtype(0.0))
        q_am4.fill(rdtype(0.0))
        q_ap5.fill(rdtype(0.0))
        q_am5.fill(rdtype(0.0))
        q_ap6.fill(rdtype(0.0))
        q_am6.fill(rdtype(0.0))

        isl = slice(0, iall - 1)
        jsl = slice(0, jall - 1)
        isl_p1 = slice(1, iall)
        jsl_p1 = slice(1, jall)
        isls  = slice(1, iall - 1)
        jsls  = slice(1, jall - 1)
        isls_m1 = slice(0, iall - 2)
        jsls_m1 = slice(0, jall - 2)

        # Vectorised over k (q/gradq/grd_xc carry a k-axis; only grd.GRD_x at K0
        # is k-independent -> broadcast with [:, :, None]). Bit-identical to the
        # original per-k loop.
        for l in (range(lall) if not _fused_remap else range(0)):

            q_ap1[isl, jsl, :] = (
                q[isl, jsl, :, l]
                + gradq[isl, jsl, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AI, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR][:, :, None])
                + gradq[isl, jsl, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AI, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR][:, :, None])
                + gradq[isl, jsl, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AI, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR][:, :, None])
            )

            q_am1[isl, jsl, :] = (
                q[isl_p1, jsl, :, l]
                + gradq[isl_p1, jsl, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AI, XDIR] - grd.GRD_x[isl_p1, jsl, K0, l, XDIR][:, :, None])
                + gradq[isl_p1, jsl, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AI, YDIR] - grd.GRD_x[isl_p1, jsl, K0, l, YDIR][:, :, None])
                + gradq[isl_p1, jsl, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AI, ZDIR] - grd.GRD_x[isl_p1, jsl, K0, l, ZDIR][:, :, None])
            )

            q_ap2[isl, jsl, :] = (
                q[isl, jsl, :, l]
                + gradq[isl, jsl, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AIJ, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR][:, :, None])
                + gradq[isl, jsl, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AIJ, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR][:, :, None])
                + gradq[isl, jsl, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AIJ, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR][:, :, None])
            )

            q_am2[isl, jsl, :] = (
                q[isl_p1, jsl_p1, :, l]
                + gradq[isl_p1, jsl_p1, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AIJ, XDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, XDIR][:, :, None])
                + gradq[isl_p1, jsl_p1, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AIJ, YDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, YDIR][:, :, None])
                + gradq[isl_p1, jsl_p1, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AIJ, ZDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, ZDIR][:, :, None])
            )

            q_ap3[isl, jsl, :] = (
                q[isl, jsl, :, l]
                + gradq[isl, jsl, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AJ, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR][:, :, None])
                + gradq[isl, jsl, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AJ, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR][:, :, None])
                + gradq[isl, jsl, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AJ, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR][:, :, None])
            )

            q_am3[isl, jsl, :] = (
                q[isl, jsl_p1, :, l]
                + gradq[isl, jsl_p1, :, l, XDIR] * (grd_xc[isl, jsl, :, l, AJ, XDIR] - grd.GRD_x[isl, jsl_p1, K0, l, XDIR][:, :, None])
                + gradq[isl, jsl_p1, :, l, YDIR] * (grd_xc[isl, jsl, :, l, AJ, YDIR] - grd.GRD_x[isl, jsl_p1, K0, l, YDIR][:, :, None])
                + gradq[isl, jsl_p1, :, l, ZDIR] * (grd_xc[isl, jsl, :, l, AJ, ZDIR] - grd.GRD_x[isl, jsl_p1, K0, l, ZDIR][:, :, None])
            )

            q_ap4[isls, jsls, :] = (
                q[isls_m1, jsls, :, l]
                + gradq[isls_m1, jsls, :, l, XDIR] * (grd_xc[isls_m1, jsls, :, l, AI, XDIR] - grd.GRD_x[isls_m1, jsls, K0, l, XDIR][:, :, None])
                + gradq[isls_m1, jsls, :, l, YDIR] * (grd_xc[isls_m1, jsls, :, l, AI, YDIR] - grd.GRD_x[isls_m1, jsls, K0, l, YDIR][:, :, None])
                + gradq[isls_m1, jsls, :, l, ZDIR] * (grd_xc[isls_m1, jsls, :, l, AI, ZDIR] - grd.GRD_x[isls_m1, jsls, K0, l, ZDIR][:, :, None])
            )

            q_am4[isls, jsls, :] = (
                q[isls, jsls, :, l]
                + gradq[isls, jsls, :, l, XDIR] * (grd_xc[isls_m1, jsls, :, l, AI, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR][:, :, None])
                + gradq[isls, jsls, :, l, YDIR] * (grd_xc[isls_m1, jsls, :, l, AI, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR][:, :, None])
                + gradq[isls, jsls, :, l, ZDIR] * (grd_xc[isls_m1, jsls, :, l, AI, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR][:, :, None])
            )

            q_ap5[isls, jsls, :] = (
                q[isls_m1, jsls_m1, :, l]
                + gradq[isls_m1, jsls_m1, :, l, XDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, XDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, XDIR][:, :, None])
                + gradq[isls_m1, jsls_m1, :, l, YDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, YDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, YDIR][:, :, None])
                + gradq[isls_m1, jsls_m1, :, l, ZDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, ZDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, ZDIR][:, :, None])
            )

            q_am5[isls, jsls, :] = (
                q[isls, jsls, :, l]
                + gradq[isls, jsls, :, l, XDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR][:, :, None])
                + gradq[isls, jsls, :, l, YDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR][:, :, None])
                + gradq[isls, jsls, :, l, ZDIR] * (grd_xc[isls_m1, jsls_m1, :, l, AIJ, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR][:, :, None])
            )

            q_ap6[isls, jsls, :] = (
                q[isls, jsls_m1, :, l]
                + gradq[isls, jsls_m1, :, l, XDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, XDIR] - grd.GRD_x[isls, jsls_m1, K0, l, XDIR][:, :, None])
                + gradq[isls, jsls_m1, :, l, YDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, YDIR] - grd.GRD_x[isls, jsls_m1, K0, l, YDIR][:, :, None])
                + gradq[isls, jsls_m1, :, l, ZDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, ZDIR] - grd.GRD_x[isls, jsls_m1, K0, l, ZDIR][:, :, None])
            )

            q_am6[isls, jsls, :] = (
                q[isls, jsls, :, l]
                + gradq[isls, jsls, :, l, XDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR][:, :, None])
                + gradq[isls, jsls, :, l, YDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR][:, :, None])
                + gradq[isls, jsls, :, l, ZDIR] * (grd_xc[isls, jsls_m1, :, l, AJ, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR][:, :, None])
            )

            q_a[isl, jsl, :, l, 0] = cmask[isl, jsl, :, l, 0] * q_am1[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 0]) * q_ap1[isl, jsl, :]
            q_a[isl, jsl, :, l, 1] = cmask[isl, jsl, :, l, 1] * q_am2[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 1]) * q_ap2[isl, jsl, :]
            q_a[isl, jsl, :, l, 2] = cmask[isl, jsl, :, l, 2] * q_am3[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 2]) * q_ap3[isl, jsl, :]
            q_a[isl, jsl, :, l, 3] = cmask[isl, jsl, :, l, 3] * q_am4[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 3]) * q_ap4[isl, jsl, :]
            q_a[isl, jsl, :, l, 4] = cmask[isl, jsl, :, l, 4] * q_am5[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 4]) * q_ap5[isl, jsl, :]
            q_a[isl, jsl, :, l, 5] = cmask[isl, jsl, :, l, 5] * q_am6[isl, jsl, :] + (rdtype(1.0) - cmask[isl, jsl, :, l, 5]) * q_ap6[isl, jsl, :]
        # end loop l

        if adm.ADM_have_pl:
            # Unit 4a: device POLE remap (pentagon). Build q_a_pl on device via the
            # compute_horizontal_remap_pl kernel (mirror of the host loop below) and
            # drain the v = gmin..gmax rows back to host (host limiter/apply still read
            # q_a_pl). Bit-exact: the kernel reproduces the host arithmetic. Gate
            # PYNICAM_RESIDENT_HADV_REMAP_PL (default OFF); asarray fallback when off.
            _remap_pl = (bk.type == "jax") and bk.resident()
            if _remap_pl:
                xp = bk.xp
                if getattr(self, "_remap_pl_kernel", None) is None:
                    self._remap_pl_kernel = bk.maybe_jit(
                        compute_horizontal_remap_pl, static_argnames=("cfg", "xp"))
                _rcp = bk.device_consts(self, "remap_gx_pl",
                                        lambda: {"gx_pl": xp.asarray(grd.GRD_x_pl[:, K0, :, :])})
                _cfgp = RemapCfgPl(n=adm.ADM_gslf_pl, gmin=adm.ADM_gmin_pl,
                                   gmax=adm.ADM_gmax_pl, XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR)
                _gqp = _gradq_pl_d if _gradq_pl_d is not None else xp.asarray(gradq_pl)
                _qp_in  = q_pl_d      if q_pl_d      is not None else xp.asarray(q_pl)       # 4c-1
                _gxp_in = grd_xc_pl_d if grd_xc_pl_d is not None else xp.asarray(grd_xc_pl)  # 4c-1
                _cmp_in = cmask_pl_d  if cmask_pl_d  is not None else xp.asarray(cmask_pl)   # 4c-1
                _qap = self._remap_pl_kernel(
                    _qp_in, _gqp, _gxp_in, _cmp_in,
                    _rcp["gx_pl"], cfg=_cfgp, xp=xp)
                self._qa_pl_remap_d = _qap   # 4c-3b: device q_a for the limiter/flux apply
                if not qa_resident_pl:       # 4c-6: host q_a_pl dead when the device q_a is carried
                    _gp0, _gp1 = adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1
                    q_a_pl[_gp0:_gp1, :, :] = bk.to_numpy(_qap)[_gp0:_gp1, :, :]
            else:
                n = adm.ADM_gslf_pl

                # Vectorised over k (geometry at K0 is k-independent; scalars
                # broadcast over the k-axis). Bit-identical to the per-k loop.
                for l in range(adm.ADM_lall_pl):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        q_ap = (
                            q_pl[n, :, l]
                            + gradq_pl[n, :, l, XDIR] * (grd_xc_pl[v, :, l, XDIR] - grd.GRD_x_pl[n, K0, l, XDIR])
                            + gradq_pl[n, :, l, YDIR] * (grd_xc_pl[v, :, l, YDIR] - grd.GRD_x_pl[n, K0, l, YDIR])
                            + gradq_pl[n, :, l, ZDIR] * (grd_xc_pl[v, :, l, ZDIR] - grd.GRD_x_pl[n, K0, l, ZDIR])
                        )

                        q_am = (
                            q_pl[v, :, l]
                            + gradq_pl[v, :, l, XDIR] * (grd_xc_pl[v, :, l, XDIR] - grd.GRD_x_pl[v, K0, l, XDIR])
                            + gradq_pl[v, :, l, YDIR] * (grd_xc_pl[v, :, l, YDIR] - grd.GRD_x_pl[v, K0, l, YDIR])
                            + gradq_pl[v, :, l, ZDIR] * (grd_xc_pl[v, :, l, ZDIR] - grd.GRD_x_pl[v, K0, l, ZDIR])
                        )

                        q_a_pl[v, :, l] = (
                            cmask_pl[v, :, l] * q_am + (rdtype(1.0) - cmask_pl[v, :, l]) * q_ap
                        )
                    # end loop v
                # end loop l
        # endif

        prf.PROF_rapend('____horizontal_adv_remap',2)

        return
        


    def vertical_limiter_thuburn_fast_maybeok(self,
        q_h, q_h_pl, q, q_pl, d, d_pl, ck, ck_pl, 
        cnst, rdtype
    ):
        prf.PROF_rapstart('_____vertical_adv_limiter', 2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        gall_pl = adm.ADM_gall_pl

        EPS = cnst.CONST_EPS
        BIG = cnst.CONST_HUGE
        UNDEF = cnst.CONST_UNDEF
        ONE = rdtype(1.0)
        HALF = rdtype(0.5)

        isl = slice(0, iall)
        jsl = slice(0, jall)

        # Allocate once
        Qout_min_km1 = np.full((iall, jall), UNDEF, dtype=rdtype)
        Qout_max_km1 = np.full((iall, jall), UNDEF, dtype=rdtype)
        Qout_min_pl =np.full(adm.ADM_shape_pl, UNDEF)
        Qout_max_pl =np.full(adm.ADM_shape_pl, UNDEF)

        for l in range(lall):
            # Preload slices for efficiency
            q_slice = q[isl, jsl, :, l]  # (i,j,k)
            d_slice = d[isl, jsl, :, l]
            ck_slice = ck[isl, jsl, :, l, :]  # (i,j,k,2)

            # k = kmin separately
            k = kmin   # 1 in p

            inflagL = HALF - np.copysign(HALF, ck_slice[:, :, k, 0])
            inflagU = HALF + np.copysign(HALF, ck_slice[:, :, k+1, 0])

            q_center = q_slice[:, :, k]
            q_below  = q_slice[:, :, k-1]
            q_above  = q_slice[:, :, k+1]

            q_minL = np.minimum(q_center, q_below)
            q_minU = np.minimum(q_center, q_above)
            q_maxL = np.maximum(q_center, q_below)
            q_maxU = np.maximum(q_center, q_above)

            Qin_minL = inflagL * q_minL + (ONE - inflagL) * BIG
            Qin_minU = inflagU * q_minU + (ONE - inflagU) * BIG
            Qin_maxL = inflagL * q_maxL + (ONE - inflagL) * -BIG
            Qin_maxU = inflagU * q_maxU + (ONE - inflagU) * -BIG

            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

            # RC-40 fix (host kmin-peeling): 2nd-Courant term at the LOCAL level k, matching
            # the main loop below, the device kernel (verticallimiter.py), main's 1e3a555, and
            # the Fortran (mod_src_tracer.f90:1925-1933). The k+1 was carried from inflagU
            # (which legitimately reads k+1,0). JW no-op (no near-surface vertical motion).
            Cin  = inflagL * ck_slice[:, :, k, 0] + inflagU * ck_slice[:, :, k, 1]
            Cout = (ONE - inflagL) * ck_slice[:, :, k, 0] + (ONE - inflagU) * ck_slice[:, :, k, 1]

            CQin_min = inflagL * ck_slice[:, :, k, 0] * Qin_minL + inflagU * ck_slice[:, :, k, 1] * Qin_minU
            CQin_max = inflagL * ck_slice[:, :, k, 0] * Qin_maxL + inflagU * ck_slice[:, :, k, 1] * Qin_maxU

            zerosw = HALF - np.copysign(HALF, np.abs(Cout) - EPS)

            Cout_safe = Cout + zerosw
            nonzero_factor = (ONE - zerosw)

            Qout_min_k = ((q_center - qnext_max) + qnext_max * (Cin + Cout - d_slice[:, :, k]) - CQin_max) / Cout_safe * nonzero_factor + q_center * zerosw
            Qout_max_k = ((q_center - qnext_min) + qnext_min * (Cin + Cout - d_slice[:, :, k]) - CQin_min) / Cout_safe * nonzero_factor + q_center * zerosw

            # Store for kmin
            Qout_min_km1[:, :] = Qout_min_k
            Qout_max_km1[:, :] = Qout_max_k

            # Loop kmin+1 to kmax
            for k in range(kmin+1, kmax+1):

                inflagL = HALF - np.copysign(HALF, ck_slice[:, :, k, 0])
                inflagU = HALF + np.copysign(HALF, ck_slice[:, :, k, 1])

                q_center = q_slice[:, :, k]
                q_below  = q_slice[:, :, k-1]
                q_above  = q_slice[:, :, k+1]

                q_minL = np.minimum(q_center, q_below)
                q_minU = np.minimum(q_center, q_above)
                q_maxL = np.maximum(q_center, q_below)
                q_maxU = np.maximum(q_center, q_above)

                Qin_minL = inflagL * q_minL + (ONE - inflagL) * BIG
                Qin_minU = inflagU * q_minU + (ONE - inflagU) * BIG
                Qin_maxL = inflagL * q_maxL + (ONE - inflagL) * -BIG
                Qin_maxU = inflagU * q_maxU + (ONE - inflagU) * -BIG

                qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
                qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

                Cin  = inflagL * ck_slice[:, :, k, 0] + inflagU * ck_slice[:, :, k, 1]
                Cout = (ONE - inflagL) * ck_slice[:, :, k, 0] + (ONE - inflagU) * ck_slice[:, :, k, 1]

                CQin_min = inflagL * ck_slice[:, :, k, 0] * Qin_minL + inflagU * ck_slice[:, :, k, 1] * Qin_minU
                CQin_max = inflagL * ck_slice[:, :, k, 0] * Qin_maxL + inflagU * ck_slice[:, :, k, 1] * Qin_maxU

                zerosw = HALF - np.copysign(HALF, np.abs(Cout) - EPS)

                Cout_safe = Cout + zerosw
                nonzero_factor = (ONE - zerosw)

                qout_min = ((q_center - qnext_max) + qnext_max * (Cin + Cout - d_slice[:, :, k]) - CQin_max) / Cout_safe * nonzero_factor + q_center * zerosw
                qout_max = ((q_center - qnext_min) + qnext_min * (Cin + Cout - d_slice[:, :, k]) - CQin_min) / Cout_safe * nonzero_factor + q_center * zerosw

                # Manual clipping
                clipped_lower = np.minimum(np.maximum(q_h[isl, jsl, k, l], Qout_min_km1), Qout_max_km1)
                clipped_upper = np.minimum(np.maximum(q_h[isl, jsl, k, l], qout_min), qout_max)

                q_h[isl, jsl, k, l] = inflagL * clipped_lower + (ONE - inflagL) * clipped_upper

                # Update km1 buffers
                Qout_min_km1[:, :] = qout_min
                Qout_max_km1[:, :] = qout_max

        if adm.ADM_have_pl:
            isl_pl = slice(0, gall_pl)

            qgkl = q_pl[isl_pl, kmin:kmax+1, :]  # (gall_pl, k, l)
            qkm1 = q_pl[isl_pl, kmin-1:kmax, :]  # k-1
            qkp1 = q_pl[isl_pl, kmin+1:kmax+2, :]  # k+1

            ck0 = ck_pl[isl_pl, kmin:kmax+1, :, 0]  # (gall_pl, k, l)
            ck1 = ck_pl[isl_pl, kmin:kmax+1, :, 1]  # (gall_pl, k, l)

            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck0)
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[isl_pl, kmin+1:kmax+2, :, 0])

            # Precompute min/max
            q_minL = np.minimum(qgkl, qkm1)
            q_minU = np.minimum(qgkl, qkp1)
            q_maxL = np.maximum(qgkl, qkm1)
            q_maxU = np.maximum(qgkl, qkp1)

            # Fuse inflag application (no np.where)
            Qin_minL = inflagL * q_minL + (rdtype(1.0) - inflagL) * BIG
            Qin_minU = inflagU * q_minU + (rdtype(1.0) - inflagU) * BIG
            Qin_maxL = inflagL * q_maxL + (rdtype(1.0) - inflagL) * -BIG
            Qin_maxU = inflagU * q_maxU + (rdtype(1.0) - inflagU) * -BIG

            # Minimize and maximize together
            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            # Fluxes
            Cin  = inflagL * ck0 + inflagU * ck1
            Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            Cout_safe = Cout + zerosw
            nonzero_factor = rdtype(1.0) - zerosw

            d_slice_pl = d_pl[isl_pl, kmin:kmax+1, :]

            # Final limiter formulas
            Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_slice_pl) - CQin_max) \
                        / Cout_safe * nonzero_factor + qgkl * zerosw

            Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_slice_pl) - CQin_min) \
                        / Cout_safe * nonzero_factor + qgkl * zerosw

            # Save output
            Qout_min_pl[isl_pl, kmin:kmax+1, :] = Qout_min
            Qout_max_pl[isl_pl, kmin:kmax+1, :] = Qout_max


            prf.PROF_rapend('_____vertical_adv_limiter',2)

        return


    def vertical_limiter_thuburn(self,
            q_h, q_h_pl,    # [INOUT]
            q, q_pl,        # [IN]
            d, d_pl,        # [IN]
            ck, ck_pl,       # [IN]
            cnst, rdtype,
            resident=False,  # RES-TP-1b: q_h/q/d/ck are device arrays; the limited
                             # regular q_h is returned on device (pole stays host).
            skip_pole=False, # RC-41r: caller limits the pole on device (reshape-reuse);
                             # skip the host pole section (its q_h_pl output is unread).
    ):

        prf.PROF_rapstart('_____vertical_adv_limiter',2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        Qout_min_km1=np.full(adm.ADM_shape[:2], cnst.CONST_UNDEF)
        Qout_max_km1=np.full(adm.ADM_shape[:2], cnst.CONST_UNDEF)
        Qout_min_pl =np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        Qout_max_pl =np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        EPS  = cnst.CONST_EPS
        BIG  = cnst.CONST_HUGE

        #print("BIG", BIG, cnst.CONST_HUGE)
        #prc.prc_mpifinish(std.io_l, std.fname_log)
        #import sys
        #sys.exit(0)

        # (A) fused jax kernel for the REGULAR vertical limiter (qout + apply),
        # gated PYNICAM_FUSE_VLIMITER (default off). No COMM (vertical). The pole
        # (_pl) section below stays on the host path. When on, the per-l numpy loop
        # is skipped (range(0)).
        _fuse_vlim = (bk.type == "jax") and os.environ.get("PYNICAM_FUSE_VLIMITER", "1") != "0"
        _qh_out_d = None   # RES-TP-1b: device q_h (regular) returned when resident
        if _fuse_vlim:
            if getattr(self, "_vlim_cfg", None) is None:
                self._vlim_cfg = VLimiterCfg(
                    iall=iall, jall=jall, kall=kall, lall=lall,
                    kmin=kmin, kmax=kmax, BIG=float(BIG), EPS=float(EPS))
                self._vlim_kernel = bk.maybe_jit(compute_vertical_limiter, static_argnames=("cfg", "xp"))
            _xpL = bk.xp
            if resident:
                # RES-TP-1b: inputs already on device; keep the limited q_h on device
                # (skip the asarray upload + to_numpy drain). Bit-identical to the
                # host path (asarray(to_numpy(.)) is a pure f64 copy). The pole
                # (_pl) section below still runs on the host with host q_h_pl.
                _qh_out_d = self._vlim_kernel(
                    q_h, q, d, ck, cfg=self._vlim_cfg, xp=_xpL)
            else:
                q_h[:, :, :, :] = bk.to_numpy(self._vlim_kernel(
                    _xpL.asarray(q_h), _xpL.asarray(q), _xpL.asarray(d), _xpL.asarray(ck),
                    cfg=self._vlim_cfg, xp=_xpL))

        for l in (range(lall) if not _fuse_vlim else range(0)):
            k = kmin  # fixed slice   # kmin = 1 in python, 2 in fortran
            # RC-40: the kmin peeling's upper (2nd-Courant) term uses ck[k,1] like the
            # Fortran (and the main loop / pole). Was ck[k+1,1] -- a level slip carried
            # from inflagU's legitimate ck[k+1,0]. (inflagU @2047 stays at k+1.)
            _ku = k

            # Define slices
            isl = slice(0, iall)
            jsl = slice(0, jall)

            # Incoming flux flags
            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck[isl, jsl, k, l, 0])
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])

            # Compute bounds with BIG trick

            Qin_minL = np.where(inflagL == rdtype(1.0),
                                np.minimum(q[isl, jsl, k, l], q[isl, jsl, k - 1, l]),
                                BIG)

            Qin_minU = np.where(inflagU == rdtype(1.0),
                                np.minimum(q[isl, jsl, k, l], q[isl, jsl, k + 1, l]),
                                BIG)

            Qin_maxL = np.where(inflagL == rdtype(1.0),
                                np.maximum(q[isl, jsl, k, l], q[isl, jsl, k - 1, l]),
                                -BIG)

            Qin_maxU = np.where(inflagU == rdtype(1.0),
                                np.maximum(q[isl, jsl, k, l], q[isl, jsl, k + 1, l]),
                                -BIG)


            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q[isl, jsl, k, l]])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q[isl, jsl, k, l]])

            # Cin = inflagL * ck[isl, jsl, k, l, 0] + inflagU * ck[isl, jsl, k + 1, l, 0]
            # Cout = (rdtype(1.0) - inflagL) * ck[isl, jsl, k, l, 0] + (rdtype(1.0) - inflagU) * ck[isl, jsl, k + 1, l, 0]

            Cin = inflagL * ck[isl, jsl, k, l, 0] + inflagU * ck[isl, jsl, _ku, l, 1]
            Cout = (rdtype(1.0) - inflagL) * ck[isl, jsl, k, l, 0] + (rdtype(1.0) - inflagU) * ck[isl, jsl, _ku, l, 1]


            # if l==1:
            #     with open(std.fname_log, 'a') as log_file:  
            #         print("Cin", Cin[6, 5], "Cout", Cout[6,5], file=log_file)   # Cout -0.005646669245168996  !!! 
            #         print("inflagL", inflagL[6, 5], file=log_file) 
            #         print("inflagU", inflagU[6, 5], file=log_file) 
            #         print("ck k+1", ck[6, 5, k+1, l, :], file=log_file)  
            #         print("ck k", ck[6, 5, k, l, :], file=log_file)

            # CQin_min = inflagL * ck[isl, jsl, k, l, 0] * Qin_minL + inflagU * ck[isl, jsl, k + 1, l, 0] * Qin_minU
            # CQin_max = inflagL * ck[isl, jsl, k, l, 0] * Qin_maxL + inflagU * ck[isl, jsl, k + 1, l, 0] * Qin_maxU

            CQin_min = inflagL * ck[isl, jsl, k, l, 0] * Qin_minL + inflagU * ck[isl, jsl, _ku, l, 1] * Qin_minU
            CQin_max = inflagL * ck[isl, jsl, k, l, 0] * Qin_maxL + inflagU * ck[isl, jsl, _ku, l, 1] * Qin_maxU


            #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout) - EPS)
            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            # Output limits
            Qout_min_k = (
                ((q[isl, jsl, k, l] - qnext_max) + qnext_max * (Cin + Cout - d[isl, jsl, k, l]) - CQin_max)
                / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q[isl, jsl, k, l] * zerosw
            )

            Qout_max_k = (
                ((q[isl, jsl, k, l] - qnext_min) + qnext_min * (Cin + Cout - d[isl, jsl, k, l]) - CQin_min)
                / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q[isl, jsl, k, l] * zerosw
            )
            
            # if l==1:
            #     with open(std.fname_log, 'a') as log_file:
            #         print("Qout_max_k", Qout_max_k[6, 5], file=log_file) #Qout_max_k  -159.38599569471765
            #         print("q", q[6, 5, k, l], "qnext_min", qnext_min[6, 5], file=log_file)    
            #         print("Cin", Cin[6, 5], "Cout", Cout[6,5], file=log_file)   # Cout -0.005646669245168996  !!! 
            #         print("zerosw", zerosw[6, 5], file=log_file)
            #         print("CQin_min", CQin_min[6, 5], file=log_file)

            # Store to arrays
            Qout_min_km1[isl, jsl] = Qout_min_k
            Qout_max_km1[isl, jsl] = Qout_max_k     #Qout_max_km1 -159.38599569471765



            for k in range(kmin + 1, kmax + 1):
                # Precompute commonly used variables
                #inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck[isl, jsl, k, l, 0])  # ck[..., 1] in Fortran = ck[..., 0] in Python
                #inflagU = rdtype(0.5) + np.sign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])
                inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck[isl, jsl, k, l, 0])  # ck[..., 1] in Fortran = ck[..., 0] in Python
                inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])

                q_center = q[isl, jsl, k, l]
                q_below  = q[isl, jsl, k - 1, l]
                q_above  = q[isl, jsl, k + 1, l]


                Qin_minL = np.where(inflagL == rdtype(1.0),
                                    np.minimum(q_center, q_below),
                                    BIG)

                Qin_minU = np.where(inflagU == rdtype(1.0),
                                    np.minimum(q_center, q_above),
                                    BIG)

                Qin_maxL = np.where(inflagL == rdtype(1.0),
                                    np.maximum(q_center, q_below),
                                    -BIG)

                Qin_maxU = np.where(inflagU == rdtype(1.0),
                                    np.maximum(q_center, q_above),
                                    -BIG)

                qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
                qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

                ck1 = ck[isl, jsl, k, l, 0]
                ck2 = ck[isl, jsl, k, l, 1]

                Cin = inflagL * ck1 + inflagU * ck2
                Cout = (rdtype(1.0) - inflagL) * ck1 + (rdtype(1.0) - inflagU) * ck2

                CQin_min = inflagL * ck1 * Qin_minL + inflagU * ck2 * Qin_minU
                CQin_max = inflagL * ck1 * Qin_maxL + inflagU * ck2 * Qin_maxU

                #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout) - EPS)
                zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

                qout_min_k = (
                    ((q_center - qnext_max) + qnext_max * (Cin + Cout - d[isl, jsl, k, l]) - CQin_max)
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q_center * zerosw
                )

                qout_max_k = (
                    ((q_center - qnext_min) + qnext_min * (Cin + Cout - d[isl, jsl, k, l]) - CQin_min)
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q_center * zerosw
                )

                # if k==2 and l==1:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("q_h", q_h[6, 5, k, l], "Qout_min_km1", Qout_min_km1[6, 5], "Qout_max_km1", Qout_max_km1[6, 5], file=log_file)    #Qout_max_km1 -159.38599569471765
                #         print("qout_min_k", qout_min_k[6, 5], "qout_max_k", qout_max_k[6,5], file=log_file)
                #         print("q_h", q_h[6, 5, k, l], file=log_file)
                #         print("inflagL", inflagL[6, 5], file=log_file)

                # Clip q_h using inflagL
                q_h[isl, jsl, k, l] = (
                    inflagL * np.clip(q_h[isl, jsl, k, l], Qout_min_km1[isl, jsl], Qout_max_km1[isl, jsl]) +
                    (rdtype(1.0) - inflagL) * np.clip(q_h[isl, jsl, k, l], qout_min_k, qout_max_k)
                )

                # Update for next level
                Qout_min_km1[isl, jsl] = qout_min_k
                Qout_max_km1[isl, jsl] = qout_max_k
            # end loop k
        # end loop l

        if adm.ADM_have_pl and not skip_pole:   # RC-41r: pole limited on device by the caller

            qgkl = q_pl[:, kmin:kmax+1, :]  # shape (g, k, l)
            qkm1 = q_pl[:, kmin-1:kmax, :]  # k-1
            qkp1 = q_pl[:, kmin+1:kmax+2, :]  # k+1

            ck0 = ck_pl[:, kmin:kmax+1, :, 0]
            ck1 = ck_pl[:, kmin:kmax+1, :, 1]

            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck0)
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[:, kmin+1:kmax+2, :, 0])
           
            Qin_minL = np.where(inflagL == rdtype(1.0), np.minimum(qgkl, qkm1), BIG)
            Qin_minU = np.where(inflagU == rdtype(1.0), np.minimum(qgkl, qkp1), BIG)
            Qin_maxL = np.where(inflagL == rdtype(1.0), np.maximum(qgkl, qkm1), -BIG)
            Qin_maxU = np.where(inflagU == rdtype(1.0), np.maximum(qgkl, qkp1), -BIG)

            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            Cin  = inflagL * ck0 + inflagU * ck1
            Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[:, kmin:kmax+1, :]) - CQin_max) \
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[:, kmin:kmax+1, :]) - CQin_min) \
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            Qout_min_pl[:, kmin:kmax+1, :] = Qout_min
            Qout_max_pl[:, kmin:kmax+1, :] = Qout_max


            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck_pl[g, k, l, 0])
                        q_h_pl[g, k, l] = (
                            inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1, l], Qout_max_pl[g, k - 1, l])
                            + (rdtype(1.0) - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k, l], Qout_max_pl[g, k, l])
                        )

            # for l in range(lall_pl):
            #     for k in range(kmin, kmax + 1):
            #         for g in range(gall_pl):
            #             #inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck_pl[g, k, l, 0])
            #             inflagL = np.copysign(rdtype(0.5), ck[g, k, l, 1])
            #             #inflagU = rdtype(0.5) + np.sign(rdtype(0.5), ck_pl[g, k + 1, l, 0])
            #             inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[g, k + 1, l, 0])

            #             qgkl = q_pl[g, k, l]
            #             Qin_minL = min(qgkl, q_pl[g, k - 1, l]) + (rdtype(1.0) - inflagL) * BIG
            #             Qin_minU = min(qgkl, q_pl[g, k + 1, l]) + (rdtype(1.0) - inflagU) * BIG
            #             Qin_maxL = max(qgkl, q_pl[g, k - 1, l]) - (rdtype(1.0) - inflagL) * BIG
            #             Qin_maxU = max(qgkl, q_pl[g, k + 1, l]) - (rdtype(1.0) - inflagU) * BIG

            #             qnext_min = np.minimum(np.minimum(Qin_minL, Qin_minU), qgkl)
            #             qnext_max = np.maximum(np.maximum(Qin_maxL, Qin_maxU), qgkl)
            #             #qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            #             #qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            #             ck0 = ck_pl[g, k, l, 0]
            #             ck1 = ck_pl[g, k, l, 1]
            #             Cin  = inflagL * ck0 + inflagU * ck1
            #             Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            #             CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            #             CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            #             zerosw = rdtype(0.5) - np.sign(rdtype(0.5), abs(Cout) - EPS)

            #             #Qout_min_pl[g, k] = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw
            #             #Qout_max_pl[g, k] = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw


            #             Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw
            #             Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            #             print(Qout_min.shape, Qout_max.shape) #, Qout_min.dtype, Qout_max.dtype)
            #             print(qgkl.shape, qnext_max.shape, qnext_min.shape) #, qgkl.dtype, qnext_max.dtype, qnext_min.dtype)
            #             print(Cout.shape, zerosw.shape) #, Cout.dtype, zerosw.dtype)
            #             print(Cin.shape, d_pl[g, k, l].shape) #, Cin.dtype, d_pl[g, k, l].dtype)
            #             print(CQin_min.shape, CQin_max.shape) #, CQin_min.dtype, CQin_max.dtype)
            #             Qout_min_pl[g, k] = Qout_min
            #             Qout_max_pl[g, k] = Qout_max
                        

            #         # end loop g
            #     # end loop k

            #     for k in range(kmin + 1, kmax + 1):
            #         for g in range(gall_pl):
            #             inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck_pl[g, k, l, 0])
            #             q_h_pl[g, k, l] = (
            #                 inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1], Qout_max_pl[g, k - 1])
            #                 + (rdtype(1.0) - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k], Qout_max_pl[g, k])
            #             )
                    # end loop g
                # end loop k
            # end loop l
        # end if 

#####
        # if adm.ADM_have_pl:
        #     npl = adm.ADM_gslf_pl
        #     for l in range(lall_pl):
        #         for k in range(kmin, kmax + 1):
        #             for g in range(gall_pl):
        #                 inflagL = 0.5 - np.sign(0.5, ck_pl[g, k, l, 0])
        #                 inflagU = 0.5 + np.sign(0.5, ck_pl[g, k + 1, l, 0])

        #                 qgkl = q_pl[g, k, l]
        #                 Qin_minL = min(qgkl, q_pl[g, k - 1, l]) + (1.0 - inflagL) * BIG
        #                 Qin_minU = min(qgkl, q_pl[g, k + 1, l]) + (1.0 - inflagU) * BIG
        #                 Qin_maxL = max(qgkl, q_pl[g, k - 1, l]) - (1.0 - inflagL) * BIG
        #                 Qin_maxU = max(qgkl, q_pl[g, k + 1, l]) - (1.0 - inflagU) * BIG

        #                 qnext_min = min(Qin_minL, Qin_minU, qgkl)
        #                 qnext_max = max(Qin_maxL, Qin_maxU, qgkl)

        #                 ck1 = ck_pl[g, k, l, 0]
        #                 ck2 = ck_pl[g, k, l, 1]
        #                 Cin  = inflagL * ck1 + inflagU * ck2
        #                 Cout = (1.0 - inflagL) * ck1 + (1.0 - inflagU) * ck2

        #                 CQin_min = inflagL * ck1 * Qin_minL + inflagU * ck2 * Qin_minU
        #                 CQin_max = inflagL * ck1 * Qin_maxL + inflagU * ck2 * Qin_maxU

        #                 zerosw = 0.5 - np.sign(0.5, abs(Cout) - EPS)

        #                 Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (1.0 - zerosw) + qgkl * zerosw
        #                 Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (1.0 - zerosw) + qgkl * zerosw

        #                 Qout_min_pl[g, k] = Qout_min
        #                 Qout_max_pl[g, k] = Qout_max
        #             # end loop g
        #         # end loop k

        #         for k in range(kmin + 1, kmax + 1):
        #             for g in range(gall_pl):
        #                 inflagL = 0.5 - np.sign(0.5, ck_pl[g, k, l, 0])
        #                 q_h_pl[g, k, l] = (
        #                     inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1], Qout_max_pl[g, k - 1])
        #                     + (1.0 - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k], Qout_max_pl[g, k])
        #                 )
        #             # end loop g
        #         # end loop k
        #     # end loop l
        # # end if 

        ###

        prf.PROF_rapend('_____vertical_adv_limiter',2)

        # RES-TP-1b: device q_h when resident (caller carries it to the update);
        # None otherwise (q_h was written in place on the host).
        return _qh_out_d

    #> Miura(2004)'s scheme with Thuburn(1996) limiter
    def horizontal_limiter_thuburn(self,
        q_a,    q_a_pl,             # [INOUT]    
        q,      q_pl,               # [IN]
        d,      d_pl,               # [IN]
        ch,     ch_pl,              # [IN]
        cmask,  cmask_pl,           # [IN]
        cnst, comm, rdtype,
        q_pl_d=None, d_pl_d=None, ch_pl_d=None, cmask_pl_d=None,  # Unit 4c-1: device POLE inputs
        qa_resident_pl=False,                                    # 4c-6: device Qout/q_a carried
    ):

        prf.PROF_rapstart('_____horizontal_adv_limiter',2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        I_min = 0
        I_max = 1

        Qin    = np.full(adm.ADM_shape + (2, 6,),  cnst.CONST_UNDEF)
        Qin_pl = np.full(adm.ADM_shape_pl + (2, 2,),  cnst.CONST_UNDEF)
        Qout   = np.full(adm.ADM_shape + (2,),  cnst.CONST_UNDEF)
        Qout_pl= np.full(adm.ADM_shape_pl + (2,),  cnst.CONST_UNDEF)

        # Qin    = np.zeros(adm.ADM_shape + (2,6,), dtype=rdtype)    # set to zero to suppress Possible bug in this scheme 
        # Qin_pl = np.zeros(adm.ADM_shape_pl + (2,2,),  dtype=rdtype)
        # Qout   = np.zeros(adm.ADM_shape + (2,),  dtype=rdtype)
        # Qout_pl= np.zeros(adm.ADM_shape_pl + (2,),  dtype=rdtype)


        EPS  = cnst.CONST_EPS
        BIG  = cnst.CONST_HUGE

        # NOTE: the matching PROF_rapend is at the function's single exit (return)
        # below, so this timer spans the ACTUAL limiter work (the i,j loops + sgp
        # correction + Qout + pole), not just the allocations above. It previously
        # sat here -> the timer read ~0 and the real cost was silently attributed
        # to the parent ____horizontal_adv, which was misleading.

        ############  WORKS, and faster, but still has i and j loops and if #########
        # Section sub-timers (level 2 so they record; level>2 is dropped by PROF).
        # They tile the limiter body so they sum to _____horizontal_adv_limiter:
        #   qin -> qout(+sgp) -> qin_pl(pole) -> apply -> apply_pl(pole).
        prf.PROF_rapstart('______hlim_qin',2)
        _hlim_vec = os.environ.get("PYNICAM_HLIM_VEC", "0") != "0"
        _fuse_hlim = (bk.type == "jax") and os.environ.get("PYNICAM_FUSE_HLIMITER", "1") != "0"
        if _fuse_hlim:
            # Stage-3: REGULAR limiter as jax kernels, SPLIT around the Qout halo
            # exchange. Kernel A here builds qin+sgp+Qout and writes the host
            # Qin/Qout arrays; the existing comm.COMM_data_transfer(Qout,Qout_pl)
            # below halo-exchanges Qout; kernel B (apply section) reads the COMM'd
            # Qout. Pole _pl sections still run on host. (A single fused kernel
            # would apply against un-exchanged Qout -> non-monotone -> NaN.)
            if getattr(self, "_hlim_cfg", None) is None:
                self._hlim_cfg = HLimiterCfg(
                    iall=iall, jall=jall, lall=lall, I_min=I_min, I_max=I_max,
                    BIG=float(BIG), EPS=float(EPS),
                    have_sgp=tuple(bool(adm.ADM_have_sgp[_l]) for _l in range(lall)),
                )
                self._hlim_qout_k = bk.maybe_jit(compute_horizontal_limiter_qout, static_argnames=("cfg", "xp"))
                self._hlim_apply_k = bk.maybe_jit(compute_horizontal_limiter_apply, static_argnames=("cfg", "xp"))
            _xpL = bk.xp
            _Qin_d, _Qout_d = self._hlim_qout_k(
                _xpL.asarray(q), _xpL.asarray(d), _xpL.asarray(ch), _xpL.asarray(cmask),
                cfg=self._hlim_cfg, xp=_xpL,
            )
            if getattr(self, "_hadv_qa_resident", False):
                # Stage-4b: keep Qin/Qout on device; Qout halo-exchanged on device
                # below, kernel B reads them resident. No host round-trip of Qin
                # (the big one) or Qout.
                self._Qin_d = _Qin_d; self._Qout_d = _Qout_d
            else:
                Qin[:]  = bk.to_numpy(_Qin_d)
                Qout[:] = bk.to_numpy(_Qout_d)
        elif _hlim_vec:
            # Stage-1: vectorized regular Qin build (replaces the i,j Python loop;
            # bit-exact -- min/max associative+exact, q read-only, scatter targets
            # unique per (source,edge)). j==0 -> q[0,0], i==0 -> q[0,0] (pentagon).
            _si = slice(0, iall-1); _sj = slice(0, jall-1)
            _sip = slice(1, iall);  _sjp = slice(1, jall)
            _q0 = q[_si, _sj]; _q2 = q[_sip, _sj]; _q3 = q[_sip, _sjp]; _qjp1 = q[_si, _sjp]
            _q1 = np.empty_like(_q0); _q1[:, 1:] = q[_si, 0:jall-2]; _q1[:, 0] = q[0, 0]
            _q4 = np.empty_like(_q0); _q4[1:, :] = q[0:iall-2, _sj]; _q4[0, :] = q[0, 0]
            _mnAI  = np.minimum(np.minimum(_q0,_q1),  np.minimum(_q2,_q3))
            _mxAI  = np.maximum(np.maximum(_q0,_q1),  np.maximum(_q2,_q3))
            _mnAIJ = np.minimum(np.minimum(_q0,_q2),  np.minimum(_q3,_qjp1))
            _mxAIJ = np.maximum(np.maximum(_q0,_q2),  np.maximum(_q3,_qjp1))
            _mnAJ  = np.minimum(np.minimum(_q0,_q3),  np.minimum(_qjp1,_q4))
            _mxAJ  = np.maximum(np.maximum(_q0,_q3),  np.maximum(_qjp1,_q4))
            _cm = cmask[_si, _sj]
            for _m, (_qmn, _qmx) in enumerate(((_mnAI,_mxAI),(_mnAIJ,_mxAIJ),(_mnAJ,_mxAJ))):
                _c = _cm[:, :, :, :, _m]
                Qin[_si, _sj, :, :, I_min, _m] = _c*_qmn + (1.0 - _c)*BIG
                Qin[_si, _sj, :, :, I_max, _m] = _c*_qmx + (1.0 - _c)*(-BIG)
            _c0 = _cm[:, :, :, :, 0]; _c1 = _cm[:, :, :, :, 1]; _c2 = _cm[:, :, :, :, 2]
            Qin[_sip, _sj,  :, :, I_min, 3] = _c0*BIG    + (1.0 - _c0)*_mnAI
            Qin[_sip, _sj,  :, :, I_max, 3] = _c0*(-BIG) + (1.0 - _c0)*_mxAI
            Qin[_sip, _sjp, :, :, I_min, 4] = _c1*BIG    + (1.0 - _c1)*_mnAIJ
            Qin[_sip, _sjp, :, :, I_max, 4] = _c1*(-BIG) + (1.0 - _c1)*_mxAIJ
            Qin[_si,  _sjp, :, :, I_min, 5] = _c2*BIG    + (1.0 - _c2)*_mnAJ
            Qin[_si,  _sjp, :, :, I_max, 5] = _c2*(-BIG) + (1.0 - _c2)*_mxAJ
        else:
            for i in range(iall-1):
                for j in range(jall-1):
                    # Build the 4-point stencil at (i,j)
                    q0 = q[i,   j,   :, :]  # center
                    if j > 0:
                        q1 = q[i,   j-1, :, :]
                    else:
                        q1 = q[0,   0,   :, :]
                
                    q2 = q[i+1, j,   :, :]
                    q3 = q[i+1, j+1, :, :]

                    # For AI
                    q_min_AI  = np.minimum.reduce([q0, q1, q2, q3])
                    q_max_AI  = np.maximum.reduce([q0, q1, q2, q3])

                    # For AIJ (no special boundary handling)
                    q_min_AIJ = np.minimum.reduce([q0, q2, q3, q[i, j+1, :, :]])
                    q_max_AIJ = np.maximum.reduce([q0, q2, q3, q[i, j+1, :, :]])

                    # For AJ
                    if i > 0:
                        q4 = q[i-1, j, :, :]
                    else:
                        q4 = q[0, 0, :, :]
                    q_min_AJ = np.minimum.reduce([q0, q3, q[i, j+1, :, :], q4])
                    q_max_AJ = np.maximum.reduce([q0, q3, q[i, j+1, :, :], q4])

                    # Now fill Qin
                    for m, (qmin, qmax) in enumerate([(q_min_AI, q_max_AI), (q_min_AIJ, q_max_AIJ), (q_min_AJ, q_max_AJ)]):
                        Qin[i, j, :, :, I_min, m] = cmask[i, j, :, :, m] * qmin + (1.0 - cmask[i, j, :, :, m]) * BIG
                        Qin[i, j, :, :, I_max, m] = cmask[i, j, :, :, m] * qmax + (1.0 - cmask[i, j, :, :, m]) * (-BIG)

                    # For shifted points (neighbors)
                    # For AI and AIJ -> (i+1, j), (i+1, j+1)
                    Qin[i+1, j,   :, :, I_min, 3] = cmask[i, j, :, :, 0] * BIG + (1.0 - cmask[i, j, :, :, 0]) * q_min_AI
                    Qin[i+1, j,   :, :, I_max, 3] = cmask[i, j, :, :, 0] * (-BIG) + (1.0 - cmask[i, j, :, :, 0]) * q_max_AI

                    Qin[i+1, j+1, :, :, I_min, 4] = cmask[i, j, :, :, 1] * BIG + (1.0 - cmask[i, j, :, :, 1]) * q_min_AIJ
                    Qin[i+1, j+1, :, :, I_max, 4] = cmask[i, j, :, :, 1] * (-BIG) + (1.0 - cmask[i, j, :, :, 1]) * q_max_AIJ

                    # For AJ -> (i, j+1)
                    Qin[i,   j+1, :, :, I_min, 5] = cmask[i, j, :, :, 2] * BIG + (1.0 - cmask[i, j, :, :, 2]) * q_min_AJ
                    Qin[i,   j+1, :, :, I_max, 5] = cmask[i, j, :, :, 2] * (-BIG) + (1.0 - cmask[i, j, :, :, 2]) * q_max_AJ

        # ###########################



        # #### WORKS, but slow ###

        # for l in range(lall):
        #     for k in range(kall):

        #         for j in range(jall - 1):     # Python: 0 to jall-2
        #             for i in range(iall - 1):

        #                 # Handling boundaries (im1j, ijm1 logic is ignored as you seem to apply 1-clamping in the original)
        #                 if i > 0 and j > 0:
        #                     q_min_AI  = np.min([q[i, j, k, l], q[i, j-1, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_max_AI  = np.max([q[i, j, k, l], q[i, j-1, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_min_AIJ = np.min([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_max_AIJ = np.max([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_min_AJ  = np.min([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[i-1, j, k, l]])
        #                     q_max_AJ  = np.max([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[i-1, j, k, l]])
        #                 else:
        #                     q_min_AI  = np.min([q[i, j, k, l], q[0, 0, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_max_AI  = np.max([q[i, j, k, l], q[0, 0, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_min_AIJ = np.min([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_max_AIJ = np.max([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_min_AJ  = np.min([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[0, 0, k, l]])
        #                     q_max_AJ  = np.max([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[0, 0, k, l]])

        #                 # Now filling Qin array
        #                 Qin[i,   j,   k, l, I_min, 0] = cmask[i, j, k, l, 0] * q_min_AI  + (1.0 - cmask[i, j, k, l, 0]) * BIG
        #                 Qin[i+1, j,   k, l, I_min, 3] = cmask[i, j, k, l, 0] * BIG       + (1.0 - cmask[i, j, k, l, 0]) * q_min_AI
        #                 Qin[i,   j,   k, l, I_max, 0] = cmask[i, j, k, l, 0] * q_max_AI  + (1.0 - cmask[i, j, k, l, 0]) * (-BIG)
        #                 Qin[i+1, j,   k, l, I_max, 3] = cmask[i, j, k, l, 0] * (-BIG)    + (1.0 - cmask[i, j, k, l, 0]) * q_max_AI

        #                 Qin[i,   j,   k, l, I_min, 1] = cmask[i, j, k, l, 1] * q_min_AIJ + (1.0 - cmask[i, j, k, l, 1]) * BIG
        #                 Qin[i+1, j+1, k, l, I_min, 4] = cmask[i, j, k, l, 1] * BIG       + (1.0 - cmask[i, j, k, l, 1]) * q_min_AIJ
        #                 Qin[i,   j,   k, l, I_max, 1] = cmask[i, j, k, l, 1] * q_max_AIJ + (1.0 - cmask[i, j, k, l, 1]) * (-BIG)
        #                 Qin[i+1, j+1, k, l, I_max, 4] = cmask[i, j, k, l, 1] * (-BIG)    + (1.0 - cmask[i, j, k, l, 1]) * q_max_AIJ

        #                 Qin[i,   j,   k, l, I_min, 2] = cmask[i, j, k, l, 2] * q_min_AJ  + (1.0 - cmask[i, j, k, l, 2]) * BIG
        #                 Qin[i,   j+1, k, l, I_min, 5] = cmask[i, j, k, l, 2] * BIG       + (1.0 - cmask[i, j, k, l, 2]) * q_min_AJ
        #                 Qin[i,   j,   k, l, I_max, 2] = cmask[i, j, k, l, 2] * q_max_AJ  + (1.0 - cmask[i, j, k, l, 2]) * (-BIG)
        #                 Qin[i,   j+1, k, l, I_max, 5] = cmask[i, j, k, l, 2] * (-BIG)    + (1.0 - cmask[i, j, k, l, 2]) * q_max_AJ


        # ####################



#         for l in range(lall):
#             for k in range(kall):
#                 # Define slices for interior region
#                 # isl = slice(1, iall - 1)   # size 16
#                 # jsl = slice(1, jall - 1)
#                 # islp1 = slice(2, iall)
#                 # jslp1 = slice(2, jall)
#                 # islm1 = slice(0, iall - 2)
#                 # jslm1 = slice(0, jall - 2)

#                 isl = slice(1, iall - 1)   # size 16
#                 jsl = slice(1, jall - 1)
#                 islp1 = slice(2, iall)
#                 jslp1 = slice(2, jall)
#                 islm1 = slice(0, iall - 2)
#                 jslm1 = slice(0, jall - 2)


#                 # Local slices for broadcasting
# #                cm1 = rdtype(1.0) - cmask[isl, jsl, k, l]   # dimension???
#                 cm1 = rdtype(1.0) - cmask[isl, jsl, k, l, :]   # dimension???  

#                 # q_min and q_max for each stencil
#                 q_min_AI  = np.minimum.reduce([q[isl, jsl, k, l], q[isl, jslm1, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[isl, jsl, k, l], q[isl, jslm1, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[isl, jsl, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[isl, jsl, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[isl, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l], q[islm1, jsl, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[isl, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l], q[islm1, jsl, k, l]])

#                 # min/max indices

#                 Qin[isl,   jsl,   k, l, I_min, 0] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[islp1, jsl,   k, l, I_min, 3] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), BIG,       q_min_AI)
#                 Qin[isl,   jsl,   k, l, I_max, 0] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[islp1, jsl,   k, l, I_max, 3] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), -BIG,      q_max_AI)

#                 Qin[isl,   jsl,   k, l, I_min, 1] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[islp1, jslp1, k, l, I_min, 4] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), BIG,       q_min_AIJ)
#                 Qin[isl,   jsl,   k, l, I_max, 1] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[islp1, jslp1, k, l, I_max, 4] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), -BIG,      q_max_AIJ)

#                 Qin[isl,   jsl,   k, l, I_min, 2] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[isl,   jslp1, k, l, I_min, 5] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), BIG,       q_min_AJ)
#                 Qin[isl,   jsl,   k, l, I_max, 2] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[isl,   jslp1, k, l, I_max, 5] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), -BIG,      q_max_AJ)


#                 #   QQQ1

#                 #         print("q_minmax", q_min_AI, q_min_AIJ, q_min_AJ, q_max_AI, q_max_AIJ, q_max_AJ, file=log_file )
#                 #         print("HLT: Qin 1st: I_min", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
#                 #         print("              I_max", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_max, :], file=log_file)
#                 #         print("              cmask", file=log_file)
#                 #         print(cmask[0, 0, k, l, :], file=log_file)
#                 #         print(cmask[1, 1, k, l, :], file=log_file)
#                 #         print(cmask[5, 5, k, l, :], file=log_file)

#                 # < edge treatment for i=0 >
#                 jv = np.arange(1, jall - 1)  # j = 2 to jall-1 (Python 0-based)
#                 i = 0
#                 ip1 = i + 1
#                 jp1 = jv + 1
#                 jm1 = jv - 1

#                 # Extract local cmask slices
#                 cmask0 = cmask[i, jv, k, l, 0]
#                 cmask1 = cmask[i, jv, k, l, 1]
#                 cmask2 = cmask[i, jv, k, l, 2]

#                 # q_min/q_max calculations
#                 q_min_AI  = np.minimum.reduce([q[i, jv,   k, l], q[i, jm1, k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[i, jv,   k, l], q[i, jm1, k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[i, jv,   k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[i, jv,   k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[i, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l], q[i, jv, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[i, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l], q[i, jv, k, l]])

#                 # Assign to Qin
#                 Qin[i, jv,    k, l, I_min, 0] = np.where(cmask0 == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[ip1, jv,  k, l, I_min, 3] = np.where(cmask0 == rdtype(1.0),     BIG,  q_min_AI)
#                 Qin[i, jv,    k, l, I_max, 0] = np.where(cmask0 == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[ip1, jv,  k, l, I_max, 3] = np.where(cmask0 == rdtype(1.0),   -BIG,  q_max_AI)

#                 Qin[i, jv,    k, l, I_min, 1] = np.where(cmask1 == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[ip1, jp1, k, l, I_min, 4] = np.where(cmask1 == rdtype(1.0),     BIG,  q_min_AIJ)
#                 Qin[i, jv,    k, l, I_max, 1] = np.where(cmask1 == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[ip1, jp1, k, l, I_max, 4] = np.where(cmask1 == rdtype(1.0),   -BIG,  q_max_AIJ)

#                 Qin[i, jv,    k, l, I_min, 2] = np.where(cmask2 == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[i, jp1,   k, l, I_min, 5] = np.where(cmask2 == rdtype(1.0),     BIG,  q_min_AJ)
#                 Qin[i, jv,    k, l, I_max, 2] = np.where(cmask2 == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[i, jp1,   k, l, I_max, 5] = np.where(cmask2 == rdtype(1.0),   -BIG,  q_max_AJ)

#                 # if l==1 and k==7:
#                 #     with open(std.fname_log, 'a') as log_file:
#                 #         print("HLT: Qin 2nd: I_min", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
#                 #         print("              I_max", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

#                 # < edge treatment for j=0 >
#                 iv = np.arange(1, iall - 1)  # i = 2 to iall-1 in Fortran
#                 j = 0
#                 ip1 = iv + 1
#                 jp1 = j + 1
#                 im1 = iv - 1

#                 # Extract cmask components
#                 cmask0 = cmask[iv, j, k, l, 0]
#                 cmask1 = cmask[iv, j, k, l, 1]
#                 cmask2 = cmask[iv, j, k, l, 2]

#                 # Compute min/max values
#                 q_min_AI  = np.minimum.reduce([q[iv, j,   k, l], q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[iv, j,   k, l], q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[iv, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l], q[im1, j, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[iv, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l], q[im1, j, k, l]])

#                 # Assign to Qin arrays
#                 Qin[iv,  j,   k, l, I_min, 0] = np.where(cmask0 == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[ip1, j,   k, l, I_min, 3] = np.where(cmask0 == rdtype(1.0),     BIG,  q_min_AI)
#                 Qin[iv,  j,   k, l, I_max, 0] = np.where(cmask0 == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[ip1, j,   k, l, I_max, 3] = np.where(cmask0 == rdtype(1.0),   -BIG,  q_max_AI)

#                 Qin[iv,  j,   k, l, I_min, 1] = np.where(cmask1 == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[ip1, jp1, k, l, I_min, 4] = np.where(cmask1 == rdtype(1.0),     BIG,  q_min_AIJ)
#                 Qin[iv,  j,   k, l, I_max, 1] = np.where(cmask1 == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[ip1, jp1, k, l, I_max, 4] = np.where(cmask1 == rdtype(1.0),   -BIG,  q_max_AIJ)

#                 Qin[iv,  j,   k, l, I_min, 2] = np.where(cmask2 == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[iv,  jp1, k, l, I_min, 5] = np.where(cmask2 == rdtype(1.0),     BIG,  q_min_AJ)
#                 Qin[iv,  j,   k, l, I_max, 2] = np.where(cmask2 == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[iv,  jp1, k, l, I_max, 5] = np.where(cmask2 == rdtype(1.0),   -BIG,  q_max_AJ)



                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         #print("QQQ1", file=log_file)
                #         print("shape", Qin.shape, Qout.shape, file=log_file)
                #         print("QQQ1 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print("QQQ1 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print("QQQ1 Qout", Qout[1, 1, k, l, :], file=log_file)

                ### CORNER treatment for i=0, j=0 missing in vectorized version $$$$$ 

                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("HLT: Qin 3rd: I_min", file=log_file)
                #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
                #         print("              I_max", file=log_file)
                #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

        prf.PROF_rapend  ('______hlim_qin',2)
        prf.PROF_rapstart('______hlim_qout',2)
        if _fuse_hlim:
            pass   # done by the jax kernel above
        elif _hlim_vec:
            # Stage-1b: vectorized sgp correction + Qout (replaces the l,k loop).
            # sgp = tiny l-only loop (sgp regions), vectorized over k; done BEFORE
            # qnext reads Qin. Main Qout fully vectorized over (i,j,k,l). Bit-exact:
            # identical minimum.reduce / sum(axis=-1) order + elementwise formulas.
            for _l in range(lall):
                if adm.ADM_have_sgp[_l]:
                    _aijmn = np.minimum.reduce([q[0,0,:,_l], q[1,1,:,_l], q[2,1,:,_l], q[0,1,:,_l]])
                    _aijmx = np.maximum.reduce([q[0,0,:,_l], q[1,1,:,_l], q[2,1,:,_l], q[0,1,:,_l]])
                    _c1 = cmask[0,0,:,_l,1]
                    Qin[0,0,:,_l,I_min,1] = np.where(_c1 == rdtype(1.0), _aijmn,  BIG)
                    Qin[1,1,:,_l,I_min,4] = np.where(_c1 == rdtype(1.0),    BIG,  _aijmn)
                    Qin[0,0,:,_l,I_max,1] = np.where(_c1 == rdtype(1.0), _aijmx, -BIG)
                    Qin[1,1,:,_l,I_max,4] = np.where(_c1 == rdtype(1.0),   -BIG,  _aijmx)
            isl = slice(1, iall - 1); jsl = slice(1, jall - 1)
            _qnmin = np.minimum.reduce([q[isl,jsl], Qin[isl,jsl,:,:,I_min,0], Qin[isl,jsl,:,:,I_min,1], Qin[isl,jsl,:,:,I_min,2], Qin[isl,jsl,:,:,I_min,3], Qin[isl,jsl,:,:,I_min,4], Qin[isl,jsl,:,:,I_min,5]])
            _qnmax = np.maximum.reduce([q[isl,jsl], Qin[isl,jsl,:,:,I_max,0], Qin[isl,jsl,:,:,I_max,1], Qin[isl,jsl,:,:,I_max,2], Qin[isl,jsl,:,:,I_max,3], Qin[isl,jsl,:,:,I_max,4], Qin[isl,jsl,:,:,I_max,5]])
            _chm = np.minimum(ch[isl,jsl], rdtype(0.0))
            _Cin  = np.sum(_chm, axis=-1)
            _Cout = np.sum(ch[isl,jsl] - _chm, axis=-1)
            _CQmin = np.sum(_chm * Qin[isl,jsl,:,:,I_min,:], axis=-1)
            _CQmax = np.sum(_chm * Qin[isl,jsl,:,:,I_max,:], axis=-1)
            _zsw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(_Cout) - EPS)
            _q = q[isl,jsl]; _d = d[isl,jsl]
            Qout[isl,jsl,:,:,I_min] = (_q - _CQmax - _qnmax*(rdtype(1.0) - _Cin - _Cout + _d)) / (_Cout + _zsw) * (rdtype(1.0) - _zsw) + _q*_zsw
            Qout[isl,jsl,:,:,I_max] = (_q - _CQmin - _qnmin*(rdtype(1.0) - _Cin - _Cout + _d)) / (_Cout + _zsw) * (rdtype(1.0) - _zsw) + _q*_zsw
            Qout[:, 0,      :, :, I_min] = q[:, 0,      :, :]; Qout[:, 0,      :, :, I_max] = q[:, 0,      :, :]
            Qout[:, jall-1, :, :, I_min] = q[:, jall-1, :, :]; Qout[:, jall-1, :, :, I_max] = q[:, jall-1, :, :]
            Qout[0,      1:jall-1, :, :, I_min] = q[0,      1:jall-1, :, :]; Qout[0,      1:jall-1, :, :, I_max] = q[0,      1:jall-1, :, :]
            Qout[iall-1, 1:jall-1, :, :, I_min] = q[iall-1, 1:jall-1, :, :]; Qout[iall-1, 1:jall-1, :, :, I_max] = q[iall-1, 1:jall-1, :, :]
        else:
            for l in range(lall):
                for k in range(kall):

                    if adm.ADM_have_sgp[l]:
                        i, j = 0, 0  

                        ip1 = i + 1
                        ip2 = i + 2
                        jp1 = j + 1

                        q_min_AIJ = np.min([
                            q[i, j, k, l],
                            q[ip1, jp1, k, l],
                            q[ip2, jp1, k, l],
                            q[i, jp1, k, l],
                        ])
                        q_max_AIJ = np.max([
                            q[i, j, k, l],
                            q[ip1, jp1, k, l],
                            q[ip2, jp1, k, l],
                            q[i, jp1, k, l],
                        ])

                        c1 = cmask[i, j, k, l, 1]

                        Qin[i,     j,    k, l, I_min, 1] = np.where(c1 == rdtype(1.0), q_min_AIJ,  BIG)
                        Qin[ip1,   jp1,  k, l, I_min, 4] = np.where(c1 == rdtype(1.0),      BIG,  q_min_AIJ)
                        Qin[i,     j,    k, l, I_max, 1] = np.where(c1 == rdtype(1.0), q_max_AIJ, -BIG)
                        Qin[ip1,   jp1,  k, l, I_max, 4] = np.where(c1 == rdtype(1.0),    -BIG,  q_max_AIJ)
                    # end if
                

                    # if l==1 and k==7:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("HLT: Qin 4th: I_min", file=log_file)
                    #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
                    #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
                    #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
                    #         print("              I_max", file=log_file)
                    #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
                    #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
                    #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

                    #---< (iii) define allowable range of q at next step, eq.(42)&(43) >---   

                    isl = slice(1, iall - 1)
                    jsl = slice(1, jall - 1)

                    qnext_min = np.minimum.reduce([
                        q[isl, jsl, k, l],
                        Qin[isl, jsl, k, l, I_min, 0],
                        Qin[isl, jsl, k, l, I_min, 1],
                        Qin[isl, jsl, k, l, I_min, 2],
                        Qin[isl, jsl, k, l, I_min, 3],
                        Qin[isl, jsl, k, l, I_min, 4],
                        Qin[isl, jsl, k, l, I_min, 5]
                    ])

                    qnext_max = np.maximum.reduce([
                        q[isl, jsl, k, l],
                        Qin[isl, jsl, k, l, I_max, 0],
                        Qin[isl, jsl, k, l, I_max, 1],
                        Qin[isl, jsl, k, l, I_max, 2],
                        Qin[isl, jsl, k, l, I_max, 3],
                        Qin[isl, jsl, k, l, I_max, 4],
                        Qin[isl, jsl, k, l, I_max, 5]
                    ])


                    # Apply masking
                    ch_masked = np.minimum(ch[isl, jsl, k, l, :], rdtype(0.0))
                    Cin_sum = np.sum(ch_masked, axis=-1)
                    Cout_sum = np.sum(ch[isl, jsl, k, l, :] - ch_masked, axis=-1)

                    CQin_min_sum = np.sum(ch_masked * Qin[isl, jsl, k, l, I_min, :], axis=-1)
                    CQin_max_sum = np.sum(ch_masked * Qin[isl, jsl, k, l, I_max, :], axis=-1)

    #                 if l==1 and k==7:
    #                     with open(std.fname_log, 'a') as log_file:
    #                         print("QQQ1x", file=log_file)
    # #                        print("MMIN", ch_masked * Qin[isl, jsl, k, l, I_min, :], file=log_file)
    #                         print("MMIN", ch_masked * Qin[0, 0, 7, 1, I_min, :], file=log_file)
    #                         #print("MMAX", ch_masked * Qin[isl, jsl, k, l, I_max, :], file=log_file)
    #                         print("MIN", Qin[0, 0, 7, 1, I_min, :], file=log_file)
    #                         print("MASK", ch_masked[0, 0, :], file=log_file)
    #                         #print("MAX", Qin[0, 0, k, l, I_max, :], file=log_file)

                    #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout_sum) - EPS)
                    zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout_sum) - EPS)

                    q_ = q[isl, jsl, k, l]
                    d_ = d[isl, jsl, k, l]

                    Qout[isl, jsl, k, l, I_min] = (
                        (q_ - CQin_max_sum - qnext_max * (rdtype(1.0) - Cin_sum - Cout_sum + d_)) /
                        (Cout_sum + zerosw) * (rdtype(1.0) - zerosw) +
                        q_ * zerosw
                    )

                    Qout[isl, jsl, k, l, I_max] = (
                        (q_ - CQin_min_sum - qnext_min * (rdtype(1.0) - Cin_sum - Cout_sum + d_)) /
                        (Cout_sum + zerosw) * (rdtype(1.0) - zerosw) +
                        q_ * zerosw
                    )

                    # if l==1 and k==7:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         #print("QQQ1", file=log_file)
                    #         print("QQQ2 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                    #         print("QQQ2 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                    #         print("QQQ2 Qout", Qout[1, 1, k, l, :], file=log_file)

                            # print("QQQ2 d_", d_[1, 1], file=log_file)
                            # print("QQQ2 q_", q_[1, 1], file=log_file)
                            # print("CQin_min_sum shape:", CQin_min_sum.shape, file=log_file)  # 16x16
                            # print("CQin_min_sum", CQin_min_sum[0,0], file=log_file)       # 0, 0 sometimes have a strange value
                            # print("CQin_max_sum", CQin_max_sum[0,0], file=log_file)
                            # print("qnext_min", qnext_min, file=log_file)
                            # print("qnext_max", qnext_max, file=log_file)

                            # print("Cin_sum", Cin_sum, file=log_file)
                            # print("Cout_sum", Cout_sum, file=log_file)
                            # #print("zerosw", zerosw, file=log_file) 

                    # j=0 and j=jall-1 edges
                    Qout[:, 0,      k, l, I_min] = q[:, 0,      k, l]
                    Qout[:, 0,      k, l, I_max] = q[:, 0,      k, l]
                    Qout[:, jall-1, k, l, I_min] = q[:, jall-1, k, l]
                    Qout[:, jall-1, k, l, I_max] = q[:, jall-1, k, l]

                    # i=0 and i=iall-1  edges (excluding corners already set)
                    Qout[0,      1:jall-1, k, l, I_min] = q[0,      1:jall-1, k, l]
                    Qout[0,      1:jall-1, k, l, I_max] = q[0,      1:jall-1, k, l]
                    Qout[iall-1, 1:jall-1, k, l, I_min] = q[iall-1, 1:jall-1, k, l]
                    Qout[iall-1, 1:jall-1, k, l, I_max] = q[iall-1, 1:jall-1, k, l]


                    # if l==1 and k==7:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("QQQ3 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                    #         print("QQQ3 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                    #         print("QQQ3 Qout", Qout[1, 1, k, l, :], file=log_file)


                # end loop k
            # end loop l

        prf.PROF_rapend  ('______hlim_qout',2)
        prf.PROF_rapstart('______hlim_qin_pl',2)
        # Unit 4b: device POLE Thuburn limiter. Gate PYNICAM_RESIDENT_HADV_LIM_PL
        # (default OFF). Build Qin_pl/Qout_pl on device (qout kernel) + the 2-stage
        # apply on device (apply kernel, after the COMM); drain the meaningful slots
        # to host so the existing Qout COMM + the flux apply still read host arrays.
        # _Qin_pl_d persists across the COMM (the COMM only touches Qout) to feed the
        # apply. Bit-exact vs the host loops; asarray fallback when off.
        _lim_pl = (bk.type == "jax") and adm.ADM_have_pl and \
            bk.resident()
        # Unit 4c-3a: keep the pole Qout on device through its halo exchange (the
        # COMM auto-routes when self._Qout_d + _Qout_pl_d are both jax arrays), so
        # the apply reads the COMM'd device Qout -- no Qout_pl drain/asarray. Needs
        # the regular Qout device handle (_hadv_qa_resident). Gate
        # PYNICAM_RESIDENT_HADV_QA_PL (default OFF).
        # 4c-6: use the caller's flag (it already requires _hadv_qa_resident +
        # REMAP/LIM/HADV/APPLY/QA all on) so the device Qout/q_a paths AND the dead
        # Qin/Qout/q_a drain skips are governed by exactly one condition.
        _qa_resident_pl = qa_resident_pl and _lim_pl
        _Qin_pl_d = None
        _Qout_pl_d = None
        _qout_pl_d = None
        if adm.ADM_have_pl:
            if _lim_pl:
                xp = bk.xp
                if getattr(self, "_hlim_pl_qout_k", None) is None:
                    self._hlim_pl_qout_k = bk.maybe_jit(
                        compute_horizontal_limiter_qout_pl, static_argnames=("cfg", "xp"))
                    self._hlim_pl_apply_k = bk.maybe_jit(
                        compute_horizontal_limiter_apply_pl, static_argnames=("cfg", "xp"))
                if getattr(self, "_hlim_cfg_pl", None) is None:
                    self._hlim_cfg_pl = HLimiterCfgPl(
                        n=adm.ADM_gslf_pl, gmin=adm.ADM_gmin_pl, gmax=adm.ADM_gmax_pl,
                        I_min=I_min, I_max=I_max, BIG=float(BIG), EPS=float(EPS))
                _q_in  = q_pl_d     if q_pl_d     is not None else xp.asarray(q_pl)      # 4c-1
                _d_in  = d_pl_d     if d_pl_d     is not None else xp.asarray(d_pl)      # 4c-1
                _ch_in = ch_pl_d    if ch_pl_d    is not None else xp.asarray(ch_pl)     # 4c-1
                _cm_in = cmask_pl_d if cmask_pl_d is not None else xp.asarray(cmask_pl)  # 4c-1
                _Qin_pl_d, _Qout_pl_d = self._hlim_pl_qout_k(
                    _q_in, _d_in, _ch_in, _cm_in,
                    cfg=self._hlim_cfg_pl, xp=xp)
                if not _qa_resident_pl:   # 4c-6: host Qin/Qout dead (device apply uses _Qin_pl_d; COMM uses _Qout_pl_d)
                    _gp0, _gp1 = adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1
                    Qin_pl[_gp0:_gp1] = bk.to_numpy(_Qin_pl_d)[_gp0:_gp1]
                    Qout_pl[adm.ADM_gslf_pl] = bk.to_numpy(_Qout_pl_d)[adm.ADM_gslf_pl]
            else:
                n = adm.ADM_gslf_pl

                for l in range(lall_pl):
                    for k in range(kall):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                            ij = v
                            ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1
                            ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                            q_min_pl = min(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])
                            q_max_pl = max(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])

                            cm = cmask_pl[ij, k, l]

                            Qin_pl[ij, k, l, I_min, 0] = np.where(cm == rdtype(1.0), q_min_pl,  BIG)         #
                            Qin_pl[ij, k, l, I_min, 1] = np.where(cm == rdtype(1.0),     BIG,  q_min_pl)
                            Qin_pl[ij, k, l, I_max, 0] = np.where(cm == rdtype(1.0), q_max_pl, -BIG)         #
                            Qin_pl[ij, k, l, I_max, 1] = np.where(cm == rdtype(1.0),    -BIG,  q_max_pl)

                        # Compute min/max over all v
                        qnext_min_pl = q_pl[n, k, l]
                        qnext_max_pl = q_pl[n, k, l]
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                            qnext_min_pl = min(qnext_min_pl, Qin_pl[v, k, l, I_min, 0])
                            qnext_max_pl = max(qnext_max_pl, Qin_pl[v, k, l, I_max, 0])
                        # end loop v

                        # Sum contributions
                        Cin_sum_pl = rdtype(0.0)
                        Cout_sum_pl = rdtype(0.0)
                        CQin_min_sum_pl = rdtype(0.0)
                        CQin_max_sum_pl = rdtype(0.0)

                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                            ch_m = cmask_pl[v, k, l] * ch_pl[v, k, l]

                            Cin_sum_pl      += ch_m
                            Cout_sum_pl     += ch_pl[v, k, l] - ch_m
                            CQin_min_sum_pl += ch_m * Qin_pl[v, k, l, I_min, 0]
                            CQin_max_sum_pl += ch_m * Qin_pl[v, k, l, I_max, 0]
                        # end loop v

                        Cout_abs = abs(Cout_sum_pl)
                        zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), Cout_abs - EPS)

                        denom = Cout_sum_pl + zerosw
                        factor = rdtype(1.0) - zerosw
                        q_nkl = q_pl[n, k, l]
                        dval = d_pl[n, k, l]

                        Qout_pl[n, k, l, I_min] = ((q_nkl - CQin_max_sum_pl -
                                                    qnext_max_pl * (rdtype(1.0) - Cin_sum_pl - Cout_sum_pl + dval))
                                                / denom * factor +
                                                q_nkl * zerosw)

                        Qout_pl[n, k, l, I_max] = ((q_nkl - CQin_min_sum_pl -
                                                    qnext_min_pl * (rdtype(1.0) - Cin_sum_pl - Cout_sum_pl + dval))
                                                / denom * factor +
                                                q_nkl * zerosw)
                    # end loop k
                # end loop l
        # endif

        if getattr(self, "_hadv_qa_resident", False):
            # Stage-4b: on-device halo exchange of the resident Qout (auto-dispatch
            # since self._Qout_d is a jax array). Qout_pl drained back for the host
            # pole apply_pl. Unit 4c-3a: under the gate pass the DEVICE pole Qout
            # (_Qout_pl_d) so the exchange stays on device end-to-end (no drain); the
            # apply reads _qout_pl_d directly.
            self._Qout_d, _qout_pl_d = comm.COMM_data_transfer(
                self._Qout_d, (_Qout_pl_d if _qa_resident_pl else Qout_pl))
            if not _qa_resident_pl and adm.ADM_have_pl:
                # non-pole ranks: Qout_pl is a local dead pole buffer (the pole apply is
                # under adm.ADM_have_pl) -> skip the dead drain. Bit-exact, and un-breaks
                # the FUSE_TRACER jit trace on non-pole ranks (same class as gradq_pl@1888).
                Qout_pl[:] = bk.to_numpy(_qout_pl_d)
        else:
            comm.COMM_data_transfer( Qout, Qout_pl )

        #---- apply inflow/outflow limiter

        prf.PROF_rapend  ('______hlim_qin_pl',2)
        prf.PROF_rapstart('______hlim_apply',2)
        if _fuse_hlim:
            # Stage-3 kernel B: apply against the halo-exchanged Qout (COMM above).
            _xpL = bk.xp
            if getattr(self, "_hadv_qa_resident", False):
                # Stage-4b: q_a/Qin/Qout all resident; result stays on device for the
                # device-drained rhogq update (one drain at the update site).
                self._q_a_d = self._hlim_apply_k(
                    self._q_a_d, self._Qin_d, self._Qout_d, _xpL.asarray(cmask),
                    cfg=self._hlim_cfg, xp=_xpL,
                )
            else:
                q_a[:, :, :, :, :] = bk.to_numpy(self._hlim_apply_k(
                    _xpL.asarray(q_a), _xpL.asarray(Qin), _xpL.asarray(Qout), _xpL.asarray(cmask),
                    cfg=self._hlim_cfg, xp=_xpL,
                ))
        elif _hlim_vec:
            # Stage-1c: vectorized apply (replaces l,k loop; body already i,j-vec).
            # Bit-exact: same per-direction read-modify-write of q_a[...,0/1/2] then
            # scatter to [...,3/4/5]; directions touch DISJOINT components (no cross
            # dep); numpy evaluates RHS before assign.
            isl = slice(0, iall - 1); jsl = slice(0, jall - 1)
            isl_p1 = slice(1, iall);  jsl_p1 = slice(1, jall)
            # Direction 1 (0 -> 3)
            q_a[isl, jsl, :, :, 0] = (
                cmask[isl, jsl, :, :, 0] * np.minimum(np.maximum(q_a[isl, jsl, :, :, 0], Qin[isl, jsl, :, :, I_min, 0]), Qin[isl, jsl, :, :, I_max, 0])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 0]) * np.minimum(np.maximum(q_a[isl, jsl, :, :, 0], Qin[isl_p1, jsl, :, :, I_min, 3]), Qin[isl_p1, jsl, :, :, I_max, 3])
            )
            q_a[isl, jsl, :, :, 0] = (
                cmask[isl, jsl, :, :, 0] * np.maximum(np.minimum(q_a[isl, jsl, :, :, 0], Qout[isl_p1, jsl, :, :, I_max]), Qout[isl_p1, jsl, :, :, I_min])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 0]) * np.maximum(np.minimum(q_a[isl, jsl, :, :, 0], Qout[isl, jsl, :, :, I_max]), Qout[isl, jsl, :, :, I_min])
            )
            q_a[isl_p1, jsl, :, :, 3] = q_a[isl, jsl, :, :, 0]
            # Direction 2 (1 -> 4)
            q_a[isl, jsl, :, :, 1] = (
                cmask[isl, jsl, :, :, 1] * np.minimum(np.maximum(q_a[isl, jsl, :, :, 1], Qin[isl, jsl, :, :, I_min, 1]), Qin[isl, jsl, :, :, I_max, 1])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 1]) * np.minimum(np.maximum(q_a[isl, jsl, :, :, 1], Qin[isl_p1, jsl_p1, :, :, I_min, 4]), Qin[isl_p1, jsl_p1, :, :, I_max, 4])
            )
            q_a[isl, jsl, :, :, 1] = (
                cmask[isl, jsl, :, :, 1] * np.maximum(np.minimum(q_a[isl, jsl, :, :, 1], Qout[isl_p1, jsl_p1, :, :, I_max]), Qout[isl_p1, jsl_p1, :, :, I_min])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 1]) * np.maximum(np.minimum(q_a[isl, jsl, :, :, 1], Qout[isl, jsl, :, :, I_max]), Qout[isl, jsl, :, :, I_min])
            )
            q_a[isl_p1, jsl_p1, :, :, 4] = q_a[isl, jsl, :, :, 1]
            # Direction 3 (2 -> 5)
            q_a[isl, jsl, :, :, 2] = (
                cmask[isl, jsl, :, :, 2] * np.minimum(np.maximum(q_a[isl, jsl, :, :, 2], Qin[isl, jsl, :, :, I_min, 2]), Qin[isl, jsl, :, :, I_max, 2])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 2]) * np.minimum(np.maximum(q_a[isl, jsl, :, :, 2], Qin[isl, jsl_p1, :, :, I_min, 5]), Qin[isl, jsl_p1, :, :, I_max, 5])
            )
            q_a[isl, jsl, :, :, 2] = (
                cmask[isl, jsl, :, :, 2] * np.maximum(np.minimum(q_a[isl, jsl, :, :, 2], Qout[isl, jsl_p1, :, :, I_max]), Qout[isl, jsl_p1, :, :, I_min])
                + (rdtype(1.0) - cmask[isl, jsl, :, :, 2]) * np.maximum(np.minimum(q_a[isl, jsl, :, :, 2], Qout[isl, jsl, :, :, I_max]), Qout[isl, jsl, :, :, I_min])
            )
            q_a[isl, jsl_p1, :, :, 5] = q_a[isl, jsl, :, :, 2]
        else:
            for l in range(lall):
                for k in range(kall):

                    isl = slice(0, iall - 1)
                    jsl = slice(0, jall - 1)
                    isl_p1 = slice(1, iall)
                    jsl_p1 = slice(1, jall)

                    # Direction 1 (index 0) → copied to index 3
                    q_a[isl, jsl, k, l, 0] = (
                        cmask[isl, jsl, k, l, 0] * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 0], Qin[isl, jsl, k, l, I_min, 0]),
                            Qin[isl, jsl, k, l, I_max, 0]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 0]) * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 0], Qin[isl_p1, jsl, k, l, I_min, 3]),
                            Qin[isl_p1, jsl, k, l, I_max, 3]
                        )
                    )
                    q_a[isl, jsl, k, l, 0] = (
                        cmask[isl, jsl, k, l, 0] * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 0], Qout[isl_p1, jsl, k, l, I_max]),
                            Qout[isl_p1, jsl, k, l, I_min]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 0]) * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 0], Qout[isl, jsl, k, l, I_max]),
                            Qout[isl, jsl, k, l, I_min]
                        )
                    )
                    q_a[isl_p1, jsl, k, l, 3] = q_a[isl, jsl, k, l, 0]

                    # Direction 2 (index 1) → copied to index 4
                    q_a[isl, jsl, k, l, 1] = (
                        cmask[isl, jsl, k, l, 1] * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 1], Qin[isl, jsl, k, l, I_min, 1]),
                            Qin[isl, jsl, k, l, I_max, 1]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 1]) * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 1], Qin[isl_p1, jsl_p1, k, l, I_min, 4]),
                            Qin[isl_p1, jsl_p1, k, l, I_max, 4]
                        )
                    )
                    q_a[isl, jsl, k, l, 1] = (
                        cmask[isl, jsl, k, l, 1] * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 1], Qout[isl_p1, jsl_p1, k, l, I_max]),
                            Qout[isl_p1, jsl_p1, k, l, I_min]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 1]) * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 1], Qout[isl, jsl, k, l, I_max]),
                            Qout[isl, jsl, k, l, I_min]
                        )
                    )
                    q_a[isl_p1, jsl_p1, k, l, 4] = q_a[isl, jsl, k, l, 1]

                    # Direction 3 (index 2) → copied to index 5
                    q_a[isl, jsl, k, l, 2] = (
                        cmask[isl, jsl, k, l, 2] * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 2], Qin[isl, jsl, k, l, I_min, 2]),
                            Qin[isl, jsl, k, l, I_max, 2]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 2]) * np.minimum(
                            np.maximum(q_a[isl, jsl, k, l, 2], Qin[isl, jsl_p1, k, l, I_min, 5]),
                            Qin[isl, jsl_p1, k, l, I_max, 5]
                        )
                    )
                    q_a[isl, jsl, k, l, 2] = (
                        cmask[isl, jsl, k, l, 2] * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 2], Qout[isl, jsl_p1, k, l, I_max]),
                            Qout[isl, jsl_p1, k, l, I_min]
                        ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 2]) * np.maximum(
                            np.minimum(q_a[isl, jsl, k, l, 2], Qout[isl, jsl, k, l, I_max]),
                            Qout[isl, jsl, k, l, I_min]
                        )
                    )
                    q_a[isl, jsl_p1, k, l, 5] = q_a[isl, jsl, k, l, 2]

                    # isl = slice(0, iall - 1)
                    # jsl = slice(0, jall - 1)

                    # #  (indices 0 and 3)
                    # cm = cmask[isl, jsl, k, l, 0]
                    # qa = q_a[isl, jsl, k, l, 0]
                    # qmin_ai = Qin[isl, jsl, k, l, I_min, 0]
                    # qmax_ai = Qin[isl, jsl, k, l, I_max, 0]
                    # qmin_ai_p = Qin[isl.start + 1, jsl, k, l, I_min, 3]
                    # qmax_ai_p = Qin[isl.start + 1, jsl, k, l, I_max, 3]

                    # q_a[isl, jsl, k, l, 0] = cm * np.minimum(np.maximum(qa, qmin_ai), qmax_ai) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_ai_p), qmax_ai_p)

                    # qmin_out = Qout[isl.start + 1, jsl, k, l, I_min]
                    # qmax_out = Qout[isl.start + 1, jsl, k, l, I_max]
                    # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                    # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                    # q_a[isl, jsl, k, l, 0] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 0], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 0], qmax_out_p), qmin_out_p)
                    # q_a[isl.start + 1, jsl, k, l, 3] = q_a[isl, jsl, k, l, 0]

                    # #  (indices 1 and 4)
                    # cm = cmask[isl, jsl, k, l, 1]
                    # qa = q_a[isl, jsl, k, l, 1]
                    # qmin = Qin[isl, jsl, k, l, I_min, 1]
                    # qmax = Qin[isl, jsl, k, l, I_max, 1]
                    # qmin_p = Qin[isl.start + 1, jsl.start + 1, k, l, I_min, 4]
                    # qmax_p = Qin[isl.start + 1, jsl.start + 1, k, l, I_max, 4]

                    # q_a[isl, jsl, k, l, 1] = cm * np.minimum(np.maximum(qa, qmin), qmax) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_p), qmax_p)

                    # qmin_out = Qout[isl.start + 1, jsl.start + 1, k, l, I_min]
                    # qmax_out = Qout[isl.start + 1, jsl.start + 1, k, l, I_max]
                    # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                    # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                    # q_a[isl, jsl, k, l, 1] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 1], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 1], qmax_out_p), qmin_out_p)
                    # q_a[isl.start + 1, jsl.start + 1, k, l, 4] = q_a[isl, jsl, k, l, 1]

                    # #  (indices 2 and 5)
                    # cm = cmask[isl, jsl, k, l, 2]
                    # qa = q_a[isl, jsl, k, l, 2]
                    # qmin = Qin[isl, jsl, k, l, I_min, 2]
                    # qmax = Qin[isl, jsl, k, l, I_max, 2]
                    # qmin_p = Qin[isl, jsl.start + 1, k, l, I_min, 5]
                    # qmax_p = Qin[isl, jsl.start + 1, k, l, I_max, 5]

                    # q_a[isl, jsl, k, l, 2] = cm * np.minimum(np.maximum(qa, qmin), qmax) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_p), qmax_p)

                    # qmin_out = Qout[isl, jsl.start + 1, k, l, I_min]
                    # qmax_out = Qout[isl, jsl.start + 1, k, l, I_max]
                    # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                    # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                    # q_a[isl, jsl, k, l, 2] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 2], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 2], qmax_out_p), qmin_out_p)
                    # q_a[isl, jsl.start + 1, k, l, 5] = q_a[isl, jsl, k, l, 2]

                # end loop k
            # end loop l

        prf.PROF_rapend  ('______hlim_apply',2)
        prf.PROF_rapstart('______hlim_apply_pl',2)
        if adm.ADM_have_pl:
            if _lim_pl:
                # Unit 4b: device pole apply, reusing the device Qin (_Qin_pl_d) from
                # the qout stage + the post-COMM host Qout_pl (asarray). Drain the
                # v=gmin..gmax rows for the downstream flux apply.
                xp = bk.xp
                _cm_in = cmask_pl_d if cmask_pl_d is not None else xp.asarray(cmask_pl)  # 4c-1
                _qout_in = _qout_pl_d if _qa_resident_pl else xp.asarray(Qout_pl)  # 4c-3a device Qout
                # 4c-3b: read the device q_a (remap output) instead of asarray(q_a_pl)
                _qa_in = self._qa_pl_remap_d if (_qa_resident_pl and getattr(self, "_qa_pl_remap_d", None) is not None) else xp.asarray(q_a_pl)
                _qa_pl_d = self._hlim_pl_apply_k(
                    _qa_in, _Qin_pl_d, _qout_in, _cm_in,
                    cfg=self._hlim_cfg_pl, xp=xp)
                if _qa_resident_pl:
                    self._qa_pl_lim_d = _qa_pl_d   # 4c-3b: device limited q_a for the flux apply
                else:                              # 4c-6: host q_a_pl dead when the device q_a is carried
                    _gp0, _gp1 = adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1
                    q_a_pl[_gp0:_gp1] = bk.to_numpy(_qa_pl_d)[_gp0:_gp1]
            else:
                n = adm.ADM_gslf_pl
                for l in range(lall_pl):
                    for k in range(kall):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                            cm = cmask_pl[v, k, l]

                            # First clamping between min/max inputs
                            q0 = np.minimum(np.maximum(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 0]),
                                            Qin_pl[v, k, l, I_max, 0])
                            q1 = np.minimum(np.maximum(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 1]),
                                            Qin_pl[v, k, l, I_max, 1])
                            q_a_pl[v, k, l] = cm * q0 + (rdtype(1.0) - cm) * q1

                            # Then further clamping with output bounds
                            q2 = np.maximum(np.minimum(q_a_pl[v, k, l], Qout_pl[v, k, l, I_max]),
                                            Qout_pl[v, k, l, I_min])
                            q3 = np.maximum(np.minimum(q_a_pl[v, k, l], Qout_pl[n, k, l, I_max]),
                                            Qout_pl[n, k, l, I_min])
                            q_a_pl[v, k, l] = cm * q2 + (rdtype(1.0) - cm) * q3
                        # end loop v
                    # end loop k
                # end loop l
        # end if

        prf.PROF_rapend  ('______hlim_apply_pl',2)
        prf.PROF_rapend  ('_____horizontal_adv_limiter',2)

        return
    