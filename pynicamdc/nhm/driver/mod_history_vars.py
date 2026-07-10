#
# mod_history_vars -- derived diagnostic (history) variables -- numpy port
#
# Ported from: nicamdc/src/nhm/driver/mod_history_vars.f90 (subroutine history_vars).
#
# Computes the standard model-level derived diagnostics from the diagnostic
# state (rho, pre, tem, vx, vy, vz, w, qv). This is the core, self-contained
# subset used for DCMIP analysis:
#   ml_rho, ml_tem, ml_pres   -- density / temperature / pressure
#   ml_u,   ml_v              -- zonal / meridional wind (vh2uv projection)
#   ml_w                      -- cell-centre vertical wind (W2Cfact interp of half-level w)
#   ml_omg                    -- omega = -g rho wc  (vertical pressure velocity)
#   ml_hgt                    -- geopotential height (GRD_Z)
#   ml_th,  ml_thv            -- potential / virtual potential temperature
#
# The pressure-level slices (sl_u850 ...) and the gated RH/MSE/th_prime variants
# are not included here.
#
import numpy as np
from pynicamdc.share.mod_adm import adm


class Hvar:

    _instance = None

    def __init__(self):
        self._tend = {}      # persisted previous-output state for the tendency diagnostics

    def history_vars(self, rho, pre, tem, vx, vy, vz, w, q,
                     grd, gmtr, vmtr, cnst, rcnf, cnvv, tdyn, satr, rdtype, dt=None):
        """Compute the core model-level diagnostics from the diagnostic state.
        All 3D inputs are (i,j,kall,l); q is the full tracer array (i,j,kall,l,ntrc)
        or None. Returns a dict {name: array}."""
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        GRAV = cnst.CONST_GRAV
        ntrc = q.shape[-1] if q is not None else 0
        qv = q[:, :, :, :, rcnf.I_QV] if ntrc > 0 else None

        # zonal / meridional wind (Cartesian -> lat/lon; pole path unused for regional out)
        vx_pl = np.zeros_like(vmtr.VMTR_GSGAM2_pl)
        u, _u_pl, v, _v_pl = cnvv.cnvvar_vh2uv(
            vx, vx_pl, vy, vx_pl, vz, vx_pl, grd, gmtr, withcos=False)

        # cell-centre vertical wind: wc = W2Cfact[...,0]*w[k+1] + W2Cfact[...,1]*w[k].
        # The half-level w at the rigid top/bottom boundaries (k=kmin, kmax+1) is 0
        # (nicamdc BNDCND_rhow); the diagnostic w carries UNDEF there, so zero the
        # non-finite entries before the interpolation.
        UNDEF = cnst.CONST_UNDEF
        w = np.where(np.isfinite(w) & (np.abs(w) < np.abs(UNDEF) * rdtype(0.1)), w, rdtype(0.0))
        W2C = vmtr.VMTR_W2Cfact
        wc = np.zeros_like(w)
        ks = slice(kmin, kmax + 1)
        wc[:, :, ks, :] = (W2C[:, :, ks, :, 0] * w[:, :, kmin + 1:kmax + 2, :]
                           + W2C[:, :, ks, :, 1] * w[:, :, kmin:kmax + 1, :])

        # omega (vertical pressure velocity)
        omg = -GRAV * rho * wc

        # geopotential height (hydrostatic assumption -> model-level height)
        hgt = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)

        # potential temperature and virtual potential temperature
        th = tdyn.THRMDYN_th(tem, pre, cnst)
        if qv is None:
            thv = th
        else:
            thv = th * (rdtype(1.0) + rdtype(0.61) * qv)

        result = {
            'ml_rho': rho, 'ml_tem': tem, 'ml_pres': pre,
            'ml_u': u, 'ml_v': v, 'ml_w': wc,
            'ml_omg': omg, 'ml_hgt': hgt,
            'ml_th': th, 'ml_thv': thv,
        }

        # pressure-level slices (log-p linear interpolation of u/v/wc/tem to a target
        # pressure). Single-level (i,j,l) fields.
        for tag, plev in (('850', 850.0e2), ('500', 500.0e2),
                          ('250', 250.0e2), ('100', 100.0e2)):
            up, vp, wp, tp = self._plev_interp(pre, (u, v, wc, tem),
                                               rdtype(plev), kmin, adm.ADM_kall)
            result['sl_u' + tag] = up
            result['sl_v' + tag] = vp
            result['sl_w' + tag] = wp
            result['sl_t' + tag] = tp

        # surface pressure (sl_ps): hydrostatic from the lowest level to the surface
        # (nicamdc sv_pre_sfc). Single-level (i,j,l).
        z_srf = grd.GRD_zs[:, :, adm.ADM_K0, :, grd.GRD_ZSFC]         # (i,j,l)
        result['sl_ps'] = pre[:, :, kmin, :] + rho[:, :, kmin, :] * GRAV * (hgt[:, :, kmin, :] - z_srf)

        # moisture-dependent diagnostics (qv=0 -> mse=dry static energy, rh=0)
        qv0 = qv if qv is not None else np.zeros_like(tem)
        # moist static energy: cp*T + phi + Lv*qv  (phi = GRD_Z*g)
        result['ml_mse'] = tem * cnst.CONST_CPdry + hgt * GRAV + qv0 * cnst.CONST_LHV
        # relative humidity (liq+ice / liq / ice): qv*rho*Rvap*T/psat*100
        Rvap = cnst.CONST_Rvap
        rh_num = qv0 * rho * Rvap * tem * rdtype(100.0)
        result['ml_rha'] = rh_num / satr.SATURATION_psat_all(tem, cnst)
        result['ml_rh'] = rh_num / satr.SATURATION_psat_liq(tem, cnst)
        result['ml_rhi'] = rh_num / satr.SATURATION_psat_ice(tem, cnst)

        # water paths (column mass integrals, kmin..kmax): sum rho*q*GSGAM2*dgz.
        # sl_pw = vapour, sl_lwp = liquid (qc+qr), sl_iwp = ice (qi+qs+qg).
        GSGAM2 = vmtr.VMTR_GSGAM2
        ks = slice(kmin, kmax + 1)
        wcol = GSGAM2[:, :, ks, :] * grd.GRD_dgz[kmin:kmax + 1][None, None, :, None]

        def colint(qx):
            return np.sum(rho[:, :, ks, :] * qx[:, :, ks, :] * wcol, axis=2)

        def sum_tracers(idxs):
            s = np.zeros_like(tem)
            for idx in idxs:
                if idx is not None and idx >= 0:
                    s = s + q[:, :, :, :, idx]
            return s

        result['sl_pw'] = colint(qv0)
        result['sl_lwp'] = colint(sum_tracers((getattr(rcnf, 'I_QC', -1), getattr(rcnf, 'I_QR', -1))))
        result['sl_iwp'] = colint(sum_tracers((getattr(rcnf, 'I_QI', -1),
                                               getattr(rcnf, 'I_QS', -1), getattr(rcnf, 'I_QG', -1))))

        # time-tendency diagnostics (nicamdc): d = (previous_output - current) * dday,
        # dday = 86400/DTL [->/day]; dq in g/kg/day. Stateful: the first call seeds the
        # reference and returns 0 (like nicamdc's history_vars_setup init at the IC).
        if dt is not None:
            dday = rdtype(86400.0) / dt
            cur = {'ml_du': (u, dday), 'ml_dv': (v, dday), 'ml_dw': (wc, dday),
                   'ml_dtem': (tem, dday), 'ml_dq': (qv0, dday * rdtype(1.0e3))}
            for nm, (c, fac) in cur.items():
                old = self._tend.get(nm)
                result[nm] = np.zeros_like(c) if old is None else (old - c) * fac
                self._tend[nm] = c.copy()

        return result

    def _plev_interp(self, pre, fields, plev, kmin, kdim):
        """Interpolate `fields` (list of (i,j,k,l)) to the pressure level `plev`
        (log-p linear), per column. Returns a list of (i,j,l). Matches nicamdc
        sv_plev_uvwt: bracket [kl,ku] with pre[kl] >= plev > pre[ku], ku = first
        k>=kmin with pre[k] < plev."""
        below = pre < plev                                          # (i,j,k,l)
        below[:, :, :kmin, :] = False                              # search starts at kmin
        ku = np.argmax(below, axis=2)                              # first True k (0 if none)
        ku = np.maximum(ku, kmin)
        kl = ku - 1
        ku_e = ku[:, :, None, :]; kl_e = kl[:, :, None, :]
        pre_ku = np.take_along_axis(pre, ku_e, axis=2)[:, :, 0, :]
        pre_kl = np.take_along_axis(pre, kl_e, axis=2)[:, :, 0, :]
        denom = np.log(pre_kl) - np.log(pre_ku)
        lp = np.log(plev)
        wght_l = (lp - np.log(pre_ku)) / denom
        wght_u = (np.log(pre_kl) - lp) / denom
        out = []
        for f in fields:
            f_ku = np.take_along_axis(f, ku_e, axis=2)[:, :, 0, :]
            f_kl = np.take_along_axis(f, kl_e, axis=2)[:, :, 0, :]
            out.append(wght_l * f_kl + wght_u * f_ku)
        return out


hvar = Hvar()
