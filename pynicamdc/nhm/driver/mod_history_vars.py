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
        self._tend = {}          # persisted previous-output state for the tendency diagnostics
        self._tend_nstep = None  # model step count at the last tendency reference

    def history_vars(self, rho, pre, tem, vx, vy, vz, w, q,
                     grd, gmtr, vmtr, cnst, rcnf, cnvv, tdyn, satr, rdtype,
                     dt=None, comm=None, items=None, nstep=None):
        """Compute the core model-level diagnostics from the diagnostic state.
        All 3D inputs are (i,j,kall,l); q is the full tracer array (i,j,kall,l,ntrc)
        or None. Returns a dict {name: array}."""
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        GRAV = cnst.CONST_GRAV
        ntrc = q.shape[-1] if q is not None else 0
        qv = q[:, :, :, :, rcnf.I_QV] if ntrc > 0 else None

        # per-variable selection: `items` is a set of requested names, or None = all.
        # W(*names) -> is any of these requested? (used to skip expensive blocks)
        want = items
        def W(*names):
            return want is None or any(n in want for n in names)
        SLICES = {'sl_' + f + lev for lev in ('850', '500', '250', '100') for f in ('u', 'v', 'w', 't')}
        need_slices = want is None or bool(want & SLICES)
        need_paths = W('sl_pw', 'sl_lwp', 'sl_iwp', 'sl_cl', 'sl_cl2', 'sl_cly')
        need_tend = dt is not None and W('ml_du', 'ml_dv', 'ml_dw', 'ml_dtem', 'ml_dq')

        result = {'ml_rho': rho, 'ml_tem': tem, 'ml_pres': pre}
        hgt = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)       # geopotential height
        result['ml_hgt'] = hgt
        qv0 = qv if qv is not None else np.zeros_like(tem)

        # zonal / meridional wind (needed by ml_u/v, the pressure-level slices, tendencies)
        u = v = None
        if W('ml_u', 'ml_v') or need_slices or need_tend:
            vx_pl = np.zeros_like(vmtr.VMTR_GSGAM2_pl)
            u, _u_pl, v, _v_pl = cnvv.cnvvar_vh2uv(vx, vx_pl, vy, vx_pl, vz, vx_pl, grd, gmtr, withcos=False)
            result['ml_u'] = u; result['ml_v'] = v

        # cell-centre vertical wind wc (needed by ml_w/omg, the w slice, tendencies). The
        # half-level w at the rigid top/bottom (k=kmin, kmax+1) is 0 (nicamdc BNDCND_rhow);
        # the diagnostic w carries UNDEF there, so zero the non-finite entries first.
        wc = None
        if W('ml_w', 'ml_omg') or need_slices or need_tend:
            UNDEF = cnst.CONST_UNDEF
            w = np.where(np.isfinite(w) & (np.abs(w) < np.abs(UNDEF) * rdtype(0.1)), w, rdtype(0.0))
            W2C = vmtr.VMTR_W2Cfact
            wc = np.zeros_like(w)
            ks = slice(kmin, kmax + 1)
            wc[:, :, ks, :] = (W2C[:, :, ks, :, 0] * w[:, :, kmin + 1:kmax + 2, :]
                               + W2C[:, :, ks, :, 1] * w[:, :, kmin:kmax + 1, :])
            result['ml_w'] = wc
            if W('ml_omg'):
                result['ml_omg'] = -GRAV * rho * wc

        # potential / virtual potential temperature (th also needed by th_prime)
        th = None
        if W('ml_th', 'ml_thv', 'ml_th_prime'):
            th = tdyn.THRMDYN_th(tem, pre, cnst)
            result['ml_th'] = th
            result['ml_thv'] = th if qv is None else th * (rdtype(1.0) + rdtype(0.61) * qv)

        # zonal/meridional wind scaled by cos(lat)
        if W('ml_ucos', 'ml_vcos'):
            vx_pl = np.zeros_like(vmtr.VMTR_GSGAM2_pl)
            ucos, _uc_pl, vcos, _vc_pl = cnvv.cnvvar_vh2uv(vx, vx_pl, vy, vx_pl, vz, vx_pl, grd, gmtr, withcos=True)
            result['ml_ucos'] = ucos; result['ml_vcos'] = vcos

        # th_prime = th - <th>: deviation from the area-weighted global mean per layer
        # (nicamdc GTL_global_sum_eachlayer: sum var*GMTR_area/VMTR_RGAM**2, MPI-reduced).
        # NOTE: the 2 pole cells are omitted here (interior only) -> a ~1e-4 relative
        # difference in the mean vs nicamdc; the perturbation structure is unaffected.
        if comm is not None and W('ml_th_prime'):
            gmin, gmax = adm.ADM_gmin, adm.ADM_gmax
            area = gmtr.GMTR_area[gmin:gmax + 1, gmin:gmax + 1, :]               # (ii,jj,l)
            wt = area[:, :, None, :] / vmtr.VMTR_RGAM[gmin:gmax + 1, gmin:gmax + 1, :, :] ** 2
            thsub = th[gmin:gmax + 1, gmin:gmax + 1, :, :]
            a_loc = np.sum(wt, axis=(0, 1, 3))                                   # (kall,)
            t_loc = np.sum(thsub * wt, axis=(0, 1, 3))
            kall = th.shape[2]
            mean = np.array([comm.Comm_Stat_sum(t_loc[k]) / comm.Comm_Stat_sum(a_loc[k])
                             for k in range(kall)], dtype=rdtype)
            result['ml_th_prime'] = th - mean[None, None, :, None]

        # pressure-level slices (log-p linear interpolation of u/v/wc/tem to a target pressure)
        if need_slices:
            for tag, plev in (('850', 850.0e2), ('500', 500.0e2), ('250', 250.0e2), ('100', 100.0e2)):
                up, vp, wp, tp = self._plev_interp(pre, (u, v, wc, tem), rdtype(plev), kmin, adm.ADM_kall)
                result['sl_u' + tag] = up; result['sl_v' + tag] = vp
                result['sl_w' + tag] = wp; result['sl_t' + tag] = tp

        # surface pressure (sl_ps): hydrostatic from the lowest level to the surface
        if W('sl_ps'):
            z_srf = grd.GRD_zs[:, :, adm.ADM_K0, :, grd.GRD_ZSFC]         # (i,j,l)
            result['sl_ps'] = pre[:, :, kmin, :] + rho[:, :, kmin, :] * GRAV * (hgt[:, :, kmin, :] - z_srf)

        # moist static energy: cp*T + phi + Lv*qv  (phi = GRD_Z*g)
        if W('ml_mse'):
            result['ml_mse'] = tem * cnst.CONST_CPdry + hgt * GRAV + qv0 * cnst.CONST_LHV

        # relative humidity (liq+ice / liq / ice): qv*rho*Rvap*T/psat*100
        if W('ml_rh', 'ml_rha', 'ml_rhi'):
            rh_num = qv0 * rho * cnst.CONST_Rvap * tem * rdtype(100.0)
            if W('ml_rha'): result['ml_rha'] = rh_num / satr.SATURATION_psat_all(tem, cnst)
            if W('ml_rh'):  result['ml_rh'] = rh_num / satr.SATURATION_psat_liq(tem, cnst)
            if W('ml_rhi'): result['ml_rhi'] = rh_num / satr.SATURATION_psat_ice(tem, cnst)

        # water paths + Terminator chem (column mass integrals sum rho*q*GSGAM2*dgz, kmin..kmax)
        if need_paths:
            ks = slice(kmin, kmax + 1)
            wcol = vmtr.VMTR_GSGAM2[:, :, ks, :] * grd.GRD_dgz[kmin:kmax + 1][None, None, :, None]

            def colint(qx):
                return np.sum(rho[:, :, ks, :] * qx[:, :, ks, :] * wcol, axis=2)

            def sum_tracers(idxs):
                s = np.zeros_like(tem)
                for idx in idxs:
                    if idx is not None and idx >= 0:
                        s = s + q[:, :, :, :, idx]
                return s

            if W('sl_pw'):  result['sl_pw'] = colint(qv0)
            if W('sl_lwp'): result['sl_lwp'] = colint(sum_tracers((getattr(rcnf, 'I_QC', -1), getattr(rcnf, 'I_QR', -1))))
            if W('sl_iwp'): result['sl_iwp'] = colint(sum_tracers((getattr(rcnf, 'I_QI', -1),
                                                                   getattr(rcnf, 'I_QS', -1), getattr(rcnf, 'I_QG', -1))))
            nchem_str = getattr(rcnf, 'NCHEM_STR', -1); nchem_end = getattr(rcnf, 'NCHEM_END', -1)
            if (getattr(rcnf, 'AF_TYPE', '') == 'DCMIP' and ntrc > 0 and nchem_str >= 0 and nchem_end >= 0
                    and W('sl_cl', 'sl_cl2', 'sl_cly')):
                rhodz = colint(np.ones_like(tem))
                result['sl_cl'] = colint(q[:, :, :, :, nchem_str]) / rhodz
                result['sl_cl2'] = colint(q[:, :, :, :, nchem_end]) / rhodz
                result['sl_cly'] = colint(q[:, :, :, :, nchem_str] + rdtype(2.0) * q[:, :, :, :, nchem_end]) / rhodz

        # time-tendency diagnostics: d = (previous_output - current) / (dn*DTL) * 86400,
        # i.e. the interval-AVERAGE rate per day (dn = model steps since the last output);
        # dq in g/kg/day. (nicamdc uses the last-STEP change; this matches it only at
        # PRGout_interval=1, otherwise it is the per-day rate averaged over the interval.)
        # Stateful: the first output seeds the reference and returns 0.
        if need_tend:
            dday = rdtype(86400.0) / dt
            prev_n = self._tend_nstep
            dn = (nstep - prev_n) if (prev_n is not None and nstep is not None) else 0
            cur = {'ml_du': (u, dday), 'ml_dv': (v, dday), 'ml_dw': (wc, dday),
                   'ml_dtem': (tem, dday), 'ml_dq': (qv0, dday * rdtype(1.0e3))}
            for nm, (c, fac) in cur.items():
                if not W(nm):
                    continue
                old = self._tend.get(nm)
                result[nm] = (np.zeros_like(c) if (old is None or dn <= 0)
                              else (old - c) * fac / rdtype(dn))
                self._tend[nm] = c.copy()
            self._tend_nstep = nstep

        # keep only the requested variables (ml_rho/tem/pres/hgt are always computed but
        # dropped here if not requested)
        if want is not None:
            result = {k: val for k, val in result.items() if k in want}
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
