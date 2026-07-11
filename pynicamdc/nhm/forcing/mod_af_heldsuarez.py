#
# mod_af_heldsuarez -- Held & Suarez (1994) artificial forcing -- numpy port
#
# Ported from: nicamdc/src/nhm/forcing/mod_af_heldsuarez.f90
#   subroutines af_heldsuarez_init, af_heldsuarez
#
# Held-Suarez idealized GCM forcing: Rayleigh friction of the low-level winds
# (boundary-layer proxy) + Newtonian relaxation of temperature toward a
# prescribed radiative-equilibrium profile. Fully pointwise except for the
# per-column surface-pressure reference; vectorized over (i,j,k,l).
#
import numpy as np


class AfHeldsuarez:

    _instance = None

    # fixed parameters (nicamdc mod_af_heldsuarez.f90 module scope)
    sigma_b = 0.7                        # sigma level of the PBL top
    Kf      = 1.0 / (1.0 * 86400.0)      # Rayleigh friction rate [1/s]
    Dth_z   = 10.0                       # vertical potential-temperature diff [K]
    Ka      = 1.0 / (40.0 * 86400.0)     # slow (free-atmosphere) relaxation rate [1/s]
    Ks      = 1.0 / (4.0 * 86400.0)      # fast (near-surface) relaxation rate [1/s]

    # set by AF_heldsuarez_init (dry vs moist HS)
    T_eq0 = 315.0                        # equatorial max equilibrium temperature [K]
    DT_y  = 60.0                         # meridional equator-pole temperature diff [K]

    def __init__(self):
        pass

    def AF_heldsuarez_init(self, moist_case=False):
        # dry: Held & Suarez (1994); moist: Thatcher & Jablonowski (2016)
        if moist_case:
            self.T_eq0 = 294.0
            self.DT_y  = 65.0
        else:
            self.T_eq0 = 315.0
            self.DT_y  = 60.0
        return

    def AF_heldsuarez(self, lat, pre, tem, vx, vy, vz, kmin, kmax, cnst, rdtype, xp=np):
        """Held-Suarez forcing tendencies. lat is (i,j,l) or (i,j,1,l); pre/tem/vx/vy/vz
        are (i,j,k,l). Returns (fvx, fvy, fvz, fe), boundaries (kmin-1, kmax+1) zeroed.
        xp = numpy or jax.numpy (device path); fully functional (no in-place mutation)."""
        def R(x): return rdtype(x)
        Rd = cnst.CONST_Rdry; Cp = cnst.CONST_CPdry
        CVdry = cnst.CONST_CVdry; PRE00 = cnst.CONST_PRE00
        sigma_b = R(self.sigma_b); Kf = R(self.Kf)
        T_eq0 = R(self.T_eq0); DT_y = R(self.DT_y); Dth_z = R(self.Dth_z)
        Ka = R(self.Ka); Ks = R(self.Ks)

        latk = lat if lat.ndim == 4 else lat[:, :, None, :]           # (i,j,1,l)

        # normalized pressure sigma = pre / (0.5*(pre[kmin]+pre[kmin-1]))  (per column)
        psref = (R(0.5) * (pre[:, :, kmin, :] + pre[:, :, kmin - 1, :]))[:, :, None, :]
        sigma = pre / psref
        factor = xp.maximum((sigma - sigma_b) / (R(1.0) - sigma_b), R(0.0))

        # Rayleigh friction (boundary-layer damping of low-level winds)
        fvx = -Kf * factor * vx
        fvy = -Kf * factor * vy
        fvz = -Kf * factor * vz

        # Newtonian temperature relaxation toward the radiative-equilibrium profile
        sinlat = xp.abs(xp.sin(latk)); coslat = xp.abs(xp.cos(latk))
        ap0 = xp.abs(pre / PRE00)
        Kt = Ka + (Ks - Ka) * factor * coslat ** 4
        # clamp the log arg away from 0 (the ghost levels, zeroed below, can have pre=0):
        # interior ap0 >> this floor so interior T_eq is bit-identical to the np.log path,
        # and we avoid log(0)->-inf + the numpy errstate warning (xp-agnostic).
        T_eq = xp.maximum(R(200.0),
                          (T_eq0 - DT_y * sinlat ** 2
                           - Dth_z * xp.log(xp.maximum(ap0, R(1e-30))) * coslat ** 2)
                          * ap0 ** (Rd / Cp))
        fe = -Kt * (tem - T_eq) * CVdry

        # zero the ghost levels functionally (nicamdc computes only kmin..kmax; no in-place)
        kk = xp.arange(pre.shape[2]).reshape(1, 1, -1, 1)
        valid = (kk >= kmin) & (kk <= kmax)
        z = R(0.0)
        fvx = xp.where(valid, fvx, z)
        fvy = xp.where(valid, fvy, z)
        fvz = xp.where(valid, fvz, z)
        fe  = xp.where(valid, fe,  z)
        return fvx, fvy, fvz, fe


afhs = AfHeldsuarez()
