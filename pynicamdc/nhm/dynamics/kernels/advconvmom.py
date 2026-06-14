"""
Pure / backend-switchable (numpy <-> jax.numpy) kernels for the two COMM-free
blocks of src_advection_convergence_momentum (mod_src.py L91-447) that bracket
the three src_advection_convergence calls:

  A. velocity merge  (the `else` / sphere branch, L158-236)
       vv{x,y,z} = v{x,y,z} + wc * GRD_x{dir} / rscale
       wc = GRD_cfact * w[k+1] + GRD_dfact * w[k]      (k = kmin .. kmax)
     ghost rows (kmin-1, kmax+1) zeroed.

  B. momentum tendency  (the `else` / sphere branch, L329-436)
       Coriolis:  dvvx += 2 rhog ohm vvy ;  dvvy -= 2 rhog ohm vvx
       horizontalize / vertical split via prd = dvv . (GRD_x/rscale)
       grhogv{x,y,z} = dvv{x,y,z} - prd * GRD_x{dir}/rscale     (k = kmin .. kmax)
       grhogwc       = prd * alpha
       grhogw[k]     = C2Wfact_a * grhogwc[k] + C2Wfact_b * grhogwc[k-1]
                                                                (k = kmin+1 .. kmax)
     ghost rows zeroed (grhogw also zeroes kmin).

Only the sphere (GRD_grid_type != on_plane) path is reproduced here; the
untested on-plane branch is left in mod_src.py. The pole (_pl) variants are
grid-type independent in the original and are reproduced as separate pure
functions.

Boundary / coverage note
------------------------
The original writes the persistent buffers only on their interior k-rows,
leaving the rest stale (never read). For the standard layout kmin == 1,
kmax == kall-2 the only non-interior rows are the two zeroed ghosts, so the
explicit zero-fill here is downstream-identical.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AdvMomCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    XDIR: int
    YDIR: int
    ZDIR: int
    rscale: float
    ohm: float
    alpha: float


# ---------------------------------------------------------------------------
# A. velocity merge
# ---------------------------------------------------------------------------
def compute_merged_velocity_reg(vx, vy, vz, w, cfact, dfact, GRD_x, cfg, xp):
    """Regional sphere velocity merge. Returns vvx, vvy, vvz (i,j,kall,l)."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    kmin, kmax = cfg.kmin, cfg.kmax
    kminp1, kmaxp1, kmaxp2 = kmin + 1, kmax + 1, kmax + 2
    i, j, kall, l = vx.shape

    cf = cfact[kmin:kmaxp1][None, None, :, None]
    df = dfact[kmin:kmaxp1][None, None, :, None]
    wc = cf * w[:, :, kminp1:kmaxp2, :] + df * w[:, :, kmin:kmaxp1, :]

    gx = GRD_x[:, :, 0, :, X][:, :, None, :]
    gy = GRD_x[:, :, 0, :, Y][:, :, None, :]
    gz = GRD_x[:, :, 0, :, Z][:, :, None, :]

    vvx_int = vx[:, :, kmin:kmaxp1, :] + wc * gx / cfg.rscale
    vvy_int = vy[:, :, kmin:kmaxp1, :] + wc * gy / cfg.rscale
    vvz_int = vz[:, :, kmin:kmaxp1, :] + wc * gz / cfg.rscale

    below = xp.zeros((i, j, kmin, l), dtype=vx.dtype)            # k = 0 .. kmin-1
    above = xp.zeros((i, j, kall - kmaxp1, l), dtype=vx.dtype)   # k = kmax+1 .. end
    vvx = xp.concatenate([below, vvx_int, above], axis=2)
    vvy = xp.concatenate([below, vvy_int, above], axis=2)
    vvz = xp.concatenate([below, vvz_int, above], axis=2)
    return vvx, vvy, vvz


def compute_merged_velocity_pl(vx_pl, vy_pl, vz_pl, w_pl, cfact, dfact, GRD_x_pl, cfg, xp):
    """Pole velocity merge. Returns vvx_pl, vvy_pl, vvz_pl (g,kall,l)."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    kmin, kmax = cfg.kmin, cfg.kmax
    kminp1, kmaxp1, kmaxp2 = kmin + 1, kmax + 1, kmax + 2
    g, kall, l = vx_pl.shape

    cf = cfact[kmin:kmaxp1][None, :, None]
    df = dfact[kmin:kmaxp1][None, :, None]
    wc = cf * w_pl[:, kminp1:kmaxp2, :] + df * w_pl[:, kmin:kmaxp1, :]

    gx = GRD_x_pl[:, 0, :, X][:, None, :]
    gy = GRD_x_pl[:, 0, :, Y][:, None, :]
    gz = GRD_x_pl[:, 0, :, Z][:, None, :]

    vvx_int = vx_pl[:, kmin:kmaxp1, :] + (wc * gx / cfg.rscale)
    vvy_int = vy_pl[:, kmin:kmaxp1, :] + (wc * gy / cfg.rscale)
    vvz_int = vz_pl[:, kmin:kmaxp1, :] + (wc * gz / cfg.rscale)

    below = xp.zeros((g, kmin, l), dtype=vx_pl.dtype)
    above = xp.zeros((g, kall - kmaxp1, l), dtype=vx_pl.dtype)
    vvx_pl = xp.concatenate([below, vvx_int, above], axis=1)
    vvy_pl = xp.concatenate([below, vvy_int, above], axis=1)
    vvz_pl = xp.concatenate([below, vvz_int, above], axis=1)
    return vvx_pl, vvy_pl, vvz_pl


# ---------------------------------------------------------------------------
# B. momentum tendency
# ---------------------------------------------------------------------------
def compute_momentum_tendency_reg(dvvx, dvvy, dvvz, rhog, vvx, vvy, GRD_x, C2Wfact, cfg, xp):
    """Regional sphere tendency. Returns grhogvx, grhogvy, grhogvz, grhogw."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    kmin, kmax = cfg.kmin, cfg.kmax
    kminp1, kmaxp1 = kmin + 1, kmax + 1
    ohm, alpha, rscale = cfg.ohm, cfg.alpha, cfg.rscale
    i, j, kall, l = dvvx.shape

    rh = rhog[:, :, kmin:kmaxp1, :]
    vvy_s = vvy[:, :, kmin:kmaxp1, :]
    vvx_s = vvx[:, :, kmin:kmaxp1, :]

    # Coriolis (in original: dvvx -= -2 rhog (ohm vvy); dvvy -= 2 rhog (ohm vvx))
    dvvx_c = dvvx[:, :, kmin:kmaxp1, :] - (-2.0 * rh * (ohm * vvy_s))
    dvvy_c = dvvy[:, :, kmin:kmaxp1, :] - (2.0 * rh * (ohm * vvx_s))
    dvvz_c = dvvz[:, :, kmin:kmaxp1, :]

    gx = GRD_x[:, :, 0, :, X][:, :, None, :] / rscale
    gy = GRD_x[:, :, 0, :, Y][:, :, None, :] / rscale
    gz = GRD_x[:, :, 0, :, Z][:, :, None, :] / rscale

    prd = dvvx_c * gx + dvvy_c * gy + dvvz_c * gz
    gvx_int = dvvx_c - prd * gx
    gvy_int = dvvy_c - prd * gy
    gvz_int = dvvz_c - prd * gz
    gwc_int = prd * alpha                                   # k = kmin .. kmax

    f1 = C2Wfact[:, :, kminp1:kmaxp1, :, 0]
    f2 = C2Wfact[:, :, kminp1:kmaxp1, :, 1]
    gw_int = f1 * gwc_int[:, :, 1:, :] + f2 * gwc_int[:, :, 0:-1, :]  # k = kmin+1 .. kmax

    below = xp.zeros((i, j, kmin, l), dtype=dvvx.dtype)
    above = xp.zeros((i, j, kall - kmaxp1, l), dtype=dvvx.dtype)
    grhogvx = xp.concatenate([below, gvx_int, above], axis=2)
    grhogvy = xp.concatenate([below, gvy_int, above], axis=2)
    grhogvz = xp.concatenate([below, gvz_int, above], axis=2)

    wbelow = xp.zeros((i, j, kminp1, l), dtype=dvvx.dtype)   # k = 0 .. kmin
    wabove = xp.zeros((i, j, kall - kmaxp1, l), dtype=dvvx.dtype)
    grhogw = xp.concatenate([wbelow, gw_int, wabove], axis=2)
    return grhogvx, grhogvy, grhogvz, grhogw


def compute_momentum_tendency_pl(dvvx_pl, dvvy_pl, dvvz_pl, rhog_pl, vvx_pl, vvy_pl,
                                 GRD_x_pl, C2Wfact_pl, cfg, xp):
    """Pole tendency. Returns grhogvx_pl, grhogvy_pl, grhogvz_pl, grhogw_pl."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    kmin, kmax = cfg.kmin, cfg.kmax
    kminp1, kmaxp1 = kmin + 1, kmax + 1
    ohm, alpha, rscale = cfg.ohm, cfg.alpha, cfg.rscale
    g, kall, l = dvvx_pl.shape

    rh = rhog_pl[:, kmin:kmaxp1, :]
    vvy_s = vvy_pl[:, kmin:kmaxp1, :]
    vvx_s = vvx_pl[:, kmin:kmaxp1, :]

    dvvx_c = dvvx_pl[:, kmin:kmaxp1, :] - (-2.0 * rh * (ohm * vvy_s))
    dvvy_c = dvvy_pl[:, kmin:kmaxp1, :] - (2.0 * rh * (ohm * vvx_s))
    dvvz_c = dvvz_pl[:, kmin:kmaxp1, :]

    gx = GRD_x_pl[:, 0, :, X][:, None, :] / rscale
    gy = GRD_x_pl[:, 0, :, Y][:, None, :] / rscale
    gz = GRD_x_pl[:, 0, :, Z][:, None, :] / rscale

    prd = dvvx_c * gx + dvvy_c * gy + dvvz_c * gz
    gvx_int = dvvx_c - prd * gx
    gvy_int = dvvy_c - prd * gy
    gvz_int = dvvz_c - prd * gz
    gwc_int = prd * alpha

    f1 = C2Wfact_pl[:, kminp1:kmaxp1, :, 0]
    f2 = C2Wfact_pl[:, kminp1:kmaxp1, :, 1]
    gw_int = f1 * gwc_int[:, 1:, :] + f2 * gwc_int[:, 0:-1, :]

    below = xp.zeros((g, kmin, l), dtype=dvvx_pl.dtype)
    above = xp.zeros((g, kall - kmaxp1, l), dtype=dvvx_pl.dtype)
    grhogvx_pl = xp.concatenate([below, gvx_int, above], axis=1)
    grhogvy_pl = xp.concatenate([below, gvy_int, above], axis=1)
    grhogvz_pl = xp.concatenate([below, gvz_int, above], axis=1)

    wbelow = xp.zeros((g, kminp1, l), dtype=dvvx_pl.dtype)
    wabove = xp.zeros((g, kall - kmaxp1, l), dtype=dvvx_pl.dtype)
    grhogw_pl = xp.concatenate([wbelow, gw_int, wabove], axis=1)
    return grhogvx_pl, grhogvy_pl, grhogvz_pl, grhogw_pl
