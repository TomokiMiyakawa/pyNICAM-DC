"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the whole
COMM-free body of src_flux_convergence (mod_src.py L545-822).

It reproduces, as a single pure function:

  1. velocity-metric products      rhogv{x,y,z}_vm = rhogv{x,y,z} * RGAM
  2. half-level vertical flux       rhogw_vmh = horiz(C2WfactGz) * RGAMH + vert
     where vert = vertical_flag * rhogw * RGSQRTH   (vertical_flag from fluxtype)
  3. horizontal flux convergence    div_rhogvh = OPRT_divergence(*_vm, coef_div)
     (the COMM-free 7-point scalar-divergence stencil is inlined here)
  4. total flux convergence         grhog = -div_rhogvh - d(rhogw_vmh)/dz

The OPRT_divergence math is inlined (not delegated) so the entire block is one
backend-switchable, jit-able function. The original code is unchanged elsewhere;
this duplicates only the divergence arithmetic, which is COMM-free.

Boundary / coverage note
------------------------
The original writes the persistent buffers rhogw_vmh / grhog only on their
interior k-rows, leaving other rows as stale (never-read) buffer contents.
This functional version zeros every row outside the computed interior, which is
downstream-identical for the standard layout kmin == 1.

  rhogw_vmh : filled on k = kmin+1 .. kmax, zeroed on kmin and kmax+1.
              flux_diff downstream reads k = kmin .. kmax+1 only.
  grhog     : filled on k = kmin .. kmax, zeroed on kmin-1 and kmax+1.

Pole (_pl) divergence: only the self row (gslf_pl) is non-zero, exactly as the
original OPRT_divergence loop (which sums v = gslf_pl .. gmax_pl into that row).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FluxConvCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    XDIR: int
    YDIR: int
    ZDIR: int
    gslf_pl: int
    gmax_pl: int
    I_SRC_default: int
    I_SRC_horizontal: int


def _divergence(vx, vy, vz, coef_div, cfg, xp):
    """Inlined COMM-free regional 7-point scalar divergence (OPRT_divergence).

    vx/y/z   : (i, j, k, l)
    coef_div : (i, j, 1, l, 3, 7)   (KNONE k-dim broadcasts over k)
    returns  : (i, j, k, l)  zero on the i/j border rows.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    i = vx.shape[0]
    j = vx.shape[1]

    # interior slices (isl = 1..i-2 etc.) and their +/- neighbours
    c = coef_div[1:i - 1, 1:j - 1]            # (i-2, j-2, 1, l, 3, 7)

    def stencil(v, d):
        return (
            c[:, :, :, :, d, 0] * v[1:i - 1, 1:j - 1, :, :] +
            c[:, :, :, :, d, 1] * v[2:i,     1:j - 1, :, :] +
            c[:, :, :, :, d, 2] * v[2:i,     2:j,     :, :] +
            c[:, :, :, :, d, 3] * v[1:i - 1, 2:j,     :, :] +
            c[:, :, :, :, d, 4] * v[0:i - 2, 1:j - 1, :, :] +
            c[:, :, :, :, d, 5] * v[0:i - 2, 0:j - 2, :, :] +
            c[:, :, :, :, d, 6] * v[1:i - 1, 0:j - 2, :, :]
        )

    interior = stencil(vx, X) + stencil(vy, Y) + stencil(vz, Z)  # (i-2, j-2, k, l)
    return xp.pad(interior, ((1, 1), (1, 1), (0, 0), (0, 0)))


def _divergence_pl(vx_pl, vy_pl, vz_pl, coef_div_pl, cfg, xp):
    """Inlined COMM-free pole scalar divergence.

    vx/y/z_pl   : (g, k, l)
    coef_div_pl : (g, 1, l, 3)
    returns     : (g, k, l)  only the self row (gslf_pl) is non-zero.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    n = cfg.gslf_pl
    v0, v1 = cfg.gslf_pl, cfg.gmax_pl + 1   # v = gslf_pl .. gmax_pl inclusive
    g = vx_pl.shape[0]

    cx = coef_div_pl[v0:v1, 0, :, X][:, None, :]   # (nv, 1, l)
    cy = coef_div_pl[v0:v1, 0, :, Y][:, None, :]
    cz = coef_div_pl[v0:v1, 0, :, Z][:, None, :]

    # accumulate (x+y+z) per stencil point v, then reduce over v -- this matches
    # the original OPRT_divergence loop's summation order (bit-exact in numpy).
    per_v = (cx * vx_pl[v0:v1, :, :] +
             cy * vy_pl[v0:v1, :, :] +
             cz * vz_pl[v0:v1, :, :])          # (nv, k, l)
    row = per_v.sum(axis=0)                     # (k, l)

    # place `row` at g-index n, zeros elsewhere (n == gslf_pl == 0 in practice)
    rowk = row[None, :, :]
    above = xp.zeros((n, row.shape[0], row.shape[1]), dtype=row.dtype)
    below = xp.zeros((g - n - 1, row.shape[0], row.shape[1]), dtype=row.dtype)
    return xp.concatenate([above, rowk, below], axis=0)


def compute_flux_convergence(
    rhogvx, rhogvy, rhogvz, rhogw,
    rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
    RGAM, RGAMH, RGSQRTH, C2WfactGz, coef_div, rdgz,
    RGAM_pl, RGAMH_pl, RGSQRTH_pl, C2WfactGz_pl, coef_div_pl,
    fluxtype, cfg: FluxConvCfg, xp,
):
    """Pure version of mod_src.py src_flux_convergence (whole COMM-free body).

    Returns (grhog, grhog_pl). grhog_pl is a placeholder (zeros_like) shape-wise
    correct array even when not have_pl.
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1 = kmin - 1
    kminp1 = kmin + 1
    kmaxp1 = kmax + 1
    kmaxp2 = kmax + 2

    if fluxtype == cfg.I_SRC_default:
        vflag = xp.asarray(1.0, dtype=rhogw.dtype)
    else:  # I_SRC_horizontal
        vflag = xp.asarray(0.0, dtype=rhogw.dtype)

    i, j, kall, l = rhogw.shape

    # --- horizontal flux: velocity-metric products (full k) ---
    vx_vm = rhogvx * RGAM
    vy_vm = rhogvy * RGAM
    vz_vm = rhogvz * RGAM

    # --- half-level vertical flux on k = kmin+1 .. kmax ---
    cz = C2WfactGz[:, :, kminp1:kmaxp1, :, :]
    horiz = (
        cz[:, :, :, :, 0] * rhogvx[:, :, kminp1:kmaxp1, :] +
        cz[:, :, :, :, 1] * rhogvx[:, :, kmin:kmax,     :] +
        cz[:, :, :, :, 2] * rhogvy[:, :, kminp1:kmaxp1, :] +
        cz[:, :, :, :, 3] * rhogvy[:, :, kmin:kmax,     :] +
        cz[:, :, :, :, 4] * rhogvz[:, :, kminp1:kmaxp1, :] +
        cz[:, :, :, :, 5] * rhogvz[:, :, kmin:kmax,     :]
    )
    horiz = horiz * RGAMH[:, :, kminp1:kmaxp1, :]
    vert = vflag * rhogw[:, :, kminp1:kmaxp1, :] * RGSQRTH[:, :, kminp1:kmaxp1, :]
    rhogw_vmh_int = horiz + vert                          # k = kmin+1 .. kmax

    # rhogw_vmh: zeros below (0..kmin), interior (kmin+1..kmax), zeros above
    lower = xp.zeros((i, j, kminp1,          l), dtype=rhogw.dtype)  # 0..kmin
    upper = xp.zeros((i, j, kall - kmaxp1,   l), dtype=rhogw.dtype)  # kmax+1..end
    rhogw_vmh = xp.concatenate([lower, rhogw_vmh_int, upper], axis=2)

    # --- horizontal flux convergence ---
    div_rhogvh = _divergence(vx_vm, vy_vm, vz_vm, coef_div, cfg, xp)

    # --- total flux convergence: grhog on k = kmin .. kmax ---
    rdgz_b = rdgz[kmin:kmaxp1][None, None, :, None]
    flux_diff = rhogw_vmh[:, :, kminp1:kmaxp2, :] - rhogw_vmh[:, :, kmin:kmaxp1, :]
    grhog_int = -div_rhogvh[:, :, kmin:kmaxp1, :] - flux_diff * rdgz_b

    glower = xp.zeros((i, j, kmin,          l), dtype=rhogw.dtype)   # 0..kmin-1
    gupper = xp.zeros((i, j, kall - kmaxp1, l), dtype=rhogw.dtype)   # kmax+1..end
    grhog = xp.concatenate([glower, grhog_int, gupper], axis=2)

    # --- pole region ---
    grhog_pl = xp.zeros_like(rhogw_pl)
    if cfg.have_pl:
        g, kall_pl, l_pl = rhogw_pl.shape

        vx_vm_pl = rhogvx_pl * RGAM_pl
        vy_vm_pl = rhogvy_pl * RGAM_pl
        vz_vm_pl = rhogvz_pl * RGAM_pl

        czp = C2WfactGz_pl[:, kminp1:kmaxp1, :, :]
        horiz_pl = (
            czp[:, :, :, 0] * rhogvx_pl[:, kminp1:kmaxp1, :] +
            czp[:, :, :, 1] * rhogvx_pl[:, kmin:kmax,     :] +
            czp[:, :, :, 2] * rhogvy_pl[:, kminp1:kmaxp1, :] +
            czp[:, :, :, 3] * rhogvy_pl[:, kmin:kmax,     :] +
            czp[:, :, :, 4] * rhogvz_pl[:, kminp1:kmaxp1, :] +
            czp[:, :, :, 5] * rhogvz_pl[:, kmin:kmax,     :]
        )
        horiz_pl = horiz_pl * RGAMH_pl[:, kminp1:kmaxp1, :]
        vert_pl = vflag * rhogw_pl[:, kminp1:kmaxp1, :] * RGSQRTH_pl[:, kminp1:kmaxp1, :]
        rhogw_vmh_pl_int = horiz_pl + vert_pl

        lower_pl = xp.zeros((g, kminp1,            l_pl), dtype=rhogw_pl.dtype)
        upper_pl = xp.zeros((g, kall_pl - kmaxp1,  l_pl), dtype=rhogw_pl.dtype)
        rhogw_vmh_pl = xp.concatenate([lower_pl, rhogw_vmh_pl_int, upper_pl], axis=1)

        div_rhogvh_pl = _divergence_pl(vx_vm_pl, vy_vm_pl, vz_vm_pl, coef_div_pl, cfg, xp)

        rdgz_bp = rdgz[kmin:kmaxp1][None, :, None]
        flux_diff_pl = rhogw_vmh_pl[:, kminp1:kmaxp2, :] - rhogw_vmh_pl[:, kmin:kmaxp1, :]
        grhog_pl_int = -div_rhogvh_pl[:, kmin:kmaxp1, :] - flux_diff_pl * rdgz_bp

        glower_pl = xp.zeros((g, kmin,             l_pl), dtype=rhogw_pl.dtype)
        gupper_pl = xp.zeros((g, kall_pl - kmaxp1, l_pl), dtype=rhogw_pl.dtype)
        grhog_pl = xp.concatenate([glower_pl, grhog_pl_int, gupper_pl], axis=1)

    return grhog, grhog_pl
