"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the whole
COMM-free body of OPRT3D_divdamp (mod_oprt.py): the 3D divergence-damping
operator used once per numfilter_divdamp call (before the lap COMM).

Structure (identical horizontal machinery to OPRT_divdamp, plus a vertical
contribution):

  1. half-level vertical mass flux  rhogw_vm  (C2WfactGz interpolation * RGAMH
     + rhogw * RGSQRTH), zero on the kmin / kmax+1 faces.
  2. velocity-metric products       rhogv{x,y,z}_vm = rhogv{x,y,z} * RGAM
  3. triangle-point intermediate    sclt[TI|TJ] = coef_intp interpolation of the
     three *_vm plus the vertical rhogw_vm difference (sclt_rhogw).
  4. singular-point fix             sclt[0,0,TI] blended with sclt[1,0,TJ] via
     pntmask (active only on the panel that owns the singular corner).
  5. cell value                     ddivd{x,y,z} = coef_diff combination of the
     6 surrounding triangle halves.

Only the vertical interior k = kmin .. kmax carries non-trivial sclt/ddivd; the
kmin-1 and kmax+1 rows are zero, matching the original.

Boundary / coverage and pole notes: same conventions as kernels/oprtdivdamp.py
(interior i,j = 1..gmax computed, i/j border padded zero -- halos are
COMM-refilled; pole spoke loop vectorised with xp.roll, sequential-equivalent
sum reduction). Bit-identical to the original under numpy.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Oprt3DDivdampCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    kmin: int
    kmax: int
    gmax: int
    gslf_pl: int
    gmin_pl: int
    gmax_pl: int
    k0: int
    TI: int
    TJ: int
    XDIR: int
    YDIR: int
    ZDIR: int


def _build_rhogw_vm(rhogvx, rhogvy, rhogvz, rhogw,
                    C2WfactGz, RGAMH, RGSQRTH, cfg, kax, xp):
    """Half-level vertical mass flux on the interior, zero-padded on kax.

    Works for both regional (kax=2, arrays (i,j,k,l)) and pole
    (kax=1, arrays (g,k,l)); C2WfactGz has a trailing size-6 axis.
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    kall = rhogw.shape[kax]

    def kslc(a, lo, hi):
        sl = [slice(None)] * a.ndim
        sl[kax] = slice(lo, hi)
        return a[tuple(sl)]

    cz = kslc(C2WfactGz, kmin + 1, kmax + 1)        # (..., kmax-kmin, ..., 6)
    horiz = (
        cz[..., 0] * kslc(rhogvx, kmin + 1, kmax + 1) +
        cz[..., 1] * kslc(rhogvx, kmin,     kmax)     +
        cz[..., 2] * kslc(rhogvy, kmin + 1, kmax + 1) +
        cz[..., 3] * kslc(rhogvy, kmin,     kmax)     +
        cz[..., 4] * kslc(rhogvz, kmin + 1, kmax + 1) +
        cz[..., 5] * kslc(rhogvz, kmin,     kmax)
    )
    interior = (horiz * kslc(RGAMH, kmin + 1, kmax + 1)
                + kslc(rhogw, kmin + 1, kmax + 1) * kslc(RGSQRTH, kmin + 1, kmax + 1))

    shp = list(rhogw.shape)
    shp_lo = shp.copy(); shp_lo[kax] = kmin + 1            # zeros k = 0..kmin
    shp_hi = shp.copy(); shp_hi[kax] = kall - (kmax + 1)   # zeros k = kmax+1..end
    lower = xp.zeros(shp_lo, dtype=rhogw.dtype)
    upper = xp.zeros(shp_hi, dtype=rhogw.dtype)
    return xp.concatenate([lower, interior, upper], axis=kax)


def _oprt3d_reg(rhogvx, rhogvy, rhogvz, rhogw,
                coef_intp, coef_diff, C2WfactGz, RGAMH, RGSQRTH, RGAM,
                rdgz, pntmask, cfg, xp):
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    TI, TJ = cfg.TI, cfg.TJ
    kmin, kmax = cfg.kmin, cfg.kmax
    i = rhogvx.shape[0]; j = rhogvx.shape[1]; kall = rhogvx.shape[2]
    G = cfg.gmax + 1

    # vertical half-level flux (full k, zeros outside interior)
    rw = _build_rhogw_vm(rhogvx, rhogvy, rhogvz, rhogw,
                         C2WfactGz, RGAMH, RGSQRTH, cfg, 2, xp)

    # velocity-metric products on k = kmin..kmax
    K = slice(kmin, kmax + 1)
    vmx = rhogvx[:, :, K, :] * RGAM[:, :, K, :]
    vmy = rhogvy[:, :, K, :] * RGAM[:, :, K, :]
    vmz = rhogvz[:, :, K, :] * RGAM[:, :, K, :]

    R   = slice(0, G)          # 0..gmax
    Rp1 = slice(1, G + 1)      # 1..gmax+1
    UP  = slice(kmin + 1, kmax + 2)
    DN  = slice(kmin,     kmax + 1)
    rdz = rdgz[K][None, None, :, None]

    # vertical rhogw difference at the TI / TJ triangle corners
    sclt_rhogw_TI = (
        (rw[R,   R,   UP, :] + rw[Rp1, R,   UP, :] + rw[Rp1, Rp1, UP, :]) -
        (rw[R,   R,   DN, :] + rw[Rp1, R,   DN, :] + rw[Rp1, Rp1, DN, :])
    ) / 3.0 * rdz
    sclt_rhogw_TJ = (
        (rw[R,   R,   UP, :] + rw[Rp1, Rp1, UP, :] + rw[R,   Rp1, UP, :]) -
        (rw[R,   R,   DN, :] + rw[Rp1, Rp1, DN, :] + rw[R,   Rp1, DN, :])
    ) / 3.0 * rdz

    c = coef_intp[R, R]        # (G, G, 1, l, 3, 3, 2)
    sclt_TI = (
        c[:, :, :, :, X, 0, TI] * vmx[R,   R]   + c[:, :, :, :, X, 1, TI] * vmx[Rp1, R]   + c[:, :, :, :, X, 2, TI] * vmx[Rp1, Rp1] +
        c[:, :, :, :, Y, 0, TI] * vmy[R,   R]   + c[:, :, :, :, Y, 1, TI] * vmy[Rp1, R]   + c[:, :, :, :, Y, 2, TI] * vmy[Rp1, Rp1] +
        c[:, :, :, :, Z, 0, TI] * vmz[R,   R]   + c[:, :, :, :, Z, 1, TI] * vmz[Rp1, R]   + c[:, :, :, :, Z, 2, TI] * vmz[Rp1, Rp1] +
        sclt_rhogw_TI
    )
    sclt_TJ = (
        c[:, :, :, :, X, 0, TJ] * vmx[R,   R]   + c[:, :, :, :, X, 1, TJ] * vmx[Rp1, Rp1] + c[:, :, :, :, X, 2, TJ] * vmx[R,   Rp1] +
        c[:, :, :, :, Y, 0, TJ] * vmy[R,   R]   + c[:, :, :, :, Y, 1, TJ] * vmy[Rp1, Rp1] + c[:, :, :, :, Y, 2, TJ] * vmy[R,   Rp1] +
        c[:, :, :, :, Z, 0, TJ] * vmz[R,   R]   + c[:, :, :, :, Z, 1, TJ] * vmz[Rp1, Rp1] + c[:, :, :, :, Z, 2, TJ] * vmz[R,   Rp1] +
        sclt_rhogw_TJ
    )
    # sclt_TI / sclt_TJ : (G, G, nk, l)

    # singular-point fix: sclt[0,0,TI] = sclt[0,0,TI]*mask0 + sclt[1,0,TJ]*mask1
    # pntmask is int32; cast to the working float dtype so the blend stays in
    # that precision (float32_array * int32_array would upcast to float64 under
    # NEP50, contaminating the whole ddivd assembly in float32 mode). pntmask is
    # 0/1, so the cast is exact and matches the original's downcast-on-assign.
    fdt = sclt_TI.dtype
    mask0 = pntmask[0, :, 0].astype(fdt)          # (l,)
    mask1 = pntmask[0, :, 1].astype(fdt)
    corner = sclt_TI[0:1, 0:1] * mask0 + sclt_TJ[1:2, 0:1] * mask1   # (1,1,nk,l)
    row0 = xp.concatenate([corner, sclt_TI[0:1, 1:]], axis=1)        # (1,G,nk,l)
    sclt_TI = xp.concatenate([row0, sclt_TI[1:]], axis=0)            # (G,G,nk,l)

    # cell assembly (identical stencil to OPRT_divdamp)
    II   = slice(1, G)         # 1..gmax
    IIm1 = slice(0, G - 1)     # 0..gmax-1
    s_ij_TI    = sclt_TI[II,   II]
    s_ij_TJ    = sclt_TJ[II,   II]
    s_im1_TI   = sclt_TI[IIm1, II]
    s_im1m1_TJ = sclt_TJ[IIm1, IIm1]
    s_im1m1_TI = sclt_TI[IIm1, IIm1]
    s_jm1_TJ   = sclt_TJ[II,   IIm1]

    pad_i = (1, i - G)
    pad_j = (1, j - G)
    pad_k = (kmin, kall - 1 - kmax)

    def ddivd(d):
        cd = coef_diff[II, II, :, :, d, :]      # (gmax, gmax, 1, l, 6)
        out = (
            cd[:, :, :, :, 0] * (s_ij_TI    + s_ij_TJ) +
            cd[:, :, :, :, 1] * (s_ij_TJ    + s_im1_TI) +
            cd[:, :, :, :, 2] * (s_im1_TI   + s_im1m1_TJ) +
            cd[:, :, :, :, 3] * (s_im1m1_TJ + s_im1m1_TI) +
            cd[:, :, :, :, 4] * (s_im1m1_TI + s_jm1_TJ) +
            cd[:, :, :, :, 5] * (s_jm1_TJ   + s_ij_TI)
        )
        return xp.pad(out, (pad_i, pad_j, pad_k, (0, 0)))

    return ddivd(X), ddivd(Y), ddivd(Z)


def _oprt3d_pl(rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
               coef_intp_pl, coef_diff_pl,
               C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl, RGAM_pl, rdgz, cfg, xp):
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    n = cfg.gslf_pl
    k0 = cfg.k0
    kmin, kmax = cfg.kmin, cfg.kmax
    v0, v1 = cfg.gmin_pl, cfg.gmax_pl + 1
    g = rhogvx_pl.shape[0]; kall = rhogvx_pl.shape[1]; l = rhogvx_pl.shape[2]

    rw = _build_rhogw_vm(rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
                         C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl, cfg, 1, xp)

    K = slice(kmin, kmax + 1)
    vmx = rhogvx_pl[:, K, :] * RGAM_pl[:, K, :]
    vmy = rhogvy_pl[:, K, :] * RGAM_pl[:, K, :]
    vmz = rhogvz_pl[:, K, :] * RGAM_pl[:, K, :]

    Vx_self, Vy_self, Vz_self = vmx[n], vmy[n], vmz[n]            # (nk, l)
    Vx_ij, Vy_ij, Vz_ij = vmx[v0:v1], vmy[v0:v1], vmz[v0:v1]     # (nv, nk, l)
    Vx_p1 = xp.roll(Vx_ij, -1, axis=0)
    Vy_p1 = xp.roll(Vy_ij, -1, axis=0)
    Vz_p1 = xp.roll(Vz_ij, -1, axis=0)

    UP = slice(kmin + 1, kmax + 2); DN = slice(kmin, kmax + 1)
    rw_n_up = rw[n, UP, :];  rw_n_dn = rw[n, DN, :]              # (nk, l)
    rw_ij_up = rw[v0:v1, UP, :]; rw_ij_dn = rw[v0:v1, DN, :]    # (nv, nk, l)
    rw_p1_up = xp.roll(rw_ij_up, -1, axis=0)
    rw_p1_dn = xp.roll(rw_ij_dn, -1, axis=0)
    sclt_rhogw = (
        (rw_n_up + rw_ij_up + rw_p1_up) - (rw_n_dn + rw_ij_dn + rw_p1_dn)
    ) / 3.0 * rdgz[K][None, :, None]                            # (nv, nk, l)

    cI = coef_intp_pl[v0:v1, k0]                                 # (nv, l, 3, 3)

    def b(a):
        return a[:, None, :]                                    # (nv, l) -> (nv, 1, l)

    sclt_v = (
        b(cI[:, :, X, 0]) * Vx_self + b(cI[:, :, X, 1]) * Vx_ij + b(cI[:, :, X, 2]) * Vx_p1 +
        b(cI[:, :, Y, 0]) * Vy_self + b(cI[:, :, Y, 1]) * Vy_ij + b(cI[:, :, Y, 2]) * Vy_p1 +
        b(cI[:, :, Z, 0]) * Vz_self + b(cI[:, :, Z, 1]) * Vz_ij + b(cI[:, :, Z, 2]) * Vz_p1 +
        sclt_rhogw
    )                                                           # (nv, nk, l)

    sclt_ijm1 = xp.roll(sclt_v, 1, axis=0)
    pair = sclt_ijm1 + sclt_v
    cd = coef_diff_pl[v0:v1, k0]                                 # (nv, l, 3)
    ddx = (b(cd[:, :, X]) * pair).sum(axis=0)                    # (nk, l)
    ddy = (b(cd[:, :, Y]) * pair).sum(axis=0)
    ddz = (b(cd[:, :, Z]) * pair).sum(axis=0)

    k_lo = xp.zeros((kmin, l), dtype=ddx.dtype)
    k_hi = xp.zeros((kall - 1 - kmax, l), dtype=ddx.dtype)

    def place(row):
        rowk = xp.concatenate([k_lo, row, k_hi], axis=0)[None]   # (1, kall, l)
        above = xp.zeros((n, kall, l), dtype=row.dtype)
        below = xp.zeros((g - n - 1, kall, l), dtype=row.dtype)
        return xp.concatenate([above, rowk, below], axis=0)

    return place(ddx), place(ddy), place(ddz)


def compute_oprt3d_divdamp(
    rhogvx, rhogvy, rhogvz, rhogw,
    rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
    coef_intp, coef_diff, coef_intp_pl, coef_diff_pl,
    C2WfactGz, RGAMH, RGSQRTH, RGAM,
    C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl, RGAM_pl,
    rdgz, pntmask, cfg: Oprt3DDivdampCfg, xp,
):
    """Pure version of OPRT3D_divdamp (whole COMM-free body).

    Returns (ddivdx, ddivdy, ddivdz, ddivdx_pl, ddivdy_pl, ddivdz_pl).
    The _pl outputs are shape-correct zero arrays when not have_pl.
    """
    dx, dy, dz = _oprt3d_reg(
        rhogvx, rhogvy, rhogvz, rhogw,
        coef_intp, coef_diff, C2WfactGz, RGAMH, RGSQRTH, RGAM,
        rdgz, pntmask, cfg, xp,
    )
    if cfg.have_pl:
        dx_pl, dy_pl, dz_pl = _oprt3d_pl(
            rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
            coef_intp_pl, coef_diff_pl,
            C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl, RGAM_pl, rdgz, cfg, xp,
        )
    else:
        dx_pl = xp.zeros_like(rhogvx_pl)
        dy_pl = xp.zeros_like(rhogvx_pl)
        dz_pl = xp.zeros_like(rhogvx_pl)
    return dx, dy, dz, dx_pl, dy_pl, dz_pl
