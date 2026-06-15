"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the whole
COMM-free body of OPRT_divdamp (mod_oprt.py): the 2D divergence-damping
operator used by numfilter_divdamp (inner lap loop) and numfilter_divdamp_2d.

It reproduces, as a single pure function:

  1. triangle-point intermediate `sclt` for the two sub-triangles TI / TJ,
     each a 9-term interpolation (coef_intp) of the three velocity components.
  2. cell value `ddivd{x,y,z}` = sum over the 6 surrounding triangle halves
     (coef_diff) of (sclt_a + sclt_b).

The OPRT_divdamp math is a horizontal-only stencil (no vertical / vmtr terms),
high arithmetic intensity, identical structure to the divergence operators
already ported. Inlined here so the whole block is one jit-able function.

Boundary / coverage note
------------------------
The original computes `sclt` on i,j = 0..gmax and writes `ddivd` on
i,j = 1..gmax+1, the gmax+1 row/col being contaminated by never-initialised
`sclt` (CONST_UNDEF) halo cells. Those cells are halos that the subsequent
COMM_data_transfer overwrites before any interior read, so this functional
version instead computes `ddivd` on the true interior i,j = 1..gmax (where all
`sclt` reads are valid) and pads the i/j border rows with zero -- downstream
identical for the standard layout.

Pole (_pl): the original loops the spokes v = gmin_pl..gmax_pl with cyclic
neighbours ijp1 / ijm1; here the spoke loop is vectorised with xp.roll and the
reduction is a plain sum over the (small) spoke axis, which matches the original
sequential accumulation bit-for-bit in numpy. Only the self row (gslf_pl) is
non-zero.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OprtDivdampCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
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


def _divdamp_reg(vx, vy, vz, coef_intp, coef_diff, cfg, xp):
    """Regional 2D divergence-damping stencil.

    vx/y/z    : (i, j, k, l)
    coef_intp : (i, j, k?, l, 3(DIR), 3(point), 2(TI/TJ))
    coef_diff : (i, j, k?, l, 3(DIR), 6)
    returns   : (ddivdx, ddivdy, ddivdz), each (i, j, k, l), zero on i/j border.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    TI, TJ = cfg.TI, cfg.TJ
    i = vx.shape[0]
    j = vx.shape[1]
    G = cfg.gmax + 1            # sclt region size: indices 0..gmax

    R   = slice(0, G)           # 0..gmax
    Rp1 = slice(1, G + 1)       # 1..gmax+1

    c = coef_intp[R, R]         # (G, G, k?, l, 3, 3, 2)

    sclt_TI = (
        c[:, :, :, :, X, 0, TI] * vx[R,   R,   :, :] +
        c[:, :, :, :, X, 1, TI] * vx[Rp1, R,   :, :] +
        c[:, :, :, :, X, 2, TI] * vx[Rp1, Rp1, :, :] +
        c[:, :, :, :, Y, 0, TI] * vy[R,   R,   :, :] +
        c[:, :, :, :, Y, 1, TI] * vy[Rp1, R,   :, :] +
        c[:, :, :, :, Y, 2, TI] * vy[Rp1, Rp1, :, :] +
        c[:, :, :, :, Z, 0, TI] * vz[R,   R,   :, :] +
        c[:, :, :, :, Z, 1, TI] * vz[Rp1, R,   :, :] +
        c[:, :, :, :, Z, 2, TI] * vz[Rp1, Rp1, :, :]
    )
    sclt_TJ = (
        c[:, :, :, :, X, 0, TJ] * vx[R,   R,   :, :] +
        c[:, :, :, :, X, 1, TJ] * vx[Rp1, Rp1, :, :] +
        c[:, :, :, :, X, 2, TJ] * vx[R,   Rp1, :, :] +
        c[:, :, :, :, Y, 0, TJ] * vy[R,   R,   :, :] +
        c[:, :, :, :, Y, 1, TJ] * vy[Rp1, Rp1, :, :] +
        c[:, :, :, :, Y, 2, TJ] * vy[R,   Rp1, :, :] +
        c[:, :, :, :, Z, 0, TJ] * vz[R,   R,   :, :] +
        c[:, :, :, :, Z, 1, TJ] * vz[Rp1, Rp1, :, :] +
        c[:, :, :, :, Z, 2, TJ] * vz[R,   Rp1, :, :]
    )
    # sclt_TI / sclt_TJ : (G, G, k, l), indices 0..gmax

    II   = slice(1, G)          # 1..gmax  (interior output rows)
    IIm1 = slice(0, G - 1)      # 0..gmax-1

    s_ij_TI    = sclt_TI[II,   II]
    s_ij_TJ    = sclt_TJ[II,   II]
    s_im1_TI   = sclt_TI[IIm1, II]
    s_im1m1_TJ = sclt_TJ[IIm1, IIm1]
    s_im1m1_TI = sclt_TI[IIm1, IIm1]
    s_jm1_TJ   = sclt_TJ[II,   IIm1]

    pad_i_after = i - G
    pad_j_after = j - G

    def ddivd(d):
        cd = coef_diff[II, II, :, :, d, :]      # (gmax, gmax, k, l, 6)
        out = (
            cd[:, :, :, :, 0] * (s_ij_TI    + s_ij_TJ) +
            cd[:, :, :, :, 1] * (s_ij_TJ    + s_im1_TI) +
            cd[:, :, :, :, 2] * (s_im1_TI   + s_im1m1_TJ) +
            cd[:, :, :, :, 3] * (s_im1m1_TJ + s_im1m1_TI) +
            cd[:, :, :, :, 4] * (s_im1m1_TI + s_jm1_TJ) +
            cd[:, :, :, :, 5] * (s_jm1_TJ   + s_ij_TI)
        )
        return xp.pad(out, ((1, pad_i_after), (1, pad_j_after), (0, 0), (0, 0)))

    return ddivd(X), ddivd(Y), ddivd(Z)


def _divdamp_pl(vx_pl, vy_pl, vz_pl, coef_intp_pl, coef_diff_pl, cfg, xp):
    """Pole 2D divergence-damping stencil.

    vx/y/z_pl    : (g, k, l)
    coef_intp_pl : (g, 1, l, 3(DIR), 3(point))
    coef_diff_pl : (g, 1, l, 3(DIR))
    returns      : (ddivdx_pl, ddivdy_pl, ddivdz_pl), each (g, k, l);
                   only the self row (gslf_pl) is non-zero.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    n = cfg.gslf_pl
    k0 = cfg.k0
    v0, v1 = cfg.gmin_pl, cfg.gmax_pl + 1     # spokes (inclusive)
    g = vx_pl.shape[0]
    kall = vx_pl.shape[1]
    l = vx_pl.shape[2]

    Vx_self, Vy_self, Vz_self = vx_pl[n], vy_pl[n], vz_pl[n]      # (k, l)
    Vx_ij, Vy_ij, Vz_ij = vx_pl[v0:v1], vy_pl[v0:v1], vz_pl[v0:v1]   # (nv, k, l)
    Vx_p1 = xp.roll(Vx_ij, -1, axis=0)
    Vy_p1 = xp.roll(Vy_ij, -1, axis=0)
    Vz_p1 = xp.roll(Vz_ij, -1, axis=0)

    cI = coef_intp_pl[v0:v1, k0]              # (nv, l, 3, 3) : [v, DIR, point]

    def b(a):
        return a[:, None, :]                  # (nv, l) -> (nv, 1, l)

    sclt_v = (
        b(cI[:, :, X, 0]) * Vx_self + b(cI[:, :, X, 1]) * Vx_ij + b(cI[:, :, X, 2]) * Vx_p1 +
        b(cI[:, :, Y, 0]) * Vy_self + b(cI[:, :, Y, 1]) * Vy_ij + b(cI[:, :, Y, 2]) * Vy_p1 +
        b(cI[:, :, Z, 0]) * Vz_self + b(cI[:, :, Z, 1]) * Vz_ij + b(cI[:, :, Z, 2]) * Vz_p1
    )                                          # (nv, k, l)

    sclt_ijm1 = xp.roll(sclt_v, 1, axis=0)
    pair = sclt_ijm1 + sclt_v                  # (nv, k, l)
    cd = coef_diff_pl[v0:v1, k0]               # (nv, l, 3) : [v, DIR]

    ddx = (b(cd[:, :, X]) * pair).sum(axis=0)  # (k, l)
    ddy = (b(cd[:, :, Y]) * pair).sum(axis=0)
    ddz = (b(cd[:, :, Z]) * pair).sum(axis=0)

    def place(row):
        rowk = row[None]                       # (1, k, l)
        above = xp.zeros((n, kall, l), dtype=row.dtype)
        below = xp.zeros((g - n - 1, kall, l), dtype=row.dtype)
        return xp.concatenate([above, rowk, below], axis=0)

    return place(ddx), place(ddy), place(ddz)


def compute_oprt_divdamp(
    vx, vy, vz, vx_pl, vy_pl, vz_pl,
    coef_intp, coef_diff, coef_intp_pl, coef_diff_pl,
    cfg: OprtDivdampCfg, xp,
):
    """Pure version of OPRT_divdamp (whole COMM-free body).

    Returns (ddivdx, ddivdy, ddivdz, ddivdx_pl, ddivdy_pl, ddivdz_pl).
    The _pl outputs are shape-correct zero arrays when not have_pl.
    """
    dx, dy, dz = _divdamp_reg(vx, vy, vz, coef_intp, coef_diff, cfg, xp)
    if cfg.have_pl:
        dx_pl, dy_pl, dz_pl = _divdamp_pl(
            vx_pl, vy_pl, vz_pl, coef_intp_pl, coef_diff_pl, cfg, xp,
        )
    else:
        dx_pl = xp.zeros_like(vx_pl)
        dy_pl = xp.zeros_like(vx_pl)
        dz_pl = xp.zeros_like(vx_pl)
    return dx, dy, dz, dx_pl, dy_pl, dz_pl
