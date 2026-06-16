"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for horizontal_flux
(the MIURA horizontal mass-flux + mass-centroid computation in
mod_src_tracer.py).

Reproduces, vectorised over (i, j, k, l) and built functionally (no in-place
writes, so it is jit-able under jax):

  1. cell-vertex interpolation of rho and rho*v on the TI/TJ sub-triangles,
     with the singular-pole point fix (pntmask) at (i, j) = (0, 0);
  2. the 6-edge horizontal mass flux flx_h (AI/AIJ/AJ "+" components 0,1,2 and
     the neighbour-cell "-" components 3,4,5 written at shifted positions);
  3. the mass-centroid position grd_xc on the AI/AIJ/AJ arcs.

The geometry (GMTR_t/a/p, GRD_xr at K0) is k-independent and broadcasts over k.
Rows/columns the original leaves untouched (CONST_UNDEF) are zero here; they are
never read downstream, so this is downstream-identical (verified by full-run
bit-exactness).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HorizFluxCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    iall: int
    jall: int
    K0: int
    TI: int
    TJ: int
    AI: int
    AIJ: int
    AJ: int
    W1: int
    W2: int
    W3: int
    HNX: int
    HNY: int
    HNZ: int
    P_RAREA: int
    XDIR: int
    YDIR: int
    ZDIR: int
    have_pl: bool
    gslf_pl: int
    gmin_pl: int
    gmax_pl: int
    EPS: float


def _pad4(xp, block, ipad, jpad, dtype):
    """Pad an (i,j,k,l) interior block to full size on the i,j axes only."""
    return xp.pad(block, (ipad, jpad, (0, 0), (0, 0)))


def compute_horizontal_flux(
    rho, rhovx, rhovy, rhovz,
    rho_pl, rhovx_pl, rhovy_pl, rhovz_pl,
    GMTR_t, GMTR_a, GMTR_p, GRD_xr, pntmask,
    GMTR_t_pl, GMTR_a_pl, GMTR_p_pl, GRD_xr_pl,
    dt, cfg: HorizFluxCfg, xp,
):
    """Returns (flx_h, grd_xc, flx_h_pl, grd_xc_pl)."""
    K0 = cfg.K0
    TI, TJ = cfg.TI, cfg.TJ
    AI, AIJ, AJ = cfg.AI, cfg.AIJ, cfg.AJ
    W1, W2, W3 = cfg.W1, cfg.W2, cfg.W3
    HNX, HNY, HNZ = cfg.HNX, cfg.HNY, cfg.HNZ
    P = cfg.P_RAREA
    XDIR, YDIR, ZDIR = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    EPS = cfg.EPS
    dtype = rho.dtype
    half = xp.asarray(0.5, dtype=dtype)
    one = xp.asarray(1.0, dtype=dtype)
    i, j, kall, l = rho.shape

    isl = slice(0, i - 1)
    jsl = slice(0, j - 1)
    isl_p = slice(1, i)
    jsl_p = slice(1, j)

    def b(g2):  # geometry (i-1,j-1,l) -> (i-1,j-1,1,l) to broadcast over k
        return g2[:, :, None, :]

    # --- vertex interpolation on the (0..i-2, 0..j-2) block, full over k,l ---
    def vert(f):
        ti = (f[isl, jsl, :, :]   * b(GMTR_t[isl, jsl, K0, :, TI, W1])
              + f[isl_p, jsl, :, :] * b(GMTR_t[isl, jsl, K0, :, TI, W2])
              + f[isl_p, jsl_p, :, :] * b(GMTR_t[isl, jsl, K0, :, TI, W3]))
        tj = (f[isl, jsl, :, :]   * b(GMTR_t[isl, jsl, K0, :, TJ, W1])
              + f[isl_p, jsl_p, :, :] * b(GMTR_t[isl, jsl, K0, :, TJ, W2])
              + f[isl, jsl_p, :, :] * b(GMTR_t[isl, jsl, K0, :, TJ, W3]))
        return ti, tj  # each (i-1, j-1, k, l)

    rhot_TI, rhot_TJ = vert(rho)
    rhovxt_TI, rhovxt_TJ = vert(rhovx)
    rhovyt_TI, rhovyt_TJ = vert(rhovy)
    rhovzt_TI, rhovzt_TJ = vert(rhovz)

    # pole-point fix at (i,j)=(0,0): TI[0,0] = TI[0,0]*m0 + TJ[1,0]*m1
    # (m0,m1 depend on l; broadcast over k). Done functionally via where on the
    # (0,0) cell of the (i-1,j-1) block.
    ii = xp.arange(i - 1)[:, None]
    jj = xp.arange(j - 1)[None, :]
    at00 = ((ii == 0) & (jj == 0))[:, :, None, None]
    m0 = pntmask[K0, :, 0][None, None, None, :]
    m1 = pntmask[K0, :, 1][None, None, None, :]

    def polefix(ti, tj):
        # tj[1,0] broadcast to the whole block, then selected only at (0,0)
        tj10 = tj[1:2, 0:1, :, :]  # (1,1,k,l)
        fixed = ti * m0 + tj10 * m1
        return xp.where(at00, fixed, ti)

    rhot_TI = polefix(rhot_TI, rhot_TJ)
    rhovxt_TI = polefix(rhovxt_TI, rhovxt_TJ)
    rhovyt_TI = polefix(rhovyt_TI, rhovyt_TJ)
    rhovzt_TI = polefix(rhovzt_TI, rhovzt_TJ)

    # convenience: index helpers on the (i-1,j-1) block
    A = slice(0, i - 1)
    Jc = slice(1, j - 1)      # 1..j-2
    Jm1 = slice(0, j - 2)     # 0..j-3
    Ic = slice(1, i - 1)      # 1..i-2
    Im1 = slice(0, i - 2)     # 0..i-3
    Jf = slice(0, j - 1)      # 0..j-2

    # --- AI edge ---
    rrhoa2 = one / xp.maximum(rhot_TJ[A, Jm1, :, :] + rhot_TI[A, Jc, :, :], EPS)
    vx2 = rhovxt_TJ[A, Jm1, :, :] + rhovxt_TI[A, Jc, :, :]
    vy2 = rhovyt_TJ[A, Jm1, :, :] + rhovyt_TI[A, Jc, :, :]
    vz2 = rhovzt_TJ[A, Jm1, :, :] + rhovzt_TI[A, Jc, :, :]
    flux_ai = half * (vx2 * b(GMTR_a[A, Jc, K0, :, AI, HNX])
                      + vy2 * b(GMTR_a[A, Jc, K0, :, AI, HNY])
                      + vz2 * b(GMTR_a[A, Jc, K0, :, AI, HNZ]))   # (i-1, j-2, k, l)
    c0 = flux_ai * b(GMTR_p[A, Jc, K0, :, P]) * dt
    c3 = -flux_ai * b(GMTR_p[isl_p, Jc, K0, :, P]) * dt
    xc_ai_x = b(GRD_xr[A, Jc, K0, :, AI, XDIR]) - vx2 * rrhoa2 * dt * half
    xc_ai_y = b(GRD_xr[A, Jc, K0, :, AI, YDIR]) - vy2 * rrhoa2 * dt * half
    xc_ai_z = b(GRD_xr[A, Jc, K0, :, AI, ZDIR]) - vz2 * rrhoa2 * dt * half

    # --- AIJ edge ---
    rrhoa2 = one / xp.maximum(rhot_TI[A, Jf, :, :] + rhot_TJ[A, Jf, :, :], EPS)
    vx2 = rhovxt_TI[A, Jf, :, :] + rhovxt_TJ[A, Jf, :, :]
    vy2 = rhovyt_TI[A, Jf, :, :] + rhovyt_TJ[A, Jf, :, :]
    vz2 = rhovzt_TI[A, Jf, :, :] + rhovzt_TJ[A, Jf, :, :]
    flux_aij = half * (vx2 * b(GMTR_a[A, Jf, K0, :, AIJ, HNX])
                       + vy2 * b(GMTR_a[A, Jf, K0, :, AIJ, HNY])
                       + vz2 * b(GMTR_a[A, Jf, K0, :, AIJ, HNZ]))   # (i-1, j-1, k, l)
    c1 = flux_aij * b(GMTR_p[A, Jf, K0, :, P]) * dt
    c4 = -flux_aij * b(GMTR_p[isl_p, jsl_p, K0, :, P]) * dt
    xc_aij_x = b(GRD_xr[A, Jf, K0, :, AIJ, XDIR]) - vx2 * rrhoa2 * dt * half
    xc_aij_y = b(GRD_xr[A, Jf, K0, :, AIJ, YDIR]) - vy2 * rrhoa2 * dt * half
    xc_aij_z = b(GRD_xr[A, Jf, K0, :, AIJ, ZDIR]) - vz2 * rrhoa2 * dt * half

    # --- AJ edge ---
    rrhoa2 = one / xp.maximum(rhot_TJ[Ic, Jf, :, :] + rhot_TI[Im1, Jf, :, :], EPS)
    vx2 = rhovxt_TJ[Ic, Jf, :, :] + rhovxt_TI[Im1, Jf, :, :]
    vy2 = rhovyt_TJ[Ic, Jf, :, :] + rhovyt_TI[Im1, Jf, :, :]
    vz2 = rhovzt_TJ[Ic, Jf, :, :] + rhovzt_TI[Im1, Jf, :, :]
    flux_aj = half * (vx2 * b(GMTR_a[Ic, Jf, K0, :, AJ, HNX])
                      + vy2 * b(GMTR_a[Ic, Jf, K0, :, AJ, HNY])
                      + vz2 * b(GMTR_a[Ic, Jf, K0, :, AJ, HNZ]))   # (i-2, j-1, k, l)
    c2 = flux_aj * b(GMTR_p[Ic, Jf, K0, :, P]) * dt
    c5 = -flux_aj * b(GMTR_p[Ic, jsl_p, K0, :, P]) * dt
    xc_aj_x = b(GRD_xr[Ic, Jf, K0, :, AJ, XDIR]) - vx2 * rrhoa2 * dt * half
    xc_aj_y = b(GRD_xr[Ic, Jf, K0, :, AJ, YDIR]) - vy2 * rrhoa2 * dt * half
    xc_aj_z = b(GRD_xr[Ic, Jf, K0, :, AJ, ZDIR]) - vz2 * rrhoa2 * dt * half

    # --- assemble flx_h (i,j,k,l,6) by padding each component to full (i,j) ---
    def P4(block, ipad, jpad):
        return xp.pad(block, (ipad, jpad, (0, 0), (0, 0)))

    comp0 = P4(c0, (0, 1), (1, 1))                 # [0:i-1, 1:j-1]
    comp3 = P4(c3, (1, 0), (1, 1))                 # [1:i,   1:j-1]
    comp1 = P4(c1, (0, 1), (0, 1))                 # [0:i-1, 0:j-1]
    comp4 = P4(c4, (1, 0), (1, 0))                 # [1:i,   1:j]
    comp2 = P4(c2, (1, 1), (0, 1))                 # [1:i-1, 0:j-1]
    comp5 = P4(c5, (1, 1), (1, 0))                 # [1:i-1, 1:j]

    # pole fix: flx_h[1,1,:,:,5] *= pntmask[K0,:,0]
    iif = xp.arange(i)[:, None]
    jjf = xp.arange(j)[None, :]
    at11 = ((iif == 1) & (jjf == 1))[:, :, None, None]
    scale5 = xp.where(at11, pntmask[K0, :, 0][None, None, None, :], one)
    comp5 = comp5 * scale5

    flx_h = xp.stack([comp0, comp1, comp2, comp3, comp4, comp5], axis=-1)

    # --- assemble grd_xc (i,j,k,l, nA, 3) ---
    def XC(bx, by, bz, ipad, jpad):
        return (P4(bx, ipad, jpad), P4(by, ipad, jpad), P4(bz, ipad, jpad))

    ai_x, ai_y, ai_z = XC(xc_ai_x, xc_ai_y, xc_ai_z, (0, 1), (1, 1))
    aij_x, aij_y, aij_z = XC(xc_aij_x, xc_aij_y, xc_aij_z, (0, 1), (0, 1))
    aj_x, aj_y, aj_z = XC(xc_aj_x, xc_aj_y, xc_aj_z, (1, 1), (0, 1))

    nA = max(AI, AIJ, AJ) + 1
    cols = []
    zero = xp.zeros((i, j, kall, l), dtype=dtype)
    per_a = {AI: (ai_x, ai_y, ai_z), AIJ: (aij_x, aij_y, aij_z), AJ: (aj_x, aj_y, aj_z)}
    for a in range(nA):
        gx, gy, gz = per_a.get(a, (zero, zero, zero))
        cols.append(xp.stack([gx, gy, gz], axis=-1))   # (i,j,k,l,3)
    grd_xc = xp.stack(cols, axis=-2)                    # (i,j,k,l,nA,3)

    # --- pole region ---
    flx_h_pl = xp.zeros_like(rho_pl)
    grd_xc_pl = xp.zeros(rho_pl.shape + (3,), dtype=dtype)
    if cfg.have_pl:
        n = cfg.gslf_pl
        v0, v1 = cfg.gmin_pl, cfg.gmax_pl + 1
        nv = v1 - v0
        # vertices v and their cyclic neighbours vp1 (v+1 wrap) for the first
        # pass and vm1 (v-1 wrap) for the flux pass.
        vidx = xp.arange(v0, v1)
        vp1 = xp.where(vidx + 1 == cfg.gmax_pl + 1, cfg.gmin_pl, vidx + 1)
        vm1 = xp.where(vidx - 1 == cfg.gmin_pl - 1, cfg.gmax_pl, vidx - 1)

        def vert_pl(f):  # f_pl (g,k,l) -> rhot_pl rows for v in [v0,v1) (nv,k,l)
            return (f[n][None, :, :] * GMTR_t_pl[v0:v1, K0, :, W1][:, None, :]
                    + f[vidx] * GMTR_t_pl[v0:v1, K0, :, W2][:, None, :]
                    + f[vp1] * GMTR_t_pl[v0:v1, K0, :, W3][:, None, :])

        rhot_p = vert_pl(rho_pl)      # (nv,k,l) indexed 0..nv-1 == v0..gmax_pl
        rhovxt_p = vert_pl(rhovx_pl)
        rhovyt_p = vert_pl(rhovy_pl)
        rhovzt_p = vert_pl(rhovz_pl)

        # map global vertex index -> local row; ij=v -> row v-v0, ijm1=vm1 -> row vm1-v0
        loc = vidx - v0
        locm1 = vm1 - v0
        rrhoa2 = one / xp.maximum(rhot_p[locm1] + rhot_p[loc], EPS)
        vx2 = rhovxt_p[locm1] + rhovxt_p[loc]
        vy2 = rhovyt_p[locm1] + rhovyt_p[loc]
        vz2 = rhovzt_p[locm1] + rhovzt_p[loc]
        flux = half * (vx2 * GMTR_a_pl[v0:v1, K0, :, HNX][:, None, :]
                       + vy2 * GMTR_a_pl[v0:v1, K0, :, HNY][:, None, :]
                       + vz2 * GMTR_a_pl[v0:v1, K0, :, HNZ][:, None, :])
        flx_rows = flux * GMTR_p_pl[n, K0, :, P][None, None, :] * dt   # (nv,k,l)
        gx = GRD_xr_pl[v0:v1, K0, :, XDIR][:, None, :] - vx2 * rrhoa2 * dt * half
        gy = GRD_xr_pl[v0:v1, K0, :, YDIR][:, None, :] - vy2 * rrhoa2 * dt * half
        gz = GRD_xr_pl[v0:v1, K0, :, ZDIR][:, None, :] - vz2 * rrhoa2 * dt * half

        g, kp, lp = rho_pl.shape
        zlo = xp.zeros((v0, kp, lp), dtype=dtype)
        zhi = xp.zeros((g - v1, kp, lp), dtype=dtype)
        flx_h_pl = xp.concatenate([zlo, flx_rows, zhi], axis=0)
        gxc = xp.stack([gx, gy, gz], axis=-1)  # (nv,k,l,3)
        zlo3 = xp.zeros((v0, kp, lp, 3), dtype=dtype)
        zhi3 = xp.zeros((g - v1, kp, lp, 3), dtype=dtype)
        grd_xc_pl = xp.concatenate([zlo3, gxc, zhi3], axis=0)

    return flx_h, grd_xc, flx_h_pl, grd_xc_pl
