"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for OPRT_diffusion
(mod_oprt.py): the tensor (kh-weighted) horizontal diffusion operator used by
numfilter_hdiffusion (tem / rho in the lap loop and lap1 filter, plus the
per-tracer section).

Regional core
-------------
1. Triangle intermediate `vt` for the two sub-triangles TI / TJ, computed on the
   block i,j = 0..iall-2 as a 3-point interpolation (coef_intp) of scl over the
   triangle corners, per direction d = 0..nxyz-1:

       vt_TI = ((2c1-c2-c3)*scl[i,j] + (-c1+2c2-c3)*scl[i+1,j]
                + (-c1-c2+2c3)*scl[i+1,j+1]) / 3
       vt_TJ = ((2c1-c2-c3)*scl[i,j] + (-c1+2c2-c3)*scl[i+1,j+1]
                + (-c1-c2+2c3)*scl[i,j+1]) / 3

2. Singular-point fix at the western pentagon vertex (block index (0,0)):
       vt[0,0,TI] = vt[0,0,TI]*pntmask[...,0] + vt[1,0,TJ]*pntmask[...,1]
   pntmask is int32 0/1 (active only on the panel that owns the corner); it is
   cast to the working float dtype so the blend stays in that precision (an
   int32 * float32 multiply would upcast to float64 under NEP50). The values are
   exactly 0/1 so the cast is exact and matches the original downcast-on-assign.

3. Cell assembly on interior i,j = gmin..gmax = 1..iall-2: a 6-term flux sum
   weighted by kf1..kf6 = 0.5*(kh[i,j] + kh[neighbour]) and coef_diff, summed
   over the three directions d. The i/j border rows are zero (original allocates
   dscl with np.zeros and writes only the interior); here the interior block is
   padded with zero, downstream identical.

Pole (_pl)
----------
Original loops spokes v = gmin_pl..gmax_pl twice (build vt_pl with cyclic-forward
neighbour, then accumulate dscl_pl[n] with cyclic-backward neighbour). Here both
spoke loops are vectorised with xp.roll; the reductions (over the nxyz axis, then
over the spoke axis) run in ascending order, matching the original's sequential
accumulation bit-for-bit in numpy. Only the self row (gslf_pl) is non-zero.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OprtDiffusionCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    gmin: int
    gmax: int
    nxyz: int
    gslf_pl: int
    gmin_pl: int
    gmax_pl: int
    k0: int
    TI: int
    TJ: int


def _diffusion_reg(scl, kh, coef_intp, coef_diff, pntmask, cfg, xp):
    """Regional kh-weighted diffusion stencil; zero on the i/j border."""
    TI, TJ = cfg.TI, cfg.TJ
    gmin, gmax = cfg.gmin, cfg.gmax
    i = scl.shape[0]
    j = scl.shape[1]

    # --- vt on the block i,j = 0..iall-2 (corner access reaches i+1, j+1) ---
    bi  = slice(0, i - 1)
    bj  = slice(0, j - 1)
    bip = slice(1, i)
    bjp = slice(1, j)

    s0   = scl[bi,  bj][..., None]            # (i-1, j-1, k, l, 1)
    s_ip = scl[bip, bj][..., None]
    s_d  = scl[bip, bjp][..., None]
    s_jp = scl[bi,  bjp][..., None]

    cti = coef_intp[bi, bj, :, :, :, :, TI]   # (i-1, j-1, k, l, d, 3)
    ctj = coef_intp[bi, bj, :, :, :, :, TJ]
    c1i, c2i, c3i = cti[..., 0], cti[..., 1], cti[..., 2]
    c1j, c2j, c3j = ctj[..., 0], ctj[..., 1], ctj[..., 2]

    term_ti = ((2.0 * c1i - c2i - c3i) * s0
               + (-c1i + 2.0 * c2i - c3i) * s_ip
               + (-c1i - c2i + 2.0 * c3i) * s_d) / 3.0
    term_tj = ((2.0 * c1j - c2j - c3j) * s0
               + (-c1j + 2.0 * c2j - c3j) * s_d
               + (-c1j - c2j + 2.0 * c3j) * s_jp) / 3.0
    # term_ti / term_tj : (i-1, j-1, k, l, d)

    # --- singular-point fix at block index (0,0), TI direction ---
    fdt = term_ti.dtype
    m0 = pntmask[0, :, 0].astype(fdt)[None, None, None, :, None]   # (1,1,1,l,1)
    m1 = pntmask[0, :, 1].astype(fdt)[None, None, None, :, None]
    corner = term_ti[0:1, 0:1] * m0 + term_tj[1:2, 0:1] * m1       # (1,1,k,l,d)
    row0 = xp.concatenate([corner, term_ti[0:1, 1:]], axis=1)      # (1,j-1,k,l,d)
    term_ti = xp.concatenate([row0, term_ti[1:]], axis=0)         # (i-1,j-1,k,l,d)

    # --- kf weights from kh (full grid), interior i,j = gmin..gmax ---
    sl  = slice(gmin,     gmax + 1)     # 1..gmax
    slp = slice(gmin + 1, gmax + 2)     # 2..gmax+1
    slm = slice(gmin - 1, gmax)         # 0..gmax-1
    kh0 = kh[sl, sl]
    kf1 = 0.5 * (kh0 + kh[slp, slp])
    kf2 = 0.5 * (kh0 + kh[sl,  slp])
    kf3 = 0.5 * (kh0 + kh[slm, sl])
    kf4 = 0.5 * (kh0 + kh[slm, slm])
    kf5 = 0.5 * (kh0 + kh[sl,  slm])
    kf6 = 0.5 * (kh0 + kh[slp, sl])

    # --- 6-term flux assembly, summed over directions ---
    II  = slice(gmin,     gmax + 1)     # 1..gmax  (block coords; block starts at 0)
    IIm = slice(gmin - 1, gmax)         # 0..gmax-1
    v_ij_ti    = term_ti[II,  II]
    v_ij_tj    = term_tj[II,  II]
    v_im1j_ti  = term_ti[IIm, II]
    v_im1m1_tj = term_tj[IIm, IIm]
    v_im1m1_ti = term_ti[IIm, IIm]
    v_ijm1_tj  = term_tj[II,  IIm]

    inner = None
    for d in range(cfg.nxyz):
        cd = coef_diff[II, II, :, :, d, :]          # (gmax, gmax, k, l, 6)
        t1 = kf1 * cd[..., 0] * (v_ij_ti[..., d]    + v_ij_tj[..., d])
        t2 = kf2 * cd[..., 1] * (v_ij_tj[..., d]    + v_im1j_ti[..., d])
        t3 = kf3 * cd[..., 2] * (v_im1j_ti[..., d]  + v_im1m1_tj[..., d])
        t4 = kf4 * cd[..., 3] * (v_im1m1_tj[..., d] + v_im1m1_ti[..., d])
        t5 = kf5 * cd[..., 4] * (v_im1m1_ti[..., d] + v_ijm1_tj[..., d])
        t6 = kf6 * cd[..., 5] * (v_ijm1_tj[..., d]  + v_ij_ti[..., d])
        block = t1 + t2 + t3 + t4 + t5 + t6
        inner = block if inner is None else inner + block

    pad_i = i - 1 - gmax
    pad_j = j - 1 - gmax
    return xp.pad(inner, ((gmin, pad_i), (gmin, pad_j), (0, 0), (0, 0)))


def _diffusion_pl(scl_pl, kh_pl, coef_intp_pl, coef_diff_pl, cfg, xp):
    """Pole kh-weighted diffusion; only the self row (gslf_pl) is non-zero."""
    n = cfg.gslf_pl
    k0 = cfg.k0
    v0, v1 = cfg.gmin_pl, cfg.gmax_pl + 1       # spokes gmin_pl..gmax_pl
    g = scl_pl.shape[0]
    kall = scl_pl.shape[1]
    l = scl_pl.shape[2]

    scl_n = scl_pl[n]                            # (k, l)
    scl_v = scl_pl[v0:v1]                        # (nv, k, l)
    scl_vp1 = xp.roll(scl_v, -1, axis=0)         # cyclic-forward neighbour (ijp1)

    cI = coef_intp_pl[v0:v1, k0]                 # (nv, l, nxyz, 3)
    c0 = cI[:, :, :, 0][:, None, :, :]           # (nv, 1, l, nxyz)
    c1 = cI[:, :, :, 1][:, None, :, :]
    c2 = cI[:, :, :, 2][:, None, :, :]

    s_self = scl_n[None, :, :, None]             # (1, k, l, 1)
    s_ij   = scl_v[..., None]                    # (nv, k, l, 1)
    s_p1   = scl_vp1[..., None]

    vt_v = ((2.0 * c0 - c1 - c2) * s_self
            + (-c0 + 2.0 * c1 - c2) * s_ij
            + (-c0 - c1 + 2.0 * c2) * s_p1) / 3.0     # (nv, k, l, nxyz)

    vt_ijm1 = xp.roll(vt_v, 1, axis=0)           # cyclic-backward neighbour (ijm1)
    vt_sum = vt_ijm1 + vt_v                       # (nv, k, l, nxyz)

    kh_n = kh_pl[n]                               # (k, l)
    kh_v = kh_pl[v0:v1]                           # (nv, k, l)
    kh_avg = 0.5 * (kh_n[None] + kh_v)            # (nv, k, l)

    cd = coef_diff_pl[v0:v1, k0]                  # (nv, l, nxyz)
    cd_b = cd[:, None, :, :]                       # (nv, 1, l, nxyz)
    contrib = kh_avg * (cd_b * vt_sum).sum(axis=-1)   # (nv, k, l)
    row = contrib.sum(axis=0)                     # (k, l)

    above = xp.zeros((n, kall, l), dtype=row.dtype)
    below = xp.zeros((g - n - 1, kall, l), dtype=row.dtype)
    return xp.concatenate([above, row[None], below], axis=0)


def compute_oprt_diffusion(scl, scl_pl, kh, kh_pl,
                           coef_intp, coef_intp_pl,
                           coef_diff, coef_diff_pl,
                           pntmask, cfg: OprtDiffusionCfg, xp):
    """Pure version of OPRT_diffusion.

    Returns (dscl, dscl_pl). dscl_pl is a shape-correct zero array when not
    have_pl.
    """
    dscl = _diffusion_reg(scl, kh, coef_intp, coef_diff, pntmask, cfg, xp)
    if cfg.have_pl:
        dscl_pl = _diffusion_pl(scl_pl, kh_pl, coef_intp_pl, coef_diff_pl, cfg, xp)
    else:
        dscl_pl = xp.zeros_like(scl_pl)
    return dscl, dscl_pl
