"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the COMM-free
"B1" island of Vi.vi_small_step's per-iteration loop (mod_vi.py vi_path1),
i.e. everything between the last numfilter_divdamp COMM and the
COMM_data_transfer(diff_vh):

  1. src_pres_gradient (horizontal only)      -> dpgrad      [reuses presgrad]
  2. tendency assembly (split: large step + pres-grad + div-damp + div-damp_2d)
       drhogv{x,y,z} = g_TEND - dpgrad + ddivd + ddivd_2d
       drhogw        = g_TEND + ddivdw * alpha
       diff_vh       = PROG_split + drhog * dt
  3. BNDCND_rhovxvyvz on diff_vh               [reuses bndcnd]

When TIME_split is False the divergence-damping / pressure-gradient terms are
absent (numfilter is not called), so the tendency reduces to the large-step
value alone -- matching the original NO-SPLITTING branch. The BNDCND treatment
is applied in both cases.

Fusing presgrad + the elementwise glue + BNDCND into one pure function means
that under jax.jit XLA emits a single graph with one host round-trip
(asarray-in / to_numpy-out) instead of three. Under numpy it is bit-for-bit
identical to the per-kernel path.

Outputs: diff_vh (post-BNDCND) and drhogw (consumed by vi_main). The original
mutates persistent buffers; here we return new arrays and let the caller write
them back, keeping the function purely functional / jit-friendly.
"""

from __future__ import annotations
from dataclasses import dataclass

from pynicamdc.nhm.dynamics.kernels.presgrad import PresGradCfg, compute_pres_gradient
from pynicamdc.nhm.dynamics.kernels.bndcnd import (
    BndCfg, compute_bndcnd_rhovxvyvz_reg, compute_bndcnd_rhovxvyvz_pl,
)


@dataclass(frozen=True)
class ViPath1Cfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    TIME_split: bool
    alpha: float
    XDIR: int
    YDIR: int
    ZDIR: int
    I_RHOGVX: int
    I_RHOGVY: int
    I_RHOGVZ: int
    I_RHOGW: int
    gradtype: int
    presgrad: PresGradCfg
    bnd: BndCfg


def _set_krow_reg(A, k, row, xp):
    """Return A (i,j,kall,l) with the single k-row replaced by row (i,j,l)."""
    return xp.concatenate([A[:, :, :k, :], row[:, :, None, :], A[:, :, k + 1:, :]], axis=2)


def _set_krow_pl(A, k, row, xp):
    """Return A (g,kall,l) with the single k-row replaced by row (g,l)."""
    return xp.concatenate([A[:, :k, :], row[:, None, :], A[:, k + 1:, :]], axis=1)


def compute_vi_path1(P, C, dt, cfg: ViPath1Cfg, xp):
    """Fused presgrad + tendency + diff_vh + BNDCND.

    P : dict of per-call device arrays
        preg, g_TEND, ddivdvx/vy/vz/w, ddivdvx/vy/vz_2d, psvx/psvy/psvz,
        prog_rhog   (+ *_pl variants)
    C : dict of device constants (same set src_pres_gradient stages)
    dt: traced scalar
    Returns dict: diff_vh, drhogw (+ diff_vh_pl, drhogw_pl when have_pl).
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kmaxp1 = kmin - 1, kmax + 1
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    IVX, IVY, IVZ, IW = cfg.I_RHOGVX, cfg.I_RHOGVY, cfg.I_RHOGVZ, cfg.I_RHOGW
    alpha = cfg.alpha

    # V4 (vi-stack-plan v1): the resident caller passes the regular g_TEND
    # momentum components directly (gt_comps); the stacked P["g_TEND"] form is
    # the numpy / non-resident interface. Same values either way (slices).
    gt = P.get("gt_comps")
    if gt is not None:
        gtvx, gtvy, gtvz, gtw = gt
    else:
        gT = P["g_TEND"]
        gtvx = gT[:, :, :, :, IVX]
        gtvy = gT[:, :, :, :, IVY]
        gtvz = gT[:, :, :, :, IVZ]
        gtw  = gT[:, :, :, :, IW]

    # -------------------------- regional --------------------------
    if cfg.TIME_split:
        Pgrad, _Pgradw, Pgrad_pl, _Pgradw_pl = compute_pres_gradient(
            P["preg"], P["preg_pl"],
            C["RGAM"], C["RGAMH"], C["C2WfactGz"], C["coef_grad"], C["GRD_x"],
            C["rdgz"], C["rdgzh"], C["GAM2H"], C["RGSGAM2"],
            C["RGAM_pl"], C["RGAMH_pl"], C["C2WfactGz_pl"], C["coef_grad_pl"],
            C["GRD_x_pl"], C["GAM2H_pl"], C["RGSGAM2_pl"],
            cfg.gradtype, cfg.presgrad, xp,
        )
        drhogvx = gtvx - Pgrad[:, :, :, :, X] + P["ddivdvx"] + P["ddivdvx_2d"]
        drhogvy = gtvy - Pgrad[:, :, :, :, Y] + P["ddivdvy"] + P["ddivdvy_2d"]
        drhogvz = gtvz - Pgrad[:, :, :, :, Z] + P["ddivdvz"] + P["ddivdvz_2d"]
        drhogw  = gtw + P["ddivdw"] * alpha
    else:
        drhogvx = gtvx
        drhogvy = gtvy
        drhogvz = gtvz
        drhogw  = gtw

    dvx = P["psvx"] + drhogvx * dt
    dvy = P["psvy"] + drhogvy * dt
    dvz = P["psvz"] + drhogvz * dt

    vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = compute_bndcnd_rhovxvyvz_reg(
        P["prog_rhog"], dvx, dvy, dvz, cfg.bnd, xp,
    )
    dvx = _set_krow_reg(_set_krow_reg(dvx, kmaxp1, vx_t, xp), kminm1, vx_b, xp)
    dvy = _set_krow_reg(_set_krow_reg(dvy, kmaxp1, vy_t, xp), kminm1, vy_b, xp)
    dvz = _set_krow_reg(_set_krow_reg(dvz, kmaxp1, vz_t, xp), kminm1, vz_b, xp)

    diff_vh = xp.stack([dvx, dvy, dvz], axis=4)
    out = {"diff_vh": diff_vh, "drhogw": drhogw}

    # ---------------------------- pole ----------------------------
    if cfg.have_pl:
        gT_pl = P["g_TEND_pl"]
        if cfg.TIME_split:
            drhogvx_pl = gT_pl[:, :, :, IVX] - Pgrad_pl[:, :, :, X] + P["ddivdvx_pl"] + P["ddivdvx_2d_pl"]
            drhogvy_pl = gT_pl[:, :, :, IVY] - Pgrad_pl[:, :, :, Y] + P["ddivdvy_pl"] + P["ddivdvy_2d_pl"]
            drhogvz_pl = gT_pl[:, :, :, IVZ] - Pgrad_pl[:, :, :, Z] + P["ddivdvz_pl"] + P["ddivdvz_2d_pl"]
            drhogw_pl  = gT_pl[:, :, :, IW] + P["ddivdw_pl"] * alpha
        else:
            drhogvx_pl = gT_pl[:, :, :, IVX]
            drhogvy_pl = gT_pl[:, :, :, IVY]
            drhogvz_pl = gT_pl[:, :, :, IVZ]
            drhogw_pl  = gT_pl[:, :, :, IW]

        dvx_pl = P["psvx_pl"] + drhogvx_pl * dt
        dvy_pl = P["psvy_pl"] + drhogvy_pl * dt
        dvz_pl = P["psvz_pl"] + drhogvz_pl * dt

        vx_t, vy_t, vz_t, vx_b, vy_b, vz_b = compute_bndcnd_rhovxvyvz_pl(
            P["prog_rhog_pl"], dvx_pl, dvy_pl, dvz_pl, cfg.bnd, xp,
        )
        dvx_pl = _set_krow_pl(_set_krow_pl(dvx_pl, kmaxp1, vx_t, xp), kminm1, vx_b, xp)
        dvy_pl = _set_krow_pl(_set_krow_pl(dvy_pl, kmaxp1, vy_t, xp), kminm1, vy_b, xp)
        dvz_pl = _set_krow_pl(_set_krow_pl(dvz_pl, kmaxp1, vz_t, xp), kminm1, vz_b, xp)

        diff_vh_pl = xp.stack([dvx_pl, dvy_pl, dvz_pl], axis=3)
        out["diff_vh_pl"] = diff_vh_pl
        out["drhogw_pl"] = drhogw_pl

    return out
