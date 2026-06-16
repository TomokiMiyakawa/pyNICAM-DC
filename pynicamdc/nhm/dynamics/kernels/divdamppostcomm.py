"""
Pure / backend-switchable fused kernel for the COMM-free island at the end of
numfilter_divdamp (mod_numfilter.py), valid for the standard configuration
lap_order_divdamp == 2 (one Laplacian iteration).

After OPRT3D_divdamp + COMM have produced vtmp2, the remaining work is one
contiguous COMM-free chain:

    vtmp  = -vtmp2
    vtmp2 = OPRT_divdamp(vtmp)            # 2-D divergence damping
    gd{x,y,z} = divdamp_coef * vtmp2
    OPRT_horizontalize_vec(gd{x,y,z})    # remove radial component (interior)

Done as separate wrapper calls this incurs three host<->device round-trips
under jax (OPRT_divdamp out, horizontalize in/out). Chaining the *pure* kernels
here keeps everything on-device: one asarray in, one to_numpy out.

The two sub-steps reuse the already-validated pure kernels compute_oprt_divdamp
and compute_horizontalize_vec, so only the glue (negate, coef multiply, and the
horizontalize interior write-back) is new.
"""

from __future__ import annotations

from pynicamdc.nhm.dynamics.kernels.oprtdivdamp import compute_oprt_divdamp
from pynicamdc.nhm.dynamics.kernels.horizontalizevec import compute_horizontalize_vec


def compute_divdamp_post_comm(
    vtmp2, vtmp2_pl,                                   # (i,j,k,l,3),(g,k,l,3) post-COMM
    divdamp_coef, divdamp_coef_pl,                     # (i,j,k,l),(g,k,l)
    coef_intp, coef_diff, coef_intp_pl, coef_diff_pl,  # OPRT_divdamp coefs
    GRD_x, GRD_x_pl, rscale,                           # horizontalize geometry
    dd_cfg, hz_cfg, xp,
):
    """Returns (gdx, gdy, gdz, gdx_pl, gdy_pl, gdz_pl)."""
    # --- vtmp = -vtmp2 ---
    vx = -vtmp2[:, :, :, :, 0]
    vy = -vtmp2[:, :, :, :, 1]
    vz = -vtmp2[:, :, :, :, 2]
    vx_pl = -vtmp2_pl[:, :, :, 0]
    vy_pl = -vtmp2_pl[:, :, :, 1]
    vz_pl = -vtmp2_pl[:, :, :, 2]

    # --- 2-D divergence damping (pure OPRT_divdamp) ---
    o0, o1, o2, o0p, o1p, o2p = compute_oprt_divdamp(
        vx, vy, vz, vx_pl, vy_pl, vz_pl,
        coef_intp, coef_diff, coef_intp_pl, coef_diff_pl,
        dd_cfg, xp,
    )

    # --- gd = divdamp_coef * vtmp2  (full field) ---
    gdx = divdamp_coef * o0
    gdy = divdamp_coef * o1
    gdz = divdamp_coef * o2
    gdx_pl = divdamp_coef_pl * o0p
    gdy_pl = divdamp_coef_pl * o1p
    gdz_pl = divdamp_coef_pl * o2p

    # --- horizontalize (interior i,j = 1..iall-2; whole pole) ---
    nvx, nvy, nvz, nvx_pl, nvy_pl, nvz_pl = compute_horizontalize_vec(
        gdx, gdy, gdz, gdx_pl, gdy_pl, gdz_pl,
        GRD_x, GRD_x_pl, rscale, hz_cfg, xp,
    )

    # write the interior result back into the full gd arrays (borders unchanged)
    iall, jall = gdx.shape[0], gdx.shape[1]
    ii = xp.arange(iall)[:, None]
    jj = xp.arange(jall)[None, :]
    inner = ((ii >= 1) & (ii < iall - 1) & (jj >= 1) & (jj < jall - 1))[:, :, None, None]

    def merge(full, inner_vals):
        padded = xp.pad(inner_vals, ((1, 1), (1, 1), (0, 0), (0, 0)))
        return xp.where(inner, padded, full)

    gdx = merge(gdx, nvx)
    gdy = merge(gdy, nvy)
    gdz = merge(gdz, nvz)

    # pole arrays are rewritten in full by horizontalize
    return gdx, gdy, gdz, nvx_pl, nvy_pl, nvz_pl
