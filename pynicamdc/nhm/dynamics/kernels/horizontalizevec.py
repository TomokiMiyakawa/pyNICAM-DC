"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for OPRT_horizontalize_vec
(mod_oprt.py): removes the radial (vertical) component of a horizontal vector
field so it lies in the local tangent plane.

For every point the operator projects (vx,vy,vz) onto the local outward unit
direction g = GRD_x / rscale and subtracts that component:

    prd        = (vx*gx + vy*gy + vz*gz) / rscale
    vx        -= prd * gx / rscale          (likewise vy, vz)

This is an INOUT operator: only the interior i,j = 1..iall-2 is modified on the
regional grid (halos untouched, COMM-refilled), and the whole pole array is
rewritten. The kernel therefore returns the *interior* regional sub-arrays and
the *full* pole arrays; the caller writes them back into the persistent buffers'
interior / full slices, preserving the original INOUT semantics exactly.

Bit-identical to the original under numpy.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HorizontalizeVecCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    XDIR: int
    YDIR: int
    ZDIR: int


def _horiz_reg(vx, vy, vz, GRD_x, rscale, cfg, xp):
    """Regional projection; returns interior (i-2, j-2, k, l) sub-arrays."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    iall = vx.shape[0]
    jall = vx.shape[1]
    isl = slice(1, iall - 1)
    jsl = slice(1, jall - 1)

    gvec = GRD_x[isl, jsl, 0, :, :]          # (i-2, j-2, l, 3)
    gx = gvec[..., X][:, :, None, :]         # (i-2, j-2, 1, l)
    gy = gvec[..., Y][:, :, None, :]
    gz = gvec[..., Z][:, :, None, :]

    vx_s = vx[isl, jsl, :, :]
    vy_s = vy[isl, jsl, :, :]
    vz_s = vz[isl, jsl, :, :]

    prd = (vx_s * gx + vy_s * gy + vz_s * gz) / rscale
    nvx = vx_s - prd * gx / rscale
    nvy = vy_s - prd * gy / rscale
    nvz = vz_s - prd * gz / rscale
    return nvx, nvy, nvz


def _horiz_pl(vx_pl, vy_pl, vz_pl, GRD_x_pl, rscale, cfg, xp):
    """Pole projection; returns full (g, k, l) arrays."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    gx = GRD_x_pl[:, 0, :, X][:, None, :]    # (g, 1, l)
    gy = GRD_x_pl[:, 0, :, Y][:, None, :]
    gz = GRD_x_pl[:, 0, :, Z][:, None, :]

    # match the original pole loop's per-term division order (a/r + b/r + c/r),
    # which is NOT bit-identical to (a+b+c)/r used on the regional grid.
    prd = (vx_pl * gx / rscale + vy_pl * gy / rscale + vz_pl * gz / rscale)
    nvx = vx_pl - prd * gx / rscale
    nvy = vy_pl - prd * gy / rscale
    nvz = vz_pl - prd * gz / rscale
    return nvx, nvy, nvz


def compute_horizontalize_vec(
    vx, vy, vz, vx_pl, vy_pl, vz_pl,
    GRD_x, GRD_x_pl, rscale, cfg: HorizontalizeVecCfg, xp,
):
    """Pure version of OPRT_horizontalize_vec.

    Returns (nvx, nvy, nvz, nvx_pl, nvy_pl, nvz_pl) where the regional outputs
    are the interior i,j = 1..iall-2 sub-arrays and the pole outputs are the
    full (g, k, l) arrays. When not have_pl the pole outputs are zeros.
    """
    nvx, nvy, nvz = _horiz_reg(vx, vy, vz, GRD_x, rscale, cfg, xp)
    if cfg.have_pl:
        nvx_pl, nvy_pl, nvz_pl = _horiz_pl(
            vx_pl, vy_pl, vz_pl, GRD_x_pl, rscale, cfg, xp,
        )
    else:
        nvx_pl = xp.zeros_like(vx_pl)
        nvy_pl = xp.zeros_like(vy_pl)
        nvz_pl = xp.zeros_like(vz_pl)
    return nvx, nvy, nvz, nvx_pl, nvy_pl, nvz_pl
