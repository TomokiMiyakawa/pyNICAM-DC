"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for OPRT_gradient
(mod_oprt.py): the 7-point gradient operator used by the tracer source
(src_tracer horizontal_adv) and at setup time by mod_vmtr.

Regional core
-------------
For interior i,j = 1..iall-2 the gradient in each direction d = X/Y/Z is a
7-term stencil sum of scl over the cell and its six neighbours, weighted by
coef_grad (iall, jall, k, l, 3(DIR), 7):

    grad[i,j,d] = sum_s coef_grad[i,j,d,s] * scl_stencil[s]

with the same 7-neighbour stencil order as OPRT_laplacian. The i/j border rows
are zero (the original does grad.fill(0) then writes only the interior); here
the interior block is computed and padded with zero, downstream identical.

Pole (_pl)
----------
The original accumulates grad_pl[n,d] += sum over spokes v = gslf_pl..gmax_pl of
coef_grad_pl[v,d] * scl_pl[v]; only the self row (gslf_pl) is written, and ONLY
when have_pl (the non-pole branch leaves grad_pl untouched). The vectorised
xp.sum over the (small) spoke axis runs in ascending order, matching the
original's sequential accumulation bit-for-bit in numpy. When not have_pl the
caller-side wrapper skips the grad_pl write-back entirely, preserving the
original's no-op behaviour.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OprtGradientCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    gslf_pl: int
    gmax_pl: int
    k0: int
    XDIR: int
    YDIR: int
    ZDIR: int


def _gradient_reg(scl, coef_grad, cfg, xp):
    """Regional 7-point gradient; zero on the i/j border. Returns (i,j,k,l,3)."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    i = scl.shape[0]
    j = scl.shape[1]

    isl,   jsl   = slice(1, i - 1), slice(1, j - 1)
    isl_p, jsl_p = slice(2, i),     slice(2, j)
    isl_m, jsl_m = slice(0, i - 2), slice(0, j - 2)

    scl_stack = xp.stack([
        scl[isl,   jsl],
        scl[isl_p, jsl],
        scl[isl_p, jsl_p],
        scl[isl,   jsl_p],
        scl[isl_m, jsl],
        scl[isl_m, jsl_m],
        scl[isl,   jsl_m],
    ], axis=4)                                  # (i-2, j-2, k, l, 7)

    gx = (coef_grad[isl, jsl, :, :, X, :] * scl_stack).sum(axis=4)
    gy = (coef_grad[isl, jsl, :, :, Y, :] * scl_stack).sum(axis=4)
    gz = (coef_grad[isl, jsl, :, :, Z, :] * scl_stack).sum(axis=4)
    inner = xp.stack([gx, gy, gz], axis=-1)     # (i-2, j-2, k, l, 3); X,Y,Z = 0,1,2

    return xp.pad(inner, ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0)))


def _gradient_pl(scl_pl, coef_grad_pl, cfg, xp):
    """Pole gradient; only the self row (gslf_pl) is non-zero. Returns (g,k,l,3)."""
    n = cfg.gslf_pl
    k0 = cfg.k0
    v0, v1 = cfg.gslf_pl, cfg.gmax_pl + 1       # spokes gslf_pl..gmax_pl (inclusive)
    g = scl_pl.shape[0]
    kall = scl_pl.shape[1]
    l = scl_pl.shape[2]

    cg = coef_grad_pl[v0:v1, k0]                # (nv, l, 3)
    sv = scl_pl[v0:v1]                          # (nv, k, l)
    row = (cg[:, None, :, :] * sv[..., None]).sum(axis=0)   # (k, l, 3)

    above = xp.zeros((n, kall, l, 3), dtype=row.dtype)
    below = xp.zeros((g - n - 1, kall, l, 3), dtype=row.dtype)
    return xp.concatenate([above, row[None], below], axis=0)


def compute_oprt_gradient(scl, scl_pl, coef_grad, coef_grad_pl,
                          cfg: OprtGradientCfg, xp):
    """Pure version of OPRT_gradient.

    Returns (grad, grad_pl). grad_pl is a shape-correct zero array when not
    have_pl; the caller-side wrapper must skip writing it back in that case to
    match the original (which leaves grad_pl untouched on non-pole ranks).
    """
    grad = _gradient_reg(scl, coef_grad, cfg, xp)
    grad_pl = _gradient_pl(scl_pl, coef_grad_pl, cfg, xp) if cfg.have_pl else None
    return grad, grad_pl
