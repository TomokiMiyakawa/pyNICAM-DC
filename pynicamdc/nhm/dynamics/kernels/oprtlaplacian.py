"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for OPRT_laplacian
(mod_oprt.py): the 7-point hexagonal Laplacian used heavily by
numfilter_hdiffusion (the high-order lap loop, the lap1 filter, and the
per-tracer diffusion section).

Regional core
-------------
For interior i,j = 1..iall-2 the operator is a 7-term stencil:

    dscl = c0*scl[i,j] + c1*scl[i+1,j] + c2*scl[i+1,j+1] + c3*scl[i,j+1]
         + c4*scl[i-1,j] + c5*scl[i-1,j-1] + c6*scl[i,j-1]

with coef_lap = (iall, jall, k, l, 7). The i/j border rows are zero (the
original allocates dscl with np.zeros and only writes the interior); here the
interior block is computed and padded with zero, downstream identical.

Pole (_pl)
----------
The original accumulates dscl_pl[n] += sum over spokes v = gslf_pl..gmax_pl of
coef_lap_pl[v] * scl_pl[v], then multiplies by ppm.plmask. plmask is the scalar
1 when this rank owns the pole (have_pl) and 0 otherwise, so the multiply is a
no-op under have_pl and a wipe otherwise. We therefore compute the spoke sum
into row n when cfg.have_pl and return shape-correct zeros when not -- exactly
matching `sum * plmask`. The plain xp.sum over the (small) spoke axis runs in
ascending order, matching the original's vectorised np.sum bit-for-bit.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OprtLaplacianCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    gslf_pl: int
    gmax_pl: int


def _laplacian_reg(scl, coef_lap, xp):
    """Regional 7-point hexagonal Laplacian; zero on the i/j border."""
    i = scl.shape[0]
    j = scl.shape[1]
    c = coef_lap[1:i - 1, 1:j - 1]            # (i-2, j-2, k, l, 7)
    inner = (
        c[..., 0] * scl[1:i - 1, 1:j - 1] +
        c[..., 1] * scl[2:i,     1:j - 1] +
        c[..., 2] * scl[2:i,     2:j]     +
        c[..., 3] * scl[1:i - 1, 2:j]     +
        c[..., 4] * scl[0:i - 2, 1:j - 1] +
        c[..., 5] * scl[0:i - 2, 0:j - 2] +
        c[..., 6] * scl[1:i - 1, 0:j - 2]
    )
    return xp.pad(inner, ((1, 1), (1, 1), (0, 0), (0, 0)))


def _laplacian_pl(scl_pl, coef_lap_pl, cfg, xp):
    """Pole Laplacian: only the self row (gslf_pl) is non-zero."""
    n = cfg.gslf_pl
    v0, v1 = cfg.gslf_pl, cfg.gmax_pl + 1     # spokes gslf_pl..gmax_pl (inclusive)
    g = scl_pl.shape[0]
    kall = scl_pl.shape[1]
    l = scl_pl.shape[2]

    row = (coef_lap_pl[v0:v1] * scl_pl[v0:v1]).sum(axis=0)   # (k, l)

    above = xp.zeros((n, kall, l), dtype=row.dtype)
    below = xp.zeros((g - n - 1, kall, l), dtype=row.dtype)
    return xp.concatenate([above, row[None], below], axis=0)


def compute_oprt_laplacian(scl, scl_pl, coef_lap, coef_lap_pl,
                           cfg: OprtLaplacianCfg, xp):
    """Pure version of OPRT_laplacian.

    Returns (dscl, dscl_pl). dscl_pl is a shape-correct zero array when not
    have_pl (matching the original's `dscl_pl * plmask` with plmask == 0).
    """
    dscl = _laplacian_reg(scl, coef_lap, xp)
    if cfg.have_pl:
        dscl_pl = _laplacian_pl(scl_pl, coef_lap_pl, cfg, xp)
    else:
        dscl_pl = xp.zeros_like(scl_pl)
    return dscl, dscl_pl
