"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the MIURA horizontal
remap reconstruction in mod_src_tracer.py (horizontal_remap).

Reproduces, vectorised over (i, j, k, l) and built functionally (no in-place
writes, so it is jit-able under jax), the upwind face value q_a:

  for each of the 6 cell edges e:
    q_ap = reconstruct(local cell  -> arc point e)   = q[C] + grad(q)[C].(xc - x[C])
    q_am = reconstruct(neighbour   -> arc point e)
    q_a[...,e] = cmask[...,e]*q_am + (1-cmask[...,e])*q_ap

The 6 edges live on the (0..i-2, 0..j-2) "isl/jsl" block; edges 3,4,5 are only
defined on the inner (1..i-2, 1..j-2) block and zero on its boundary (the
original fills q_ap4/5/6 with 0 there). Both are reproduced by xp.pad. The last
row/col of q_a (i-1 / j-1) is left 0, matching the original (CONST_UNDEF there,
never read downstream -> downstream-identical, as for horizontalflux.py).

GRD_x at K0 is k-independent and broadcasts over k via [:, :, None, :].
The pole (_pl) branch is small and stays on the host path in mod_src_tracer.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RemapCfg:
    """Static (hashable) parameters: safe as static_argnames under jax.jit."""
    AI: int
    AIJ: int
    AJ: int
    XDIR: int
    YDIR: int
    ZDIR: int


def compute_horizontal_remap(q, gradq, grd_xc, cmask, grd_x_k0, cfg: RemapCfg, xp):
    """q_a (i,j,k,l,6). Inputs:
       q        (i,j,k,l)            cell-centre tracer
       gradq    (i,j,k,l,3)          grad(q) (after COMM)
       grd_xc   (i,j,k,l,3,3)        arc-point positions [...,arc,dir]
       cmask    (i,j,k,l,6)          upwind mask
       grd_x_k0 (i,j,l,3)            cell-centre positions at K0 (k-independent)
    """
    AI, AIJ, AJ = cfg.AI, cfg.AIJ, cfg.AJ
    XDIR, YDIR, ZDIR = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    i, j, kall, l = q.shape
    one = xp.asarray(1.0, dtype=q.dtype)

    isl, jsl = slice(0, i - 1), slice(0, j - 1)
    isl_p1, jsl_p1 = slice(1, i), slice(1, j)
    isls, jsls = slice(1, i - 1), slice(1, j - 1)
    isls_m1, jsls_m1 = slice(0, i - 2), slice(0, j - 2)

    def rec(ci, cj, ai, aj, arc):
        gx = grd_x_k0[ci, cj, :, :]                       # (I,J,l,3)
        gxx = gx[:, :, None, :]                           # (I,J,1,l) per dir below
        return (
            q[ci, cj, :, :]
            + gradq[ci, cj, :, :, XDIR] * (grd_xc[ai, aj, :, :, arc, XDIR] - gxx[..., XDIR])
            + gradq[ci, cj, :, :, YDIR] * (grd_xc[ai, aj, :, :, arc, YDIR] - gxx[..., YDIR])
            + gradq[ci, cj, :, :, ZDIR] * (grd_xc[ai, aj, :, :, arc, ZDIR] - gxx[..., ZDIR])
        )

    def pad_inner(block):
        # (i-2,j-2,k,l) inner block -> (i-1,j-1,k,l) isl/jsl frame, 0 on low edge
        return xp.pad(block, ((1, 0), (1, 0), (0, 0), (0, 0)))

    cm = cmask[isl, jsl, :, :, :]   # (i-1,j-1,k,l,6) upwind mask on the isl/jsl frame

    # edges 0,1,2 : full isl/jsl frame
    ap1 = rec(isl, jsl, isl, jsl, AI);      am1 = rec(isl_p1, jsl, isl, jsl, AI)
    ap2 = rec(isl, jsl, isl, jsl, AIJ);     am2 = rec(isl_p1, jsl_p1, isl, jsl, AIJ)
    ap3 = rec(isl, jsl, isl, jsl, AJ);      am3 = rec(isl, jsl_p1, isl, jsl, AJ)
    # edges 3,4,5 : inner block, padded to isl/jsl frame (0 on low boundary)
    ap4 = pad_inner(rec(isls_m1, jsls, isls_m1, jsls, AI))
    am4 = pad_inner(rec(isls,    jsls, isls_m1, jsls, AI))
    ap5 = pad_inner(rec(isls_m1, jsls_m1, isls_m1, jsls_m1, AIJ))
    am5 = pad_inner(rec(isls,    jsls,    isls_m1, jsls_m1, AIJ))
    ap6 = pad_inner(rec(isls, jsls_m1, isls, jsls_m1, AJ))
    am6 = pad_inner(rec(isls, jsls,    isls, jsls_m1, AJ))

    def sel(e, am, ap):
        c = cm[:, :, :, :, e]
        return c * am + (one - c) * ap

    e0 = sel(0, am1, ap1); e1 = sel(1, am2, ap2); e2 = sel(2, am3, ap3)
    e3 = sel(3, am4, ap4); e4 = sel(4, am5, ap5); e5 = sel(5, am6, ap6)

    q_a_frame = xp.stack([e0, e1, e2, e3, e4, e5], axis=-1)   # (i-1,j-1,k,l,6)
    # pad to full (i,j,k,l,6): last row/col stay 0 (downstream-identical)
    return xp.pad(q_a_frame, ((0, 1), (0, 1), (0, 0), (0, 0), (0, 0)))
