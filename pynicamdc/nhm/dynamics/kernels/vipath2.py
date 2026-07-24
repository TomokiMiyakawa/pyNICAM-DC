"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the COMM-free
"C2" island at the tail of Vi.vi_small_step's per-iteration loop (mod_vi.py
vi_path2), i.e. everything after the COMM_data_transfer(diff_we):

  PROG_split[..., RHOG ]  = diff_we[..., 0]
  PROG_split[..., RHOGVX] = diff_vh[..., 0]
  PROG_split[..., RHOGVY] = diff_vh[..., 1]
  PROG_split[..., RHOGVZ] = diff_vh[..., 2]
  PROG_split[..., RHOGW ] = diff_we[..., 1]
  PROG_split[..., RHOGE ] = diff_we[..., 2]
  PROG_mean[..., RHOG:RHOGW+1] += PROG_split[..., RHOG:RHOGW+1] * rweight_itr

This island is almost pure data movement (six copies + one fused-multiply-add
with no arithmetic intensity), so fusing it in isolation does not reduce memory
traffic and is expected to be neutral-to-negative under the current
COMM-forces-host architecture. Its real value is (1) bit-exact validation and
(2) being ready for Win B, where diff_vh / diff_we are already device-resident
and the writeback becomes free. Kept behind a default-OFF toggle in mod_vi.py.

The original mutates persistent buffers in place; here we return new PROG_split
and PROG_mean arrays and let the caller write them back, keeping the function
purely functional / jit-friendly.

The component order of PROG_split / PROG_mean is the prognostic index order
[I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW, I_RHOGE] = [0..5]; the running
mean accumulates indices 0..4 (RHOG..RHOGW), leaving RHOGE (index 5) untouched.
"""

from __future__ import annotations
from dataclasses import dataclass

from pynicamdc.share.mod_backend import backend as bk


@dataclass(frozen=True)
class ViPath2Cfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    have_pl: bool
    I_RHOG: int
    I_RHOGVX: int
    I_RHOGVY: int
    I_RHOGVZ: int
    I_RHOGW: int
    I_RHOGE: int


def _update(diff_vh, diff_we, PROG_mean, rweight_itr, cfg, stack_axis, mean_lo, mean_hi, xp):
    """Shared regional/pole body. diff_* have a trailing size-3 axis at
    stack_axis-1; PROG_split / PROG_mean component axis is stack_axis."""
    # new PROG_split in prognostic index order [RHOG, VX, VY, VZ, W, E]
    g  = _take(diff_we, 0, stack_axis)
    vx = _take(diff_vh, 0, stack_axis)
    vy = _take(diff_vh, 1, stack_axis)
    vz = _take(diff_vh, 2, stack_axis)
    gw = _take(diff_we, 1, stack_axis)
    ge = _take(diff_we, 2, stack_axis)
    PROG_split = xp.stack([g, vx, vy, vz, gw, ge], axis=stack_axis)

    # running mean over RHOG..RHOGW (mean_lo:mean_hi); RHOGE left untouched
    lo = _slice_last(PROG_mean, stack_axis, mean_lo, mean_hi)
    ps = _slice_last(PROG_split, stack_axis, mean_lo, mean_hi)
    updated = lo + ps * rweight_itr
    # write the updated component slice back via set_at instead of
    # concatenating [updated, rest] (vi-stack-plan v1, part of S4).
    # at_base: PROG_mean is caller-owned (numpy copies).
    _idx = (slice(None),) * stack_axis + (slice(mean_lo, mean_hi),)
    PROG_mean_out = bk.set_at(bk.at_base(PROG_mean), _idx, updated)

    return PROG_split, PROG_mean_out


def _take(a, idx, comp_axis):
    """a[..., idx] along the last axis (comp_axis is the PROG component axis,
    i.e. one past diff_*'s trailing axis)."""
    sl = [slice(None)] * a.ndim
    sl[-1] = idx
    return a[tuple(sl)]


def _slice_last(a, axis, lo, hi):
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lo, hi)
    return a[tuple(sl)]


def compute_vi_path2_components(P, dt_unused, cfg: ViPath2Cfg, xp):
    """Component-carry variant of compute_vi_path2_update (vi-stack-plan v1, V3a).

    The regular PROG_split / PROG_mean are threaded through the resident ns-loop
    as SEPARATE component arrays instead of one stacked (i,j,k,l,6) array, so
    this island never re-stacks them: it takes the post-COMM diff_vh / diff_we
    components (cheap trailing-axis slices) and updates the 5 mean components
    elementwise. Values are bit-identical to the stacked kernel (same take, same
    lo + ps*rw per element); only the data layout of the carry changes.
    The pole arrays stay stacked (tiny) and reuse the shared _update body.

    P : diff_vh (i,j,k,l,3), diff_we (i,j,k,l,3),
        pm0..pm4 (the 5 PROG_mean components RHOG..RHOGW), rweight_itr
        (+ stacked diff_vh_pl / diff_we_pl / PROG_mean_pl when have_pl)
    Returns dict: ps (6-tuple RHOG,VX,VY,VZ,W,E), pm (5-tuple)
        (+ stacked PROG_split_pl / PROG_mean_pl when have_pl).
    """
    mean_lo, mean_hi = cfg.I_RHOG, cfg.I_RHOGW + 1
    rw = P["rweight_itr"]

    dvh, dwe = P["diff_vh"], P["diff_we"]
    ps = (dwe[..., 0], dvh[..., 0], dvh[..., 1], dvh[..., 2], dwe[..., 1], dwe[..., 2])
    pm = tuple(P[f"pm{i}"] + ps[i] * rw for i in range(5))
    out = {"ps": ps, "pm": pm}

    if cfg.have_pl:
        PROG_split_pl, PROG_mean_pl = _update(
            P["diff_vh_pl"], P["diff_we_pl"], P["PROG_mean_pl"], rw, cfg,
            stack_axis=3, mean_lo=mean_lo, mean_hi=mean_hi, xp=xp,
        )
        out["PROG_split_pl"] = PROG_split_pl
        out["PROG_mean_pl"] = PROG_mean_pl

    return out


def compute_vi_path2_update(P, dt_unused, cfg: ViPath2Cfg, xp):
    """Fused PROG_split writeback + PROG_mean accumulation.

    P : dict of per-call device arrays
        diff_vh (i,j,k,l,3), diff_we (i,j,k,l,3), PROG_mean (i,j,k,l,nm),
        rweight_itr (scalar)  (+ *_pl variants)
    Returns dict: PROG_split, PROG_mean (+ *_pl when have_pl).
    """
    mean_lo, mean_hi = cfg.I_RHOG, cfg.I_RHOGW + 1
    rw = P["rweight_itr"]

    PROG_split, PROG_mean = _update(
        P["diff_vh"], P["diff_we"], P["PROG_mean"], rw, cfg,
        stack_axis=4, mean_lo=mean_lo, mean_hi=mean_hi, xp=xp,
    )
    out = {"PROG_split": PROG_split, "PROG_mean": PROG_mean}

    if cfg.have_pl:
        PROG_split_pl, PROG_mean_pl = _update(
            P["diff_vh_pl"], P["diff_we_pl"], P["PROG_mean_pl"], rw, cfg,
            stack_axis=3, mean_lo=mean_lo, mean_hi=mean_hi, xp=xp,
        )
        out["PROG_split_pl"] = PROG_split_pl
        out["PROG_mean_pl"] = PROG_mean_pl

    return out
