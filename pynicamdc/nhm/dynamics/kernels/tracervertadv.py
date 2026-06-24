"""
Pure / backend-switchable (numpy <-> jax.numpy) kernels for the per-tracer
vertical-advection body of src_tracer_advection (mod_src_tracer.py), the two
Strang fractional vertical steps. These replace the literal Python (l,k) loops
that compute, for each tracer iq:

  q   = rhogq[...,iq] / rho_den                       (cell-center ratio)
  q_h = GRD_afact[k]*q[k] + GRD_bfact[k]*q[k-1]       (k = kmin..kmax+1; q_h[kmin-1]=0)
        --> handed to the (existing) vertical Thuburn limiter --
  rhogq[...,iq] -= (flx_v[k+1]*q_h[k+1] - flx_v[k]*q_h[k]) * GRD_rdgz[k]
                                                       (k = kmin..kmax, with
                                                        q_h[kmin]=q_h[kmax+1]=0
                                                        applied first; the two
                                                        rhogq ghost rows zeroed)

The limiter sits between the two halves, so this is split into a pre-limiter
kernel (q + q_h) and a post-limiter kernel (rhogq update), mirroring the
horizontal-limiter Stage-3 split. No COMM (vertical/k only). rho_den is the
caller's denominator: rhog_in for the 1st step, rhog (updated) for the 2nd.

Assembly is backend-agnostic (xp.concatenate along k + xp.where for the q_h
ghost zeroing), so the same code runs bit-identically under numpy and jit-able
under jax. Non-interior rhogq rows (k<kmin-1, k>kmax+1) are preserved; q_h is
scratch and fully (re)built each call.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TracerVertAdvCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool


def _build_qh(q, kax, kmin, kmax, a, b, xp):
    """Assemble q_h = a*q[k] + b*q[k-1] on k=kmin..kmax+1, zero elsewhere.

    kax = the k axis index (2 regional, 1 pole); a/b already broadcast-shaped.
    """
    sl = [slice(None)] * q.ndim
    def kslice(lo, hi):
        s = list(sl); s[kax] = slice(lo, hi); return tuple(s)
    kall = q.shape[kax]
    inner = a[kslice(kmin, kmax + 2)] * q[kslice(kmin, kmax + 2)] \
        + b[kslice(kmin, kmax + 2)] * q[kslice(kmin - 1, kmax + 1)]
    below_shape = list(q.shape); below_shape[kax] = kmin              # k=0..kmin-1
    above_shape = list(q.shape); above_shape[kax] = kall - (kmax + 2)  # k=kmax+2..
    below = xp.zeros(tuple(below_shape), dtype=q.dtype)
    above = xp.zeros(tuple(above_shape), dtype=q.dtype)
    return xp.concatenate([below, inner, above], axis=kax)


def compute_vert_qh(rhogq_iq, rho_den, afact, bfact, cfg: TracerVertAdvCfg, xp):
    """Pure pre-limiter step (regional): returns (q, q_h) for one tracer."""
    q = rhogq_iq / rho_den
    a = afact[None, None, :, None]
    b = bfact[None, None, :, None]
    q_h = _build_qh(q, 2, cfg.kmin, cfg.kmax, a, b, xp)
    return q, q_h


def compute_vert_qh_pl(rhogq_iq_pl, rho_den_pl, afact, bfact, cfg: TracerVertAdvCfg, xp):
    """Pole q and q_h. _pl arrays are (g, k, l)."""
    if not cfg.have_pl:
        z = xp.zeros_like(rhogq_iq_pl)
        return z, z
    q = rhogq_iq_pl / rho_den_pl
    a = afact[None, :, None]
    b = bfact[None, :, None]
    q_h = _build_qh(q, 1, cfg.kmin, cfg.kmax, a, b, xp)
    return q, q_h


def _update(rhogq_iq, flx_v, q_h, rdgz, kax, kmin, kmax, xp):
    """Flux-divergence rhogq update along axis kax; q_h ghosts (kmin, kmax+1)
    zeroed first; the two rhogq ghost rows (kmin-1, kmax+1) zeroed; other rows
    preserved."""
    sl = [slice(None)] * rhogq_iq.ndim
    def kslice(lo, hi):
        s = list(sl); s[kax] = slice(lo, hi); return tuple(s)
    def kone(k):
        s = list(sl); s[kax] = slice(k, k + 1); return tuple(s)
    kall = rhogq_iq.shape[kax]
    # zero q_h at kmin and kmax+1 via a k-mask (backend-agnostic)
    kk = xp.arange(kall)
    keep = (kk != kmin) & (kk != (kmax + 1))
    kshape = [1] * rhogq_iq.ndim; kshape[kax] = kall
    q_h0 = xp.where(keep.reshape(tuple(kshape)), q_h, xp.zeros((), dtype=q_h.dtype))
    g = rdgz.reshape(tuple(kshape))
    div = (flx_v[kslice(kmin + 1, kmax + 2)] * q_h0[kslice(kmin + 1, kmax + 2)]
           - flx_v[kslice(kmin, kmax + 1)] * q_h0[kslice(kmin, kmax + 1)]) * g[kslice(kmin, kmax + 1)]
    modified = rhogq_iq[kslice(kmin, kmax + 1)] - div
    zrow = xp.zeros(rhogq_iq[kone(kmin - 1)].shape, dtype=rhogq_iq.dtype)
    return xp.concatenate([
        rhogq_iq[kslice(0, kmin - 1)],   # preserve k < kmin-1
        zrow,                            # k = kmin-1 -> 0
        modified,                        # k = kmin..kmax
        zrow,                            # k = kmax+1 -> 0
        rhogq_iq[kslice(kmax + 2, kall)],  # preserve k > kmax+1
    ], axis=kax)


def compute_vert_update(rhogq_iq, flx_v, q_h, rdgz, cfg: TracerVertAdvCfg, xp):
    """Pure post-limiter step (regional): returns updated rhogq_iq."""
    return _update(rhogq_iq, flx_v, q_h, rdgz, 2, cfg.kmin, cfg.kmax, xp)


def compute_vert_update_pl(rhogq_iq_pl, flx_v_pl, q_h_pl, rdgz, cfg: TracerVertAdvCfg, xp):
    """Pole post-limiter update. _pl arrays are (g, k, l)."""
    if not cfg.have_pl:
        return rhogq_iq_pl
    return _update(rhogq_iq_pl, flx_v_pl, q_h_pl, rdgz, 1, cfg.kmin, cfg.kmax, xp)
