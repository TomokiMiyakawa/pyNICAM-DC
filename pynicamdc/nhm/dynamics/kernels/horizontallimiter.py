"""Backend-switchable horizontal Thuburn limiter (regular grid; pole stays host).

Functional (jax-safe) port of the vectorized horizontal_limiter_thuburn body.
SPLIT INTO TWO STAGES around the Qout halo exchange, because the host caller does
`comm.COMM_data_transfer(Qout, Qout_pl)` between the Qout build and the apply
(mod_src_tracer.py): the apply reads neighbour-rank Qout values at the ring. A
single fused kernel would apply against un-exchanged local Qout -> non-monotone
limiter -> blow-up/NaN. So:

  compute_horizontal_limiter_qout(q, d, ch, cmask, cfg, xp) -> (Qin, Qout)
      qin build + singular-point correction + Qout. Qin/Qout drained to the host
      numpy arrays; the host then halo-exchanges Qout (only Qout needs COMM --
      Qin's ring is computed locally).
  compute_horizontal_limiter_apply(q_a, Qin, Qout, cmask, cfg, xp) -> q_a
      the 3-direction apply, reading the COMM'd Qout. q_a is INOUT (returned).

Bit-exact vs the vectorized numpy path on CPU; machine-precision on GPU (min/max
are exact, but cmask blends / Qout divides / 6-edge sums may FMA-fuse). Mirrors
the validated PYNICAM_HLIM_VEC numpy code one-to-one, with in-place writes
replaced by .at[].set() and the static per-l sgp loop unrolled at trace time.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce as _reduce


@dataclass(frozen=True)
class HLimiterCfg:
    iall: int
    jall: int
    lall: int
    I_min: int
    I_max: int
    BIG: float
    EPS: float
    have_sgp: tuple   # per-l bool (static -> sgp loop unrolls)


@dataclass(frozen=True)
class HLimiterCfgPl:
    """Static (hashable) parameters for the POLE Thuburn limiter kernels."""
    n: int       # gslf_pl  (pentagon centre index, = 0)
    gmin: int    # gmin_pl  (first neighbour vertex, = 1)
    gmax: int    # gmax_pl  (last  neighbour vertex, = 5)
    I_min: int
    I_max: int
    BIG: float
    EPS: float


def compute_horizontal_limiter_qout_pl(q_pl, d_pl, ch_pl, cmask_pl, cfg: HLimiterCfgPl, xp):
    """POLE (pentagon) Qin build + Qout. Device/functional mirror of the host pole
    loop (Qin_pl over the 5 ring neighbours v=gmin..gmax, Qout_pl at the centre n).
    The cyclic ring neighbours ijm1/ijp1 are roll(+-1) over the v-axis. Returns
    (Qin_pl, Qout_pl) full pole arrays; only the v=gmin..gmax rows of Qin and the
    centre n of Qout are meaningful (the rest are 0 / filled by the Qout COMM).
       q_pl/d_pl/ch_pl/cmask_pl  (gall_pl, kall, lall_pl)
       Qin_pl   (gall_pl, kall, lall_pl, 2, 2)
       Qout_pl  (gall_pl, kall, lall_pl, 2)
    """
    n, gmin, gmax = cfg.n, cfg.gmin, cfg.gmax
    I_min, I_max = cfg.I_min, cfg.I_max
    BIG, EPS = cfg.BIG, cfg.EPS
    ONE = 1.0
    gall_pl, kall, lall_pl = q_pl.shape
    vsl = slice(gmin, gmax + 1)

    qn = q_pl[vsl]                          # (V,k,l) self
    qc = q_pl[n]                            # (k,l)   centre
    qn_p1 = xp.roll(qn, -1, axis=0)         # v+1 (cyclic over the ring)
    qn_m1 = xp.roll(qn,  1, axis=0)         # v-1
    q_min = xp.minimum(xp.minimum(qc[None], qn), xp.minimum(qn_m1, qn_p1))
    q_max = xp.maximum(xp.maximum(qc[None], qn), xp.maximum(qn_m1, qn_p1))

    cm = cmask_pl[vsl]                      # (V,k,l)
    is1 = cm == ONE
    qin_min0 = xp.where(is1, q_min,  BIG)
    qin_min1 = xp.where(is1,  BIG,  q_min)
    qin_max0 = xp.where(is1, q_max, -BIG)
    qin_max1 = xp.where(is1, -BIG,  q_max)

    qnext_min = xp.minimum(qc, xp.min(qin_min0, axis=0))   # (k,l)
    qnext_max = xp.maximum(qc, xp.max(qin_max0, axis=0))

    chn = ch_pl[vsl]                        # (V,k,l)
    ch_m = cm * chn
    Cin  = xp.sum(ch_m, axis=0)            # (k,l)
    Cout = xp.sum(chn - ch_m, axis=0)
    CQin_min = xp.sum(ch_m * qin_min0, axis=0)
    CQin_max = xp.sum(ch_m * qin_max0, axis=0)

    zerosw = 0.5 - xp.copysign(0.5, xp.abs(Cout) - EPS)
    denom = Cout + zerosw
    factor = ONE - zerosw
    dval = d_pl[n]                          # (k,l)
    Qout_n_min = (qc - CQin_max - qnext_max * (ONE - Cin - Cout + dval)) / denom * factor + qc * zerosw
    Qout_n_max = (qc - CQin_min - qnext_min * (ONE - Cin - Cout + dval)) / denom * factor + qc * zerosw

    Qin = xp.zeros((gall_pl, kall, lall_pl, 2, 2), dtype=q_pl.dtype)
    Qin = Qin.at[vsl, :, :, I_min, 0].set(qin_min0)
    Qin = Qin.at[vsl, :, :, I_min, 1].set(qin_min1)
    Qin = Qin.at[vsl, :, :, I_max, 0].set(qin_max0)
    Qin = Qin.at[vsl, :, :, I_max, 1].set(qin_max1)

    Qout = xp.zeros((gall_pl, kall, lall_pl, 2), dtype=q_pl.dtype)
    Qout = Qout.at[n, :, :, I_min].set(Qout_n_min)
    Qout = Qout.at[n, :, :, I_max].set(Qout_n_max)
    return Qin, Qout


def compute_horizontal_limiter_apply_pl(q_a_pl, Qin_pl, Qout_pl, cmask_pl, cfg: HLimiterCfgPl, xp):
    """POLE 2-stage apply against the (halo-exchanged) Qout_pl. q_a_pl is INOUT
    (returned). Only the v=gmin..gmax rows are updated; the centre n is preserved."""
    n, gmin, gmax = cfg.n, cfg.gmin, cfg.gmax
    I_min, I_max = cfg.I_min, cfg.I_max
    ONE = 1.0
    vsl = slice(gmin, gmax + 1)

    cm = cmask_pl[vsl]                       # (V,k,l)
    qa = q_a_pl[vsl]                         # (V,k,l)
    # stage 1: clamp to the Qin range (edge 0 if cm==1, edge 1 otherwise)
    q0 = xp.minimum(xp.maximum(qa, Qin_pl[vsl, :, :, I_min, 0]), Qin_pl[vsl, :, :, I_max, 0])
    q1 = xp.minimum(xp.maximum(qa, Qin_pl[vsl, :, :, I_min, 1]), Qin_pl[vsl, :, :, I_max, 1])
    qa = cm * q0 + (ONE - cm) * q1
    # stage 2: clamp to the Qout range (neighbour v slot if cm==1, centre n otherwise)
    q2 = xp.maximum(xp.minimum(qa, Qout_pl[vsl, :, :, I_max]), Qout_pl[vsl, :, :, I_min])
    q3 = xp.maximum(xp.minimum(qa, Qout_pl[n, :, :, I_max][None]), Qout_pl[n, :, :, I_min][None])
    qa = cm * q2 + (ONE - cm) * q3
    return q_a_pl.at[vsl].set(qa)


def compute_horizontal_limiter_qout(q, d, ch, cmask, cfg: HLimiterCfg, xp):
    """qin build + sgp correction + Qout. Returns (Qin, Qout); host COMMs Qout."""
    iall, jall, lall = cfg.iall, cfg.jall, cfg.lall
    I_min, I_max = cfg.I_min, cfg.I_max
    BIG, EPS = cfg.BIG, cfg.EPS
    ONE = 1.0

    si  = slice(0, iall - 1); sj  = slice(0, jall - 1)
    sip = slice(1, iall);     sjp = slice(1, jall)

    # ---------- Qin build (regular) ----------
    q0 = q[si, sj]; q2 = q[sip, sj]; q3 = q[sip, sjp]; qjp1 = q[si, sjp]
    # q1 = q[i,j-1] (j==0 -> q[0,0]);  q4 = q[i-1,j] (i==0 -> q[0,0])
    q1 = xp.zeros_like(q0)
    q1 = q1.at[:, 1:].set(q[si, 0:jall - 2])
    q1 = q1.at[:, 0].set(q[0, 0])
    q4 = xp.zeros_like(q0)
    q4 = q4.at[1:, :].set(q[0:iall - 2, sj])
    q4 = q4.at[0, :].set(q[0, 0])

    mnAI  = xp.minimum(xp.minimum(q0, q1), xp.minimum(q2, q3))
    mxAI  = xp.maximum(xp.maximum(q0, q1), xp.maximum(q2, q3))
    mnAIJ = xp.minimum(xp.minimum(q0, q2), xp.minimum(q3, qjp1))
    mxAIJ = xp.maximum(xp.maximum(q0, q2), xp.maximum(q3, qjp1))
    mnAJ  = xp.minimum(xp.minimum(q0, q3), xp.minimum(qjp1, q4))
    mxAJ  = xp.maximum(xp.maximum(q0, q3), xp.maximum(qjp1, q4))

    Qin = xp.zeros((iall, jall) + q.shape[2:] + (2, 6), dtype=q.dtype)
    cm = cmask[si, sj]
    for m, (qmn, qmx) in enumerate(((mnAI, mxAI), (mnAIJ, mxAIJ), (mnAJ, mxAJ))):
        c = cm[:, :, :, :, m]
        Qin = Qin.at[si, sj, :, :, I_min, m].set(c * qmn + (ONE - c) * BIG)
        Qin = Qin.at[si, sj, :, :, I_max, m].set(c * qmx + (ONE - c) * (-BIG))
    c0 = cm[:, :, :, :, 0]; c1 = cm[:, :, :, :, 1]; c2 = cm[:, :, :, :, 2]
    Qin = Qin.at[sip, sj,  :, :, I_min, 3].set(c0 * BIG    + (ONE - c0) * mnAI)
    Qin = Qin.at[sip, sj,  :, :, I_max, 3].set(c0 * (-BIG) + (ONE - c0) * mxAI)
    Qin = Qin.at[sip, sjp, :, :, I_min, 4].set(c1 * BIG    + (ONE - c1) * mnAIJ)
    Qin = Qin.at[sip, sjp, :, :, I_max, 4].set(c1 * (-BIG) + (ONE - c1) * mxAIJ)
    Qin = Qin.at[si,  sjp, :, :, I_min, 5].set(c2 * BIG    + (ONE - c2) * mnAJ)
    Qin = Qin.at[si,  sjp, :, :, I_max, 5].set(c2 * (-BIG) + (ONE - c2) * mxAJ)

    # ---------- singular-point correction (static per-l, unrolled) ----------
    for l in range(lall):
        if cfg.have_sgp[l]:
            aijmn = _reduce(xp.minimum, [q[0, 0, :, l], q[1, 1, :, l], q[2, 1, :, l], q[0, 1, :, l]])
            aijmx = _reduce(xp.maximum, [q[0, 0, :, l], q[1, 1, :, l], q[2, 1, :, l], q[0, 1, :, l]])
            sc1 = cmask[0, 0, :, l, 1]
            Qin = Qin.at[0, 0, :, l, I_min, 1].set(xp.where(sc1 == ONE, aijmn,  BIG))
            Qin = Qin.at[1, 1, :, l, I_min, 4].set(xp.where(sc1 == ONE,   BIG,  aijmn))
            Qin = Qin.at[0, 0, :, l, I_max, 1].set(xp.where(sc1 == ONE, aijmx, -BIG))
            Qin = Qin.at[1, 1, :, l, I_max, 4].set(xp.where(sc1 == ONE,  -BIG,  aijmx))

    # ---------- Qout ----------
    isl = slice(1, iall - 1); jsl = slice(1, jall - 1)
    qnmin = _reduce(xp.minimum, [q[isl, jsl]] + [Qin[isl, jsl, :, :, I_min, e] for e in range(6)])
    qnmax = _reduce(xp.maximum, [q[isl, jsl]] + [Qin[isl, jsl, :, :, I_max, e] for e in range(6)])
    chm = xp.minimum(ch[isl, jsl], 0.0)
    Cin  = xp.sum(chm, axis=-1)
    Cout = xp.sum(ch[isl, jsl] - chm, axis=-1)
    CQmin = xp.sum(chm * Qin[isl, jsl, :, :, I_min, :], axis=-1)
    CQmax = xp.sum(chm * Qin[isl, jsl, :, :, I_max, :], axis=-1)
    zsw = 0.5 - xp.copysign(0.5, xp.abs(Cout) - EPS)
    qi = q[isl, jsl]; di = d[isl, jsl]
    Qout = xp.zeros((iall, jall) + q.shape[2:] + (2,), dtype=q.dtype)
    Qout = Qout.at[isl, jsl, :, :, I_min].set(
        (qi - CQmax - qnmax * (ONE - Cin - Cout + di)) / (Cout + zsw) * (ONE - zsw) + qi * zsw)
    Qout = Qout.at[isl, jsl, :, :, I_max].set(
        (qi - CQmin - qnmin * (ONE - Cin - Cout + di)) / (Cout + zsw) * (ONE - zsw) + qi * zsw)
    for II in (I_min, I_max):
        Qout = Qout.at[:, 0, :, :, II].set(q[:, 0])
        Qout = Qout.at[:, jall - 1, :, :, II].set(q[:, jall - 1])
        Qout = Qout.at[0, 1:jall - 1, :, :, II].set(q[0, 1:jall - 1])
        Qout = Qout.at[iall - 1, 1:jall - 1, :, :, II].set(q[iall - 1, 1:jall - 1])

    return Qin, Qout


def compute_horizontal_limiter_apply(q_a, Qin, Qout, cmask, cfg: HLimiterCfg, xp):
    """3-direction apply against the (halo-exchanged) Qout. q_a is INOUT."""
    iall, jall = cfg.iall, cfg.jall
    I_min, I_max = cfg.I_min, cfg.I_max
    ONE = 1.0
    ai = slice(0, iall - 1); aj = slice(0, jall - 1)
    aip = slice(1, iall);    ajp = slice(1, jall)
    # dir 1: 0 -> 3
    t = cmask[ai, aj, :, :, 0] * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 0], Qin[ai, aj, :, :, I_min, 0]), Qin[ai, aj, :, :, I_max, 0]) \
        + (ONE - cmask[ai, aj, :, :, 0]) * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 0], Qin[aip, aj, :, :, I_min, 3]), Qin[aip, aj, :, :, I_max, 3])
    t = cmask[ai, aj, :, :, 0] * xp.maximum(xp.minimum(t, Qout[aip, aj, :, :, I_max]), Qout[aip, aj, :, :, I_min]) \
        + (ONE - cmask[ai, aj, :, :, 0]) * xp.maximum(xp.minimum(t, Qout[ai, aj, :, :, I_max]), Qout[ai, aj, :, :, I_min])
    q_a = q_a.at[ai, aj, :, :, 0].set(t)
    q_a = q_a.at[aip, aj, :, :, 3].set(t)
    # dir 2: 1 -> 4
    t = cmask[ai, aj, :, :, 1] * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 1], Qin[ai, aj, :, :, I_min, 1]), Qin[ai, aj, :, :, I_max, 1]) \
        + (ONE - cmask[ai, aj, :, :, 1]) * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 1], Qin[aip, ajp, :, :, I_min, 4]), Qin[aip, ajp, :, :, I_max, 4])
    t = cmask[ai, aj, :, :, 1] * xp.maximum(xp.minimum(t, Qout[aip, ajp, :, :, I_max]), Qout[aip, ajp, :, :, I_min]) \
        + (ONE - cmask[ai, aj, :, :, 1]) * xp.maximum(xp.minimum(t, Qout[ai, aj, :, :, I_max]), Qout[ai, aj, :, :, I_min])
    q_a = q_a.at[ai, aj, :, :, 1].set(t)
    q_a = q_a.at[aip, ajp, :, :, 4].set(t)
    # dir 3: 2 -> 5
    t = cmask[ai, aj, :, :, 2] * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 2], Qin[ai, aj, :, :, I_min, 2]), Qin[ai, aj, :, :, I_max, 2]) \
        + (ONE - cmask[ai, aj, :, :, 2]) * xp.minimum(xp.maximum(q_a[ai, aj, :, :, 2], Qin[ai, ajp, :, :, I_min, 5]), Qin[ai, ajp, :, :, I_max, 5])
    t = cmask[ai, aj, :, :, 2] * xp.maximum(xp.minimum(t, Qout[ai, ajp, :, :, I_max]), Qout[ai, ajp, :, :, I_min]) \
        + (ONE - cmask[ai, aj, :, :, 2]) * xp.maximum(xp.minimum(t, Qout[ai, aj, :, :, I_max]), Qout[ai, aj, :, :, I_min])
    q_a = q_a.at[ai, aj, :, :, 2].set(t)
    q_a = q_a.at[ai, ajp, :, :, 5].set(t)

    return q_a
