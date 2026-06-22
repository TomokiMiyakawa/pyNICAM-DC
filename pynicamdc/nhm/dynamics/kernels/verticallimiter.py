"""Backend-switchable vertical Thuburn limiter (regular grid; pole stays host).

Functional (jax-safe) port of vertical_limiter_thuburn's regular body. No halo
COMM (vertical/k-direction only). The original walks k sequentially carrying the
previous level's Qout, but Qout at level k depends only on q/ck/d at k-1,k,k+1 --
no inter-k dependence in the COMPUTE -- so we precompute Qout for all interior
levels (k=kmin..kmax) at once, then apply (k=kmin+1..kmax) reading Qout[k-1],Qout[k].

FAITHFUL QUIRK: the original's k==kmin special block uses ck[k+1,l,1] for the
upper term (Cin/Cout/CQin), whereas the main loop (and the pole) use ck[k,l,1].
That latent inconsistency is baked into the gold, so we reproduce it: the
vectorized ck2 has its kmin level overridden to ck[kmin+1,l,1].

Bit-exact vs the numpy path on CPU; machine-precision on GPU. q_h is INOUT
(returned); only the apply levels kmin+1..kmax are modified.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class VLimiterCfg:
    iall: int
    jall: int
    kall: int
    lall: int
    kmin: int
    kmax: int
    BIG: float
    EPS: float


def _clip(x, lo, hi, xp):
    # np.clip semantics: minimum(maximum(x, lo), hi)
    return xp.minimum(xp.maximum(x, lo), hi)


def compute_vertical_limiter(q_h, q, d, ck, cfg: VLimiterCfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    BIG, EPS = cfg.BIG, cfg.EPS
    ONE = 1.0

    kc   = slice(kmin,     kmax + 1)   # interior levels k = kmin..kmax
    kcm1 = slice(kmin - 1, kmax)       # k-1
    kcp1 = slice(kmin + 1, kmax + 2)   # k+1

    # ---------- Qout at every interior level (vectorized over i,j,k,l) ----------
    inflagL = 0.5 - xp.copysign(0.5, ck[:, :, kc,   :, 0])
    inflagU = 0.5 + xp.copysign(0.5, ck[:, :, kcp1, :, 0])

    q_c = q[:, :, kc,   :]
    q_b = q[:, :, kcm1, :]
    q_a = q[:, :, kcp1, :]

    ck1 = ck[:, :, kc, :, 0]
    ck2 = ck[:, :, kc, :, 1]
    # kmin special case: upper term uses ck[kmin+1,l,1] (see module docstring).
    ck2 = ck2.at[:, :, 0, :].set(ck[:, :, kmin + 1, :, 1])

    QinminL = xp.where(inflagL == ONE, xp.minimum(q_c, q_b),  BIG)
    QinminU = xp.where(inflagU == ONE, xp.minimum(q_c, q_a),  BIG)
    QinmaxL = xp.where(inflagL == ONE, xp.maximum(q_c, q_b), -BIG)
    QinmaxU = xp.where(inflagU == ONE, xp.maximum(q_c, q_a), -BIG)

    qnmin = xp.minimum(xp.minimum(QinminL, QinminU), q_c)
    qnmax = xp.maximum(xp.maximum(QinmaxL, QinmaxU), q_c)

    Cin  = inflagL * ck1 + inflagU * ck2
    Cout = (ONE - inflagL) * ck1 + (ONE - inflagU) * ck2
    CQmin = inflagL * ck1 * QinminL + inflagU * ck2 * QinminU
    CQmax = inflagL * ck1 * QinmaxL + inflagU * ck2 * QinmaxU

    zsw = 0.5 - xp.copysign(0.5, xp.abs(Cout) - EPS)
    dkc = d[:, :, kc, :]

    Qout_min = ((q_c - qnmax) + qnmax * (Cin + Cout - dkc) - CQmax) \
        / (Cout + zsw) * (ONE - zsw) + q_c * zsw
    Qout_max = ((q_c - qnmin) + qnmin * (Cin + Cout - dkc) - CQmin) \
        / (Cout + zsw) * (ONE - zsw) + q_c * zsw

    # ---------- apply (k = kmin+1..kmax): inflagL[k] picks Qout[k-1] vs Qout[k] ----------
    ka = slice(kmin + 1, kmax + 1)
    infL_a = inflagL[:, :, 1:, :]
    qh_a = q_h[:, :, ka, :]
    clip_km1 = _clip(qh_a, Qout_min[:, :, :-1, :], Qout_max[:, :, :-1, :], xp)
    clip_k   = _clip(qh_a, Qout_min[:, :, 1:,  :], Qout_max[:, :, 1:,  :], xp)
    qh_new = infL_a * clip_km1 + (ONE - infL_a) * clip_k

    q_h = q_h.at[:, :, ka, :].set(qh_new)
    return q_h
