"""
Validation harness for kernels/advconv.py.

Checks that the pure backend-switchable kernel (compute_scaled_fluxes)
reproduces, bit-for-bit, the original in-place scaling block of
src_advection_convergence (mod_src.py L469-522) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Both fluxtype branches are exercised:
    I_SRC_default    (=3): weighted half-level w-flux
    I_SRC_horizontal (=1): zero w-flux

Run:
    .../envs/jax_nomtl_mpi/bin/python test_advconv_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.advconv import AdvConvCfg, compute_scaled_fluxes

I_SRC_HORIZONTAL = 1
I_SRC_DEFAULT = 3


# ---------------------------------------------------------------------------
# Reference: exact transcription of mod_src.py scaling block (in-place, numpy).
# Writes into zeros buffers; original leaves rows outside the computed interior
# as stale buffer contents that are never read downstream -> zeros is the
# downstream-equivalent reference.
# ---------------------------------------------------------------------------
def compute_scaled_fluxes_ref(rhogvx, rhogvy, rhogvz, rhogw, scl,
                              rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl,
                              afact_v, bfact_v, fluxtype, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1 = kmin - 1
    kmaxp1 = kmax + 1
    kmaxp2 = kmax + 2

    vxscl = rhogvx * scl
    vyscl = rhogvy * scl
    vzscl = rhogvz * scl

    wscl = np.zeros_like(rhogw)
    if fluxtype == cfg.I_SRC_default:
        afact = afact_v[kmin:kmaxp2][None, None, :, None]
        bfact = bfact_v[kmin:kmaxp2][None, None, :, None]
        weighted = afact * scl[:, :, kmin:kmaxp2, :] + bfact * scl[:, :, kminm1:kmaxp1, :]
        wscl[:, :, kmin:kmaxp2, :] = rhogw[:, :, kmin:kmaxp2, :] * weighted
        wscl[:, :, kminm1, :] = 0.0
    # else I_SRC_horizontal: wscl stays zero

    vxscl_pl = rhogvx_pl * scl_pl
    vyscl_pl = rhogvy_pl * scl_pl
    vzscl_pl = rhogvz_pl * scl_pl
    wscl_pl = np.zeros_like(rhogw_pl)
    if cfg.have_pl and fluxtype == cfg.I_SRC_default:
        afact = afact_v[kmin:kmaxp2][None, :, None]
        bfact = bfact_v[kmin:kmaxp2][None, :, None]
        weighted_pl = afact * scl_pl[:, kmin:kmaxp2, :] + bfact * scl_pl[:, kminm1:kmaxp1, :]
        wscl_pl[:, kmin:kmaxp2, :] = rhogw_pl[:, kmin:kmaxp2, :] * weighted_pl
        wscl_pl[:, kminm1, :] = 0.0

    return vxscl, vyscl, vzscl, wscl, vxscl_pl, vyscl_pl, vzscl_pl, wscl_pl


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 5, 4, 10, 3
    gall_pl, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(dt)

    rhogvx = R(iall, jall, kall, lall)
    rhogvy = R(iall, jall, kall, lall)
    rhogvz = R(iall, jall, kall, lall)
    rhogw = R(iall, jall, kall, lall)
    scl = R(iall, jall, kall, lall)
    rhogvx_pl = R(gall_pl, kall, lall_pl)
    rhogvy_pl = R(gall_pl, kall, lall_pl)
    rhogvz_pl = R(gall_pl, kall, lall_pl)
    rhogw_pl = R(gall_pl, kall, lall_pl)
    scl_pl = R(gall_pl, kall, lall_pl)

    afact = rng.uniform(0.3, 0.7, (kall,)).astype(dt)
    bfact = (1.0 - afact).astype(dt)

    cfg = AdvConvCfg(kmin=kmin, kmax=kmax, have_pl=True, I_SRC_default=I_SRC_DEFAULT)
    return (rhogvx, rhogvy, rhogvz, rhogw, scl,
            rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl,
            afact, bfact, cfg)


def report(name, ref, got, rtol=0.0, atol=0.0):
    names = ["vxscl", "vyscl", "vzscl", "wscl",
             "vxscl_pl", "vyscl_pl", "vzscl_pl", "wscl_pl"]
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, r, g in zip(names, ref, got):
        g = np.asarray(g)
        denom = np.maximum(np.abs(r), 1e-300)
        max_abs = np.max(np.abs(g - r))
        max_rel = np.max(np.abs(g - r) / denom)
        exact = np.array_equal(g, r)
        passed = exact or np.allclose(g, r, rtol=rtol, atol=atol)
        ok = ok and passed
        flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
        print(f"  {nm:9s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def run_for_fluxtype(fluxtype, label):
    inp = make_inputs()
    *arrays, afact, bfact, cfg = inp
    (rhogvx, rhogvy, rhogvz, rhogw, scl,
     rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl) = arrays

    ref = compute_scaled_fluxes_ref(rhogvx, rhogvy, rhogvz, rhogw, scl,
                                    rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl,
                                    afact, bfact, fluxtype, cfg)

    # (1) numpy backend
    import numpy as xnp
    got_np = compute_scaled_fluxes(
        rhogvx, rhogvy, rhogvz, rhogw, scl,
        rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl,
        afact, bfact, fluxtype, cfg, xnp)
    ok1 = report(f"{label}: numpy backend (eager)", ref, got_np)

    # (2) + (3) jax backend
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    j = lambda a: jnp.asarray(a)
    jargs = (j(rhogvx), j(rhogvy), j(rhogvz), j(rhogw), j(scl),
             j(rhogvx_pl), j(rhogvy_pl), j(rhogvz_pl), j(rhogw_pl), j(scl_pl),
             j(afact), j(bfact))

    got_jnp = compute_scaled_fluxes(*jargs, fluxtype, cfg, jnp)
    ok2 = report(f"{label}: jax.numpy backend (eager)", ref, got_jnp)

    kernel_jit = jax.jit(compute_scaled_fluxes, static_argnames=("fluxtype", "cfg", "xp"))
    got_jit = kernel_jit(*jargs, fluxtype=fluxtype, cfg=cfg, xp=jnp)
    jax.block_until_ready(got_jit)
    ok3 = report(f"{label}: jax.jit (compiled)", ref, got_jit, rtol=1e-12, atol=1e-12)

    return ok1, ok2, ok3


def main():
    results = []
    results += run_for_fluxtype(I_SRC_DEFAULT, "I_SRC_default")
    results += run_for_fluxtype(I_SRC_HORIZONTAL, "I_SRC_horizontal")

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
