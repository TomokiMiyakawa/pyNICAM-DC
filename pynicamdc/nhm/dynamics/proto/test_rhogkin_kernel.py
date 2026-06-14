"""
Validation harness for kernels/rhogkin.py.

Checks that the pure backend-switchable kernels reproduce, bit-for-bit, the
COMM-free cnvvar_rhogkin total kinetic-energy assembly (mod_cnvvar.py), for the
full (i,j,kall,l) / (g,kall,l) output arrays, under:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_rhogkin_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.rhogkin import (
    RhogkinCfg, compute_rhogkin_reg, compute_rhogkin_pl,
)


# ---------------------------------------------------------------------------
# References: exact transcription of mod_cnvvar.cnvvar_rhogkin (full arrays).
# ---------------------------------------------------------------------------
def rhogkin_reg_ref(rhog, rhogvx, rhogvy, rhogvz, rhogw,
                    C2Wfact, W2Cfact, cfg, UNDEF):
    kmin, kmax = cfg.kmin, cfg.kmax
    rdtype = np.float64
    rhogkin   = np.full_like(rhog, UNDEF)
    rhogkin_h = np.full_like(rhog, UNDEF)
    rhogkin_v = np.full_like(rhog, UNDEF)

    rhogkin_h[:, :, kmin:kmax+1, :] = rdtype(0.5) * (
        rhogvx[:, :, kmin:kmax+1, :] ** 2 +
        rhogvy[:, :, kmin:kmax+1, :] ** 2 +
        rhogvz[:, :, kmin:kmax+1, :] ** 2
    ) / rhog[:, :, kmin:kmax+1, :]

    denom = (
        C2Wfact[:, :, kmin+1:kmax+1, :, 0] * rhog[:, :, kmin+1:kmax+1, :] +
        C2Wfact[:, :, kmin+1:kmax+1, :, 1] * rhog[:, :, kmin:kmax, :]
    )
    rhogkin_v[:, :, kmin+1:kmax+1, :] = rdtype(0.5) * rhogw[:, :, kmin+1:kmax+1, :] ** 2 / denom
    rhogkin_v[:, :, kmin, :] = rdtype(0.0)
    rhogkin_v[:, :, kmax+1, :] = rdtype(0.0)

    rhogkin[:, :, kmin:kmax+1, :] = (
        rhogkin_h[:, :, kmin:kmax+1, :] +
        W2Cfact[:, :, kmin:kmax+1, :, 0] * rhogkin_v[:, :, kmin+1:kmax+2, :] +
        W2Cfact[:, :, kmin:kmax+1, :, 1] * rhogkin_v[:, :, kmin:kmax+1, :]
    )
    rhogkin[:, :, kmin-1, :] = rdtype(0.0)
    rhogkin[:, :, kmax+1, :] = rdtype(0.0)
    return rhogkin


def rhogkin_pl_ref(rhog, rhogvx, rhogvy, rhogvz, rhogw,
                   C2Wfact, W2Cfact, cfg, UNDEF):
    kmin, kmax = cfg.kmin, cfg.kmax
    rdtype = np.float64
    rhogkin   = np.full_like(rhog, UNDEF)
    rhogkin_h = np.full_like(rhog, UNDEF)
    rhogkin_v = np.full_like(rhog, UNDEF)

    rhogkin_h[:, kmin:kmax+1, :] = rdtype(0.5) * (
        rhogvx[:, kmin:kmax+1, :] ** 2 +
        rhogvy[:, kmin:kmax+1, :] ** 2 +
        rhogvz[:, kmin:kmax+1, :] ** 2
    ) / rhog[:, kmin:kmax+1, :]

    denom = (
        C2Wfact[:, kmin+1:kmax+1, :, 0] * rhog[:, kmin+1:kmax+1, :] +
        C2Wfact[:, kmin+1:kmax+1, :, 1] * rhog[:, kmin:kmax, :]
    )
    rhogkin_v[:, kmin+1:kmax+1, :] = rdtype(0.5) * rhogw[:, kmin+1:kmax+1, :] ** 2 / denom
    rhogkin_v[:, kmin, :] = rdtype(0.0)
    rhogkin_v[:, kmax+1, :] = rdtype(0.0)

    rhogkin[:, kmin:kmax+1, :] = (
        rhogkin_h[:, kmin:kmax+1, :] +
        W2Cfact[:, kmin:kmax+1, :, 0] * rhogkin_v[:, kmin+1:kmax+2, :] +
        W2Cfact[:, kmin:kmax+1, :, 1] * rhogkin_v[:, kmin:kmax+1, :]
    )
    rhogkin[:, kmin-1, :] = rdtype(0.0)
    rhogkin[:, kmax+1, :] = rdtype(0.0)
    return rhogkin


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 10, 3
    gall_pl, lall_pl = 6, 2
    kmin, kmax = 1, kall - 2
    UNDEF = -999.9

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    data = dict(
        rhog=R(iall, jall, kall, lall),
        rhogvx=R(iall, jall, kall, lall), rhogvy=R(iall, jall, kall, lall),
        rhogvz=R(iall, jall, kall, lall), rhogw=R(iall, jall, kall, lall),
        C2Wfact=R(iall, jall, kall, lall, 2), W2Cfact=R(iall, jall, kall, lall, 2),
        rhog_pl=R(gall_pl, kall, lall_pl),
        rhogvx_pl=R(gall_pl, kall, lall_pl), rhogvy_pl=R(gall_pl, kall, lall_pl),
        rhogvz_pl=R(gall_pl, kall, lall_pl), rhogw_pl=R(gall_pl, kall, lall_pl),
        C2Wfact_pl=R(gall_pl, kall, lall_pl, 2), W2Cfact_pl=R(gall_pl, kall, lall_pl, 2),
    )
    cfg = RhogkinCfg(kmin=kmin, kmax=kmax, have_pl=True, UNDEF=UNDEF)
    return data, cfg, UNDEF


def report(name, ref, got, rtol=0.0, atol=0.0):
    got = np.asarray(got)
    denom = np.maximum(np.abs(ref), 1e-300)
    # ignore UNDEF rows when measuring (they should match exactly anyway)
    max_abs = np.max(np.abs(got - ref))
    max_rel = np.max(np.abs(got - ref) / denom)
    exact = np.array_equal(got, ref)
    passed = exact or np.allclose(got, ref, rtol=rtol, atol=atol)
    flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
    print(f"  [{name}]: {'OK ' if passed else 'BAD'} {flag}")
    return passed


def main():
    d, cfg, UNDEF = make_inputs()

    ref_r = rhogkin_reg_ref(d["rhog"], d["rhogvx"], d["rhogvy"], d["rhogvz"], d["rhogw"],
                            d["C2Wfact"], d["W2Cfact"], cfg, UNDEF)
    ref_p = rhogkin_pl_ref(d["rhog_pl"], d["rhogvx_pl"], d["rhogvy_pl"], d["rhogvz_pl"], d["rhogw_pl"],
                           d["C2Wfact_pl"], d["W2Cfact_pl"], cfg, UNDEF)

    results = []

    def run_backend(xp, label, rtol=0.0, atol=0.0, conv=lambda a: a):
        c = {k: (conv(v) if hasattr(v, "shape") else v) for k, v in d.items()}
        r = compute_rhogkin_reg(c["rhog"], c["rhogvx"], c["rhogvy"], c["rhogvz"], c["rhogw"],
                                c["C2Wfact"], c["W2Cfact"], cfg, xp)
        p = compute_rhogkin_pl(c["rhog_pl"], c["rhogvx_pl"], c["rhogvy_pl"], c["rhogvz_pl"], c["rhogw_pl"],
                               c["C2Wfact_pl"], c["W2Cfact_pl"], cfg, xp)
        print(f"\n[{label}]")
        results.append(report("reg", ref_r, r, rtol, atol))
        results.append(report("pl", ref_p, p, rtol, atol))

    import numpy as xnp
    run_backend(xnp, "numpy (eager)")

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    run_backend(jnp, "jax.numpy (eager)", conv=lambda a: jnp.asarray(a))

    f_r = jax.jit(compute_rhogkin_reg, static_argnames=("cfg", "xp"))
    f_p = jax.jit(compute_rhogkin_pl, static_argnames=("cfg", "xp"))
    j = lambda a: jnp.asarray(a)
    r = f_r(j(d["rhog"]), j(d["rhogvx"]), j(d["rhogvy"]), j(d["rhogvz"]), j(d["rhogw"]),
            j(d["C2Wfact"]), j(d["W2Cfact"]), cfg=cfg, xp=jnp)
    p = f_p(j(d["rhog_pl"]), j(d["rhogvx_pl"]), j(d["rhogvy_pl"]), j(d["rhogvz_pl"]), j(d["rhogw_pl"]),
            j(d["C2Wfact_pl"]), j(d["W2Cfact_pl"]), cfg=cfg, xp=jnp)
    jax.block_until_ready((r, p))
    print("\n[jax.jit]")
    results.append(report("reg", ref_r, r, 1e-11, 1e-11))
    results.append(report("pl", ref_p, p, 1e-11, 1e-11))

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
