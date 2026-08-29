"""
Validation harness for kernels/tracervertflux.py.

Checks that the pure backend-switchable kernel (compute_tracer_vert_flux)
reproduces, bit-for-bit, the tracer-independent part of the 1st vertical
fractional step in mod_src_tracer.py (L106-353: flx_v, ck, d, rhog and their
pole counterparts), for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_tracervertflux_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.tracervertflux import (  # noqa: E402
    TracerVertFluxCfg, compute_tracer_vert_flux,
)


# ---------------------------------------------------------------------------
# Reference: faithful transcription of mod_src_tracer.py L106-353 (regional +
# pole). Writes into zero buffers; rows the original leaves stale (CONST_UNDEF)
# are never read downstream, so zeros is the downstream-equivalent reference.
# ---------------------------------------------------------------------------
def ref(rhogvx_mean, rhogvy_mean, rhogvz_mean, rhogw_mean,
        rhogvx_mean_pl, rhogvy_mean_pl, rhogvz_mean_pl, rhogw_mean_pl,
        rhog_in, rhog_in_pl, frhog, frhog_pl,
        C2WfactGz, RGAMH, RGSQRTH, C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl,
        rdgz, dt, b1, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    rdt = rhog_in.dtype.type
    half = rdt(0.5)
    gall_pl, kall_pl, lall_pl = rhog_in_pl.shape

    flx_v = np.zeros_like(rhog_in)
    ck = np.zeros(rhog_in.shape + (2,), dtype=rhog_in.dtype)
    rhog = np.zeros_like(rhog_in)

    kslice = slice(kmin + 1, kmax + 1)
    kslice_m1 = slice(kmin, kmax)

    flx_v[:, :, kslice, :] = (
        (
            C2WfactGz[:, :, kslice, :, 0] * rhogvx_mean[:, :, kslice, :] +
            C2WfactGz[:, :, kslice, :, 1] * rhogvx_mean[:, :, kslice_m1, :] +
            C2WfactGz[:, :, kslice, :, 2] * rhogvy_mean[:, :, kslice, :] +
            C2WfactGz[:, :, kslice, :, 3] * rhogvy_mean[:, :, kslice_m1, :] +
            C2WfactGz[:, :, kslice, :, 4] * rhogvz_mean[:, :, kslice, :] +
            C2WfactGz[:, :, kslice, :, 5] * rhogvz_mean[:, :, kslice_m1, :]
        ) * RGAMH[:, :, kslice, :] +
        rhogw_mean[:, :, kslice, :] * RGSQRTH[:, :, kslice, :]
    ) * half * dt
    flx_v[:, :, kmin, :] = rdt(0.0)
    flx_v[:, :, kmax + 1, :] = rdt(0.0)

    d = b1 * frhog / rhog_in * dt

    ck[:, :, kmin:kmax + 1, :, 0] = -flx_v[:, :, kmin:kmax + 1, :] / rhog_in[:, :, kmin:kmax + 1, :] * rdgz[kmin:kmax + 1][None, None, :, None]
    ck[:, :, kmin:kmax + 1, :, 1] = flx_v[:, :, kmin + 1:kmax + 2, :] / rhog_in[:, :, kmin:kmax + 1, :] * rdgz[kmin:kmax + 1][None, None, :, None]
    ck[:, :, kmin - 1, :, 0] = rdt(0.0)
    ck[:, :, kmin - 1, :, 1] = rdt(0.0)
    ck[:, :, kmax + 1, :, 0] = rdt(0.0)
    ck[:, :, kmax + 1, :, 1] = rdt(0.0)

    for k in range(kmin, kmax + 1):
        rhog[:, :, k, :] = (
            rhog_in[:, :, k, :]
            - (flx_v[:, :, k + 1, :] - flx_v[:, :, k, :]) * rdgz[k]
            + b1 * frhog[:, :, k, :] * dt
        )
    rhog[:, :, kmin - 1, :] = rhog_in[:, :, kmin, :]
    rhog[:, :, kmax + 1, :] = rhog_in[:, :, kmax, :]

    flx_v_pl = np.zeros_like(rhog_in_pl)
    ck_pl = np.zeros(rhog_in_pl.shape + (2,), dtype=rhog_in_pl.dtype)
    d_pl = np.zeros_like(rhog_in_pl)
    rhog_pl = np.zeros_like(rhog_in_pl)
    if cfg.have_pl:
        for l in range(lall_pl):
            for k in range(kmin + 1, kmax + 1):
                for g in range(gall_pl):
                    flx_v_pl[g, k, l] = (
                        (
                            C2WfactGz_pl[g, k, l, 0] * rhogvx_mean_pl[g, k, l] +
                            C2WfactGz_pl[g, k, l, 1] * rhogvx_mean_pl[g, k - 1, l] +
                            C2WfactGz_pl[g, k, l, 2] * rhogvy_mean_pl[g, k, l] +
                            C2WfactGz_pl[g, k, l, 3] * rhogvy_mean_pl[g, k - 1, l] +
                            C2WfactGz_pl[g, k, l, 4] * rhogvz_mean_pl[g, k, l] +
                            C2WfactGz_pl[g, k, l, 5] * rhogvz_mean_pl[g, k - 1, l]
                        ) * RGAMH_pl[g, k, l] +
                        rhogw_mean_pl[g, k, l] * RGSQRTH_pl[g, k, l]
                    ) * half * dt
            flx_v_pl[:, kmin, l] = rdt(0.0)
            flx_v_pl[:, kmax + 1, l] = rdt(0.0)

            d_pl[:, :, l] = b1 * frhog_pl[:, :, l] / rhog_in_pl[:, :, l] * dt

            for k in range(kmin, kmax + 1):
                ck_pl[:, k, l, 0] = -flx_v_pl[:, k, l] / rhog_in_pl[:, k, l] * rdgz[k]
                ck_pl[:, k, l, 1] = flx_v_pl[:, k + 1, l] / rhog_in_pl[:, k, l] * rdgz[k]
            ck_pl[:, kmin - 1, l, 0] = rdt(0.0)
            ck_pl[:, kmin - 1, l, 1] = rdt(0.0)
            ck_pl[:, kmax + 1, l, 0] = rdt(0.0)
            ck_pl[:, kmax + 1, l, 1] = rdt(0.0)

        for k in range(kmin, kmax + 1):
            rhog_pl[:, k, :] = (
                rhog_in_pl[:, k, :]
                - (flx_v_pl[:, k + 1, :] - flx_v_pl[:, k, :]) * rdgz[k]
                + b1 * frhog_pl[:, k, :] * dt
            )
        rhog_pl[:, kmin - 1, :] = rhog_in_pl[:, kmin, :]
        rhog_pl[:, kmax + 1, :] = rhog_in_pl[:, kmax, :]

    return flx_v, ck, d, rhog, flx_v_pl, ck_pl, d_pl, rhog_pl


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    i = j = 8
    kall, lall = 10, 3
    g, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    def R(*s):
        return rng.uniform(0.5, 1.5, s).astype(dt)

    arrs = dict(
        rhogvx_mean=R(i, j, kall, lall), rhogvy_mean=R(i, j, kall, lall),
        rhogvz_mean=R(i, j, kall, lall), rhogw_mean=R(i, j, kall, lall),
        rhogvx_mean_pl=R(g, kall, lall_pl), rhogvy_mean_pl=R(g, kall, lall_pl),
        rhogvz_mean_pl=R(g, kall, lall_pl), rhogw_mean_pl=R(g, kall, lall_pl),
        rhog_in=R(i, j, kall, lall), rhog_in_pl=R(g, kall, lall_pl),
        frhog=R(i, j, kall, lall), frhog_pl=R(g, kall, lall_pl),
        C2WfactGz=R(i, j, kall, lall, 6), RGAMH=R(i, j, kall, lall), RGSQRTH=R(i, j, kall, lall),
        C2WfactGz_pl=R(g, kall, lall_pl, 6), RGAMH_pl=R(g, kall, lall_pl), RGSQRTH_pl=R(g, kall, lall_pl),
        rdgz=R(kall),
    )
    # b1 defaults to 0.0 in the model; test with a nonzero value to exercise the
    # frhog/d/rhog terms too.
    scal = dict(dt=dt(20.0), b1=dt(0.3))
    cfg = TracerVertFluxCfg(kmin=kmin, kmax=kmax, have_pl=True)
    return arrs, scal, cfg


ARG_ORDER = [
    "rhogvx_mean", "rhogvy_mean", "rhogvz_mean", "rhogw_mean",
    "rhogvx_mean_pl", "rhogvy_mean_pl", "rhogvz_mean_pl", "rhogw_mean_pl",
    "rhog_in", "rhog_in_pl", "frhog", "frhog_pl",
    "C2WfactGz", "RGAMH", "RGSQRTH", "C2WfactGz_pl", "RGAMH_pl", "RGSQRTH_pl",
    "rdgz",
]
OUT_NAMES = ["flx_v", "ck", "d", "rhog", "flx_v_pl", "ck_pl", "d_pl", "rhog_pl"]


def report(name, r, g, rtol=0.0, atol=0.0):
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, a, b in zip(OUT_NAMES, r, g):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            print(f"  {nm:9s}: BAD shape {b.shape} != {a.shape}")
            ok = False
            continue
        denom = np.maximum(np.abs(a), 1e-300)
        max_abs = np.max(np.abs(b - a))
        max_rel = np.max(np.abs(b - a) / denom)
        exact = np.array_equal(b, a)
        passed = exact or np.allclose(b, a, rtol=rtol, atol=atol)
        ok = ok and passed
        flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
        print(f"  {nm:9s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    arrs, scal, cfg = make_inputs()
    r = ref(*[arrs[k] for k in ARG_ORDER], scal["dt"], scal["b1"], cfg)

    import numpy as xnp
    got_np = compute_tracer_vert_flux(
        *[arrs[k] for k in ARG_ORDER], scal["dt"], scal["b1"], cfg=cfg, xp=xnp)
    ok1 = report("numpy backend (eager)", r, got_np)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jargs = [jnp.asarray(arrs[k]) for k in ARG_ORDER]

    got_jnp = compute_tracer_vert_flux(
        *jargs, scal["dt"], scal["b1"], cfg=cfg, xp=jnp)
    ok2 = report("jax.numpy backend (eager)", r, got_jnp)

    kernel_jit = jax.jit(compute_tracer_vert_flux, static_argnames=("cfg", "xp"))
    got_jit = kernel_jit(*jargs, scal["dt"], scal["b1"], cfg=cfg, xp=jnp)
    jax.block_until_ready(got_jit)
    ok3 = report("jax.jit (compiled)", r, got_jit, rtol=1e-12, atol=1e-12)

    print("\n========================================")
    allok = ok1 and ok2 and ok3
    print(f"all checks: {'PASS' if allok else 'FAIL'}")
    print("========================================")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
