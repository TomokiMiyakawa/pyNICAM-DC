#
# terminator -- DCMIP "toy" chemistry (Terminator) -- numpy/jax port
#
# Ported from: nicamdc/src/nhm/share/dcmip/Terminator.f90
#   k_vals, tendency_Terminator, initial_value_Terminator
#
# A self-contained reversible two-species reaction  Cl2 <-> 2 Cl  with a
# photolysis rate k1 that depends on (lat,lon) and a constant recombination
# rate k2. Designed so that the "family" cly = cl + 2*cl2 is EXACTLY conserved
# (cl2_f == -cl_f/2 by construction) and the pair relaxes to a lat/lon-dependent
# equilibrium; the analytic time-integration below is unconditionally stable.
#
# Elementwise + xp-agnostic: every function is a pure function of its array args
# and works under numpy OR jax (broadcasting handles the (Ncol,1) lat/lon vs
# (Ncol,nk) cl/cl2 shapes), so it composes inside the resident/jit device forcing
# path (_get_dcmip_jit) with no host ops.
#
# Precision is NOT hardcoded: the working precision arrives as the callable `rdtype`
# (the bk.ndtype contract -- np.float32/np.float64) and every constant/literal is cast
# to it, so the reaction runs in the field precision on both backends (no silent numpy
# float64 promotion, no jax weak-type surprises). The Fortran computes the chemistry in real(8) regardless
# of RP; a float32 run that wants that faithfulness upcasts cl/cl2/dt to float64
# at the call site (the module then follows those float64 inputs) -- the choice
# lives with the caller, not baked in here.
#
import numpy as np

# DCMIP Terminator defining constants (dimensionless magic numbers from
# Terminator.f90). Cast to the working dtype at use -- never used raw in arithmetic.
CLY_CONSTANT_VAL = 4.0e-6
K1_LAT_CENTER_DEG = 20.0
K1_LON_CENTER_DEG = 300.0


def _consts(rdtype):
    """Terminator constants materialized in the working dtype. `rdtype` is a CALLABLE working-
    precision scalar type (np.float32/np.float64) per the bk.ndtype contract -- so constants land
    in the field precision on either backend."""
    d2r = rdtype(np.pi / 180.0)
    return {
        "d2r": d2r,
        "cly": rdtype(CLY_CONSTANT_VAL),
        "lat_c": rdtype(K1_LAT_CENTER_DEG) * d2r,
        "lon_c": rdtype(K1_LON_CENTER_DEG) * d2r,
        "eps": rdtype(1.0e-16),
        "zero": rdtype(0.0), "one": rdtype(1.0), "two": rdtype(2.0), "four": rdtype(4.0),
    }


def k_vals(lat_rad, lon_rad, rdtype, xp=np, _c=None):
    """Solar photolysis rate k1 and recombination rate k2 (Terminator.f90 k_vals).
    lat_rad, lon_rad in RADIANS. k2 is the constant 1; k1 is the (clipped) solar
    zenith projection toward the (k1_lat_center, k1_lon_center) sub-solar point."""
    c = _c if _c is not None else _consts(rdtype)
    k1 = xp.maximum(c["zero"],
                    xp.sin(lat_rad) * xp.sin(c["lat_c"])
                    + xp.cos(lat_rad) * xp.cos(c["lat_c"]) * xp.cos(lon_rad - c["lon_c"]))
    k2 = c["one"]
    return k1, k2


def tendency_terminator(lat_deg, lon_deg, cl, cl2, dt, rdtype, xp=np):
    """Time rate of change of cl and cl2 (Terminator.f90 tendency_Terminator).
    lat_deg, lon_deg in DEGREES (matching the Fortran interface; the af_dcmip
    caller passes lat/d2r). cl, cl2 are molar mixing ratios. Returns (cl_f, cl2_f),
    per-second tendencies with cl2_f == -cl_f/2 (cly exactly conserved)."""
    c = _consts(rdtype)
    dt = rdtype(dt)
    k1, k2 = k_vals(lat_deg * c["d2r"], lon_deg * c["d2r"], rdtype, xp, _c=c)
    r   = k1 / (c["four"] * k2)
    cly = cl + c["two"] * cl2
    det = xp.sqrt(r * r + c["two"] * r * cly)
    expdt = xp.exp(-c["four"] * k2 * det * dt)
    # el = (1 - expdt)/(det*dt) where |det*k2*dt| > eps, else the det->0 limit 4*k2.
    # Guard the divisor so the unused (det~0) branch does not create NaN under xp.where
    # (both branches are evaluated eagerly).
    mask = xp.abs(det * k2 * dt) > c["eps"]
    det_safe = xp.where(mask, det, xp.ones_like(det))
    el = xp.where(mask, (c["one"] - expdt) / det_safe / dt, xp.full_like(det, c["four"] * k2))
    cl_f  = -el * (cl - det + r) * (cl + det + r) / (c["one"] + expdt + dt * el * (cl + r))
    cl2_f = -cl_f / c["two"]
    return cl_f, cl2_f


def initial_value_terminator(lat_deg, lon_deg, rdtype, xp=np):
    """Equilibrium initial cl, cl2 for cly_constant total (Terminator.f90
    initial_value_Terminator). lat_deg, lon_deg in DEGREES."""
    c = _consts(rdtype)
    k1, k2 = k_vals(lat_deg * c["d2r"], lon_deg * c["d2r"], rdtype, xp, _c=c)
    r   = k1 / (c["four"] * k2)
    det = xp.sqrt(r * r + c["two"] * c["cly"] * r)
    cl  = det - r
    cl2 = c["cly"] / c["two"] - (det - r) / c["two"]
    return cl, cl2
