"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the whole
COMM-free body of src_pres_gradient (mod_src.py L604-803).

It reproduces, as a single pure function, the four [OUT] arrays
(Pgrad, Pgradw, Pgrad_pl, Pgradw_pl) from P / P_pl:

  1. P_vm = P * RGAM                                    (full k)
  2. horizontal gradient, horizontal contribution      OPRT_gradient(P_vm)
     (the COMM-free 7-point vector-gradient stencil is inlined here)
  3. horizontal gradient, vertical contribution        P_vmh from C2WfactGz
     Pgrad[k] += (P_vmh[k+1]-P_vmh[k]) * rdgz[k]  on k = kmin .. kmax
     boundary: zero kmin-1 / kmax+1; first_layer_remedy -> Pgrad[kmin]=Pgrad[kmin+1]
  4. horizontalize                                       OPRT_horizontalize_vec
     (tangent-plane projection, inlined; interior i/j only, border preserved)
  5. vertical gradient (half level)                      Pgradw
     gradtype==I_SRC_default : GAM2H*(P*RGSGAM2 - P[k-1]*RGSGAM2[k-1])*rdgzh
     gradtype==I_SRC_horizontal : zero

Both OPRT_gradient and OPRT_horizontalize_vec math are inlined (not delegated)
so the entire block is one backend-switchable, jit-able function. The original
code is unchanged elsewhere; this duplicates only COMM-free arithmetic.

Boundary / coverage note
------------------------
The original writes the persistent [OUT] buffers only on the computed interior
k-rows, leaving the rest as stale (never-read) buffer contents. This functional
version reconstructs every row explicitly; for the standard layout
kmin == 1, kmax == kall-2 there are no rows above kmax+1, so the explicit
zero-fill of out-of-interior rows is downstream-identical.

Summation order
---------------
The pole OPRT_gradient loop uses a SEPARATE accumulator per direction, so a
per-direction reduction (`.sum(axis=0)` independently for X/Y/Z) is bit-exact.
The regional 7-point stencil uses np.sum over the stacked stencil axis, matching
the original exactly.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PresGradCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    XDIR: int
    YDIR: int
    ZDIR: int
    gslf_pl: int
    gmax_pl: int
    nxyz: int
    first_layer_remedy: bool
    rscale: float
    plmask: int
    horizontalize: bool          # grid_type != on_plane
    I_SRC_default: int
    I_SRC_horizontal: int


def _gradient_interior(P_vm, coef_grad, cfg, xp):
    """Inlined COMM-free regional 7-point vector gradient (OPRT_gradient).

    P_vm      : (i, j, k, l)
    coef_grad : (i, j, 1, l, 3, 7)   (KNONE k-dim broadcasts over k)
    returns   : (i, j, k, l, 3)  zero on the i/j border rows.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    i = P_vm.shape[0]
    j = P_vm.shape[1]

    c = coef_grad[1:i - 1, 1:j - 1]            # (i-2, j-2, 1, l, 3, 7)

    # 7-point stencil stack (same point ordering as OPRT_gradient)
    scl_stack = xp.stack([
        P_vm[1:i - 1, 1:j - 1, :, :],
        P_vm[2:i,     1:j - 1, :, :],
        P_vm[2:i,     2:j,     :, :],
        P_vm[1:i - 1, 2:j,     :, :],
        P_vm[0:i - 2, 1:j - 1, :, :],
        P_vm[0:i - 2, 0:j - 2, :, :],
        P_vm[1:i - 1, 0:j - 2, :, :],
    ], axis=4)                                 # (i-2, j-2, k, l, 7)

    def comp(d):
        return xp.sum(c[:, :, :, :, d, :] * scl_stack, axis=4)  # (i-2, j-2, k, l)

    gx = comp(X)
    gy = comp(Y)
    gz = comp(Z)
    interior = xp.stack([gx, gy, gz], axis=4)  # (i-2, j-2, k, l, 3)

    # pad i/j border with zeros (k, l, dir axes untouched)
    return xp.pad(interior, ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0)))


def _gradient_pl_row(P_vm_pl, coef_grad_pl, cfg, xp):
    """Inlined COMM-free pole vector gradient (OPRT_gradient pole loop).

    P_vm_pl      : (g, k, l)
    coef_grad_pl : (g, 1, l, 3)
    returns      : (g, k, l, 3)  only the self row (gslf_pl) is non-zero.
    Per-direction accumulation (bit-exact with the original separate accumulators).
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    n = cfg.gslf_pl
    v0, v1 = cfg.gslf_pl, cfg.gmax_pl + 1
    g = P_vm_pl.shape[0]

    def comp(d):
        cd = coef_grad_pl[v0:v1, 0, :, d][:, None, :]   # (nv, 1, l)
        per_v = cd * P_vm_pl[v0:v1, :, :]               # (nv, k, l)
        return per_v.sum(axis=0)                        # (k, l)

    row = xp.stack([comp(X), comp(Y), comp(Z)], axis=2)  # (k, l, 3)

    rowk = row[None, :, :, :]                            # (1, k, l, 3)
    above = xp.zeros((n,) + row.shape, dtype=row.dtype)
    below = xp.zeros((g - n - 1,) + row.shape, dtype=row.dtype)
    return xp.concatenate([above, rowk, below], axis=0)  # (g, k, l, 3)


def compute_pres_gradient(
    P, P_pl,
    RGAM, RGAMH, C2WfactGz, coef_grad, GRD_x, rdgz, rdgzh, GAM2H, RGSGAM2,
    RGAM_pl, RGAMH_pl, C2WfactGz_pl, coef_grad_pl, GRD_x_pl, GAM2H_pl, RGSGAM2_pl,
    gradtype, cfg: PresGradCfg, xp,
):
    """Pure version of mod_src.py src_pres_gradient (whole COMM-free body).

    Returns (Pgrad, Pgradw, Pgrad_pl, Pgradw_pl). The *_pl arrays are
    shape-correct placeholders (zeros) when not have_pl.
    """
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1 = kmin - 1
    kminp1 = kmin + 1
    kmaxp1 = kmax + 1
    kmaxp2 = kmax + 2
    rscale = cfg.rscale

    i, j, kall, l = P.shape

    # =====================================================================
    # regional
    # =====================================================================
    P_vm = P * RGAM

    # --- horizontal contribution: 7-point gradient stencil (full k) ---
    grad_h = _gradient_interior(P_vm, coef_grad, cfg, xp)   # (i,j,k,l,3)

    # --- vertical contribution: P_vmh on k = kmin .. kmax+1 ---
    cz = C2WfactGz[:, :, kmin:kmaxp2, :, :]                 # (i,j,nk,l,6)
    rgh = RGAMH[:, :, kmin:kmaxp2, :]                       # (i,j,nk,l)
    Pk = P[:, :, kmin:kmaxp2, :]
    Pkm1 = P[:, :, kminm1:kmaxp1, :]
    pvmh_x = (cz[:, :, :, :, 0] * Pk + cz[:, :, :, :, 1] * Pkm1) * rgh
    pvmh_y = (cz[:, :, :, :, 2] * Pk + cz[:, :, :, :, 3] * Pkm1) * rgh
    pvmh_z = (cz[:, :, :, :, 4] * Pk + cz[:, :, :, :, 5] * Pkm1) * rgh
    P_vmh = xp.stack([pvmh_x, pvmh_y, pvmh_z], axis=4)      # (i,j,nk,l,3), k=kmin..kmax+1

    # vterm on k = kmin .. kmax : (P_vmh[k+1] - P_vmh[k]) * rdgz[k]
    rdgz_b = rdgz[kmin:kmaxp1][None, None, :, None, None]
    vterm = (P_vmh[:, :, 1:, :, :] - P_vmh[:, :, 0:-1, :, :]) * rdgz_b  # (i,j,kmax-kmin+1,l,3)

    # grad_full on k = kmin .. kmax
    gf = grad_h[:, :, kmin:kmaxp1, :, :] + vterm           # index 0->kmin .. ->kmax

    # remedy / boundary assembly along k
    zrow = xp.zeros((i, j, 1, l, cfg.nxyz), dtype=P.dtype)
    if cfg.first_layer_remedy:
        row_kmin = gf[:, :, 1:2, :, :]                     # = grad_full[kmin+1]
    else:
        row_kmin = gf[:, :, 0:1, :, :]                     # = grad_full[kmin]
    rows_mid = gf[:, :, 1:, :, :]                          # k = kmin+1 .. kmax

    parts = []
    if kminm1 > 0:
        parts.append(grad_h[:, :, 0:kminm1, :, :])
    parts.append(zrow)                                     # k = kmin-1
    parts.append(row_kmin)                                 # k = kmin
    parts.append(rows_mid)                                 # k = kmin+1 .. kmax
    parts.append(zrow)                                     # k = kmax+1
    if kmaxp2 < kall:
        parts.append(grad_h[:, :, kmaxp2:kall, :, :])
    Pgrad = xp.concatenate(parts, axis=2)                  # (i,j,kall,l,3)

    # --- horizontalize (interior i/j, all k; border preserved) ---
    if cfg.horizontalize:
        Pgrad = _horizontalize(Pgrad, GRD_x, cfg, xp)

    # --- vertical gradient (half level): Pgradw ---
    if gradtype == cfg.I_SRC_default:
        gh = GAM2H[:, :, kminp1:kmaxp1, :]
        pr = P[:, :, kminp1:kmaxp1, :] * RGSGAM2[:, :, kminp1:kmaxp1, :]
        prm1 = P[:, :, kmin:kmax, :] * RGSGAM2[:, :, kmin:kmax, :]
        rdh = rdgzh[kminp1:kmaxp1][None, None, :, None]
        pgw_int = gh * (pr - prm1) * rdh                   # k = kmin+1 .. kmax
        low = xp.zeros((i, j, kminp1, l), dtype=P.dtype)   # k = 0 .. kmin
        up = xp.zeros((i, j, kall - kmaxp1, l), dtype=P.dtype)  # k = kmax+1 .. end
        Pgradw = xp.concatenate([low, pgw_int, up], axis=2)
    else:
        Pgradw = xp.zeros((i, j, kall, l), dtype=P.dtype)

    # =====================================================================
    # pole
    # =====================================================================
    Pgrad_pl = xp.zeros(P_pl.shape + (cfg.nxyz,), dtype=P_pl.dtype)
    Pgradw_pl = xp.zeros_like(P_pl)
    if cfg.have_pl:
        g, kall_pl, l_pl = P_pl.shape
        P_vm_pl = P_pl * RGAM_pl * cfg.plmask

        grad_h_pl = _gradient_pl_row(P_vm_pl, coef_grad_pl, cfg, xp)  # (g,k,l,3)

        czp = C2WfactGz_pl[:, kmin:kmaxp2, :, :]
        rghp = RGAMH_pl[:, kmin:kmaxp2, :]
        Pkp = P_pl[:, kmin:kmaxp2, :]
        Pkm1p = P_pl[:, kminm1:kmaxp1, :]
        pvmh_xp = (czp[:, :, :, 0] * Pkp + czp[:, :, :, 1] * Pkm1p) * rghp
        pvmh_yp = (czp[:, :, :, 2] * Pkp + czp[:, :, :, 3] * Pkm1p) * rghp
        pvmh_zp = (czp[:, :, :, 4] * Pkp + czp[:, :, :, 5] * Pkm1p) * rghp
        P_vmh_pl = xp.stack([pvmh_xp, pvmh_yp, pvmh_zp], axis=3)  # (g,nk,l,3)

        rdgz_bp = rdgz[kmin:kmaxp1][None, :, None, None]
        vterm_pl = (P_vmh_pl[:, 1:, :, :] - P_vmh_pl[:, 0:-1, :, :]) * rdgz_bp
        gf_pl = grad_h_pl[:, kmin:kmaxp1, :, :] + vterm_pl

        zrow_pl = xp.zeros((g, 1, l_pl, cfg.nxyz), dtype=P_pl.dtype)
        if cfg.first_layer_remedy:
            row_kmin_pl = gf_pl[:, 1:2, :, :]
        else:
            row_kmin_pl = gf_pl[:, 0:1, :, :]
        rows_mid_pl = gf_pl[:, 1:, :, :]

        parts_pl = []
        if kminm1 > 0:
            parts_pl.append(grad_h_pl[:, 0:kminm1, :, :])
        parts_pl.append(zrow_pl)
        parts_pl.append(row_kmin_pl)
        parts_pl.append(rows_mid_pl)
        parts_pl.append(zrow_pl)
        if kmaxp2 < kall_pl:
            parts_pl.append(grad_h_pl[:, kmaxp2:kall_pl, :, :])
        Pgrad_pl = xp.concatenate(parts_pl, axis=1)

        if cfg.horizontalize:
            Pgrad_pl = _horizontalize_pl(Pgrad_pl, GRD_x_pl, cfg, xp)

        if gradtype == cfg.I_SRC_default:
            ghp = GAM2H_pl[:, kminp1:kmaxp1, :]
            prp = P_pl[:, kminp1:kmaxp1, :] * RGSGAM2_pl[:, kminp1:kmaxp1, :]
            prm1p = P_pl[:, kmin:kmax, :] * RGSGAM2_pl[:, kmin:kmax, :]
            rdhp = rdgzh[kminp1:kmaxp1][None, :, None]
            pgw_int_pl = ghp * (prp - prm1p) * rdhp
            low_pl = xp.zeros((g, kminp1, l_pl), dtype=P_pl.dtype)
            up_pl = xp.zeros((g, kall_pl - kmaxp1, l_pl), dtype=P_pl.dtype)
            Pgradw_pl = xp.concatenate([low_pl, pgw_int_pl, up_pl], axis=1)
        else:
            Pgradw_pl = xp.zeros_like(P_pl)

    return Pgrad, Pgradw, Pgrad_pl, Pgradw_pl


def _horizontalize(Pgrad, GRD_x, cfg, xp):
    """Inlined COMM-free regional tangent-plane projection (interior i/j only)."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    rscale = cfg.rscale
    i = Pgrad.shape[0]
    j = Pgrad.shape[1]

    vx = Pgrad[:, :, :, :, X]
    vy = Pgrad[:, :, :, :, Y]
    vz = Pgrad[:, :, :, :, Z]

    # direction vector on interior: (i-2, j-2, l) -> (i-2, j-2, 1, l)
    gv = GRD_x[1:i - 1, 1:j - 1, 0, :, :]
    gx = gv[..., X][:, :, None, :]
    gy = gv[..., Y][:, :, None, :]
    gz = gv[..., Z][:, :, None, :]

    vxi = vx[1:i - 1, 1:j - 1, :, :]
    vyi = vy[1:i - 1, 1:j - 1, :, :]
    vzi = vz[1:i - 1, 1:j - 1, :, :]

    prd = (vxi * gx + vyi * gy + vzi * gz) / rscale
    nxi = vxi - prd * gx / rscale
    nyi = vyi - prd * gy / rscale
    nzi = vzi - prd * gz / rscale

    nvx = _replace_interior(vx, nxi, xp)
    nvy = _replace_interior(vy, nyi, xp)
    nvz = _replace_interior(vz, nzi, xp)
    return xp.stack([nvx, nvy, nvz], axis=4)


def _horizontalize_pl(Pgrad_pl, GRD_x_pl, cfg, xp):
    """Inlined COMM-free pole tangent-plane projection (all g, all k)."""
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    rscale = cfg.rscale

    vx = Pgrad_pl[:, :, :, X]
    vy = Pgrad_pl[:, :, :, Y]
    vz = Pgrad_pl[:, :, :, Z]

    # GRD_x_pl: (g, 1, l, 3) -> direction (g, l) -> (g, 1, l)
    gx = GRD_x_pl[:, 0, :, X][:, None, :]
    gy = GRD_x_pl[:, 0, :, Y][:, None, :]
    gz = GRD_x_pl[:, 0, :, Z][:, None, :]

    # NOTE: the original pole loop divides each term by rscale separately
    # (mod_oprt.py L2058-2061), unlike the regional vectorized version which
    # divides the summed dot product. Match it term-by-term for bit-exactness.
    prd = vx * gx / rscale + vy * gy / rscale + vz * gz / rscale
    nvx = vx - prd * gx / rscale
    nvy = vy - prd * gy / rscale
    nvz = vz - prd * gz / rscale
    return xp.stack([nvx, nvy, nvz], axis=3)


def _replace_interior(A, B, xp):
    """Return A with its [1:i-1, 1:j-1] interior replaced by B, border preserved.

    A : (i, j, k, l)   B : (i-2, j-2, k, l)
    """
    i = A.shape[0]
    j = A.shape[1]
    mid = xp.concatenate([A[1:i - 1, 0:1, :, :], B, A[1:i - 1, j - 1:j, :, :]], axis=1)
    return xp.concatenate([A[0:1, :, :, :], mid, A[i - 1:i, :, :, :]], axis=0)
