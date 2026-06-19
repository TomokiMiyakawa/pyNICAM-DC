#!/usr/bin/env python3
"""
Generate a per-resolution run directory (config + driversettings) for the
pyNICAM-DC f90-vs-pyNICAM resolution sweep, from config/nhm_driver.template.toml.

For glevel g (rlevel=1, pe=4):
  dtl   = 1200.0 / 2**(g-5)     # CFL: halve the timestep per glevel
  paths -> the bundled npz boundary/restart, vgrid, and mnginfo (absolute)
  output dir: run/gl0g/  with nhm_driver.toml + driversettings.toml

Usage:
  python scripts/make_config.py 7
  python scripts/make_config.py 7 --backend jax --lstep 12 --output on
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # package root (parent of scripts/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glevel", type=int, help="grid level 5..9")
    ap.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    ap.add_argument("--precision", default="float64")
    ap.add_argument("--lstep", type=int, default=12, help="number of large steps (default 12)")
    ap.add_argument("--output", choices=("off", "on"), default="off",
                    help="off (default): minimise I/O for clean timing; "
                         "on: write one restart snapshot + history (validated cadence)")
    ap.add_argument("--label", default=None,
                    help="run-dir / timer-CSV suffix (default: the backend name). "
                         "Use to separate variants, e.g. --label jax_be for the "
                         "best-effort hybrid so it does not overwrite the plain jax run.")
    a = ap.parse_args()

    g = a.glevel
    glpad = f"{g:02d}"
    dtl = 1200.0 / (2 ** (g - 5))                  # CFL-scaled timestep

    # Horizontal hyperdiffusion / divergence-damping coefficient (DIRECT, lap_order=2)
    # is resolution-dependent, NOT fixed. Values match the f90 NICAM-DC ICOMEX_JW
    # reference namelists (.../test/case/ICOMEX_JW/gl0Nrl01z40pe04_48steps/nhm_driver.cnf);
    # gl05 keeps the original validated value (no f90 48-step ref exists for gl05).
    # gamma_h == alpha_d at every level in the reference.
    HDIFF = {5: 1.20e16, 6: 1.50e15, 7: 2.00e14, 8: 2.50e13, 9: 3.00e12}
    if g not in HDIFF:
        raise SystemExit(f"ERROR: no hdiff/divdamp coefficient defined for glevel {g} "
                         f"(known: {sorted(HDIFF)})")
    gamma_h = HDIFF[g]
    alpha_d = HDIFF[g]

    # PRGout write guard in the driver is `if n % PRGout_interval == 1`, and
    # mod_io sizes the zarr time axis as nt = int(lstep_max / PRGout_interval)
    # (must stay >= 1). interval=1 => guard never true => no writes, nt=lstep_max.
    if a.output == "on":
        prgint, hstep = a.lstep, 3          # one snapshot (n=1); validated history cadence
    else:
        prgint, hstep = 1, a.lstep          # disable writes; keep nt>=1

    data = os.path.join(ROOT, "data")
    hgrid = os.path.join(data, "boundary", f"gl{glpad}rl01pe04", f"bboundary_GL{glpad}RL01.pe")
    restart = os.path.join(data, "restart", f"gl{glpad}rl01pe04", f"restart_all_GL{glpad}RL01z40.pe")
    vgrid = os.path.join(data, "vgrid40_stretch_45km.json")
    mnginfo = os.path.join(data, "mnginfo", "rl01-prc000004.toml")

    for p, what in [(hgrid + "00000000.npz", "boundary npz"),
                    (restart + "00000000.npz", "restart npz"),
                    (vgrid, "vgrid"), (mnginfo, "mnginfo")]:
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: missing {what}: {p}")

    label = a.label or a.backend
    rundir = os.path.join(ROOT, "run", f"gl{glpad}_{label}")
    os.makedirs(rundir, exist_ok=True)
    cfg_path = os.path.join(rundir, "nhm_driver.toml")

    with open(os.path.join(ROOT, "config", "nhm_driver.template.toml")) as f:
        tmpl = f.read()
    cfg = (tmpl
           .replace("@GLEVEL@", str(g))
           .replace("@GLPAD@", glpad)
           .replace("@DTL@", repr(dtl))
           .replace("@GAMMA_H@", repr(gamma_h))
           .replace("@ALPHA_D@", repr(alpha_d))
           .replace("@LSTEP@", str(a.lstep))
           .replace("@PRGINT@", str(prgint))
           .replace("@HSTEP@", str(hstep))
           .replace("@HGRID_FNAME@", hgrid)
           .replace("@VGRID_FNAME@", vgrid)
           .replace("@INPUT_BASENAME@", restart)
           .replace("@MNGINFO@", mnginfo)
           .replace("@SELF@", cfg_path))
    with open(cfg_path, "w") as f:
        f.write(cfg)

    drv = (f'[driver]\n'
           f'backend = "{a.backend}"\n'
           f'precision = "{a.precision}"\n'
           f'nhm_driver_cnf = "{cfg_path}"\n')
    with open(os.path.join(rundir, "driversettings.toml"), "w") as f:
        f.write(drv)

    print(f"gl{glpad}: dtl={dtl:g} gamma_h=alpha_d={gamma_h:g} "
          f"lstep={a.lstep} backend={a.backend} output={a.output}")
    print(f"  -> {rundir}/nhm_driver.toml  + driversettings.toml")


if __name__ == "__main__":
    main()
