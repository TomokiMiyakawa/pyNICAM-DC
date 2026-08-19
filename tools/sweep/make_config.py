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
                    help="off (default): PRGout_interval past lstep_max, so NO snapshot is "
                         "written -- clean timing. testout_tmp.zarr is then created but never "
                         "filled, so it reads back as all-NaN: that is an empty store, NOT a "
                         "diverged solution, and it must never be compared against a gold. "
                         "on: PRGout_interval = lstep_max, one snapshot at the final step "
                         "(the shape the golds hold) -- use this for any run you intend to "
                         "validate with proto/cmp_prec.py.")
    ap.add_argument("--label", default=None,
                    help="run-dir / timer-CSV suffix (default: the backend name). "
                         "Use to separate variants, e.g. --label jax_be for the "
                         "best-effort hybrid so it does not overwrite the plain jax run.")
    ap.add_argument("--step0", action="store_true",
                    help="also emit the initial condition (t=0) as zarr frame 0, like "
                         "nicamdc doout_step0 (PRGout_step0). With the periodic output this "
                         "gives frames at TIME_cstep = 0, interval, 2*interval, ...")
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

    # Periodic output fires when TIME_cstep (= loop index n+1) is a multiple of PRGout_interval
    # (driver `_is_out_*`), i.e. at TIME_cstep = interval, 2*interval, ... (nicamdc phase); mod_io
    # sizes the zarr time axis nt = lstep_max//interval + step0 (kept >= 1). interval=lstep_max =>
    # one snapshot at the final step; interval > lstep_max => no periodic write at all, and the one
    # slot prg_output_nslots' floor keeps stays at zarr's NaN fill (see the --output help).
    if a.output == "on":
        prgint, hstep = a.lstep, 3          # one snapshot at the final step; validated history cadence
    else:
        prgint, hstep = a.lstep + 1, a.lstep   # past lstep_max => no write at all

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
           .replace("@STEP0@", "true" if a.step0 else "false")
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

    frames = ("none (zarr stays at its NaN fill -- not a result)" if a.output == "off"
              else f"1 at TIME_cstep={a.lstep}" + (" (+ step0)" if a.step0 else ""))
    print(f"gl{glpad}: dtl={dtl:g} gamma_h=alpha_d={gamma_h:g} "
          f"lstep={a.lstep} backend={a.backend} output={a.output}")
    print(f"  PRGout_interval={prgint} -> snapshots: {frames}")
    print(f"  -> {rundir}/nhm_driver.toml  + driversettings.toml")


if __name__ == "__main__":
    main()
