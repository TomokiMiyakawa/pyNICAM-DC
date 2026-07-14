#!/bin/bash
# Generate the CHARACTERISTIC figure(s) + movie for a tutorial case, tuned per case
# in viz_spec.txt (field, level/latitude, centering, zonal-anomaly, step count).
#
#   ./run_viz.sh <case>            # quick default (onset / early evolution)
#   ./run_viz.sh <case> --full     # full_steps: push toward the mature feature (SLOW)
#
# Each run gets its OWN sandbox  runs/<case>/  (the shared inputs are symlinked in as
# ./case, so nothing is copied and runs never clobber each other). Everything for the
# run -- the config actually used (_viz_<case>.toml), the log (run.log), the raw zarr
# and msg files -- lives in runs/<case>/. The rendered figures + movie go to viz/<case>/.
#
# Fast cases (gw, sc, tracers) show their feature at the quick setting; slow cases
# (jw/jm/tc/mw) need --full (baroclinic wave breaks ~day 8, etc.) -- and --full at a
# few hundred steps on numpy is minutes+; run on more cores / a GPU build for speed
# (qsub -v CASE=<case> run_viz_gpu.pbs runs the same thing on the GPU).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$CODE"
DRIVER="$CODE/pynicamdc/nhm/driver/driver-dc.py"
RENDER="$CODE/pynicamdc/nhm/driver/render_zarr.py"

case="${1:?usage: ./run_viz.sh <case> [--full]   (cases: see viz_spec.txt)}"
full=0; [ "${2:-}" = "--full" ] && full=1
[ -f "$HERE/case/config/nhm_${case}.toml" ] || { echo "unknown case '$case'"; exit 2; }
[ -f "$HERE/case/grid_gl05rl00pe01/vgrid30_dcmip2016v2.json" ] || { echo "dataset missing -> ./download_inputs.sh"; exit 2; }

spec=$(grep -E "^${case}\|" "$HERE/viz_spec.txt" | head -1)
[ -z "$spec" ] && { echo "no viz_spec.txt entry for '$case'"; exit 2; }
var=$(echo "$spec"|cut -d'|' -f2); mode=$(echo "$spec"|cut -d'|' -f3); arg=$(echo "$spec"|cut -d'|' -f4)
steps=$(echo "$spec"|cut -d'|' -f5); fsteps=$(echo "$spec"|cut -d'|' -f6)
dtl=$(echo "$spec"|cut -d'|' -f7); opts=$(echo "$spec"|cut -d'|' -f8)
[ "$full" = 1 ] && steps="$fsteps"

# --- per-run sandbox: runs/<case>/ with the shared inputs symlinked in as ./case ---
rundir="$HERE/runs/${case}"
rm -rf "$rundir"; mkdir -p "$rundir"
ln -sfn "$HERE/case" "$rundir/case"
cfg="$rundir/_viz_${case}.toml"
cp "$HERE/case/config/nhm_${case}.toml" "$cfg"
# viz config: correct DTL for the case, diagnostics on, output every step, run `steps`; tracers on if plotting one
[ -n "$dtl" ] && sed -i "s/[Dd][Tt][Ll] *= *[0-9.eED]*/dtl = ${dtl}.0/" "$cfg"
sed -i "s/lstep_max *= *[0-9]*/lstep_max = ${steps}/" "$cfg"
# Cap output to ~150 frames EVENLY SAMPLED across the whole run (coarser cadence, not
# the first N steps): a smaller, sensible movie whose render fits in memory. render_zarr
# loads the selected timesteps into RAM, so one-frame-per-step of a long z40-60 run OOMs.
oint=$(( (steps + 149) / 150 )); [ "$oint" -lt 1 ] && oint=1
sed -i "s/PRGout_interval *= *[0-9]*/PRGout_interval=${oint}/g" "$cfg"
grep -q 'PRGout_diagnostics' "$cfg" && sed -i "s/PRGout_diagnostics *= *false/PRGout_diagnostics=true/" "$cfg" \
    || sed -i "s#\(PRGout_name.*\)#\1\nPRGout_diagnostics=true#" "$cfg"
[[ "$var" == passive* ]] && sed -i "s/PRGout_tracers *= *false/PRGout_tracers=true/" "$cfg"
printf '[driver]\nbackend="numpy"\nprecision="float64"\nnhm_driver_cnf="./_viz_%s.toml"\n' "$case" > "$rundir/drv_viz.toml"

echo "=== VIZ [$case] var=$var mode=$mode arg=$arg steps=$steps${opts:+ opts=$opts}$([ $full = 1 ] && echo ' (FULL)') -> runs/${case}/ ==="
cd "$rundir"
if command -v mpirun >/dev/null 2>&1; then MP="mpirun -np 1"; else MP=""; fi
$MP python "$DRIVER" --driver-setting "drv_viz.toml" < /dev/null > run.log 2>&1
grep -q 'peacefully done' run.log || { echo "RUN FAILED -- see runs/${case}/run.log"; tail -5 run.log; exit 1; }

echo "=== rendering -> viz/${case}/ ==="
vizout="$HERE/viz/${case}"
if [ "$mode" = "x" ]; then
  python "$RENDER" ./testout_tmp.zarr --var "$var" --cross-section "$arg" $opts \
      --outdir "$vizout" --movie "$vizout/${case}_${var}_xsec.mp4" --fps 4
else
  kopt=""; [ "$arg" != "-" ] && kopt="--k $arg"
  python "$RENDER" ./testout_tmp.zarr --var "$var" $kopt $opts \
      --outdir "$vizout" --movie "$vizout/${case}_${var}.mp4" --fps 4
fi
echo "=== done: viz/${case}/  (run sandbox kept at runs/${case}/) ==="
