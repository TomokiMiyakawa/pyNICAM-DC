#!/bin/bash
# Generate the CHARACTERISTIC figure(s) + movie for a tutorial case.
# Reads viz_spec.txt for the field/view that reveals each case's signature feature
# (temperature perturbation for baroclinic waves, surface pressure for the TC,
# vertical velocity for the supercell, a lon-height cross-section for mountain/
# gravity waves, the passive tracer for advection cases). Runs the case with the
# physical DIAGNOSTICS enabled, then renders with render_zarr.py.
#
#   ./run_viz.sh <case> [var] [arg] [steps]   # CLI args override viz_spec.txt
# Output: viz/<case>/  (PNG frames + an mp4).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$CODE"
cd "$HERE"
DRIVER="$CODE/pynicamdc/nhm/driver/driver-dc.py"
RENDER="$CODE/pynicamdc/nhm/driver/render_zarr.py"

case="${1:?usage: ./run_viz.sh <case> [var] [arg] [steps]  (cases: see viz_spec.txt)}"
[ -f "case/config/nhm_${case}.toml" ] || { echo "unknown case '$case'"; exit 2; }
[ -f case/grid_gl05rl00pe01/vgrid30_dcmip2016v2.json ] || { echo "dataset missing -> ./download_inputs.sh"; exit 2; }

# defaults from viz_spec.txt
spec=$(grep -E "^${case}\|" viz_spec.txt | head -1)
svar=$(echo "$spec" | cut -d'|' -f2); smode=$(echo "$spec" | cut -d'|' -f3)
sarg=$(echo "$spec" | cut -d'|' -f4); ssteps=$(echo "$spec" | cut -d'|' -f5)
var="${2:-$svar}"; arg="${3:-$sarg}"; steps="${4:-$ssteps}"
mode="$smode"                                   # geometry is a property of the case
: "${var:=RHOGE}" "${arg:=-}" "${steps:=12}" "${mode:=h}"

# viz config: same case, diagnostics ON, output every step, run `steps`; tracers on if needed
cfg="case/config/_viz_${case}.toml"
cp "case/config/nhm_${case}.toml" "$cfg"
sed -i "s/lstep_max *= *[0-9]*/lstep_max = ${steps}/" "$cfg"
sed -i "s/PRGout_interval *= *[0-9]*/PRGout_interval=1/" "$cfg"
grep -q 'PRGout_diagnostics' "$cfg" && sed -i "s/PRGout_diagnostics *= *false/PRGout_diagnostics=true/" "$cfg" \
    || sed -i "s#\(PRGout_name.*\)#\1\nPRGout_diagnostics=true#" "$cfg"
[[ "$var" == passive* ]] && sed -i "s/PRGout_tracers *= *false/PRGout_tracers=true/" "$cfg"
printf '[driver]\nbackend="numpy"\nprecision="float64"\nnhm_driver_cnf="./case/config/_viz_%s.toml"\n' "$case" > "/tmp/drv_viz_${case}.toml"

echo "=== VIZ [$case] var=$var mode=$mode arg=$arg steps=$steps -- running (numpy, diagnostics on) ==="
rm -rf ./testout_tmp.zarr ./msg.pe*
if command -v mpirun >/dev/null 2>&1; then MP="mpirun -np 1"; else MP=""; fi
$MP python "$DRIVER" --driver-setting "/tmp/drv_viz_${case}.toml" < /dev/null > "run_viz_${case}.log" 2>&1
rm -f "$cfg"
grep -q 'peacefully done' "run_viz_${case}.log" || { echo "RUN FAILED -- see run_viz_${case}.log"; tail -5 "run_viz_${case}.log"; exit 1; }

echo "=== rendering -> viz/${case}/ ==="
if [ "$mode" = "x" ]; then
  python "$RENDER" ./testout_tmp.zarr --var "$var" --cross-section "$arg" \
      --outdir "viz/${case}" --movie "viz/${case}/${case}_${var}_xsec.mp4" --fps 4
else
  kopt=""; [ "$arg" != "-" ] && kopt="--k $arg"
  python "$RENDER" ./testout_tmp.zarr --var "$var" $kopt \
      --outdir "viz/${case}" --movie "viz/${case}/${case}_${var}.mp4" --fps 4
fi
echo "=== done: viz/${case}/ ==="
