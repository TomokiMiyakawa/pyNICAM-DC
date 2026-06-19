#!/usr/bin/env bash
# Best-effort hybrid CPU sweep: jax backend + PYNICAM_BESTEFFORT=1 (divdamp ->
# numpy), gl05-09, one isolated node per group. Writes run/gl0N_jax_be/ and
# run/timers_jax_be.csv -- does NOT touch the plain numpy/jax runs.
#
#   ./besteffort_sweep.sh            # submit the sweep (auto-picks idle P28 nodes)
#   ./besteffort_sweep.sh collect    # after jobs finish: build timers_jax_be.csv + compare
#
# Step counts match the main sweep (gl05-07=12, gl08=6, gl09=4) so the per-step
# numbers line up with run/timers_{numpy,jax}.csv and run/f90_reference.csv.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
PY="$HOME/micromamba/envs/pynicam-sweep/bin/python"

if [ "${1:-}" = "collect" ]; then
    "$PY" "$ROOT/scripts/collect_timers.py" --label jax_be 5 6 7 8 9 | tee "$ROOT/run/timers_jax_be.csv"
    echo "=== compare vs f90 + the other backends (timers_*.csv) ==="
    "$PY" "$ROOT/scripts/compare_f90.py"
    exit 0
fi

# --- pick 3 distinct idle P28 nodes (most free memory first) ---
mapfile -t IDLE < <(sinfo -p P28 -N -h -o "%t %e %n" | awk '$1=="idle"{print $2, $3}' | sort -k1 -nr | awk '{print $2}')
if [ "${#IDLE[@]}" -lt 3 ]; then
    echo "WARN: only ${#IDLE[@]} idle P28 nodes; heavy grids may queue or share."; fi
N1="${IDLE[0]:-}"; N2="${IDLE[1]:-$N1}"; N3="${IDLE[2]:-$N2}"
nd(){ [ -n "$1" ] && echo "--nodelist=$1" || true; }

sub(){ # name tlimit mem glevels lstep node
    GLEVELS="$4" BACKEND=jax LSTEP="$5" RUNLABEL=jax_be PYNICAM_BESTEFFORT=1 \
    sbatch --parsable --export=ALL -c1 $(nd "$6") -J "$1" -t "$2" --mem="$3" sweep.sbatch
}
J1=$(sub be_lo  02:00:00 32G  "5 6 7" 12 "$N1")
J2=$(sub be_g08 04:00:00 64G  "8"     6  "$N2")
J3=$(sub be_g09 12:00:00 400G "9"     4  "$N3")
echo "best-effort sweep submitted: lo(gl05-07)=$J1 g08=$J2 g09=$J3"
echo "nodes: $N1 $N2 $N3"
echo "$J1 $J2 $J3" > .besteffort_jids
echo "when all finish:  ./besteffort_sweep.sh collect"
