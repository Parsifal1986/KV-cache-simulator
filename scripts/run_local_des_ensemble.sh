#!/usr/bin/env bash
# des_sim closed loop over every arm of a machine manifest x noise seeds (same layout as run_local_sgl_ensemble.sh,
# for a head-to-head against sgl_sim). Schedule policy per arm is read from the GPU arm's server.log.
# usage: run_local_des_ensemble.sh OUT [FILTER_REGEX]   (env: SEEDS="1 2 3" MAXJ=6 PROFILE=... MANIFEST=... IDS=1)
# IDS=1: token ids the GPU arm actually sent/generated (--ids-from, 10-23 GB per process -> MAXJ<=4 under 100 GB)
set -euo pipefail
SIMU=${SIMU:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}
PY=${PY:-$DATA_ROOT/.venv/bin/python}
MANIFEST=${MANIFEST:-$SIMU/profiles/machines/local8xh100_arms.tsv}
PROFILE=${PROFILE:-$SIMU/profiles/machines/local8xh100.json}
SEEDS=${SEEDS:-"1 2 3"}; MAXJ=${MAXJ:-6}
OUT=$(realpath -m "$1"); FILTER=${2:-.}
export LD_LIBRARY_PATH=/extra/chuxi/libs:${LD_LIBRARY_PATH:-} TMPDIR=${TMPDIR:-/home/chuxi/tmp}
cd "$DATA_ROOT"; mkdir -p "$OUT"
while IFS=$'\t' read -r tag dir stream conc seed recconc recspan tgap arm; do
  [[ "$tag" =~ $FILTER ]] || continue
  log=$(ls "$dir"/server.log.gz "$dir"/server.log 2>/dev/null | head -1 || true)
  sched=$(zgrep -o -m1 "schedule_policy='[a-z]*'" "$log" | head -1 | cut -d"'" -f2 || true)
  if [ "${IDS:-0}" = 1 ]; then tok="--ids-from $dir"; else tok="--drops-from $dir"; fi
  for n in $SEEDS; do
    d="$OUT/${tag}_n$n"; [ -s "$d/req_$arm.csv" ] && continue
    echo "mkdir -p $d && $PY -W ignore $SIMU/sim/des_sim.py $stream --kv 750000 --conc $conc --seed $seed --rec-conc $recconc --rec-span $recspan --trial-gap $tgap --arms $arm --sched ${sched:-fcfs} --params-json $PROFILE --noise-seed $n $tok --out-dir $d --csv > $d/run.log 2>&1"
  done
done < "$MANIFEST" | xargs -P "$MAXJ" -I{} bash -c {}
