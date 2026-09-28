#!/usr/bin/env bash
# sgl_sim (real SGLang scheduler) closed loop over every arm of a machine manifest x noise seeds.
# Each run takes its ServerArgs and token ids / client drops from the GPU arm it is compared with; host load is not
# modelled (runs are at the machine profile's conditions).
# Output layout matches analysis/local_eval.py: $OUT/<tag>_n<seed>/req_<arm>.csv
# usage: run_local_sgl_ensemble.sh OUT [FILTER_REGEX]   (env: SEEDS="1 2 3" MAXJ=6 PROFILE=... MANIFEST=... IDS=1)
# IDS=1: token ids = what the GPU arm actually sent/generated (--ids-from, 10-18 GB per process -> MAXJ<=4 under the
# 100 GB limit); IDS_FROM_TAG=<tag suffix map> not used -- cross-arm token traces are run separately.
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
  for n in $SEEDS; do
    d="$OUT/${tag}_n$n"; [ -s "$d/req_$arm.csv" ] && continue
    if [ "${IDS:-0}" = 1 ]; then tok="--ids-from $dir"; else tok="--drops-from $dir --client-from $dir"; fi
    echo "mkdir -p $d && $PY -W ignore $SIMU/sim/sgl_sim.py $stream --kv 750000 --conc $conc --seed $seed --rec-conc $recconc --rec-span $recspan --trial-gap $tgap --arm $arm --params-json $PROFILE --noise-seed $n --server-args-from $dir $tok --out-dir $d --csv > $d/run.log 2>&1"
  done
done < "$MANIFEST" | xargs -P "$MAXJ" -I{} bash -c {}
