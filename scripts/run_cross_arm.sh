#!/usr/bin/env bash
# Cross-arm token test: simulate GPU arm TARGET with the token ids another GPU arm (SOURCE, same work point, other
# policy) actually sent/generated -- "run the real model once, simulate every policy". ServerArgs come from TARGET;
# host load is not modelled. Output layout matches analysis/local_eval.py ($OUT/<target>_n<seed>/req_<arm>.csv).
# usage: run_cross_arm.sh OUT PAIRS.tsv   (PAIRS: target_tag<TAB>source_tag; env SEEDS="1 2 3" MAXJ=4)
set -euo pipefail
SIMU=${SIMU:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}
PY=${PY:-$DATA_ROOT/.venv/bin/python}
MANIFEST=${MANIFEST:-$SIMU/profiles/machines/local8xh100_arms.tsv}
PROFILE=${PROFILE:-$SIMU/profiles/machines/local8xh100.json}
SEEDS=${SEEDS:-"1 2 3"}; MAXJ=${MAXJ:-4}          # --ids-from: 10-18 GB per process
OUT=$(realpath -m "$1"); PAIRS=$(realpath "$2")
export LD_LIBRARY_PATH=/extra/chuxi/libs:${LD_LIBRARY_PATH:-} TMPDIR=${TMPDIR:-/home/chuxi/tmp}
cd "$DATA_ROOT"; mkdir -p "$OUT"
while IFS=$'\t' read -r tgt src; do
  IFS=$'\t' read -r tag dir stream conc seed recconc recspan tgap arm < <(grep -P "^$tgt\t" "$MANIFEST")
  sdir=$(grep -P "^$src\t" "$MANIFEST" | cut -f2)
  for n in $SEEDS; do
    d="$OUT/${tag}_n$n"; [ -s "$d/req_$arm.csv" ] && continue
    echo "mkdir -p $d && $PY -W ignore $SIMU/sim/sgl_sim.py $stream --kv 750000 --conc $conc --seed $seed --rec-conc $recconc --rec-span $recspan --trial-gap $tgap --arm $arm --params-json $PROFILE --noise-seed $n --server-args-from $dir --ids-from $sdir --out-dir $d --csv > $d/run.log 2>&1"
  done
done < "$PAIRS" | xargs -P "$MAXJ" -I{} bash -c {}
