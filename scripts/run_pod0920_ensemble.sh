#!/bin/bash
# Closed-loop des_sim ensemble over the pod0920 arms (profiles/machines/pod0920_arms.tsv), 8 noise seeds per arm.
# Paths in the manifest (GPU arm dirs, stream pickles) are relative to DATA_ROOT (the sglang repo holding results/).
SIMU=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}
PY=${PY:-$DATA_ROOT/.venv/bin/python}
cd "$DATA_ROOT"
export LD_LIBRARY_PATH=${LIBNUMA_DIR:-/extra/chuxi/libs}:$LD_LIBRARY_PATH TMPDIR=${TMPDIR:-/home/chuxi/tmp}
S=${S:-results/des_sim_validation/pod0920_runs}
run() { # tag dir stream conc seed rec_conc rec_span trial_gap arm j
  out=$S/ens2/$1_n${10}; [ -f $out/summary.json ] && return
  $PY $SIMU/sim/des_sim.py $3 --kv 750000 --conc $4 --seed $5 --rec-conc $6 --rec-span $7 --trial-gap $8 --arms $9 --sched fcfs --params-json $SIMU/profiles/machines/pod0920.json --drops-from $2 --noise-seed ${10} --out-dir $out --csv > $out.log 2>&1
}
for j in 1 2 3 4 5 6 7 8; do
  while IFS=$'\t' read tag dir st conc seed rc rs tg arm; do
    while [ $(jobs -r | wc -l) -ge ${MAXJ:-12} ]; do sleep 5; done   # user limit: <=30 threads, <=100 GB total
    run $tag $dir $st $conc $seed $rc $rs $tg $arm $j &
  done < $SIMU/profiles/machines/pod0920_arms.tsv
done
wait; echo ALL DONE
