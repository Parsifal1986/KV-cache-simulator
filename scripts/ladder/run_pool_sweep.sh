#!/usr/bin/env bash
# Escape-probability curve at a wall point: sgl_sim over pool sizes x policy arms x noise seeds on ONE token stream.
# usage: run_pool_sweep.sh TOKEN_PKL OUT_DIR   (env: CONC=48 POOLS="700000 ..." ARMS="rrdeclband0.001 rrdeclchunk1024" NSEEDS="1..10"
#        PROF=profiles/machines/pod0920.json MAXJ=8 METHODS="sgl" (sgl and/or des) -- user limit: all jobs <= 30 threads / 100 GB)
set -euo pipefail
T=$(realpath "${1:?token pkl}"); OUT=$(realpath -m "${2:?out dir}")
SIMU=$(cd "$(dirname "$0")/../.." && pwd); DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}; PY=${PY:-$DATA_ROOT/.venv/bin/python}
CONC=${CONC:-48}; POOLS=${POOLS:-"700000 725000 775000 800000 825000 850000"}; ARMS=${ARMS:-"rrdeclband0.001 rrdeclchunk1024"}
NSEEDS=${NSEEDS:-"1 2 3 4 5 6 7 8 9 10"}; PROF=${PROF:-$SIMU/profiles/machines/pod0920.json}; MAXJ=${MAXJ:-8}
export LD_LIBRARY_PATH=/extra/chuxi/libs:${LD_LIBRARY_PATH:-} TMPDIR=${TMPDIR:-/home/chuxi/tmp} PYTHONPATH=$SIMU/sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$DATA_ROOT"; J=$OUT/jobs.txt; mkdir -p "$OUT"; : > "$J"
CM="--conc $CONC --seed 0 --rec-conc 4 --rec-span 9765.1 --trial-gap 0"
if [[ " ${METHODS:-sgl} " == *" des "* ]]; then for K in $POOLS; do for N in $NSEEDS; do   # des_sim runs all arms in one process
    D=$OUT/kv$K/des_n$N; [ -n "$(ls $D/req_* 2>/dev/null)" ] && continue
    echo "mkdir -p $OUT/kv$K && $PY -W ignore $SIMU/sim/des_sim.py $T --kv $K $CM --arms ${ARMS// /,} --no-stitch --params-json $PROF --noise-seed $N --out-dir $D --csv > $OUT/kv$K/des_n$N.log 2>&1" >> "$J"
done; done; fi
[[ " ${METHODS:-sgl} " == *" sgl "* ]] || POOLS=""
for K in $POOLS; do for A in $ARMS; do for N in $NSEEDS; do
    D=$OUT/kv$K/sgl_n$N; [ -s "$D/req_$A.csv" ] && continue
    echo "mkdir -p $D && $PY -W ignore $SIMU/scripts/ladder/sgl_tok.py $T --kv $K $CM --arm $A --sched fcfs --context-length 57344 --params-json $PROF --noise-seed $N --out-dir $D --csv > $D/../sgl_n${N}_$A.log 2>&1" >> "$J"
done; done; done
echo "$(wc -l < "$J") jobs, $MAXJ in parallel"
tr '\n' '\0' < "$J" | xargs -0 -P "$MAXJ" -I{} bash -c '{}' || true
echo "ALL DONE"
