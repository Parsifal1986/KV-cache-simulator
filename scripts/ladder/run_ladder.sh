#!/usr/bin/env bash
# Motivation "baseline ladder": one workload (spider2 terminus x2, 750K pool), a concurrency sweep from slack to the
# capacity wall, every simulator class vs the GPU arms of the same machine.
#   B1  trace-driven cache replay at recorded times         sim/offline_replay.py (retime_for_concurrency)
#   B2  open-loop DES: des_sim mechanics, recorded arrivals  scripts/ladder/b2_openloop.py
#   B3  naive closed loop                                    sim/closed_loop_sim.py, step costs projected from the profile
#   DES closed-loop des_sim (prediction mode, 3 noise seeds)
#   SGL closed-loop sgl_sim (real SGLang Scheduler; normal loop, no noise)
# All five use the same machine profile (B1 has no step costs: it replays recorded service times).
#
# usage: run_ladder.sh OUT_DIR            (env: MAXJ=7 PY=... DATA_ROOT=... SGL=path/to/sgl_sim.py)
set -euo pipefail
OUT=${1:?out dir}; MAXJ=${MAXJ:-7}
SIMU=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}
PY=${PY:-$DATA_ROOT/.venv/bin/python}
SGL=${SGL:-$SIMU/sim/sgl_sim.py}
export LD_LIBRARY_PATH=/extra/chuxi/libs:${LD_LIBRARY_PATH:-} TMPDIR=${TMPDIR:-/home/chuxi/tmp} PYTHONPATH=$SIMU/sim
# one core per simulation (user rule: all CPU jobs <= 30 threads, <= 100 GB): offline_replay.py / closed_loop_sim.py do
# not pin torch/OpenMP themselves and otherwise start ~128 threads each
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ST=results/terminus_recordings/spider2_terminus/trace_x/spider2_terminus_x2.pkl
REC="--rec-conc 4 --rec-span 9765.1 --trial-gap 0"
mkdir -p "$OUT"; JOBS=$OUT/jobs.txt; : > "$JOBS"

b3args() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join('--'+k.replace('_','-')+' '+repr(v) for k,v in d.items() if not k.startswith('_')))" "$1"; }

add_point() {   # machine conc seed arms
    local M=$1 C=$2 S=$3 ARMS=$4 D="$OUT/$1/c$2_s$3"
    local PROF=$SIMU/profiles/machines/$M.json B3=$(b3args "$SIMU/profiles/ladder/b3_$M.json")
    local CM="$ST --kv 750000 --conc $C --seed $S $REC"
    echo "$PY -W ignore $SIMU/sim/offline_replay.py $ST --kv 750000 --conc $C --seeds $S --arms $ARMS --out-dir $D/b1 --csv --no-gate --quiet > $D/b1.log 2>&1" >> "$JOBS"
    echo "$PY -W ignore $SIMU/sim/closed_loop_sim.py $CM --arms $ARMS $B3 --out-dir $D/b3 --csv > $D/b3.log 2>&1" >> "$JOBS"
    for N in 1 2 3; do
        echo "$PY -W ignore $SIMU/sim/des_sim.py $CM --arms $ARMS --params-json $PROF --noise-seed $N --out-dir $D/des_n$N --csv > $D/des_n$N.log 2>&1" >> "$JOBS"
    done
    for A in ${ARMS//,/ }; do
        echo "$PY -W ignore $SIMU/scripts/ladder/b2_openloop.py $CM --arms $A --params-json $PROF --out-dir $D/b2_$A > $D/b2_$A.log 2>&1" >> "$JOBS"
        echo "$PY -W ignore $SGL $CM --arm $A --no-overlap --context-length 57344 --params-json $PROF --out-dir $D/sgl --csv > $D/sgl_$A.log 2>&1" >> "$JOBS"
    done
    mkdir -p "$D"
}

ALL=lru,rrdeclband0.001,rrdeclchunk1024
for C in 16 24 32 40 46 48 56; do add_point local8xh100 $C 0 $ALL; done
for S in 1 2; do add_point local8xh100 32 $S $ALL; done
for C in 32 40 44 48 52; do add_point pod0920 $C 0 rrdeclband0.001,rrdeclchunk1024; done

echo "$(wc -l < "$JOBS") jobs, $MAXJ in parallel"
cd "$DATA_ROOT"
tr '\n' '\0' < "$JOBS" | xargs -0 -P "$MAXJ" -I{} bash -c '{}' || true
echo "ALL DONE"
