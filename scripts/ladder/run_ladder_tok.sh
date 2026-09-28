#!/usr/bin/env bash
# Baseline ladder, v2 (2026-09-28): every method replays the SAME server-side token input -- the ids a GPU arm actually
# sent/generated (make_token_stream.py) -- so only the timing / scheduling model differs between methods.
#   B1  trace-driven cache replay at recorded times       sim/offline_replay.py        (token stream pickle)
#   B2  open-loop DES at recorded arrivals                scripts/ladder/b2_openloop.py --no-stitch
#   B3  naive closed loop                                 sim/closed_loop_sim.py       (profile-projected step costs)
#   DES des_sim closed loop, 3 noise seeds                sim/des_sim.py --no-stitch
#   SGL sgl_sim closed loop, 3 noise seeds, real SGLang Scheduler + overlap loop   scripts/ladder/sgl_tok.py
# One token arm per point (all methods and all policy arms of that point use it); sim-only points borrow the nearest
# GPU arm's tokens ("cross-point"). Host load is NOT modelled (user rule 2026-09-28): every simulator runs at the machine
# profile's conditions; GPU runs made under abnormal host load are flagged in the report instead.
# usage: run_ladder_tok.sh OUT_DIR     (env: MAXJ=3 RUNS=runs_noload POINTS="add_point ...; ..." to run only those points -- user limit: all jobs <= 30 threads / 100 GB)
set -euo pipefail
OUT=$(realpath -m "${1:?out dir}"); MAXJ=${MAXJ:-3}
SIMU=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/chuxi/Project/sglang}
PY=${PY:-$DATA_ROOT/.venv/bin/python}
export LD_LIBRARY_PATH=/extra/chuxi/libs:${LD_LIBRARY_PATH:-} TMPDIR=${TMPDIR:-/home/chuxi/tmp} PYTHONPATH=$SIMU/sim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ST=results/terminus_recordings/spider2_terminus/trace_x/spider2_terminus_x2.pkl
REC="--rec-conc 4 --rec-span 9765.1 --trial-gap 0"
X=results/terminus_recordings/spider2_terminus/x2
declare -A TOK=(
  [sp24]=$X/setup_24c_750k_s0/lru [sp32s0]=$X/setup_32c_750k_s0/lru [sp32s1]=$X/setup_32c_750k_s1/lru [sp32s2]=$X/setup_32c_750k_s2/lru
  [sp46]=$X/setup_46c_750k_s0/lru [sp46rr]=${X}_local0923/setup_46c_750k_s0/rrdeclchunk1024 [sp48]=$X/setup_48c_750k_s0/lru
  [pod40]=$X/setup_40c_750k_s0_cloud/rrdeclband0.001 [pod48]=results/cloud_20260920/terminus_recordings/spider2_terminus/x2/setup_48c_750k_s0/rrdeclband0.001 )
cd "$DATA_ROOT"; mkdir -p "$OUT/tok"
# 1) token streams (one parse per token arm)
for k in "${!TOK[@]}"; do [ -s "$OUT/tok/$k.pkl" ] || echo "$PY -W ignore $SIMU/scripts/ladder/make_token_stream.py $ST ${TOK[$k]} $OUT/tok/$k.pkl > $OUT/tok/$k.log 2>&1"; done \
  | xargs -P "$MAXJ" -I{} bash -c {}
# 2) methods per point
b3args() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join('--'+k.replace('_','-')+' '+repr(v) for k,v in d.items() if not k.startswith('_')))" "$1"; }
JOBS=$OUT/jobs.txt; : > "$JOBS"
add_point() {   # machine conc seed arms token-key
    local M=$1 C=$2 S=$3 ARMS=$4 K=$5 D="$OUT/${RUNS:-runs_noload}/$1/c$2_s$3"; mkdir -p "$D"; echo "$K ${TOK[$5]}" > "$D/token_arm.txt"
    local PROF=$SIMU/profiles/machines/$M.json B3=$(b3args "$SIMU/profiles/ladder/b3_$M.json") T=$OUT/tok/$K.pkl
    local CM="$T --kv 750000 --conc $C --seed $S $REC"
    [ -s "$D/b1/summary.json" ] || [ -n "$(ls $D/b1/req_* 2>/dev/null)" ] || echo "$PY -W ignore $SIMU/sim/offline_replay.py $T --kv 750000 --conc $C --seeds $S --arms $ARMS --out-dir $D/b1 --csv --no-gate --quiet > $D/b1.log 2>&1" >> "$JOBS"
    [ -n "$(ls $D/b3/req_* 2>/dev/null)" ] || echo "$PY -W ignore $SIMU/sim/closed_loop_sim.py $CM --arms $ARMS $B3 --out-dir $D/b3 --csv > $D/b3.log 2>&1" >> "$JOBS"
    for N in 1 2 3; do
        [ -n "$(ls $D/des_n$N/req_* 2>/dev/null)" ] || echo "$PY -W ignore $SIMU/sim/des_sim.py $CM --arms $ARMS --no-stitch --params-json $PROF --noise-seed $N --out-dir $D/des_n$N --csv > $D/des_n$N.log 2>&1" >> "$JOBS"
    done
    for A in ${ARMS//,/ }; do
        [ -n "$(ls $D/b2_$A/req_* 2>/dev/null)" ] || echo "$PY -W ignore $SIMU/scripts/ladder/b2_openloop.py $CM --arms $A --no-stitch --params-json $PROF --out-dir $D/b2_$A > $D/b2_$A.log 2>&1" >> "$JOBS"
        for N in 1 2 3; do
            [ -s "$D/sgl_n$N/req_$A.csv" ] || echo "mkdir -p $D/sgl_n$N && $PY -W ignore $SIMU/scripts/ladder/sgl_tok.py $CM --arm $A --sched fcfs --context-length 57344 --params-json $PROF --noise-seed $N --out-dir $D/sgl_n$N --csv > $D/sgl_n${N}_$A.log 2>&1" >> "$JOBS"
        done
    done
}
ALL=lru,rrdeclband0.001,rrdeclchunk1024; POD=rrdeclband0.001,rrdeclchunk1024
if [ -n "${POINTS:-}" ]; then eval "$POINTS"; else
add_point local8xh100 16 0 $ALL sp24; add_point local8xh100 24 0 $ALL sp24
add_point local8xh100 32 0 $ALL sp32s0; add_point local8xh100 32 1 $ALL sp32s1; add_point local8xh100 32 2 $ALL sp32s2
add_point local8xh100 40 0 $ALL sp46; add_point local8xh100 46 0 $ALL sp46; add_point local8xh100 48 0 $ALL sp48; add_point local8xh100 56 0 $ALL sp48
add_point pod0920 32 0 $POD pod40; add_point pod0920 40 0 $POD pod40; add_point pod0920 44 0 $POD pod40
add_point pod0920 48 0 $POD pod48; add_point pod0920 52 0 $POD pod48
fi
echo "$(wc -l < "$JOBS") jobs, $MAXJ in parallel"
tr '\n' '\0' < "$JOBS" | xargs -0 -P "$MAXJ" -I{} bash -c '{}' || true
echo "ALL DONE"
