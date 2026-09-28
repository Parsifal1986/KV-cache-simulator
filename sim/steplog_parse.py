#!/usr/bin/env python3
"""Parse SGLang scheduler step lines (Prefill batch / Decode batch / retract warnings) from a server.log[.gz|.zst]
into per-step CSVs, as the calibration source for the event-driven simulator (des_sim.py).

Timing semantics (python/sglang/srt/observability/scheduler_metrics_mixin.py):
  * one "Prefill batch" line per prefill step, written when that step's RESULT is processed; its
    "input throughput" = tokens of the PREVIOUS prefill line / time since the previous prefill line.
    -> gap_i = new_tok_{i-1} / thr_i is the exact (sub-second) spacing of lines i-1 and i. When line i-1 was a
       full chunk (#new-token == chunked_prefill_size, i.e. a chunked request is still open) the scheduler MUST run
       another prefill next, so no decode step sits in between and gap_i is a pure prefill-step duration.
  * one "Decode batch" line every decode_log_interval (40) decode steps; "gen throughput" = tokens generated since
    the previous decode line / time since it. With no prefill line in between, the interval is 40 pure decode steps:
    step_ms = gap / 40, gap = generated / thr, generated ~= 40 * mean(running).
Log timestamps are whole seconds; the throughput fields give the precise gaps. A per-line absolute time is rebuilt
by anchoring each line to its second and accumulating precise gaps inside runs of consecutive same-kind lines.

usage: steplog_parse.py SERVER_LOG [--out-dir DIR] [--chunk 8192]
"""
import os as _os
# one process = one core (user limit: all simulation jobs together <= 30 threads)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): _os.environ.setdefault(_v, "1")
import argparse, gzip, io, os, re, subprocess, sys
from datetime import datetime

PRE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP\d+)? TP0[^\]]*\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+),.*?cuda graph: (\w+), input throughput \(token/s\): ([\d.]+)")
DEC = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP\d+)? TP0[^\]]*\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*?cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
RET = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP\d+)? TP0[^\]]*\].*KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+)")


def open_log(path):
    if path.endswith(".gz"): return io.TextIOWrapper(gzip.open(path), errors="replace")
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        return io.TextIOWrapper(p.stdout, errors="replace")
    return open(path, errors="replace")


def parse(path, chunk=8192):
    steps = []; t0 = None
    for line in open_log(path):
        if "TP0" not in line[:40]: continue
        m = PRE.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            if t0 is None: t0 = ts
            steps.append(dict(kind="P", sec=ts - t0, new_seq=int(m.group(2)), new_tok=int(m.group(3)), cached=int(m.group(4)), usage=float(m.group(5)), running=int(m.group(6)), queue=int(m.group(7)), cg=m.group(8) == "True", thr=float(m.group(9))))
            continue
        m = DEC.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            if t0 is None: t0 = ts
            steps.append(dict(kind="D", sec=ts - t0, running=int(m.group(2)), tokens=int(m.group(3)), usage=float(m.group(4)), cg=m.group(5) == "True", thr=float(m.group(6)), queue=int(m.group(7))))
            continue
        m = RET.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            if t0 is None: t0 = ts
            steps.append(dict(kind="R", sec=ts - t0, retracted=int(m.group(2))))
    # annotate prefill gaps and decode gaps
    lastP = None; lastD = None; nP_since_D = 0
    for i, s in enumerate(steps):
        if s["kind"] == "P":
            s["prev_new_tok"] = lastP["new_tok"] if lastP else None
            s["gap"] = (lastP["new_tok"] / s["thr"]) if (lastP and s["thr"] > 0) else None
            s["pure"] = bool(lastP and lastP["new_tok"] == chunk and s["thr"] > 0)   # previous step left a chunked request open
            s["prev_new_seq"] = lastP["new_seq"] if lastP else None; s["prev_cached"] = lastP["cached"] if lastP else None
            lastP = s; nP_since_D += 1
        elif s["kind"] == "D":
            s["n_prefill_between"] = nP_since_D if lastD else None
            s["prev_running"] = lastD["running"] if lastD else None
            if lastD and s["thr"] > 0:
                gen = 40 * 0.5 * (s["running"] + lastD["running"]); s["gap"] = gen / s["thr"]; s["step_ms"] = 1e3 * s["gap"] / 40
            else: s["gap"] = None; s["step_ms"] = None
            lastD = s; nP_since_D = 0
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log"); ap.add_argument("--out-dir", default=None); ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    steps = parse(a.log, a.chunk)
    out = a.out_dir or os.path.dirname(os.path.abspath(a.log)); os.makedirs(out, exist_ok=True)
    import csv
    P = [s for s in steps if s["kind"] == "P"]; D = [s for s in steps if s["kind"] == "D"]; R = [s for s in steps if s["kind"] == "R"]
    for name, rows in (("steps_prefill.csv", P), ("steps_decode.csv", D), ("steps_retract.csv", R)):
        if not rows: continue
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
        with open(os.path.join(out, name), "w") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{a.log}: {len(P)} prefill lines ({sum(s['pure'] for s in P)} pure-gap), {len(D)} decode lines ({sum(1 for s in D if s.get('n_prefill_between') == 0)} pure), {len(R)} retract events -> {out}")


if __name__ == "__main__":
    main()
