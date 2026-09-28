#!/usr/bin/env python3
"""Baseline B2 of the motivation ladder: OPEN-LOOP event-driven simulation.

Scheduler, radix cache and step costs are des_sim.py's (the same machine profile as the closed-loop runs), but every
(program, turn) enters the server at a time fixed in advance from the RECORDING -- exactly the times the trace-driven
cache replay (B1, offline_replay.py) uses: rhs.retime_for_concurrency packs trials behind a `conc`-slot semaphore
(same admission-order shuffle seed as des_sim) and keeps every turn's recorded offset inside its trial. The
simulator's own completion times never feed back into arrivals. Only a request's first-token/finish times and its
cache outcome are simulated.

usage: b2_openloop.py STREAM.pkl --kv 750000 --conc 32 --seed 0 --arms lru --trial-gap 0 \
           --params-json profiles/machines/local8xh100.json --out-dir DIR
"""
import argparse, csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
import des_sim as DS
OR = DS.OR; rhs = DS.rhs


def recorded_arrivals(reqs, conc, seed, trial_gap):
    """(program, turn) -> arrival time from the recording, re-timed to `conc` concurrent trials (B1's timing)."""
    timed = rhs.retime_for_concurrency(reqs, conc, seed, trial_gap)
    byp = defaultdict(list)
    for r in timed: byp[r.program_id or r.rid].append(r)
    arr = {}
    for pid, lst in byp.items():
        lst.sort(key=lambda r: r.arrival)
        for k, r in enumerate(lst): arr[(pid, k)] = r.arrival
    t0 = min(arr.values())
    return {k: v - t0 for k, v in arr.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream"); ap.add_argument("--kv", type=int, required=True); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="lru"); ap.add_argument("--rec-conc", type=int, default=4); ap.add_argument("--rec-span", type=float, default=None); ap.add_argument("--trial-gap", type=float, default=None)
    ap.add_argument("--params-json", required=True); ap.add_argument("--noise-seed", type=int, default=-1); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-stitch", action="store_true", help="the stream is already the server-side token input (make_token_stream.py): replay ids verbatim")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    P = dict(DS.DEFAULTS); P.update({k: v for k, v in json.load(open(a.params_json)).items() if not k.startswith("_")})
    P["noise_seed"] = a.noise_seed; P["probe_dead"] = False
    reqs = OR.load_stream(a.stream)
    if not a.no_stitch: reqs = DS.stitch_like_replay(reqs)      # as des_sim's default (program-unique outputs)
    trial_gap = a.trial_gap if a.trial_gap is not None else rhs.calibrate_trial_gap(reqs, a.rec_conc, a.rec_span)
    P["arrivals"] = recorded_arrivals(reqs, a.conc, a.seed, trial_gap); P["open_until"] = 0.0
    print(f"B2 open-loop: {len(P['arrivals'])} recorded arrivals, conc {a.conc} seed {a.seed} trial_gap {trial_gap:.0f}s", flush=True)
    out = dict(kv=a.kv, conc=a.conc, seed=a.seed, trial_gap=trial_gap, arms={})
    for name in a.arms.split(","):
        rec, occ, steps, st = DS.simulate(reqs, OR.ArmSpec.parse(name), a.kv, a.conc, a.seed, trial_gap, P)
        s, wins = DS.summarize(rec, occ, st, a.kv); out["arms"][name] = dict(summary=s, windows=wins)
        print(f"[{name}] miss {s['miss']:.2f}% U p99 {s['U_p99']:.0f} n>16k {s['n16k']} sTTFT p90 {s['sTTFT_p90']:.3f}s", flush=True)
        with open(os.path.join(a.out_dir, f"req_{name}.csv"), "w") as f:
            wr = csv.DictWriter(f, fieldnames=list(rec[0].keys())); wr.writeheader(); wr.writerows(rec)
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
