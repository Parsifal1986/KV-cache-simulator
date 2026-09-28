#!/usr/bin/env python3
"""Pair simulator requests (des_sim.py --csv req_<arm>.csv) with a GPU arm's requests on (program_id, turn) and report,
on the common warm turns inside the GPU arm's [cut0, end-cut1] arrival window:
  * both sides' miss / U tail / sTTFT tail / queue, and the per-request agreement of U (exact-hit share, corr);
  * optionally a second sim/GPU arm pair so the RR-vs-baseline DELTAS can be compared (the thing the sim must get right).

usage: sim_vs_gpu_pair.py --sim SIM_DIR/req_A.csv --gpu GPU_ARM_A [--sim2 SIM_DIR/req_B.csv --gpu2 GPU_ARM_B] [--cut0 300 --cut1 60]
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_windows import requests, breakdown, tm


def load_sim(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[(r["program"], int(r["turn"]))] = dict(pt=float(r["prompt_len"]), unc=float(r["prompt_len"]) - float(r["cached"]), srv=float(r["ttft"]), queue=float(r["first_sched"] if "first_sched" in r else r["admit"]) - float(r["arrival"]), recv=float(r["arrival"]))
    return out


def load_gpu(d):
    log = next(os.path.join(d, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(d, x)))
    return breakdown(requests(log))


def metrics(X, keys):
    pt = np.array([X[k]["pt"] for k in keys]); u = np.array([X[k]["unc"] for k in keys]); s = np.array([X[k]["srv"] for k in keys]); q = np.array([X[k]["queue"] for k in keys])
    return dict(miss=100 * u.sum() / pt.sum(), U_p90=np.percentile(u, 90), U_p99=np.percentile(u, 99), U_TM99=tm(u, 99), n16k=int((u > 16000).sum()),
                sTTFT_p50=np.percentile(s, 50), sTTFT_p90=np.percentile(s, 90), sTTFT_p99=np.percentile(s, 99), sTTFT_TM99=tm(s, 99), queue_p99=np.percentile(q, 99), dq_p99=np.percentile(s[q <= 0.05], 99))


def stall_keys(d, G, kv_dummy=None):
    """User rule (2026-09-23): requests with U<1000 and sTTFT>1.5 s, >=3 within 30 s -> candidate window (+-5 s); if every
    scheduler step line in that window shows #queue-req 0 it is a host stall -> drop every request received in it."""
    import gzip, re, datetime
    from steplog_parse import parse
    log = next(os.path.join(d, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(d, x)))
    t0 = min(v["recv"] for v in G.values()); fl = None
    for line in (gzip.open(log, "rt", errors="replace") if log.endswith(".gz") else open(log, errors="replace")):
        if "Prefill batch" in line or "Decode batch" in line:
            fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
    st = [(x["sec"] + fl - t0, x["queue"]) for x in parse(log) if x["kind"] in "PD"]
    slow = sorted(v["recv"] - t0 for v in G.values() if v["unc"] < 1000 and v["srv"] > 1.5)
    wins = []
    for i, x in enumerate(slow):
        c = [y for y in slow if x <= y <= x + 30]
        if len(c) >= 3:
            lo, hi = c[0] - 5, c[-1] + 5
            if all(q == 0 for ts, q in st if lo <= ts <= hi): wins.append((lo, hi))
    drop = {k for k, v in G.items() if any(lo <= v["recv"] - t0 <= hi for lo, hi in wins)}
    merged = []
    for lo, hi in sorted(wins):
        if merged and lo <= merged[-1][1]: merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else: merged.append((lo, hi))
    return drop, merged


def window_keys(G, keys, cut0, cut1):
    t0 = min(v["recv"] for v in G.values()); t1 = max(v["recv"] for v in G.values())
    return [k for k in keys if k[1] > 0 and cut0 <= G[k]["recv"] - t0 <= t1 - t0 - cut1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True); ap.add_argument("--gpu", required=True); ap.add_argument("--sim2"); ap.add_argument("--gpu2")
    ap.add_argument("--drop-stalls", action="store_true"); ap.add_argument("--cut0", type=float, default=300); ap.add_argument("--cut1", type=float, default=60)
    a = ap.parse_args()
    S1, G1 = load_sim(a.sim), load_gpu(a.gpu)
    keys = set(S1) & set(G1)
    if a.sim2:
        S2, G2 = load_sim(a.sim2), load_gpu(a.gpu2); keys &= set(S2) & set(G2)
    keys = window_keys(G1, sorted(keys), a.cut0, a.cut1)
    if a.sim2: keys = [k for k in keys if k in set(window_keys(G2, keys, a.cut0, a.cut1))]
    if a.drop_stalls:
        drop = set()
        for d, G in ((a.gpu, G1),) + (((a.gpu2, G2),) if a.sim2 else ()):
            dk, w = stall_keys(d, G); drop |= dk
            print(f"stall windows {os.path.basename(d.rstrip('/'))}: " + ", ".join(f"{lo/60:.1f}-{hi/60:.1f}min" for lo, hi in w))
        n0 = len(keys); keys = [k for k in keys if k not in drop]; print(f"dropped {n0-len(keys)} requests inside host-stall windows")
    cols = [("sim A", S1), ("GPU A", G1)] + ([("sim B", S2), ("GPU B", G2)] if a.sim2 else [])
    M = {n: metrics(X, keys) for n, X in cols}
    print(f"common warm turns in GPU window: n={len(keys)}")
    hdr = list(M["sim A"].keys())
    print(f"{'metric':<11}" + "".join(f"{n:>10}" for n, _ in cols) + ("   Δsim(B-A)  ΔGPU(B-A)" if a.sim2 else ""))
    for h in hdr:
        row = f"{h:<11}" + "".join(f"{M[n][h]:>10.3f}" if isinstance(M[n][h], float) and M[n][h] < 100 else f"{M[n][h]:>10.0f}" for n, _ in cols)
        if a.sim2:
            def d(x, y): return f"{y-x:+.2f}pp" if h == "miss" else (f"{int(y-x):+d}" if h == "n16k" else f"{100*(y/x-1):+.0f}%")
            row += f"   {d(M['sim A'][h], M['sim B'][h]):>9}  {d(M['GPU A'][h], M['GPU B'][h]):>9}"
        print(row)
    for tag, S, G in (("A", S1, G1),) + ((("B", S2, G2),) if a.sim2 else ()):
        us = np.array([S[k]["unc"] for k in keys]); ug = np.array([G[k]["unc"] for k in keys])
        big_s = us > 4000; big_g = ug > 4000
        print(f"per-request U agreement {tag}: |Δ|<=256 {100*np.mean(np.abs(us-ug)<=256):.1f}%  corr {np.corrcoef(us, ug)[0,1]:.3f}  U>4K both/sim-only/GPU-only {int((big_s&big_g).sum())}/{int((big_s&~big_g).sum())}/{int((~big_s&big_g).sum())}")


if __name__ == "__main__":
    main()
