#!/usr/bin/env python3
"""Same summary + 5-minute window metrics as des_sim.summarize(), computed from a GPU arm's server.log, so simulator
and GPU can be compared window by window. Per-request: warm turns, U = prompt - cached, sTTFT = response_sent -
received (streamed first token), queue = reconstructed as in cloud/tools/clean_pair.py. Per-window server state from
the step lines (steplog_parse.py): running p50, #queue-req max, in-use tokens (usage * pool) p50/max.

usage: gpu_windows.py ARM_DIR [ARM_DIR ...] --kv 750000 [--cut0 300 --cut1 60] [--json OUT]
"""
import argparse, gzip, json, math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steplog_parse import parse as parse_steps

CHUNK = 8192; GAP = 0.005; dec = json.JSONDecoder()


def requests(path):
    fh = gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace"); out = {}
    for line in fh:
        if "request.finished" not in line: continue
        m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
        if not m: continue
        k = line.find("{", line.rfind('"meta_info"'))
        try: meta, _ = dec.raw_decode(line, k)
        except Exception: continue
        if meta.get("cached_tokens") is None: continue
        out[(m.group(1), int(m.group(2)))] = (meta["request_received_ts"], meta["api_server_dispatch_finish_ts"], meta["response_sent_to_client_ts"], meta["prompt_tokens"], meta["prompt_tokens"] - meta["cached_tokens"], meta.get("request_finished_ts"))
    return out


def breakdown(ts):
    items = sorted((v[2], k, v[0], v[1], v[4]) for k, v in ts.items()); ends = []; bidx = {}
    for sent, k, *_ in items:
        if not ends or sent - ends[-1] > GAP: ends.append(sent)
        else: ends[-1] = sent
        bidx[k] = len(ends) - 1
    res = {}
    for sent, k, recv, disp, unc in items:
        n = max(1, math.ceil(unc / CHUNK)); b = bidx[k]; start = ends[b - n] if b - n >= 0 else -math.inf
        res[k] = dict(srv=sent - recv, queue=max(0.0, start - disp), unc=unc, pt=ts[k][3], recv=recv, e2e=(ts[k][5] or sent) - recv)
    return res


def tm(x, p): x = np.sort(np.asarray(x)); return float(x[int(len(x) * p / 100):].mean()) if len(x) else float("nan")


def arm_metrics(arm_dir, kv, cut0, cut1, win=300.0):
    log = next(os.path.join(arm_dir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm_dir, x)))
    R = breakdown(requests(log)); t0 = min(v["recv"] for v in R.values()); t1 = max(v["recv"] for v in R.values())
    sel = [v for k, v in R.items() if k[1] > 0 and cut0 <= v["recv"] - t0 <= t1 - t0 - cut1]
    pt = np.array([v["pt"] for v in sel], float); u = np.array([v["unc"] for v in sel], float); s = np.array([v["srv"] for v in sel]); q = np.array([v["queue"] for v in sel]); e2e = np.array([v["e2e"] for v in sel])
    steps = parse_steps(log); first_line = None
    # align step-line clock (whole seconds since first step line) to request clock via the first request received
    import datetime
    for line in (gzip.open(log, "rt", errors="replace") if log.endswith(".gz") else open(log, errors="replace")):
        m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if m and ("Prefill batch" in line or "Decode batch" in line):
            first_line = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(); break
    off = first_line - t0 if first_line else 0.0
    D = [x for x in steps if x["kind"] in ("P", "D")]
    ts = np.array([x["sec"] + off for x in D]); use = np.array([x["usage"] * kv for x in D]); run = np.array([x["running"] for x in D]); qq = np.array([x["queue"] for x in D])
    dstep = np.array([x["step_ms"] if x["kind"] == "D" and x.get("n_prefill_between") == 0 and x.get("step_ms") else np.nan for x in D])
    mid = (ts > cut0) & (ts < t1 - t0 - cut1)
    out = dict(n=len(sel), makespan_min=(max(v["recv"] + v["e2e"] for v in R.values()) - t0) / 60, miss=100 * u.sum() / pt.sum(), U_p90=np.percentile(u, 90), U_p99=np.percentile(u, 99), U_TM99=tm(u, 99), U_CVaR95=tm(u, 95),
               n16k=int((u > 16000).sum()), sTTFT_p50=np.percentile(s, 50), sTTFT_p90=np.percentile(s, 90), sTTFT_p99=np.percentile(s, 99), sTTFT_TM99=tm(s, 99),
               queue_p99=np.percentile(q, 99), queue_share=100 * q.sum() / s.sum(), queued50=100 * (q > 0.05).mean(), dequeued_p99=np.percentile(s[q <= 0.05], 99),
               e2e_p50=np.percentile(e2e, 50), inuse_max=float(use.max()), inuse_p90=float(np.percentile(use[mid], 90)), waiting_max=int(qq.max()),
               decode_step_ms=float(np.nanmedian(dstep[mid])), lines_queue_gt0=100 * float((qq[mid] > 0).mean()))
    wins = []
    Rl = list(R.items())
    for w0 in np.arange(0, t1 - t0 + 1, win):
        m = [v for k, v in Rl if k[1] > 0 and w0 <= v["recv"] - t0 < w0 + win]
        if len(m) < 5: continue
        pw = np.array([v["pt"] for v in m], float); uw = np.array([v["unc"] for v in m], float); sw = np.array([v["srv"] for v in m]); qw = np.array([v["queue"] for v in m])
        mm = (ts >= w0) & (ts < w0 + win)
        wins.append(dict(w=int(w0 // 60), n=len(m), miss=100 * uw.sum() / pw.sum(), U_p99=float(np.percentile(uw, 99)), sTTFT_p99=float(np.percentile(sw, 99)), queue_p99=float(np.percentile(qw, 99)),
                         inuse_p50=float(np.percentile(use[mm], 50)) if mm.any() else 0, inuse_max=float(use[mm].max()) if mm.any() else 0, running_p50=float(np.percentile(run[mm], 50)) if mm.any() else 0,
                         waiting_max=int(qq[mm].max()) if mm.any() else 0, lines_queue=100 * float((qq[mm] > 0).mean()) if mm.any() else 0,
                         dstep=float(np.nanmedian(dstep[mm])) if mm.any() and np.isfinite(dstep[mm]).any() else float("nan")))
    return out, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+"); ap.add_argument("--kv", type=int, required=True); ap.add_argument("--cut0", type=float, default=300); ap.add_argument("--cut1", type=float, default=60); ap.add_argument("--json", default=None)
    a = ap.parse_args(); res = {}
    for d in a.arms:
        s, wins = arm_metrics(d, a.kv, a.cut0, a.cut1); res[d] = dict(summary=s, windows=wins)
        print(f"[{d}] makespan {s['makespan_min']:.1f} min n={s['n']} | miss {s['miss']:.2f}% U p90/p99/TM99 {s['U_p90']:.0f}/{s['U_p99']:.0f}/{s['U_TM99']:.0f} n>16k {s['n16k']} | sTTFT p50/p90/p99/TM99 {s['sTTFT_p50']:.3f}/{s['sTTFT_p90']:.3f}/{s['sTTFT_p99']:.2f}/{s['sTTFT_TM99']:.2f}s queue p99 {s['queue_p99']:.2f}s share {s['queue_share']:.0f}% dq p99 {s['dequeued_p99']:.2f} | in-use max/p90 {s['inuse_max']/1e3:.0f}/{s['inuse_p90']/1e3:.0f}K queue-lines {s['lines_queue_gt0']:.1f}% wait max {s['waiting_max']} | decode {s['decode_step_ms']:.0f}ms e2e p50 {s['e2e_p50']:.1f}s")
        for w in wins: print(f"    {w['w']:3d}-{w['w']+5:3d} n={w['n']:3d} miss {w['miss']:5.2f}% U p99 {w['U_p99']:6.0f} sTTFT p99 {w['sTTFT_p99']:5.2f}s q99 {w['queue_p99']:5.2f}s | in-use p50/max {w['inuse_p50']/1e3:4.0f}/{w['inuse_max']/1e3:4.0f}K running {w['running_p50']:.0f} queue max {w['waiting_max']} queue-lines {w['lines_queue']:.0f}% dstep {w['dstep']:.0f}ms")
    if a.json: json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
