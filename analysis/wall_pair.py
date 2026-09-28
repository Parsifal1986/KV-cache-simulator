"""Paired per-request comparison in a GPU time window: real GPU queue_time (meta_info) vs sim queue, uncached, TTFT.
usage: wall_pair.py GPU_ARM_DIR 'SIM_GLOB/req_*.csv' LO_MIN HI_MIN"""
import sys, glob, gzip, re, json, numpy as np, pandas as pd
gdir, sglob, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
dec = json.JSONDecoder(); G = {}
for line in gzip.open(gdir + "/server.log.gz", "rt", errors="replace"):
    if "request.finished" not in line: continue
    m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
    if not m: continue
    k = line.find("{", line.rfind('"meta_info"')); meta, _ = dec.raw_decode(line, k)
    if meta.get("cached_tokens") is None: continue
    G[(m.group(1), int(m.group(2)))] = dict(recv=meta["request_received_ts"], q=meta["queue_time"] or 0.0, srv=meta["response_sent_to_client_ts"] - meta["request_received_ts"],
                                            unc=meta["prompt_tokens"] - meta["cached_tokens"], pt=meta["prompt_tokens"], out=meta["completion_tokens"])
t0 = min(v["recv"] for v in G.values())
keys = [k for k, v in G.items() if lo * 60 <= v["recv"] - t0 < hi * 60 and k[1] > 0]
g = pd.DataFrame([dict(key=k, **G[k]) for k in keys])
print(f"GPU {lo:.0f}-{hi:.0f} min: n={len(g)} miss {100*g.unc.sum()/g.pt.sum():.2f}%  queued>0.2s {int((g.q>0.2).sum())}  queue sum {g.q.sum():.0f}s p90 {g.q.quantile(.9):.2f} p99 {g.q.quantile(.99):.2f}  srv p50/p90/p99 {g.srv.quantile(.5):.2f}/{g.srv.quantile(.9):.2f}/{g.srv.quantile(.99):.2f}")
rows = []
for f in sorted(glob.glob(sglob)):
    s = pd.read_csv(f); s["key"] = list(zip(s.program, s.turn)); s = s.set_index("key"); ks = [k for k in keys if k in s.index]; x = s.loc[ks]
    q = x.first_sched - x.arrival; u = x.prompt_len - x.cached; gg = g.set_index("key").loc[ks]
    rows.append(dict(n=len(ks), miss=100 * u.sum() / x.prompt_len.sum(), gmiss=100 * gg.unc.sum() / gg.pt.sum(), queued=int((q > 0.2).sum()), qsum=q.sum(), q90=q.quantile(.9), q99=q.quantile(.99),
                     t50=x.ttft.quantile(.5), t90=x.ttft.quantile(.9), t99=x.ttft.quantile(.99)))
R = pd.DataFrame(rows)
print(f"SIM x{len(R)}: miss {R.miss.mean():.2f}±{R.miss.std():.2f} (GPU same keys {R.gmiss.mean():.2f})  queued>0.2s {R.queued.mean():.0f}±{R.queued.std():.0f}  queue sum {R.qsum.mean():.0f}s  p90 {R.q90.mean():.2f} p99 {R.q99.mean():.2f}  ttft p50/p90/p99 {R.t50.mean():.2f}/{R.t90.mean():.2f}/{R.t99.mean():.2f}")
