import sys, os, csv, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from gpu_windows import requests, breakdown
from steplog_parse import parse
import gzip, re, datetime
simdir, gpudir, arm, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); KV = 750000
log = next(os.path.join(gpudir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(gpudir, x)))
G = breakdown(requests(log)); t0 = min(v["recv"] for v in G.values())
fl = None
for line in gzip.open(log, "rt", errors="replace"):
    if "Prefill batch" in line or "Decode batch" in line:
        fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
st = [x for x in parse(log) if x["kind"] in "PD"]; off = fl - t0
gs = pd.DataFrame([dict(m=int((x["sec"] + off) // 60), use=x["usage"] * KV, q=x["queue"], run=x["running"]) for x in st])
gr = pd.DataFrame([dict(m=int((v["recv"] - t0) // 60), s=v["srv"], q=v["queue"], u=v["unc"], warm=k[1] > 0) for k, v in G.items()])
ss = pd.read_csv(f"{simdir}/steps_{arm}.csv"); ss["m"] = (ss.t // 60).astype(int)
sr = pd.read_csv(f"{simdir}/req_{arm}.csv"); s0 = sr.arrival.min(); sr["m"] = ((sr.arrival - s0) // 60).astype(int); sr["q"] = sr.first_sched - sr.arrival; sr["u"] = sr.prompt_len - sr.cached
print(f"{'min':>4} | {'GPU inuse max':>13} {'qmax':>4} {'run':>4} {'sTTFT99':>7} {'q99':>5} {'U>4K':>4} | {'SIM infl max':>12} {'wait':>4} {'full%':>5} {'run':>4} {'sTTFT99':>7} {'q99':>5} {'U>4K':>4}")
for m in range(lo, hi):
    a = gs[gs.m == m]; b = gr[gr.m == m]; c = ss[ss.m == m]; d = sr[sr.m == m]
    if not len(b) or not len(d): continue
    print(f"{m:>4} | {a.use.max()/1e3:>12.0f}K {a.q.max():>4} {a.run.median():>4.0f} {np.percentile(b.s,99):>7.2f} {np.percentile(b.q,99):>5.2f} {int((b.u>4000).sum()):>4} | {c.inflight.max()/1e3:>11.0f}K {c.waiting.max():>4} {100*c.full.mean():>5.0f} {c.running.median():>4.0f} {np.percentile(d.ttft,99):>7.2f} {np.percentile(d.q,99):>5.2f} {int((d.u>4000).sum()):>4}")
