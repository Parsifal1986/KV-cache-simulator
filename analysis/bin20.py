"""20-s bins in a window: GPU step log vs sim steps (mean over sim replicates). usage: bin20.py GPU_ARM 'SIMGLOB' ARM LO_MIN HI_MIN [BIN_S]"""
import sys, os, glob, gzip, datetime, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from gpu_windows import requests
from steplog_parse import parse
gdir, sg, arm, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]); B = float(sys.argv[6]) if len(sys.argv) > 6 else 20
log = gdir + "/server.log.gz"; R = requests(log); t0 = min(v[0] for v in R.values())
for line in gzip.open(log, "rt", errors="replace"):
    if "Prefill batch" in line or "Decode batch" in line: fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
st = parse(log); off = fl - t0
P = pd.DataFrame([dict(t=x["sec"] + off, new=x["new_tok"], cached=x["cached"], q=x["queue"], run=x["running"], use=x["usage"]) for x in st if x["kind"] == "P"])
D = pd.DataFrame([dict(t=x["sec"] + off, q=x["queue"], run=x["running"], use=x["usage"], ms=x["step_ms"]) for x in st if x["kind"] == "D"])
S = []
for f in sorted(glob.glob(sg)):
    s = pd.read_csv(f); S.append(s)
print(f"{'t(min)':>7} | GPU: pre_new  cached qmax run  use   dec_ms | SIM: pre_new  cached wmax run  infl  dec_ms full%")
for b0 in np.arange(lo * 60, hi * 60, B):
    p = P[(P.t >= b0) & (P.t < b0 + B)]; d = D[(D.t >= b0) & (D.t < b0 + B)]
    sv = []
    for s in S:
        x = s[(s.t >= b0) & (s.t < b0 + B)]; xp = x[x.kind == "P"]; xd = x[x.kind == "D"]
        sv.append((xp.new_or_k.sum(), xp.cached.sum(), x.waiting.max(), x.running.median(), x.inflight.max(), 1e3 * (xd.dur.sum() / xd.new_or_k.sum()) if len(xd) else np.nan, 100 * x.full.mean()))
    sv = np.nanmean(np.array(sv, float), axis=0)
    gq = max(p.q.max() if len(p) else 0, d.q.max() if len(d) else 0)
    print(f"{b0/60:7.2f} | {p.new.sum()/1e3:9.0f}K {p.cached.sum()/1e3:6.0f}K {gq:4.0f} {d.run.median() if len(d) else np.nan:3.0f} {750*max(p.use.max() if len(p) else 0, d.use.max() if len(d) else 0):4.0f}K {d.ms.median() if len(d) else np.nan:6.1f} | {sv[0]/1e3:9.0f}K {sv[1]/1e3:6.0f}K {sv[2]:4.1f} {sv[3]:3.0f} {sv[4]/1e3:4.0f}K {sv[5]:6.1f} {sv[6]:5.0f}")
