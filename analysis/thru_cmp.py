import sys, gzip, datetime, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from steplog_parse import parse
from gpu_windows import requests
simdir, gpudir, arm, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
log = f"{gpudir}/server.log.gz"; R = requests(log); t0 = min(v[0] for v in R.values())
for line in gzip.open(log, "rt", errors="replace"):
    if "Prefill batch" in line or "Decode batch" in line: fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
st = parse(log); off = fl - t0
P = pd.DataFrame([dict(m=int((x["sec"] + off) // 60), new=x["new_tok"]) for x in st if x["kind"] == "P"])
D = pd.DataFrame([dict(m=int((x["sec"] + off) // 60), gen=x["thr"] * (x["gap"] or 0), run=x["running"]) for x in st if x["kind"] == "D" and x.get("gap")])
fin = pd.Series([int((v[5] - t0) // 60) for v in R.values() if v[5]]).value_counts()
ss = pd.read_csv(f"{simdir}/steps_{arm}.csv"); ss["m"] = (ss.t // 60).astype(int)
sr = pd.read_csv(f"{simdir}/req_{arm}.csv"); s0 = sr.arrival.min(); sfin = ((sr.arrival + sr.e2e - s0) // 60).astype(int).value_counts()
print(f"{'min':>4} | GPU prefillK/min finishes | SIM prefillK/min finishes decode-ms prefill-share")
for m in range(lo, hi):
    a = P[P.m == m].new.sum() / 1e3; c = ss[(ss.m == m) & (ss.kind == "P")]; d = ss[(ss.m == m) & (ss.kind == "D")]
    dms = 1e3 * d.dur.sum() / max(d.new_or_k.sum(), 1)
    print(f"{m:>4} | {a:>8.0f}K {fin.get(m, 0):>6} | {c.new_or_k.sum()/1e3:>8.0f}K {sfin.get(m, 0):>6} {dms:>6.1f} {100*c.dur.sum()/60:>5.0f}%")
