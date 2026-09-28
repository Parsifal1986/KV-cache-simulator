"""Pure-decode intervals (no prefill inside, bs<=32 both ends) of mldev logs: residual vs pooled model, against candidate features."""
import sys, gzip, datetime, json, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from steplog_parse import parse
from gpu_windows import requests as reqs
P = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "profiles", "machines", "pod0920.json")))
out = []
for L in sys.argv[1:]:
    st = parse(L); R = reqs(L); t0 = min(v[0] for v in R.values())
    for line in gzip.open(L, "rt", errors="replace"):
        if "Prefill batch" in line or "Decode batch" in line: fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
    iv = np.array([(v[2] - t0, (v[5] or v[2]) - t0, v[3]) for v in R.values()]); off = fl - t0
    nP = 0
    for s in st:
        if s["kind"] == "P": nP += 1; continue
        if s["kind"] != "D": continue
        ok = s.get("gap") and s["prev_running"] is not None and 60 < s["sec"] and s["gap"] < 20 and nP == 0 and max(s["running"], s["prev_running"]) <= 32
        nP = 0
        if not ok: continue
        tt = s["sec"] + off - 0.5; m = (iv[:, 0] <= tt) & (iv[:, 1] >= tt); c = np.sort(iv[m, 2])[::-1] / 1e3
        if len(c) == 0: continue
        bs = 0.5 * (s["running"] + s["prev_running"]); K = s["tokens"] / 1e3
        pred = P["d0"] + P["d_tok"] * K + P["d_req"] * bs + P["d_max"] * c[0]
        out.append(dict(t=s["sec"] / 60, y=1e3 * s["gap"] / 40, pred=pred, bs=bs, K=K, c1=c[0], c2=c[1] if len(c) > 1 else 0, n32=(c > 32).sum(), n64=(c > 64).sum(), sq=(c ** 2).sum() / 1e3, nreq=len(c), log=L[-40:]))
import pandas as pd
D = pd.DataFrame(out); D["r"] = D.y - D.pred; D["c1bs"] = D.c1 * D.bs / 100; D["c1inv"] = D.c1 / D.bs; D["kc1"] = D.K / D.c1
print(f"n={len(D)} resid median {D.r.median():+.2f} ms MAD {np.median(np.abs(D.r - D.r.median())):.2f}")
for w in range(0, 45, 5):
    g = D[(D.t >= w) & (D.t < w + 5)]
    if len(g): print(f"  {w:2d}-{w+5:<2d} n={len(g):4d} resid {g.r.median():+5.2f} ms  y {g.y.median():5.1f}  c1 {g.c1.median():5.1f}K c2 {g.c2.median():5.1f}K n32 {g.n32.median():.0f} K {g.K.median():4.0f} bs {g.bs.median():4.1f} nreq(log) {g.nreq.median():.0f}")
X = np.c_[np.ones(len(D)), D[["K", "bs", "c1"]].values]
for extra in ([], ["c2"], ["c1bs"], ["c1inv"], ["kc1"], ["c2", "c1bs"]):
    Xe = np.c_[X, D[extra].values] if extra else X
    b, *_ = np.linalg.lstsq(Xe, D.y.values, rcond=None); r = D.y.values - Xe @ b
    byw = [np.median(r[(D.t >= w) & (D.t < w + 5)]) for w in (10, 15, 20, 25, 30)]; byl = [round(float(np.median(r[D.log == l])), 2) for l in sorted(D.log.unique())]
    print(f"  refit +{extra}: MAD {np.median(np.abs(r - np.median(r))):.3f}  coef {np.round(b, 4)}  resid 10/15/20/25/30min {np.round(byw, 2)} per-log {byl}")
print("per-minute (all logs pooled), 17-28 min:")
for w in range(17, 28):
    g = D[(D.t >= w) & (D.t < w + 1)]
    if len(g): print(f"  {w:2d} n={len(g):3d} resid {g.r.median():+5.2f} ms  y {g.y.median():5.1f} c1 {g.c1.median():5.1f}K c2 {g.c2.median():5.1f}K K {g.K.median():4.0f} bs {g.bs.median():4.1f}  per-log resid {[round(x,1) for x in g.groupby('log').r.median().values]}")
