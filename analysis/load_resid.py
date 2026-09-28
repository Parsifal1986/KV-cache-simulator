"""Within-run regression of step-interval residual on the host load at that moment (cotenant.log snapshots)."""
import sys, re, gzip, datetime, numpy as np
man, npz = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(man)]
d = np.load(npz); y, p, src, t, keep, chain = (d[k] for k in ("y", "pred", "src", "tsec", "keep", "is_chain"))
X, Y, C, G = [], [], [], []
for i, r in enumerate(rows):
    L = r[1] + "/server.log.gz"
    for line in gzip.open(L, "rt", errors="replace"):
        if "Prefill batch" in line or "Decode batch" in line: t0 = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
    snaps = [(datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").timestamp(), float(b)) for a, b in re.findall(r"== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) load=([\d.]+)", open(r[1] + "/cotenant.log", errors="replace").read())]
    if len(snaps) < 5: continue
    st, sl = np.array(snaps).T
    for isch in (False, True):
        m = (src == i) & keep & (chain == isch) & (t > 300)
        if m.sum() < 30: continue
        ld = np.interp(t0 + t[m], st, sl); r_ = np.log(y[m] / p[m])
        X.append(ld - ld.mean()); Y.append(r_ - r_.mean()); C.append(np.full(m.sum(), isch)); G.append(np.full(m.sum(), i))
X, Y, C = np.concatenate(X), np.concatenate(Y), np.concatenate(C)
for name, m in (("decode intervals", ~C), ("prefill chunk chains", C)):
    b = np.polyfit(X[m], Y[m], 1)[0]; bins = np.percentile(X[m], [0, 20, 40, 60, 80, 100])
    prof = [f"{np.median(X[m][(X[m]>=a)&(X[m]<c)]):+.0f}:{100*np.median(Y[m][(X[m]>=a)&(X[m]<c)]):+.1f}%" for a, c in zip(bins[:-1], bins[1:])]
    print(f"{name}: n={m.sum()} within-run slope {100*b:.4f} %/load-unit  (x100 load = {100*b*100:+.1f}%)  binned (load dev: resid) {prof}")
