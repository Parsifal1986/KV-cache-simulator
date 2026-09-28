"""Per sim replicate: arrival drift entering the wall (median sim-GPU arrival offset of turns the GPU received in [D0,D1) min)
vs miss on the common keys of the wall window [W0,W1) min and of the whole run. usage: gain.py GPU_ARM 'SIMGLOB...' (several globs ok)"""
import sys, glob, gzip, re, json, numpy as np, pandas as pd
gdir = sys.argv[1]; files = sorted(sum((glob.glob(x) for x in sys.argv[2:]), []))
D0, D1, W0, W1 = 27, 31, 31, 38
dec = json.JSONDecoder(); G = {}
for line in gzip.open(gdir + "/server.log.gz", "rt", errors="replace"):
    if "request.finished" not in line: continue
    m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
    if not m: continue
    k = line.find("{", line.rfind('"meta_info"')); meta, _ = dec.raw_decode(line, k)
    if meta.get("cached_tokens") is None: continue
    G[(m.group(1), int(m.group(2)))] = (meta["request_received_ts"], meta["prompt_tokens"], meta["prompt_tokens"] - meta["cached_tokens"])
t0 = min(v[0] for v in G.values())
S = []
for f in files:
    s = pd.read_csv(f); s0 = s.arrival.min(); S.append((f, {(r.program, r.turn): (r.arrival - s0, r.prompt_len, r.prompt_len - r.cached) for r in s.itertuples()}))
keys = set(k for k in G if k[1] > 0 and 300 <= G[k][0] - t0)
for _, d in S: keys &= set(d)
wk = [k for k in keys if W0 * 60 <= G[k][0] - t0 < W1 * 60]; dk = [k for k in keys if D0 * 60 <= G[k][0] - t0 < D1 * 60]
gw = 100 * sum(G[k][2] for k in wk) / sum(G[k][1] for k in wk); ga = 100 * sum(G[k][2] for k in keys) / sum(G[k][1] for k in keys)
print(f"GPU: wall {W0}-{W1} min miss {gw:.2f}% (n={len(wk)}), all {ga:.2f}%")
X = []
for f, d in S:
    dr = np.median([d[k][0] - (G[k][0] - t0) for k in dk]); mw = 100 * sum(d[k][2] for k in wk) / sum(d[k][1] for k in wk); ma = 100 * sum(d[k][2] for k in keys) / sum(d[k][1] for k in keys)
    X.append((dr, mw, ma)); print(f"  {f.split('/')[-3]}/{f.split('/')[-2]:<10} drift {dr:+6.1f}s  wall miss {mw:6.2f}%  all {ma:5.2f}%")
X = np.array(X)
if len(X) > 2:
    b = np.polyfit(X[:, 0], X[:, 1], 1); c = np.polyfit(X[:, 0], X[:, 2], 1); r = np.corrcoef(X[:, 0], X[:, 2])[0, 1]
    print(f"fit: wall miss = {b[1]:.2f} {b[0]:+.3f}*drift  |  all miss = {c[1]:.2f} {c[0]:+.4f}*drift  (r={r:+.2f}); at drift 0: all {c[1]:.2f} vs GPU {ga:.2f}")
