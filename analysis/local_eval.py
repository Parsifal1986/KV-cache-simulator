"""Per-arm GPU vs sim (mean±sd over replicates) on the common warm turns. usage: local_eval.py MANIFEST SIMROOT [FILTER]"""
import sys, os, glob, re, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from sim_vs_gpu_pair import load_sim, load_gpu, window_keys, stall_keys
from des_noise_band import metrics
man, root = sys.argv[1], sys.argv[2]; flt = sys.argv[3] if len(sys.argv) > 3 else ""
H = ["miss", "U_p90", "U_p99", "U_CVaR90", "sTTFT_p50", "sTTFT_p80", "sTTFT_p90", "sTTFT_p99"]
print(f"{'arm':<32} {'n':>5} | " + " | ".join(f"{h:>18}" for h in H))
for line in open(man):
    f = line.rstrip("\n").split("\t"); tag, d, arm = f[0], f[1], f[8]
    if flt and not re.search(flt, tag): continue
    files = sorted(glob.glob(f"{root}/{tag}_n*/req_{arm}.csv"))
    if not files: continue
    G = load_gpu(d); S = [load_sim(x) for x in files]
    keys = set(G)
    for X in S: keys &= set(X)
    keys = window_keys(G, sorted(keys), 300, 60); drop = stall_keys(d, G)[0]; keys = [k for k in keys if k not in drop]
    g = metrics(G, keys); s = [metrics(X, keys) for X in S]
    cells = []
    for h in H:
        v = np.array([m[h] for m in s]); fmt = "{:.2f}" if h == "miss" else ("{:.3f}" if "TTFT" in h else "{:.0f}")
        rel = "" if h == "miss" else f"{100*(v.mean()/g[h]-1):+.0f}%"
        cells.append(f"{fmt.format(g[h])}/{fmt.format(v.mean())} {('%+.1fpp' % (v.mean()-g[h])) if h=='miss' else rel}")
    print(f"{tag:<32} {len(keys):>5} | " + " | ".join(f"{c:>18}" for c in cells))
