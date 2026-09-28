import sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from sim_vs_gpu_pair import load_sim, load_gpu, window_keys, stall_keys
sim, gpu = sys.argv[1], sys.argv[2]
S = load_sim(sim); G = load_gpu(gpu); keys = window_keys(G, sorted(set(S) & set(G)), 300, 60)
drop, _ = stall_keys(gpu, G); keys = [k for k in keys if k not in drop]
print(f"n={len(keys)}   bins by GPU-side U | sTTFT p50/p90 sim vs GPU | queue>50ms share sim vs GPU | own-prefill p50 (srv-queue)")
for lo, hi in ((0, 200), (200, 1000), (1000, 4000), (4000, 8000), (8000, 16000), (16000, 1e9)):
    k = [x for x in keys if lo <= G[x]["unc"] < hi]
    if len(k) < 5: continue
    s = np.array([S[x]["srv"] for x in k]); g = np.array([G[x]["srv"] for x in k]); qs = np.array([S[x]["queue"] for x in k]); qg = np.array([G[x]["queue"] for x in k])
    print(f"U {lo:>6.0f}-{hi:<6.0f} n={len(k):4d} | {np.percentile(s,50):.3f}/{np.percentile(s,90):.3f} vs {np.percentile(g,50):.3f}/{np.percentile(g,90):.3f} | {100*(qs>0.05).mean():4.1f}% vs {100*(qg>0.05).mean():4.1f}% | {np.percentile(s-qs,50):.3f} vs {np.percentile(g-qg,50):.3f}")
