import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from sim_vs_gpu_pair import load_gpu
sim, gpu, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
sr = pd.read_csv(sim); G = load_gpu(gpu); t0 = min(v["recv"] for v in G.values())
sr = sr.sort_values(["program", "turn"]); sr["gap"] = sr.arrival - (sr.arrival + sr.e2e).groupby(sr.program).shift()
sr["key"] = list(zip(sr.program, sr.turn)); sr = sr[sr.key.isin(G.keys()) & (sr.turn > 0)].copy()
sr["tg"] = [(G[k]["recv"] - t0) / 60 for k in sr.key]; sr = sr[(sr.tg >= lo) & (sr.tg < hi)]
sr["us"] = sr.prompt_len - sr.cached; sr["ug"] = [G[k]["unc"] for k in sr.key]; sr["qs"] = sr.first_sched - sr.arrival; sr["qg"] = [G[k]["queue"] for k in sr.key]
print(f"GPU window {lo:.0f}-{hi:.0f} min n={len(sr)}  uncached sim {sr.us.sum()/1e3:.0f}K  gpu {sr.ug.sum()/1e3:.0f}K  extra {(sr.us.sum()-sr.ug.sum())/1e3:+.0f}K")
def tab(name, col, bins):
    print(f"  by {name}:")
    for a, b in bins:
        g = sr[(sr[col] >= a) & (sr[col] < b)]
        if len(g): print(f"    {a:>6}-{b:<6} n={len(g):4d} sim {g.us.sum()/1e3:6.0f}K gpu {g.ug.sum()/1e3:6.0f}K extra {(g.us.sum()-g.ug.sum())/1e3:+6.0f}K | whole-history losses (U>80% prompt) sim {int((g.us>0.8*g.prompt_len).sum())} gpu {int((g.ug>0.8*g.prompt_len).sum())}")
tab("tool gap before turn (s)", "gap", ((0, 2), (2, 5), (5, 15), (15, 60), (60, 1e9)))
tab("sim queue wait (s)", "qs", ((0, 0.2), (0.2, 1), (1, 5), (5, 1e9)))
tab("GPU queue wait (s)", "qg", ((0, 0.2), (0.2, 1), (1, 5), (5, 1e9)))
