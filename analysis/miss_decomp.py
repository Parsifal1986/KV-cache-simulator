import sys, csv, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from sim_vs_gpu_pair import load_gpu
simcsv, gpudir = sys.argv[1], sys.argv[2]
sr = pd.read_csv(simcsv); G = load_gpu(gpudir); t0 = min(v["recv"] for v in G.values())
sr["key"] = list(zip(sr.program, sr.turn)); sr = sr[sr.key.isin(G.keys())]
sr["u_sim"] = sr.prompt_len - sr.cached; sr["u_gpu"] = [G[k]["unc"] for k in sr.key]; sr["m"] = [int((G[k]["recv"] - t0) // 300) * 5 for k in sr.key]
# gap before this turn = arrival - previous turn finish, from the sim record (same recorded gap)
sr = sr.sort_values(["program", "turn"]); prev_fin = sr.groupby("program").apply(lambda g: (g.arrival + g.e2e).shift()).reset_index(level=0, drop=True)
sr["gap"] = sr.arrival - prev_fin
w = sr[sr.turn > 0]
print("window  n   miss sim/gpu   extra-unc sim-gpu (K)")
for m, g in w.groupby("m"): print(f"{m:3d}-{m+5:<3d} {len(g):4d}  {100*g.u_sim.sum()/g.prompt_len.sum():5.2f}/{100*g.u_gpu.sum()/g.prompt_len.sum():5.2f}   {(g.u_sim.sum()-g.u_gpu.sum())/1e3:+8.0f}K")
print("\nby tool gap before the turn (all windows): extra-uncached sim-gpu, and where sim loses >4K but gpu <1K")
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 30), (30, 100), (100, 1e9)):
    g = w[(w.gap >= lo) & (w.gap < hi)]
    print(f"gap {lo:>4}-{hi:<5} n={len(g):5d}  extra {(g.u_sim.sum()-g.u_gpu.sum())/1e3:+8.0f}K  sim-only big {int(((g.u_sim>4000)&(g.u_gpu<1000)).sum()):4d}  gpu-only big {int(((g.u_gpu>4000)&(g.u_sim<1000)).sum()):4d}")
