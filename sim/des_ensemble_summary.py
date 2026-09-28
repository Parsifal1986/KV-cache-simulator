#!/usr/bin/env python3
"""Ensemble (jitter replicates) of des_sim.py vs a GPU arm pair: per metric mean±sd [min,max] for both arms and the RR-C+D minus LRU+D delta, on the common warm turns of the GPU window (stall-dropped).
usage: des_ensemble_summary.py ENS_DIR CONC GPU_ARM_LRUD GPU_ARM_RRCD   (ENS_DIR/c<CONC>_<j>/req_<arm>.csv)"""
import sys, os, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from sim_vs_gpu_pair import load_sim, load_gpu, window_keys, stall_keys, metrics
ens, conc, gA, gB = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
A, B = "rrdeclband0.001", "rrdeclchunk1024"
GA, GB = load_gpu(gA), load_gpu(gB)
reps = sorted({int(d.split("_")[-1]) for d in os.listdir(ens) if d.startswith(f"c{conc}_") and os.path.isdir(os.path.join(ens, d))})
SA = {j: load_sim(f"{ens}/c{conc}_{j}/req_{A}.csv") for j in reps}; SB = {j: load_sim(f"{ens}/c{conc}_{j}/req_{B}.csv") for j in reps}
keys = set(GA) & set(GB)
for j in reps: keys &= set(SA[j]) & set(SB[j])
keys = window_keys(GA, sorted(keys), 300, 60); keys = [k for k in keys if k in set(window_keys(GB, keys, 300, 60))]
drop = stall_keys(gA, GA)[0] | stall_keys(gB, GB)[0]; keys = [k for k in keys if k not in drop]
mGA, mGB = metrics(GA, keys), metrics(GB, keys)
MA = [metrics(SA[j], keys) for j in reps]; MB = [metrics(SB[j], keys) for j in reps]
print(f"c{conc}: {len(reps)} replicates, common warm turns n={len(keys)}")
print(f"{'metric':<11}{'sim LRU+D mean±sd [min,max]':>34}{'GPU':>9} |{'sim RR-C+D mean±sd [min,max]':>34}{'GPU':>9} |{'sim Δ mean±sd [min,max]':>30}{'GPU Δ':>9}")
for h in ("miss", "U_p90", "U_p99", "U_TM99", "n16k", "sTTFT_p90", "sTTFT_p99", "sTTFT_TM99", "queue_p99"):
    a = np.array([m[h] for m in MA], float); b = np.array([m[h] for m in MB], float)
    if h == "miss": d = b - a; gd = mGB[h] - mGA[h]; unit = "pp"
    elif h == "n16k": d = b - a; gd = mGB[h] - mGA[h]; unit = ""
    else: d = 100 * (b / a - 1); gd = 100 * (mGB[h] / mGA[h] - 1); unit = "%"
    f = (lambda x: f"{x:.2f}") if (h == "miss" or "TTFT" in h or "queue" in h) else (lambda x: f"{x:.0f}")
    print(f"{h:<11}{f(a.mean()):>10}±{f(a.std()):<6}[{f(a.min())},{f(a.max())}]".ljust(45) + f"{f(mGA[h]):>9} | " + f"{f(b.mean()):>8}±{f(b.std()):<6}[{f(b.min())},{f(b.max())}]".ljust(32) + f"{f(mGB[h]):>9} | " + f"{d.mean():+7.1f}±{d.std():.1f}{unit} [{d.min():+.1f},{d.max():+.1f}]".ljust(28) + f"{gd:+8.1f}{unit}")
