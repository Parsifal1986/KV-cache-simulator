#!/usr/bin/env python3
"""Is the GPU inside the simulator's noise band? For one working point (two GPU arms A/B run on the same machine) and
an ensemble of des_sim.py runs per arm (different --noise-seed), compute each metric on the common warm turns of the
GPU window (first cut0 / last cut1 s dropped, host-stall windows dropped), then report per metric:
GPU value, simulated mean ± sd [min, max], z = (GPU − mean) / sd — for arm A, arm B and the B − A delta
(all cross pairs A_i x B_j: the GPU arms are independent runs).

usage: des_noise_band.py --gpu-a DIR --gpu-b DIR --sim-a 'glob/req_A.csv' --sim-b 'glob/req_B.csv' [--label NAME] [--json OUT]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_vs_gpu_pair import load_sim, load_gpu, window_keys, stall_keys
from gpu_windows import tm

METRICS = ["miss", "U_p90", "U_p99", "U_TM99", "U_CVaR90", "n16k", "sTTFT_p50", "sTTFT_p80", "sTTFT_p90", "sTTFT_p99"]


def metrics(X, keys):
    pt = np.array([X[k]["pt"] for k in keys]); u = np.array([X[k]["unc"] for k in keys]); s = np.array([X[k]["srv"] for k in keys])
    return dict(miss=100 * u.sum() / pt.sum(), U_p90=np.percentile(u, 90), U_p99=np.percentile(u, 99), U_TM99=tm(u, 99), U_CVaR90=tm(u, 90),
                n16k=float((u > 16000).sum()), sTTFT_p50=np.percentile(s, 50), sTTFT_p80=np.percentile(s, 80), sTTFT_p90=np.percentile(s, 90), sTTFT_p99=np.percentile(s, 99))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-a", required=True); ap.add_argument("--gpu-b", required=True); ap.add_argument("--sim-a", required=True); ap.add_argument("--sim-b", required=True)
    ap.add_argument("--cut0", type=float, default=300); ap.add_argument("--cut1", type=float, default=60); ap.add_argument("--label", default=""); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    GA, GB = load_gpu(a.gpu_a), load_gpu(a.gpu_b)
    fa, fb = sorted(glob.glob(a.sim_a)), sorted(glob.glob(a.sim_b))
    SA, SB = [load_sim(f) for f in fa], [load_sim(f) for f in fb]
    keys = set(GA) & set(GB)
    for X in SA + SB: keys &= set(X)
    keys = window_keys(GA, sorted(keys), a.cut0, a.cut1); keys = [k for k in keys if k in set(window_keys(GB, keys, a.cut0, a.cut1))]
    drop = stall_keys(a.gpu_a, GA)[0] | stall_keys(a.gpu_b, GB)[0]; keys = [k for k in keys if k not in drop]
    mA, mB = metrics(GA, keys), metrics(GB, keys)
    sA, sB = [metrics(X, keys) for X in SA], [metrics(X, keys) for X in SB]
    n = min(len(sA), len(sB)); out = dict(label=a.label, n_keys=len(keys), n_rep=n, rows={})
    print(f"{a.label}: {len(fa)}/{len(fb)} sim replicates, common warm turns n={len(keys)}")
    print(f"{'metric':<11}| {'GPU A':>8} {'sim A mean±sd':>17} {'z':>5} | {'GPU B':>8} {'sim B mean±sd':>17} {'z':>5} | {'GPU Δ':>8} {'sim Δ mean±sd':>17} {'z':>5}")
    for h in METRICS:
        a_ = np.array([m[h] for m in sA]); b_ = np.array([m[h] for m in sB])
        rel = h != "miss" and h != "n16k"
        # GPU arms are independent runs -> judge the delta against ALL cross pairs (A_i, B_j), not same-seed pairs
        # (same-seed pairs share one noise stream and understate the delta's spread)
        A2, B2 = np.meshgrid(a_, b_, indexing="ij"); A2, B2 = A2.ravel(), B2.ravel()
        d = (100 * (B2 / A2 - 1)) if rel else (B2 - A2); gd = (100 * (mB[h] / mA[h] - 1)) if rel else (mB[h] - mA[h])
        def z(g, x): return (g - x.mean()) / x.std() if x.std() > 0 else float("nan")
        f = (lambda v: f"{v:8.3f}") if ("TTFT" in h or h == "miss") else (lambda v: f"{v:8.0f}")
        unit = "%" if rel else ("pp" if h == "miss" else "")
        print(f"{h:<11}| {f(mA[h])} {f(a_.mean())}±{f(a_.std()).strip():<7} {z(mA[h], a_):+5.1f} | {f(mB[h])} {f(b_.mean())}±{f(b_.std()).strip():<7} {z(mB[h], b_):+5.1f} | {gd:+7.1f}{unit:<2}{d.mean():+8.1f}±{d.std():<6.1f} {z(gd, d):+5.1f}")
        out["rows"][h] = dict(gpu_a=mA[h], sim_a=a_.tolist(), gpu_b=mB[h], sim_b=b_.tolist(), gpu_d=gd, sim_d=d.tolist())
    if a.json: json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
