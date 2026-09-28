#!/usr/bin/env python3
"""Wall dynamics, window by window: GPU 5-minute windows (gpu_windows.py --json) next to the des_sim / sgl_sim windows of one
ladder point, mean ± sd over noise seeds. Shows WHEN a wall point diverges (onset, depth, recovery) rather than only the total.

usage: wall_windows.py GPU_JSON POINT_DIR [POINT_DIR ...] --arms rrdeclband0.001,rrdeclchunk1024 [--key miss]
  GPU_JSON arms are matched to an arm name by the last path component of their directory."""
import argparse, glob, json, os, re
import numpy as np


def sim_windows(point, key):
    """-> {(method, arm): [[window values] per seed]}"""
    out = {}
    for f in glob.glob(os.path.join(point, "des_n*", "summary.json")):
        for arm, v in json.load(open(f))["arms"].items():
            out.setdefault(("DES", arm), []).append([w[key] for w in v["windows"]])
    for f in glob.glob(os.path.join(point, "sgl_n*", "summary_*.json")):
        v = json.load(open(f)); out.setdefault(("SGL", v["arm"]), []).append([w[key] for w in v["windows"]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpu_json"); ap.add_argument("points", nargs="+"); ap.add_argument("--arms", required=True); ap.add_argument("--key", default="miss")
    a = ap.parse_args(); arms = a.arms.split(",")
    G = {os.path.basename(d.rstrip("/")): [w[a.key] for w in v["windows"]] for d, v in json.load(open(a.gpu_json)).items()}
    for arm in arms:
        g = G.get(arm, [])
        print(f"== {arm}   ({a.key} per 5-min window)")
        print("   " + "".join(f"{f'{5 * i}-{5 * i + 5}':>13}" for i in range(max(len(g), 8))))
        print("GPU" + "".join(f"{x:13.2f}" for x in g))
        for p in a.points:
            tok = open(os.path.join(p, "token_arm.txt")).read().split()[0] if os.path.exists(os.path.join(p, "token_arm.txt")) else "?"
            for (meth, ar), runs in sorted(sim_windows(p, a.key).items()):
                if ar != arm: continue
                n = max(map(len, runs)); M = np.full((len(runs), n), np.nan)
                for i, r in enumerate(runs): M[i, :len(r)] = r
                mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
                print(f"{meth} tok={tok} ×{len(runs)}".ljust(3) + "\n   " + "".join(f"{m:8.2f}±{s:4.1f}" for m, s in zip(mu, sd)))
        print()


if __name__ == "__main__":
    main()
