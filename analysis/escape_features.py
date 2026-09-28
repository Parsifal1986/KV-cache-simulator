#!/usr/bin/env python3
"""What decides whether a run escapes a bistable wall wave? Per run (simulator req_*.csv or GPU arm): miss in the wave, and
state features measured BEFORE / AT the wave from per-request records only:
  w1        miss in the first wave window
  prog      turns completed by the wave start (schedule progress)
  done      programs finished by the wave start
  dem_max   max context demand in the wave (sum of latest prompt of alive programs, K tokens)
  inf_p90 / inf_max   in-flight KV in the wave: sum of prompt+output of turns in service (arrival..arrival+e2e), K tokens, 10 s grid
  inf_pre   in-flight p90 in the 2 minutes before the wave
  srv_n     turns in service p90 in the wave
Prints runs sorted by wave miss and the Spearman correlation of each feature with it.

usage: escape_features.py --sim LABEL=GLOB ... [--gpu LABEL=ARM_DIR ...] [--w1 14 19] [--w2 21 26]"""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wave_minutes as WM

FEATS = ("w1", "prog", "done", "dem_max", "inf_pre", "inf_p90", "inf_max", "srv_n")


def feats(rows, w1, w2):
    def miss(a, b):
        s = [r for r in rows if a * 60 <= r[2] < b * 60 and r[1] > 0]; return 100 * sum(r[4] for r in s) / max(1, sum(r[3] for r in s))
    t0 = w2[0] * 60; by_p = defaultdict(list)
    for r in rows: by_p[r[0]].append(r)
    ends = {p: max(r[2] + r[5] for r in v) for p, v in by_p.items()}
    grid = np.arange(w2[0] * 60, w2[1] * 60, 10.0); pre = np.arange(t0 - 120, t0, 10.0)
    A = np.array([r[2] for r in rows]); E = A + np.array([r[5] for r in rows]); P = np.array([r[3] for r in rows])  # prompt (output is small)
    def infl(ts): return np.array([P[(A <= t) & (E > t)].sum() for t in ts]) / 1e3, np.array([((A <= t) & (E > t)).sum() for t in ts])
    ig, ng = infl(grid); ip, _ = infl(pre)
    dem = []
    for t in grid[::6]:
        d = 0.0
        for p, v in by_p.items():
            if min(r[2] for r in v) <= t <= ends[p]: d += max((r[3] for r in v if r[2] <= t), default=0)
        dem.append(d / 1e3)
    return dict(w2=miss(*w2), w1=miss(*w1), prog=int(sum(1 for e in E if e <= t0)), done=sum(1 for e in ends.values() if e <= t0),
                dem_max=max(dem), inf_pre=float(np.percentile(ip, 90)), inf_p90=float(np.percentile(ig, 90)), inf_max=float(ig.max()), srv_n=float(np.percentile(ng, 90)))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", nargs="*", default=[]); ap.add_argument("--gpu", nargs="*", default=[])
    ap.add_argument("--w1", nargs=2, type=int, default=[14, 19]); ap.add_argument("--w2", nargs=2, type=int, default=[21, 26])
    a = ap.parse_args(); runs = []
    for spec in a.gpu:
        lab, d = spec.split("=", 1); runs.append((lab, feats(WM.gpu_rows(d), a.w1, a.w2)))
    for spec in a.sim:
        lab, g = spec.split("=", 1)
        for f in sorted(glob.glob(g)):
            n = os.path.basename(os.path.dirname(f)); runs.append((f"{lab} {n}", feats(WM.sim_rows(f), a.w1, a.w2)))
    print(f"{'run':28s}{'w2':>6s}" + "".join(f"{k:>9s}" for k in FEATS))
    for lab, f in sorted(runs, key=lambda x: x[1]["w2"]):
        print(f"{lab[:28]:28s}{f['w2']:6.1f}" + "".join(f"{f[k]:9.1f}" for k in FEATS))
    sim = [f for lab, f in runs if not lab.startswith("GPU")]
    y = np.array([f["w2"] for f in sim])
    print("\nSpearman with wave-2 miss over the simulated runs (n=%d):" % len(sim))
    print("  " + "  ".join(f"{k} {spearman(np.array([f[k] for f in sim]), y):+.2f}" for k in FEATS))


if __name__ == "__main__":
    main()
