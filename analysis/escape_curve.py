#!/usr/bin/env python3
"""Escape-probability curve of a wall point: per pool size and arm, total miss (ladder scoring), second-wave miss per seed,
escape rate (second-wave miss <= --thr) and the seed-paired policy delta. Input = run_pool_sweep.sh layout
OUT/kv<POOL>/sgl_n<N>/req_<arm>.csv; --extra POOL=DIR adds a pool whose runs live elsewhere (DIR/sgl_n<N>/req_<arm>.csv).

usage: escape_curve.py SWEEP_DIR [--extra 750000=DIR] [--arms rrdeclband0.001,rrdeclchunk1024] [--w2 21 26] [--thr 25]"""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_report as L, wave_minutes as WM

ARMN = {"lru": "LRU", "rrdeclband0.001": "LRU+D", "rrdeclchunk1024": "RR-C+D"}


def wave(path, a, b):
    s = [r for r in WM.sim_rows(path) if a * 60 <= r[2] < b * 60 and r[1] > 0]
    return 100 * sum(r[4] for r in s) / max(1, sum(r[3] for r in s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep"); ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--arms", default="rrdeclband0.001,rrdeclchunk1024")
    ap.add_argument("--w2", nargs=2, type=int, default=[21, 26]); ap.add_argument("--thr", type=float, default=25.0); ap.add_argument("--method", default="sgl", help="sgl or des")
    a = ap.parse_args(); arms = a.arms.split(",")
    dirs = {int(re.search(r"kv(\d+)", d).group(1)): d for d in glob.glob(os.path.join(a.sweep, "kv*"))}
    for e in a.extra: k, d = e.split("=", 1); dirs[int(k)] = d
    print(f"escape = wave-2 ({a.w2[0]}-{a.w2[1]} min) miss <= {a.thr:.0f} %; Δ = {ARMN.get(arms[1])} - {ARMN.get(arms[0])}, paired by noise seed (mean ± se)\n")
    print(f"{'pool K':>7s} | " + " | ".join(f"{ARMN.get(x, x)} miss  esc  wave-2 per seed" for x in arms) + " | Δmiss  ΔU p99 | Δwave2")
    for kv in sorted(dirs):
        per = {}
        for arm in arms:
            per[arm] = {}
            for f in glob.glob(os.path.join(dirs[kv], f"{a.method}_n*", f"req_{arm}.csv")):
                n = int(re.search(r"_n(\d+)", f).group(1)); m = L.metrics_csv(f); per[arm][n] = (m["miss"], m["U_p99"], wave(f, *a.w2))
        cells = []
        for arm in arms:
            v = per[arm]; w = [v[n][2] for n in sorted(v)]
            cells.append(f"{np.mean([x[0] for x in v.values()]):5.2f}±{np.std([x[0] for x in v.values()], ddof=1) if len(v) > 1 else 0:4.2f} {sum(x <= a.thr for x in w):2d}/{len(w):<2d} "
                         + " ".join(f"{x:2.0f}" for x in w))
        common = sorted(set(per[arms[0]]) & set(per[arms[1]]))
        d = np.array([per[arms[1]][n][0] - per[arms[0]][n][0] for n in common]); du = np.array([100 * (per[arms[1]][n][1] / per[arms[0]][n][1] - 1) for n in common])
        dw = np.array([per[arms[1]][n][2] - per[arms[0]][n][2] for n in common]); se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        print(f"{kv // 1000:7d} | " + " | ".join(cells) + f" | {d.mean():+5.2f}±{se(d):4.2f} {du.mean():+4.0f}% | {dw.mean():+5.1f}±{se(dw):3.1f}")


if __name__ == "__main__":
    main()
