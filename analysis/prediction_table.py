#!/usr/bin/env python3
"""Prospective prediction table for a GPU experiment: per point (pool dir), method and arm, the seed mean ± sd of miss,
U p99, U CVaR90, n>16k, TTFT p80 / p90 (ladder scoring: warm turns, 300 s head / 60 s tail cut), and the seed-paired
policy delta (mean ± se). Written BEFORE the GPU runs so the comparison is a prediction, not a fit.

usage: prediction_table.py LABEL=POOL_DIR [...] [--arms rrdeclband0.001,rrdeclchunk1024] [--methods sgl,des]"""
import argparse, csv, glob, os, re, sys
import numpy as np
ARMN = {"lru": "LRU", "rrdeclband0.001": "LRU+D", "rrdeclchunk1024": "RR-C+D"}
KEYS = ("miss", "U_p99", "U_cvar90", "n16k", "ttft_p80", "ttft_p90")


def metrics(path, cut0=300.0, cut1=60.0):
    rows = list(csv.DictReader(open(path)))
    arr = np.array([float(r["arrival"]) for r in rows]); t0, t1 = arr.min(), arr.max()
    sel = [r for r, a in zip(rows, arr) if int(r["turn"]) > 0 and cut0 <= a - t0 <= t1 - t0 - cut1]
    pt = np.array([float(r["prompt_len"]) for r in sel]); u = pt - np.array([float(r["cached"]) for r in sel]); s = np.array([float(r["ttft"]) for r in sel])
    us = np.sort(u)
    return dict(miss=100 * u.sum() / pt.sum(), U_p99=float(np.percentile(u, 99)), U_cvar90=float(us[int(len(us) * 0.9):].mean()), n16k=float((u > 16000).sum()),
                ttft_p80=float(np.percentile(s, 80)), ttft_p90=float(np.percentile(s, 90)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", nargs="+"); ap.add_argument("--arms", default="rrdeclband0.001,rrdeclchunk1024"); ap.add_argument("--methods", default="sgl,des")
    a = ap.parse_args(); arms = a.arms.split(",")
    fmt = {"miss": "{:.2f}", "U_p99": "{:.0f}", "U_cvar90": "{:.0f}", "n16k": "{:.0f}", "ttft_p80": "{:.3f}", "ttft_p90": "{:.3f}"}
    print("| point | method | arm | n | " + " | ".join(KEYS) + " |"); print("|" + "---|" * (4 + len(KEYS)))
    for spec in a.points:
        lab, d = spec.split("=", 1)
        for meth in a.methods.split(","):
            per = {arm: {int(re.search(r"_n(\d+)$", os.path.dirname(f)).group(1)): metrics(f) for f in glob.glob(os.path.join(d, f"{meth}_n*", f"req_{arm}.csv"))} for arm in arms}
            if not all(per.values()): continue
            for arm in arms:
                v = per[arm]
                print(f"| {lab} | {meth} | {ARMN.get(arm, arm)} | {len(v)} | " + " | ".join(fmt[k].format(np.mean([x[k] for x in v.values()])) + " ± " + fmt[k].format(np.std([x[k] for x in v.values()], ddof=1)) for k in KEYS) + " |")
            common = sorted(set(per[arms[0]]) & set(per[arms[1]])); cells = []
            for k in KEYS:
                if k == "miss" or k == "n16k": x = np.array([per[arms[1]][n][k] - per[arms[0]][n][k] for n in common]); f_ = "{:+.2f} ± {:.2f}"
                else: x = np.array([100 * (per[arms[1]][n][k] / per[arms[0]][n][k] - 1) for n in common]); f_ = "{:+.1f}% ± {:.1f}"
                cells.append(f_.format(x.mean(), x.std(ddof=1) / np.sqrt(len(x))) + f" ({int((x < 0).sum())}/{len(x)} neg)")
            print(f"| {lab} | {meth} | **Δ** | {len(common)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
