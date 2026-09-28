#!/usr/bin/env python3
"""Per-minute dynamics of a wall wave, GPU vs simulators, from per-request records only (so both sides are measured the same
way): turns arriving, prompt / uncached tokens (prefill work) and miss, median e2e, programs alive, and the context DEMAND =
sum over alive programs of their latest prompt length (a program is alive from its first to its last turn arrival).
Demand above the pool is what pushes a point over the wall; recovery needs demand to fall, i.e. programs to finish.

usage: wave_minutes.py --gpu LABEL=ARM_DIR ... --sim LABEL=GLOB ... [--t0 10 --t1 30]
  GLOB matches simulator req_*.csv files (e.g. .../sgl_n*/req_rrdeclchunk1024.csv); a label's runs are averaged."""
import argparse, csv, glob, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))


def gpu_rows(arm_dir):
    import gpu_windows as GW
    log = next(os.path.join(arm_dir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm_dir, x)))
    R = GW.breakdown(GW.requests(log)); t0 = min(v["recv"] for v in R.values())
    return [(p, t, v["recv"] - t0, v["pt"], v["unc"], v["e2e"]) for (p, t), v in R.items()]


def sim_rows(path):
    rows = list(csv.DictReader(open(path)))
    return [(r["program"], int(r["turn"]), float(r["arrival"]), float(r["prompt_len"]), float(r["prompt_len"]) - float(r["cached"]), float(r["e2e"])) for r in rows]


def minutes(rows, t0, t1):
    """-> dict of per-minute arrays for minutes t0..t1-1"""
    by_p = defaultdict(list)
    for p, t, a, pt, u, e in rows: by_p[p].append((a, pt, a + e))
    for v in by_p.values(): v.sort()
    M = np.arange(t0, t1); out = {k: np.zeros(len(M)) for k in ("arr", "ptok", "unc", "e2e", "alive", "demand")}
    for i, m in enumerate(M):
        sel = [r for r in rows if m * 60 <= r[2] < (m + 1) * 60 and r[1] > 0]
        out["arr"][i] = len(sel); out["ptok"][i] = sum(r[3] for r in sel) / 1e3; out["unc"][i] = sum(r[4] for r in sel) / 1e3
        out["e2e"][i] = np.median([r[5] for r in sel]) if sel else np.nan
        t = (m + 0.5) * 60; alive = 0; dem = 0.0
        for v in by_p.values():
            if v[0][0] <= t <= v[-1][2]:
                alive += 1; dem += max((pt for a, pt, _ in v if a <= t), default=0)
        out["alive"][i] = alive; out["demand"][i] = dem / 1e3
    out["miss"] = 100 * out["unc"] / np.maximum(out["ptok"], 1e-9)
    return M, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", nargs="*", default=[]); ap.add_argument("--sim", nargs="*", default=[])
    ap.add_argument("--t0", type=int, default=10); ap.add_argument("--t1", type=int, default=30)
    a = ap.parse_args(); series = []
    for spec in a.gpu:
        lab, d = spec.split("=", 1); M, o = minutes(gpu_rows(d), a.t0, a.t1); series.append((lab, o, None))
    for spec in a.sim:
        lab, g = spec.split("=", 1); runs = [minutes(sim_rows(f), a.t0, a.t1)[1] for f in sorted(glob.glob(g))]
        if not runs: print("no files:", g); continue
        M = np.arange(a.t0, a.t1)
        series.append((f"{lab} ×{len(runs)}", {k: np.nanmean([r[k] for r in runs], 0) for k in runs[0]}, {k: np.nanstd([r[k] for r in runs], 0) for k in runs[0]}))
    M = np.arange(a.t0, a.t1)
    for key, fmt, title in (("miss", "{:6.1f}", "miss % (turns arriving in the minute)"), ("unc", "{:6.0f}", "uncached K tokens (prefill work)"),
                            ("arr", "{:6.0f}", "turns arriving"), ("e2e", "{:6.1f}", "median e2e s"), ("alive", "{:6.0f}", "programs alive"),
                            ("demand", "{:6.0f}", "context demand K tokens (pool 750)")):
        print(f"== {title}"); print(" " * 22 + "".join(f"{m:>6d}" for m in M))
        for lab, mu, sd in series: print(f"{lab[:22]:22s}" + "".join(fmt.format(x) for x in mu[key]))
        print()


if __name__ == "__main__":
    main()
