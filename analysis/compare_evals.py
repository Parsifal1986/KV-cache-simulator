"""Side-by-side error vs GPU from several local_eval.py outputs. usage: compare_evals.py NAME=eval.txt [NAME=eval.txt ...]"""
import sys, re
import numpy as np
H = ["miss", "U_p90", "U_p99", "U_CVaR90", "sTTFT_p50", "sTTFT_p80", "sTTFT_p90", "sTTFT_p99"]
runs = {}
for a in sys.argv[1:]:
    name, path = a.split("=", 1); rows = {}
    for line in open(path):
        if line.startswith("arm") or "|" not in line: continue
        parts = [p.strip() for p in line.split("|")]; tag = parts[0].split()[0]; cells = {}
        for h, c in zip(H, parts[1:]):
            m = re.match(r"([\d.]+)/([\d.]+)", c)
            if m: g, s = float(m.group(1)), float(m.group(2)); cells[h] = (s - g) if h == "miss" else 100 * (s / g - 1)
        rows[tag] = cells
    runs[name] = rows
names = list(runs); tags = [t for t in runs[names[0]] if all(t in runs[n] for n in names)]
cols = ["miss", "U_p99", "U_CVaR90", "sTTFT_p90"]
print(f"{'arm':<32} " + " ".join(f"{c + '[' + n + ']':>18}" for c in cols for n in names))
for t in tags:
    print(f"{t:<32} " + " ".join(f"{runs[n][t][c]:>+17.2f}{'pp' if c == 'miss' else '%'}" if c == "miss" else f"{runs[n][t][c]:>+17.0f}%" for c in cols for n in names))
print("\nsummary over arms: median |error| (share of arms within +-0.4pp miss / +-10% others)")
for n in names:
    s = []
    for c in cols:
        v = np.abs([runs[n][t][c] for t in tags]); thr = 0.4 if c == "miss" else 10
        s.append(f"{c} {np.median(v):.2f}{'pp' if c == 'miss' else '%'} ({100*(v <= thr).mean():.0f}%)")
    print(f"  {n:<8} " + " | ".join(s))
