"""Busy-time closure per bin: model cost of the steps the GPU actually ran (from its step log) vs bin length.
usage: busy.py SERVER_LOG PARAMS_JSON LO_MIN HI_MIN [BIN_S]"""
import sys, gzip, datetime, json, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from steplog_parse import parse
from gpu_windows import requests as reqs
L, P = sys.argv[1], json.load(open(sys.argv[2])); lo, hi = float(sys.argv[3]), float(sys.argv[4]); B = float(sys.argv[5]) if len(sys.argv) > 5 else 30
st = parse(L); R = reqs(L); t0 = min(v[0] for v in R.values())
for line in gzip.open(L, "rt", errors="replace"):
    if "Prefill batch" in line or "Decode batch" in line: fl = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
iv = np.array([(v[0] - t0, (v[5] or v[2]) - t0, v[3]) for v in R.values()]); off = fl - t0
def maxctx(t):
    m = (iv[:, 0] <= t) & (iv[:, 1] >= t); return iv[m, 2].max() / 1e3 if m.any() else 0.0
bins = {}; ctx = None; prev = None
for s in st:
    t = s["sec"] + off; b = int(t // B)
    if s["kind"] == "P":
        cont = prev is not None and prev["new_tok"] == 8192 and ctx is not None
        n = s["new_tok"]; seqs = max(s["new_seq"], 1); per = n / seqs
        attn = (n * (ctx + n / 2) if cont else seqs * per * (s["cached"] / seqs + per / 2)) / 1e9
        ms = P["p0"] + P["p_tok"] * n / 1e3 + P["p_attn"] * attn + P["p_open"] * (n == 8192) + P.get("p_pre", 0) * ((ctx if cont else s["cached"]) / 1e3)
        ctx = ((ctx + 8192) if cont else s["cached"] + 8192) if (n == 8192 and seqs == 1) else None
        x = bins.setdefault(b, [0, 0, 0, 0, 0]); x[0] += ms / 1e3; x[2] += n; x[4] = max(x[4], s["queue"]); prev = s
    elif s["kind"] == "D":
        bs = s["running"]; K = s["tokens"] / 1e3; mc = maxctx(t)
        ms = (P["d0"] + P["d_tok"] * K + P["d_req"] * bs + P["d_max"] * mc) if bs <= 32 else (P["n0"] + P["n_tok"] * K + P["n_req"] * bs + P["n_max"] * mc)
        x = bins.setdefault(b, [0, 0, 0, 0, 0]); x[1] += 40 * max(ms, 12) / 1e3; x[3] += 40; x[4] = max(x[4], s["queue"])
print(f"{'t(min)':>7} | model busy: prefill + decode = total / {B:.0f}s | prefill tok | qmax")
for b in sorted(bins):
    if lo * 60 <= b * B < hi * 60:
        x = bins[b]; print(f"{b*B/60:7.2f} | {x[0]:6.1f} + {x[1]:5.1f} = {x[0]+x[1]:5.1f}  ({100*(x[0]+x[1])/B:3.0f}%) | {x[2]/1e3:5.0f}K | {x[4]}")
