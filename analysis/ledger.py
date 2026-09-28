import sys, json, gzip, datetime, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from steplog_parse import parse
log, pj = sys.argv[1], sys.argv[2]; P = json.load(open(pj))
st = parse(log); ctx = None; prev = None; rows = []
for s in st:
    if s["kind"] == "P":
        # prefix of the (single) chunked request continues along a chain of full chunks
        if prev is not None and prev["new_tok"] == 8192 and ctx is not None: pre = ctx
        else: pre = s["cached"] / max(s["new_seq"], 1)
        n = s["new_tok"]; seqs = s["new_seq"]; per = n / seqs
        attn = (n * (pre + n / 2) if (prev is not None and prev["new_tok"] == 8192 and ctx is not None) else seqs * per * (pre + per / 2)) / 1e9
        ms = P["p0"] + P["p_tok"] * n / 1e3 + P["p_attn"] * attn + P["p_open"] * (n == 8192)
        if n == 8192 and seqs == 1: ctx = (ctx + 8192) if (prev is not None and prev["new_tok"] == 8192 and ctx is not None) else s["cached"] + 8192
        elif n == 8192: ctx = None
        else: ctx = None
        rows.append(dict(sec=s["sec"], kind="P", ms=ms, new=n, meas=(s["gap"] * 1e3 if s["pure"] else np.nan)))
        prev = s
    elif s["kind"] == "D":
        bs = s["running"]; K = s["tokens"] / 1e3
        ms = (P["d0"] + P["d_tok"] * K + P["d_req"] * bs) if bs <= 32 else (P["n0"] + P["n_tok"] * K + P["n_req"] * bs)
        rows.append(dict(sec=s["sec"], kind="D", ms=40 * max(ms, 12.0), new=0, meas=np.nan))
R = pd.DataFrame(rows); R["m"] = (R.sec // 60).astype(int)
print(" min | model busy s: prefill + decode = total | prefill tok | pure-chunk steps: n, measured/model")
for m, g in R.groupby("m"):
    if m < int(sys.argv[3]) or m > int(sys.argv[4]): continue
    p = g[g.kind == "P"]; d = g[g.kind == "D"]; pc = p[p.meas.notna()]
    r = (pc.meas / pc.ms).median() if len(pc) else float("nan")
    print(f"{m:4d} | {p.ms.sum()/1e3:5.1f} + {d.ms.sum()/1e3:5.1f} = {(p.ms.sum()+d.ms.sum())/1e3:5.1f} | {p.new.sum()/1e3:5.0f}K | {len(pc):3d} {r:5.2f}")
