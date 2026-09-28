#!/usr/bin/env python3
"""Fit des_sim.py step-cost parameters from one or more GPU server logs (their Prefill/Decode step lines).

  prefill ms = p0 + p_tok*Knew + p_attn*G(new*(prefix+new/2)) + p_open*[chunk left open]
      fitted on single-sequence chunk chains (a full 8192 chunk forces another prefill next, so the gap between the
      two lines is a pure prefill step; prefix depth of continuation chunks is rebuilt along the chain).
  decode  ms = d0 + d_tok*Ktokens (CUDA graph, bs <= 32)   fitted on 40-step decode intervals with no prefill between
             = n0 + n_tok*Ktokens + n_req*bs (eager, bs > 32)  fitted on intervals WITH prefills, after subtracting the
               fitted prefill model for the prefill lines inside them (the only place bs > 32 decode occurs).
Pass --decode-from to fit decode on other logs than prefill (e.g. prefill pooled over a pod, decode per run).

usage: fit_step_costs.py --prefill-from LOG [LOG ...] [--decode-from LOG ...] --out params.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steplog_parse import parse


def chain_rows(steps, chunk=8192):
    rows = []; ctx = None; prev = None
    for s in steps:
        if s["kind"] != "P": continue
        if prev is not None and prev["new_tok"] == chunk and ctx is not None and s["new_seq"] == 1 and s["thr"] > 0:
            rows.append(dict(sec=s["sec"], prefix=ctx, new=s["new_tok"], gap=prev["new_tok"] / s["thr"]))
        if s["new_tok"] == chunk and s["new_seq"] == 1:
            ctx = ctx + chunk if (prev is not None and prev["new_tok"] == chunk and ctx is not None) else s["cached"] + chunk
        else: ctx = None
        prev = s
    return rows


def fit_prefill(logs, skip=180, max_gap=1.5):
    R = [r for L in logs for r in chain_rows(parse(L)) if r["sec"] > skip and r["gap"] < max_gap]
    new = np.array([r["new"] for r in R], float); pre = np.array([r["prefix"] for r in R], float); y = 1e3 * np.array([r["gap"] for r in R])
    X = np.c_[np.ones(len(R)), new / 1e3, new * (pre + new / 2) / 1e9, (new == 8192).astype(float)]
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return dict(p0=b[0], p_tok=b[1], p_attn=b[2], p_open=b[3]), dict(n=len(R), r2=1 - r.var() / y.var(), mad_ms=float(np.median(np.abs(r))))


def fit_decode(logs, pp, skip=300, skip_eager=60):
    pure = []; mixed = []
    def pre_ms(s):
        per = s["new_tok"] / max(s["new_seq"], 1); a = s["new_seq"] * per * (s["cached"] / max(s["new_seq"], 1) + per / 2) / 1e9
        return pp["p0"] + pp["p_tok"] * s["new_tok"] / 1e3 + pp["p_attn"] * a + pp["p_open"] * (s["new_tok"] == 8192)
    for L in logs:
        st = parse(L); seg = []
        for s in st:
            if s["kind"] == "P": seg.append(s); continue
            if s["kind"] != "D": continue
            if s.get("gap") and s["prev_running"] is not None and s["gap"] < 20:
                bs = 0.5 * (s["running"] + s["prev_running"])
                if s["sec"] > skip and not seg and s["running"] == s["prev_running"] and s["running"] <= 32: pure.append((s["tokens"] / 1e3, s["step_ms"], s["running"]))
                elif s["sec"] > skip_eager and min(s["running"], s["prev_running"]) > 32 and not s["cg"]:
                    mixed.append((s["tokens"] / 1e3, bs, (1e3 * s["gap"] - sum(pre_ms(x) for x in seg)) / 40))
            seg = []
    a = np.array(pure); X = np.c_[np.ones(len(a)), a[:, 0], a[:, 2]]; b, *_ = np.linalg.lstsq(X, a[:, 1], rcond=None); r = a[:, 1] - X @ b
    out = dict(d0=b[0], d_tok=b[1], d_req=b[2]); info = dict(n_pure=len(a), mad_ms=float(np.median(np.abs(r))))
    if len(mixed) >= 8:
        m = np.array(mixed); X = np.c_[np.ones(len(m)), m[:, 0], m[:, 1]]; c, *_ = np.linalg.lstsq(X, m[:, 2], rcond=None)
        out.update(n0=c[0], n_tok=c[1], n_req=c[2]); info["n_eager"] = len(m)
    return out, info


def fit_joint(logs, skip=60, chunk=8192, tmax=1e9, out_path="params.json", prefix_term=False):
    """Joint least squares over (a) every interval between consecutive decode lines = 40 decode steps + the prefill
    steps logged inside it, and (b) the pure prefill chunk-chain gaps. Uses the whole operating region instead of the
    quiet stretches that pure decode intervals select (those under-represent busy, high-bs periods)."""
    rows, ys, w, src, tsec = [], [], [], [], []
    from gpu_windows import requests as _reqs
    import gzip as _gz, datetime as _dt
    for li, L in enumerate(logs):
        st = parse(L); ctx = None; prev = None; seg = []
        R = _reqs(L); t0 = min(v[0] for v in R.values())
        for line in (_gz.open(L, "rt", errors="replace") if L.endswith(".gz") else open(L, errors="replace")):
            if "Prefill batch" in line or "Decode batch" in line: fl = _dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(); break
        iv = np.array([(v[2] - t0, (v[5] or v[2]) - t0, v[3]) for v in R.values()]); off = fl - t0
        def maxctx(sec):
            t = sec + off - 0.5; m = (iv[:, 0] <= t) & (iv[:, 1] >= t)
            return iv[m, 2].max() / 1e3 if m.any() else 0.0
        def pfeat(s, pre_known):
            n = s["new_tok"]; seqs = max(s["new_seq"], 1); per = n / seqs
            attn = (n * (pre_known + n / 2) if pre_known is not None else seqs * per * (s["cached"] / seqs + per / 2)) / 1e9
            f = [1.0, n / 1e3, attn, float(n == chunk)]
            # prefix_term: cost linear in the prefix read (K tokens), independent of the new-token count -- small extends on a
            # long prefix give FA3 only a few q tiles, so the step is paced by streaming the prefix KV, not by new x prefix
            if prefix_term: f.append((pre_known if pre_known is not None else s["cached"]) / 1e3)
            return np.array(f)
        for s in st:
            if s["kind"] == "P":
                pre = ctx if (prev is not None and prev["new_tok"] == chunk and ctx is not None) else None
                f = pfeat(s, pre)
                if s["pure"] and s["sec"] > skip and s["gap"] < 1.5 and s["new_seq"] == 1 and pre is not None:
                    rows.append(np.r_[np.zeros(8), f]); ys.append(1e3 * s["gap"]); w.append(1.0); src.append(li); tsec.append(s["sec"])
                seg.append(f)
                if s["new_tok"] == chunk and s["new_seq"] == 1: ctx = (ctx + chunk) if pre is not None else s["cached"] + chunk
                else: ctx = None
                prev = s
            elif s["kind"] == "D":
                if s.get("gap") and s["prev_running"] is not None and skip < s["sec"] < tmax and s["gap"] < 20:
                    bs = 0.5 * (s["running"] + s["prev_running"]); K = s["tokens"] / 1e3
                    eager = min(s["running"], s["prev_running"]) > 32 and not s["cg"]
                    if eager or max(s["running"], s["prev_running"]) <= 32:
                        mc = maxctx(s["sec"]); dfe = 40 * np.array([1.0, K, bs, mc]); df = np.r_[np.zeros(4), dfe] if eager else np.r_[dfe, np.zeros(4)]
                        pf = np.sum(seg, axis=0) if seg else np.zeros(5 if prefix_term else 4)
                        rows.append(np.r_[df, pf]); ys.append(1e3 * s["gap"]); w.append(1.0); src.append(li); tsec.append(s["sec"])
                seg = []
    X = np.array(rows); y = np.array(ys)
    keep = np.ones(len(y), bool)
    for _ in range(3):                                   # drop gross outliers (host stalls) and refit
        b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None); r = y - X @ b
        mad = np.median(np.abs(r[keep])); keep = np.abs(r) < 6 * mad + 50
    names = ["d0", "d_tok", "d_req", "d_max", "n0", "n_tok", "n_req", "n_max", "p0", "p_tok", "p_attn", "p_open"] + (["p_pre"] if prefix_term else [])
    out = dict(zip(names, map(float, b)))
    src = np.array(src); tsec = np.array(tsec); pred = X @ b
    print(f"{'log':<60} {'n':>6} {'bias %':>7} {'MAD %':>6} {'outliers %':>10}  outlier minutes")
    for li, L in enumerate(logs):
        m = src == li; k = m & keep
        bias = 100 * np.median((y[k] - pred[k]) / pred[k]); mad = 100 * np.median(np.abs(y[k] - pred[k]) / pred[k])
        om = sorted(set((tsec[m & ~keep] // 60).astype(int)))
        print(f"{L[-60:]:<60} {m.sum():6d} {bias:+7.1f} {mad:6.1f} {100 * (m & ~keep).sum() / m.sum():10.1f}  {om[:25]}")
    np.savez(os.path.splitext(out_path)[0] + "_resid.npz", y=y, pred=pred, src=src, tsec=tsec, keep=keep, is_chain=(X[:, :8] == 0).all(axis=1), n_prefill=X[:, 8] * (X[:, :8] != 0).any(axis=1))
    return out, dict(n=int(keep.sum()), dropped=int((~keep).sum()), mad_ms=float(mad))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefill-from", nargs="+", required=True); ap.add_argument("--decode-from", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--fit-window", nargs=2, type=float, default=None, help="joint: only decode intervals with log time in [t0, t1] s"); ap.add_argument("--joint", action="store_true", help="joint fit over all decode intervals + chunk chains of the --prefill-from logs")
    ap.add_argument("--prefix-term", action="store_true", help="joint: add p_pre, a prefill cost linear in the prefix length (ms per K prefix tokens)")
    a = ap.parse_args()
    if a.joint:
        res, info = fit_joint(a.prefill_from, out_path=a.out, prefix_term=a.prefix_term, **({"skip": a.fit_window[0], "tmax": a.fit_window[1]} if a.fit_window else {})); json.dump(res, open(a.out, "w"), indent=1)
        print(f"joint {info}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items())); return
    pp, pi = fit_prefill(a.prefill_from)
    dp, di = fit_decode(a.decode_from or a.prefill_from, pp)
    res = {**{k: float(v) for k, v in pp.items()}, **{k: float(v) for k, v in dp.items()}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"prefill {pi}: " + ", ".join(f"{k}={v:.3f}" for k, v in pp.items()))
    print(f"decode  {di}: " + ", ".join(f"{k}={v:.4f}" for k, v in dp.items()))


if __name__ == "__main__":
    main()
