#!/usr/bin/env python3
"""Collect the baseline-ladder runs (scripts/ladder/run_ladder.sh) and the GPU arms into one table + figure.

Every method is scored with the same selection as des_sim.summarize / gpu_windows.py: warm turns (turn > 0) arriving
between 300 s after the first request and 60 s before the last. Metrics: miss (token-weighted), U p99 (uncached tokens
per request), n>16k, TTFT p90 (B1 has none: it replays recorded service times).

usage: ladder_report.py RUNS_DIR --gpu-json a.json [b.json ...] --md docs/LADDER.md --png fig.png
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict
import numpy as np

SIMU = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = ["B1", "B2", "B3", "DES", "SGL"]
LABEL = {"B1": "B1 trace replay (recorded times)", "B2": "B2 open-loop DES", "B3": "B3 naive closed loop",
         "DES": "des_sim closed loop", "SGL": "sgl_sim closed loop", "GPU": "GPU"}
ARMN = {"lru": "LRU", "rrdeclband0.001": "LRU+D", "rrdeclchunk1024": "RR-C+D"}


def metrics_csv(path, cut0=300.0, cut1=60.0):
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    arr = np.array([float(r["arrival"]) for r in rows]); t0, t1 = arr.min(), arr.max()
    sel = [r for r, a in zip(rows, arr) if int(r["turn"]) > 0 and cut0 <= a - t0 <= t1 - t0 - cut1]
    pt = np.array([float(r["prompt_len"]) for r in sel]); u = pt - np.array([float(r["cached"]) for r in sel])
    out = dict(n=len(sel), miss=100 * u.sum() / pt.sum(), U_p99=float(np.percentile(u, 99)), n16k=int((u > 16000).sum()))
    if "ttft" in sel[0] and sel[0]["ttft"] not in ("", "None"):
        s = np.array([float(r["ttft"]) for r in sel if r["ttft"] not in ("", "None")]); out["ttft_p90"] = float(np.percentile(s, 90))
    return out


def collect_sims(runs):
    """-> {(machine, conc, seed, arm): {method: [metrics per replicate]}}"""
    R = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(runs, "*", "c*_s*"))):
        machine = os.path.basename(os.path.dirname(d)); c, s = map(int, re.match(r"c(\d+)_s(\d+)", os.path.basename(d)).groups())
        for f in glob.glob(os.path.join(d, "b1", "req_*_s*.csv")):
            arm = re.match(r"req_(.*)_s\d+\.csv", os.path.basename(f)).group(1); m = metrics_csv(f)
            if m: R[(machine, c, s, arm)]["B1"].append(m)
        for f in glob.glob(os.path.join(d, "b2_*", "req_*.csv")):
            arm = os.path.basename(f)[4:-4]; m = metrics_csv(f)
            if m: R[(machine, c, s, arm)]["B2"].append(m)
        subs = (("b3", "B3"), ("sgl", "SGL")) + tuple((os.path.basename(x), "DES") for x in glob.glob(os.path.join(d, "des_n*"))) \
            + tuple((os.path.basename(x), "SGL") for x in glob.glob(os.path.join(d, "sgl_n*")))
        for sub, meth in subs:
            for f in glob.glob(os.path.join(d, sub, "req_*.csv")):
                arm = os.path.basename(f)[4:-4]; m = metrics_csv(f)
                if m: R[(machine, c, s, arm)][meth].append(m)
    return R


LOAD_MEDIAN_MAX, LOAD_SPREAD_MAX = 100.0, 60.0   # host-load envelope of the machine profile (local8xh100 fitted around ~80)


def host_load(arm_dir):
    """Host load of a GPU run from its cotenant.log snapshots: (median, p10, p90) or None (pod arms have no log)."""
    f = os.path.join(SIMU, "..", "sglang", arm_dir, "cotenant.log") if not os.path.isabs(arm_dir) else os.path.join(arm_dir, "cotenant.log")
    if not os.path.exists(f): return None
    v = np.array([float(x) for x in re.findall(r"load=([\d.]+)", open(f, errors="replace").read())])
    return (float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))) if len(v) >= 5 else None


def load_flag(ld):
    return ld is not None and (ld[0] > LOAD_MEDIAN_MAX or ld[2] - ld[1] > LOAD_SPREAD_MAX)


def collect_gpu(jsons):
    man = {}
    for mach in ("local8xh100", "pod0920"):
        for line in open(os.path.join(SIMU, "profiles", "machines", f"{mach}_arms.tsv")):
            f = line.rstrip("\n").split("\t")
            man[f[1]] = (mach, int(f[3]), int(f[4]), f[8])
    G = {}
    for j in jsons:
        for d, v in json.load(open(j)).items():
            if d in man:
                s = v["summary"]; ld = host_load(d)
                G[man[d]] = dict(miss=s["miss"], U_p99=s["U_p99"], n16k=s["n16k"], ttft_p90=s["sTTFT_p90"], dir=d, load=ld, hot=load_flag(ld))
    return G


def agg(ms, k):
    v = [m[k] for m in ms if k in m]
    return (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


def fmt(x, nd=2): return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def token_arms(runs):
    out = {}
    for f in glob.glob(os.path.join(runs, "*", "c*_s*", "token_arm.txt")):
        d = os.path.dirname(f); c, s = map(int, re.match(r"c(\d+)_s(\d+)", os.path.basename(d)).groups())
        out[(os.path.basename(os.path.dirname(d)), c, s)] = open(f).read().split()[1]
    return out


def token_label(tk, mach, c, s):
    """short label of the GPU arm whose tokens a point replays; '*' = borrowed from another concurrency/seed."""
    d = tk.get((mach, c, s))
    if not d: return "recorded"
    m = re.search(r"setup_(\d+)c_750k_s(\d+)", d); tc, ts = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    lab = f"c{tc}s{ts}/{os.path.basename(d)}"
    return lab + ("" if (tc, ts) == (c, s) else " *")


def write_md(R, G, path, png, tk=None, prev_md=None):
    tk = tk or {}
    L = ["# Motivation baseline ladder — spider2 terminus x2, 750K pool, fcfs",
         "", "Generated by `analysis/ladder_report.py` from `scripts/ladder/run_ladder_tok.sh` (v2, 2026-09-28). **Every method replays the same "
         "server-side token input**: the ids one GPU arm of that point actually sent/generated (`scripts/ladder/make_token_stream.py` = "
         "`des_sim.ids_from_gpu`), so only the timing / scheduling model differs between methods. Scoring = warm turns, 300 s head / 60 s tail "
         "cut, same as `gpu_windows.py`.",
         "", "- **B1** trace-driven cache replay (`offline_replay.py`): every turn at its recorded time and recorded service time, trials re-packed to the target concurrency (`retime_for_concurrency`); no server model, so no TTFT.",
         "- **B2** open-loop DES (`scripts/ladder/b2_openloop.py --no-stitch`): des_sim's scheduler/cache/step costs, but each (program, turn) arrives at B1's recorded time; completions never feed back.",
         "- **B3** naive closed loop (`closed_loop_sim.py`, defaults incl. decode_overlap 0.5) with its 6 step-cost parameters least-squares projected from the same machine profile (`scripts/ladder/fit_b3_params.py`); no host-load model.",
         "- **DES** des_sim closed loop, **SGL** sgl_sim closed loop (real SGLang Scheduler + event_loop_overlap, via `scripts/ladder/sgl_tok.py`); both mean ± sd over 3 noise seeds, v1 machine profiles (`local8xh100.json`, `pod0920.json`), host-load timeline of the token arm (local box only).",
         "- **GPU** `gpu_windows.py` on the arm's server.log (local8xh100 = shared 8×H100 box; pod0920 = cloud pod).",
         f"- **Host load is not modelled**: every simulator runs at its machine profile's conditions (no `--load-from`). GPU runs whose host load (cotenant.log `load=`) was outside the profile's envelope -- median > {LOAD_MEDIAN_MAX:.0f} or p10–p90 spread > {LOAD_SPREAD_MAX:.0f} -- are marked **环境超出档案条件**; they are listed but not used for conclusions, and their gaps are not read as simulator error.",
         "- **token** = GPU arm whose ids the point replays (all methods, all policy arms). `*` = borrowed from another concurrency/seed (sim-only points, or a point whose GPU arm is another seed) — same stream, same client type where possible; GPU arms of one point agree on 74–98 % of outputs and on every prompt length.", "",
         f"Figure: `{png}`", ""]
    for mach in ("local8xh100", "pod0920"):
        keys = sorted(k for k in R if k[0] == mach)
        if not keys: continue
        L += [f"## {mach}", "", "| conc | seed | token | arm | GPU host load p50 (p10–p90) | GPU miss | B1 | B2 | B3 | DES | SGL | GPU U p99 | B1 | B2 | B3 | DES | SGL | GPU TTFT p90 | B2 | B3 | DES | SGL |",
              "|" + "---|" * 22]
        for k in keys:
            g = G.get(k, {}); ld = g.get("load")
            lcell = "–" if not g else ("n/a (pod, steady)" if ld is None else f"{ld[0]:.0f} ({ld[1]:.0f}–{ld[2]:.0f})" + (" **环境超出档案条件**" if g.get("hot") else ""))
            row = [str(k[1]), str(k[2]), token_label(tk, *k[:3]), ARMN.get(k[3], k[3]), lcell, fmt(g.get("miss"))]
            for meth in METHODS:
                mu, sd = agg(R[k][meth], "miss"); e = f" ({mu - g['miss']:+.2f})" if g and np.isfinite(mu) else ""
                row.append(fmt(mu) + (f"±{sd:.2f}" if sd else "") + e)
            row.append(fmt(g.get("U_p99"), 0))
            for meth in METHODS:
                mu, sd = agg(R[k][meth], "U_p99"); e = f" ({100 * (mu / g['U_p99'] - 1):+.0f}%)" if g and np.isfinite(mu) else ""
                row.append(fmt(mu, 0) + e)
            row.append(fmt(g.get("ttft_p90"), 3))
            for meth in ("B2", "B3", "DES", "SGL"):
                mu, sd = agg(R[k][meth], "ttft_p90"); e = f" ({100 * (mu / g['ttft_p90'] - 1):+.0f}%)" if g and np.isfinite(mu) else ""
                row.append(fmt(mu, 3) + e)
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    # policy-delta sign agreement
    L += ["## Does the simulator get the policy decision right? Δ = RR-C+D − LRU+D (miss pp / U p99 %)", "",
          "| machine | conc | seed | GPU Δmiss | GPU ΔU p99 | " + " | ".join(f"{m} Δmiss / ΔU p99" for m in METHODS) + " |", "|" + "---|" * (5 + len(METHODS))]
    for k in sorted({(m, c, s) for (m, c, s, a) in R}):
        a, b = k + ("rrdeclband0.001",), k + ("rrdeclchunk1024",)
        if a not in R or b not in R: continue
        ga, gb = G.get(a), G.get(b)
        gdm = gb["miss"] - ga["miss"] if ga and gb else None; gdu = 100 * (gb["U_p99"] / ga["U_p99"] - 1) if ga and gb else None
        hot = bool((ga and ga.get("hot")) or (gb and gb.get("hot")))
        row = [k[0], str(k[1]), str(k[2]) + (" (环境超出档案条件)" if hot else ""), fmt(gdm), fmt(gdu, 0)]
        for meth in METHODS:
            ma, mb = agg(R[a][meth], "miss")[0], agg(R[b][meth], "miss")[0]; ua, ub = agg(R[a][meth], "U_p99")[0], agg(R[b][meth], "U_p99")[0]
            dm, du = mb - ma, 100 * (ub / ua - 1)
            ok = ""
            if gdm is not None and np.isfinite(dm): ok = " ✓" if (np.sign(dm) == np.sign(gdm) and np.sign(du) == np.sign(gdu)) else " ✗"
            row.append(f"{fmt(dm)} / {fmt(du, 0)}%{ok}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    if prev_md and os.path.exists(prev_md):
        old = open(prev_md).read(); i = old.find("## Findings")
        if i >= 0:
            L += ["## Previous pass (v1, kept for comparison)", "",
                  "v1 differed in the token input and in sgl_sim: prediction mode on the RECORDED stream (des_sim's stitching emulation with program-unique "
                  "outputs; the local c24/c32/c48 GPU arms were run by the pre-2026-09-20 unstitched client, so that emulation does not describe them), "
                  "and sgl_sim was a snapshot run with the normal (non-overlap) loop, no noise model, no host load. Its findings:", "",
                  old[i:].split("\n", 1)[1].strip(), ""]
    open(path, "w").write("\n".join(L))


def plot(R, G, png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # reference categorical palette (dataviz references/palette.md), fixed slot per entity; ours solid, baselines dashed
    COL = {"DES": "#2a78d6", "SGL": "#eb6834", "B1": "#1baf7a", "B2": "#eda100", "B3": "#e87ba4"}
    STY = {"DES": "-", "SGL": "-", "B1": "--", "B2": "--", "B3": "--"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    panels = [("local8xh100", "lru", "shared 8×H100 box · LRU"), ("pod0920", "rrdeclband0.001", "cloud pod · LRU+D")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="#fcfcfb")
    for row, (mach, arm, title) in enumerate(panels):
        for col, (key, ylab) in enumerate((("miss", "miss rate (%)"), ("ttft_p90", "TTFT p90 (s)"))):
            ax = axes[row][col]; ax.set_facecolor("#fcfcfb")
            concs = sorted({c for (m, c, s, a) in R if m == mach and a == arm and s == 0})
            for meth in METHODS:
                if key == "ttft_p90" and meth == "B1": continue   # B1 has no server model
                xs, mu, sd = [], [], []
                for c in concs:
                    ms = R[(mach, c, 0, arm)][meth]
                    if ms: m_, s_ = agg(ms, key); xs.append(c); mu.append(m_); sd.append(s_)
                if not xs: continue
                mu, sd = np.array(mu), np.array(sd)
                ax.plot(xs, mu, STY[meth], color=COL[meth], lw=2, marker="o", ms=5, label=LABEL[meth], zorder=3)
                if meth in ("DES", "SGL") and sd.any(): ax.fill_between(xs, mu - sd, mu + sd, color=COL[meth], alpha=0.15, lw=0)
            gx = sorted({c for (m, c, s, a) in G if m == mach and a == arm})
            for c in gx:
                vals = [G[k][key] for k in G if k[0] == mach and k[1] == c and k[3] == arm]
                y = vals[[k[2] for k in G if k[0] == mach and k[1] == c and k[3] == arm].index(0)] if 0 in [k[2] for k in G if k[0] == mach and k[1] == c and k[3] == arm] else np.mean(vals)
                k0 = (mach, c, 0, arm); hot = G[k0].get("hot") if k0 in G else all(G[k].get("hot") for k in G if k[0] == mach and k[1] == c and k[3] == arm)
                ax.plot([c], [y], "D", color="#fcfcfb" if hot else ink, ms=8, zorder=5, markeredgecolor=ink, markeredgewidth=1.5,
                        label="GPU, host load outside the profile's conditions (hollow)" if hot else "GPU (seed 0; bar = seeds 0–2)")
                if len(vals) > 1: ax.vlines(c, min(vals), max(vals), color=ink, lw=2, zorder=4)
            ax.set_title(f"{title} — {ylab.split(' (')[0]}", color=ink, fontsize=11, loc="left")
            ax.set_xlabel("concurrent agent programs", color=ink2); ax.set_ylabel(ylab, color=ink2)
            ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): ax.spines[sp].set_color(grid)
            ax.tick_params(colors=ink2)
            if key == "ttft_p90": ax.set_yscale("log")
    seen = {}
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()): seen.setdefault(l_, h_)
    h, l = list(seen.values()), list(seen.keys())
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=ink)
    fig.suptitle("Simulator error vs GPU along one pressure sweep (spider2, 750K pool)", color=ink, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96)); fig.savefig(png, dpi=150); print("wrote", png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs"); ap.add_argument("--gpu-json", nargs="+", required=True); ap.add_argument("--md", required=True); ap.add_argument("--png", required=True)
    ap.add_argument("--prev-md", default=None, help="earlier MOTIVATION_LADDER.md whose Findings section is kept as a comparison section")
    a = ap.parse_args()
    R = collect_sims(a.runs); G = collect_gpu(a.gpu_json)
    write_md(R, G, a.md, a.png, token_arms(a.runs), a.prev_md); plot(R, G, a.png); print("wrote", a.md)


if __name__ == "__main__":
    main()
