#!/usr/bin/env python3
"""Build a machine profile for des_sim.py from the GPU arms run on ONE machine (profile once per machine).

Stage A -- from the server logs + the replayed traces only:
  * step costs      fit_step_costs.fit_joint over every arm (decode intervals + prefill chunk chains)
  * noise           decode speed factor per ~40-step interval (robust sd) + 60 s slow drift; prefill-chain step sd
  * client overhead per turn: (GPU recv_k - GPU finish_{k-1}) - recorded tool gap_k
                    = turn_overhead + ovh_tok * Kprompt_k + ovh_out * Kout_{k-1}  (ms, L1 fit), lognormal ovh_sigma
  * ingress         request_received -> api_server_dispatch_finish = ingress_base + ingress_tok * Kprompt (ms, L1 fit)
  Host load is NOT modelled: the profile describes the machine as it was while the arms were run (their host load is
  recorded in _machine.host_load_median for reference only).
Stage B -- needs one OPEN-LOOP des_sim run per arm with the stage-A profile (ftd_median 0):
  * TTFT residual of de-queued requests: GPU TTFT - sim ttft_raw = extra per-token ingress (added to ingress_tok)
    + first-token delay (lognormal ftd_median / ftd_sigma)

MANIFEST: TSV rows  tag, arm_dir, stream.pkl, conc, seed, rec_conc, rec_span, trial_gap, arm[, ...]
usage: profile_machine.py fit  MANIFEST --out machines/NAME.json [--about TEXT]
       profile_machine.py ttft MANIFEST --profile machines/NAME.json --sim-dir DIR   (DIR/<tag>/req_<arm>.csv)
"""
import os as _os
# one process = one core (user limit: all simulation jobs together <= 30 threads)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): _os.environ.setdefault(_v, "1")
import argparse, datetime, gzip, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEC = json.JSONDecoder()


def read_manifest(path):
    rows = []
    for line in open(path):
        if not line.strip() or line.startswith("#"): continue
        f = line.rstrip("\n").split("\t")
        rows.append(dict(tag=f[0], dir=f[1], stream=f[2], conc=int(f[3]), seed=int(f[4]), arm=f[8]))
    return rows


def gpu_requests(arm_dir):
    """(program, turn) -> meta_info fields of every finished replay request."""
    log = next(os.path.join(arm_dir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm_dir, x)))
    out = {}
    for line in (gzip.open(log, "rt", errors="replace") if log.endswith(".gz") else open(log, errors="replace")):
        if '"request.finished"' not in line: continue
        m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
        if not m: continue
        meta, _ = DEC.raw_decode(line, line.find("{", line.rfind('"meta_info"')))
        if meta.get("cached_tokens") is None: continue
        out[(m.group(1), int(m.group(2)))] = meta
    return out


def l1_fit(X, y, iters=50):
    """Least absolute deviations via IRLS (median regression)."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - X @ b), 1e-3)
        b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return b


def rsd(x): x = np.asarray(x); return float(1.4826 * np.median(np.abs(x - np.median(x))))


def noise_from_resid(npz):
    d = np.load(npz); y, p, src, t, keep, chain = (d[k] for k in ("y", "pred", "src", "tsec", "keep", "is_chain"))
    pf = d["n_prefill"] if "n_prefill" in d.files else np.zeros(len(y))
    dec_r, drift, periods = [], [], []
    for s in np.unique(src):
        m = (src == s) & ~chain & keep & (pf == 0)                         # pure decode intervals only
        if m.sum() < 20: continue
        o = np.argsort(t[m]); ts, r = t[m][o], np.log(y[m][o] / p[m][o])
        r = r - np.median(r)                                               # per-arm bias is not noise
        roll = np.array([r[(ts >= a - 30) & (ts <= a + 30)].mean() for a in ts])
        dec_r.append(r - roll); drift.append(roll); periods.append(y[m] / 1e3)
    mc = chain & keep
    return dict(dec_noise=rsd(np.concatenate(dec_r)), drift_noise=rsd(np.concatenate(drift)), dec_period=float(np.median(np.concatenate(periods))),
                drift_tau=60.0, pre_noise=rsd(np.log(y[mc] / p[mc])))


def client_and_ingress(rows, skip=300.0):
    import offline_replay as OR
    streams = {}; ovh, ing = [], []
    for r in rows:
        if r["stream"] not in streams:
            byp = defaultdict(list)
            for q in OR.load_stream(r["stream"]): byp[q.program_id or q.rid].append(q)
            streams[r["stream"]] = {pid: sorted(l, key=lambda q: q.arrival) for pid, l in byp.items()}
        rec = streams[r["stream"]]; G = gpu_requests(r["dir"]); t0 = min(m["request_received_ts"] for m in G.values())
        for (pid, k), m in G.items():
            if m["request_received_ts"] - t0 < skip: continue
            ing.append((m["prompt_tokens"] / 1e3, 1e3 * (m["api_server_dispatch_finish_ts"] - m["request_received_ts"])))
            prev = G.get((pid, k - 1)); lst = rec.get(pid)
            if k == 0 or prev is None or lst is None or k >= len(lst): continue
            gap_rec = max(0.0, lst[k].arrival - lst[k - 1].finish)
            y = 1e3 * ((m["request_received_ts"] - prev["request_finished_ts"]) - gap_rec)
            if -1e3 < y < 5e3: ovh.append((m["prompt_tokens"] / 1e3, prev["completion_tokens"] / 1e3, y, r["tag"]))
    O = np.array([o[:3] for o in ovh], float); I = np.array(ing, float)
    X = np.c_[np.ones(len(O)), O[:, 0], O[:, 1]]
    b = l1_fit(X, O[:, 2]); pr = X @ b; ok = (O[:, 2] > 0) & (pr > 0)
    bi = l1_fit(np.c_[np.ones(len(I)), I[:, 0]], I[:, 1])
    per_arm = {}
    for tag in sorted(set(o[3] for o in ovh)):
        m = np.array([o[3] == tag for o in ovh]); per_arm[tag] = float(np.median(O[m, 2] - pr[m]))
    out = dict(turn_overhead=float(b[0]) / 1e3, ovh_tok=float(b[1]), ovh_out=float(b[2]), ovh_sigma=rsd(np.log(O[ok, 2] / pr[ok])),
               ingress_base=float(bi[0]) / 1e3, ingress_tok=float(bi[1]))
    # run-to-run offset of the client overhead (arm medians scatter beyond per-turn noise; e.g. the client's pinned cores)
    out["ovh_run_sd"] = rsd(list(per_arm.values())) / 1e3
    return out, dict(n_turns=len(O), n_ingress=len(I), ovh_median_resid_ms_per_arm=per_arm)


def machine_info(rows):
    info = defaultdict(list)
    for r in rows:
        G = gpu_requests(r["dir"]); m = next(iter(G.values()))
        info["boot_fingerprint"].append(round(m["request_finished_ts"] - m["inference_time"]) if m.get("inference_time") else None)
        info["date"].append(datetime.datetime.fromtimestamp(m["request_received_ts"]).strftime("%Y-%m-%d %H:%M"))
        g = os.path.join(r["dir"], "gpus.txt"); info["gpus"].append(open(g).read().strip() if os.path.exists(g) else "")
        c = os.path.join(r["dir"], "cotenant.log")
        ld = [float(x) for x in re.findall(r"load=([\d.]+)", open(c, errors="replace").read())] if os.path.exists(c) else []
        info["host_load_median"].append(float(np.median(ld)) if ld else None)
    return {k: dict(zip([r["tag"] for r in rows], v)) for k, v in info.items()}


def cmd_fit(a):
    from fit_step_costs import fit_joint
    rows = read_manifest(a.manifest); logs = [os.path.join(r["dir"], "server.log.gz") for r in rows]
    tmp = os.path.splitext(a.out)[0] + "_steps.json"
    steps, sinfo = fit_joint(logs, out_path=tmp)
    noise = noise_from_resid(os.path.splitext(tmp)[0] + "_resid.npz")
    cli, cinfo = client_and_ingress(rows)
    prof = {**steps, "ttft_floor": 0.0, "ftd_median": 0.0, "ftd_sigma": 1.0, **cli, **noise,
            "_about": a.about or f"profile_machine.py fit over {len(rows)} arms of {a.manifest}", "_stage": "A (run `ttft` for ftd + extra ingress)",
            "_fit": {"steps": sinfo, **cinfo}, "_machine": machine_info(rows)}
    json.dump(prof, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in prof.items() if not k.startswith("_")}, indent=1, default=float))


def cmd_client(a):
    rows = read_manifest(a.manifest); prof = json.load(open(a.profile))
    cli, cinfo = client_and_ingress(rows); prof.update(cli); prof.setdefault("_fit", {}).update(cinfo)
    json.dump(prof, open(a.profile, "w"), indent=1, default=float); print(cli); print(cinfo["ovh_median_resid_ms_per_arm"])


def cmd_ttft(a):
    """First-token path + residual ingress from de-queued small requests of OPEN-LOOP sims run with the stage-A profile:
    d = GPU TTFT - sim ttft_raw = b*Kprompt (extra ingress) + ftd (median f0); ftd_sigma chosen so that sim_raw + ftd matches the GPU TTFT quantiles (ftd is added
    per request after the dynamics, so no re-simulation is needed)."""
    import pandas as pd
    rows = read_manifest(a.manifest); prof = json.load(open(a.profile)); D = []
    for r in rows:
        f = os.path.join(a.sim_dir, r["tag"] + a.suffix, f"req_{r['arm']}.csv")
        if not os.path.exists(f): print("missing", f); continue
        G = gpu_requests(r["dir"]); t0 = min(m["request_received_ts"] for m in G.values())
        s = pd.read_csv(f)
        for q in s.itertuples():
            m = G.get((q.program, q.turn))
            if m is None or m["request_received_ts"] - t0 < 300: continue
            ing = prof.get("ingress_base", 0.0) + prof["ingress_tok"] * q.prompt_len / 1e6
            if q.first_sched - q.arrival - ing > a.q_max or q.prompt_len - q.cached > 2048: continue      # de-queued, small extend
            gt = m["response_sent_to_client_ts"] - m["request_received_ts"]
            D.append((q.prompt_len / 1e3, 1e3 * (gt - q.ttft_raw), q.ttft_raw, gt))
    D = np.array(D); K, d, sraw, gt = D.T
    f0, bk = l1_fit(np.c_[np.ones(len(D)), K], d)
    med = max(1.0, f0) / 1e3; base = sraw + bk * K / 1e3
    z = np.random.default_rng(0).standard_normal(len(D)); qs = [50, 80, 90, 99]; gq = np.percentile(gt, qs)
    # match the body quantiles only: the GPU p99 of "de-queued" requests is set by GPU-side queueing / host stalls
    sig = min(np.arange(0.0, 1.51, 0.05), key=lambda sg: np.abs(np.log(np.percentile(base + med * np.exp(sg * z), qs[:3]) / gq[:3])).sum())
    prof["ingress_tok"] += float(bk); prof.update(ftd_median=max(float(f0), 1.0) / 1e3, ftd_sigma=float(sig))
    fq = np.percentile(base + med * np.exp(sig * z), qs)
    prof["_stage"] = "A+B"; prof.setdefault("_fit", {})["ttft"] = dict(n=len(D), extra_ingress_ms_per_K=float(bk), gpu_ttft_q=dict(zip(map(str, qs), gq.tolist())), fitted_q=dict(zip(map(str, qs), fq.tolist())))
    json.dump(prof, open(a.profile, "w"), indent=1, default=float)
    print(f"ttft: n={len(D)} extra ingress {bk:+.3f} ms/K -> ingress_tok {prof['ingress_tok']:.3f}; ftd median {f0:.1f} ms, sigma {sig:.2f}")
    print("  de-queued TTFT quantiles p50/p80/p90/p99  GPU " + " ".join(f"{x:.3f}" for x in gq) + "   fitted " + " ".join(f"{x:.3f}" for x in fq))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("manifest"); f.add_argument("--out", required=True); f.add_argument("--about", default=None)
    t = sp.add_parser("ttft"); t.add_argument("manifest"); t.add_argument("--profile", required=True); t.add_argument("--sim-dir", required=True); t.add_argument("--suffix", default="", help="sim run dir = SIM_DIR/<tag><suffix>"); t.add_argument("--q-max", type=float, default=0.05, help="max sim wait (s) beyond ingress for a request to count as de-queued")
    c = sp.add_parser("client"); c.add_argument("manifest"); c.add_argument("--profile", required=True)
    a = ap.parse_args(); dict(fit=cmd_fit, ttft=cmd_ttft, client=cmd_client)[a.cmd](a)


if __name__ == "__main__":
    main()
