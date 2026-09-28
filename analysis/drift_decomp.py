"""Where does closed-loop arrival drift come from? Per (program, turn>0) pair: drift change across the turn =
(sim e2e_{k-1} - gpu e2e_{k-1}) [service] + (sim client gap - gpu client gap) [client], gap = arrival_k - finish_{k-1}.
Aggregated by GPU 5-min window of turn k, mean over sim replicates. usage: drift_decomp.py GPU_ARM 'SIMGLOB'"""
import types, sys, glob, gzip, re, json, numpy as np, pandas as pd
gdir, sg = sys.argv[1], sys.argv[2]
dec = json.JSONDecoder(); G = {}
for line in gzip.open(gdir + "/server.log.gz", "rt", errors="replace"):
    if "request.finished" not in line: continue
    m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
    if not m: continue
    k = line.find("{", line.rfind('"meta_info"')); meta, _ = dec.raw_decode(line, k)
    if meta.get("cached_tokens") is None: continue
    G[(m.group(1), int(m.group(2)))] = (meta["request_received_ts"], meta["request_finished_ts"], meta["response_sent_to_client_ts"], meta["prompt_tokens"], meta["completion_tokens"], meta["prompt_tokens"] - meta["cached_tokens"])
t0 = min(v[0] for v in G.values())
rows = []
for f in sorted(glob.glob(sg)):
    s = pd.read_csv(f); s0 = s.arrival.min(); d = {(x["program"], int(x["turn"])): types.SimpleNamespace(**x) for x in s.to_dict("records")}
    for (p, t), r in d.items():
        if t == 0 or (p, t - 1) not in d or (p, t) not in G or (p, t - 1) not in G: continue
        q = d[(p, t - 1)]; g1, g0 = G[(p, t)], G[(p, t - 1)]
        rows.append(dict(f=f, w=int((g1[0] - t0) // 300) * 5, svc=(q.e2e) - (g0[1] - g0[0]), gap=(r.arrival - (q.arrival + q.e2e)) - (g1[0] - g0[1]),
                         drift=(r.arrival - s0) - (g1[0] - t0), ttft=q.ttft - (g0[2] - g0[0]), dec=(q.e2e - q.ttft) - (g0[1] - g0[2]), out=g0[4], unc_s=q.prompt_len - q.cached, unc_g=g0[5]))
R = pd.DataFrame(rows)
print(f"{'win':>5} {'n/rep':>6} | {'drift med':>9} | per-turn mean error (s): {'service':>8} = {'ttft':>7} + {'decode':>7} | {'client gap':>10} | sum svc+gap per program-turn")
for w, g in R.groupby("w"):
    n = len(g) / R.f.nunique()
    print(f"{w:>3}-{w+5:<3} {n:6.0f} | {g.drift.median():+9.1f} | {g.svc.mean():+8.3f} = {g.ttft.mean():+7.3f} + {g.dec.mean():+7.3f} | {g.gap.mean():+10.3f} | {g.svc.mean()+g.gap.mean():+.3f}")
