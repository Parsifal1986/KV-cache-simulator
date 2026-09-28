import sys, glob, gzip, re, json, numpy as np, pandas as pd
gdir, sg = sys.argv[1], sys.argv[2]
dec = json.JSONDecoder(); G = {}
for line in gzip.open(gdir + "/server.log.gz", "rt", errors="replace"):
    if "request.finished" not in line: continue
    m = re.search(r'"rid":\s*"(.*?)#t(\d+)"', line)
    if not m: continue
    k = line.find("{", line.rfind('"meta_info"')); meta, _ = dec.raw_decode(line, k)
    if meta.get("cached_tokens") is None: continue
    G[(m.group(1), int(m.group(2)))] = (meta["request_received_ts"], meta["response_sent_to_client_ts"], meta["request_finished_ts"], meta["completion_tokens"])
t0 = min(v[0] for v in G.values()); acc = {}
for f in sorted(glob.glob(sg)):
    s = pd.read_csv(f)
    for r in s.itertuples():
        g = G.get((r.program, r.turn))
        if g is None or r.output_len < 20: continue
        w = int((g[0] - t0) // 300) * 5; a = acc.setdefault(w, [0, 0, 0, 0])
        a[0] += r.e2e - r.ttft_raw; a[1] += g[2] - g[1]; a[2] += 1; a[3] += g[3]
print(" win  | n/rep | GPU decode s/turn (ms/tok) | sim/GPU")
nf = len(glob.glob(sg))
for w in sorted(acc):
    a = acc[w]; print(f"{w:2d}-{w+5:<3d}| {a[2]/nf:5.0f} | {a[1]/a[2]:6.2f} s ({1e3*a[1]/a[3]:5.1f}) | {a[0]/a[1]:.3f}")
