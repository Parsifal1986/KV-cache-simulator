"""Clean single-request prefill cost from GPU logs: requests that were the only new seq of a prefill step run on an IDLE
server (#running-req 0, #queue-req 0). TTFT - dispatch delay = scheduler pickup + one prefill step + first-token path.
usage: idle_prefill.py PROFILE.json ARM_DIR..."""
import sys, gzip, re, json, datetime, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from profile_machine import gpu_requests
P = json.load(open(sys.argv[1])); rows = []
PRE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP\d+)? TP0[^\]]*\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: [\d.]+, #running-req: (\d+), #queue-req: (\d+)")
for d in sys.argv[2:]:
    G = gpu_requests(d); idx = {}
    for line in gzip.open(d + "/server.log.gz", "rt", errors="replace"):
        m = PRE.match(line)
        if m and m.group(2) == "1" and m.group(5) == "0" and m.group(6) == "0":
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(); idx.setdefault((int(m.group(3)), int(m.group(4))), []).append(ts)
    for k, mt in G.items():
        new = mt["prompt_tokens"] - mt["cached_tokens"]; c = mt["cached_tokens"]
        if new > 8192 or new < 1: continue
        hit = [ts for ts in idx.get((new, c), []) if -1.5 <= ts - mt["request_received_ts"] <= 1.5]
        if len(hit) != 1: continue
        ttft = mt["response_sent_to_client_ts"] - mt["request_received_ts"]; disp = mt["api_server_dispatch_finish_ts"] - mt["request_received_ts"]
        model = (P["p0"] + P["p_tok"] * new / 1e3 + P["p_attn"] * new * (c + new / 2) / 1e9) / 1e3
        rows.append((c / 1e3, new / 1e3, 1e3 * (ttft - disp), 1e3 * model))
R = np.array(rows); print(f"n={len(R)} idle single-request prefills")
print(" prefix K | n   | measured (TTFT-dispatch) ms median | model prefill ms | excess")
for a in range(0, 80, 10):
    m = (R[:, 0] >= a) & (R[:, 0] < a + 10)
    if m.sum() >= 5: print(f" {a:2d}-{a+10:<3d}   | {m.sum():4d} | {np.median(R[m,2]):7.1f} | {np.median(R[m,3]):6.1f} | {np.median(R[m,2]-R[m,3]):+6.1f}")
X = np.c_[np.ones(len(R)), R[:, 1], R[:, 1] * (R[:, 0] + R[:, 1] / 2), R[:, 0]]
from profile_machine import l1_fit
b = l1_fit(X, R[:, 2]); print("L1 fit TTFT-dispatch ms = %.1f + %.2f*Knew + %.3f*Knew*(Kpre+Knew/2) + %.3f*Kpre" % tuple(b))
