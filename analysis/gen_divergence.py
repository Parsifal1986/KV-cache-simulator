"""Where does the server's own generation leave the recorded output? d = LCP(generated, recorded output) per turn.
usage: gen_divergence.py STREAM.pkl ARM_DIR [ARM_DIR ...] [--json OUT]   (run from DATA_ROOT)
--json writes the pooled d/len distribution (101 quantiles) for sgl_sim.py --diverge-json (unstitched replay clients)."""
import sys, os, re, gzip, tempfile, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import offline_replay as OR
rhs = OR.rhs


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i + 256 <= n and a[i:i + 256] == b[i:i + 256]: i += 256
    while i < n and a[i] == b[i]: i += 1
    return i


args = sys.argv[1:]; out_json = None
if "--json" in args: i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
rec = {}; pooled = []
for r in OR.load_stream(args[0]):
    by = rec.setdefault(r.program_id or r.rid, []); by.append(r)
for v in rec.values(): v.sort(key=lambda r: r.arrival)
for arm in args[1:]:
    log = next(os.path.join(arm, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm, x)))
    fd, src = tempfile.mkstemp(suffix=".log", dir=os.environ.get("TMPDIR"))
    with (gzip.open(log, "rb") if log.endswith(".gz") else open(log, "rb")) as f, os.fdopen(fd, "wb") as g:
        for line in f:
            if b'"event": "request.' in line: g.write(line)
    G = rhs.parse_stream(src); os.unlink(src)
    d = []; frac = []; L = []; full = 0
    for q in G:
        m = re.match(r"(.*)#t(\d+)$", q.rid)
        if not m or m.group(1) not in rec: continue
        t = int(m.group(2)); lst = rec[m.group(1)]
        if t >= len(lst): continue
        ro = list(lst[t].output_ids); go = list(q.output_ids)
        if not ro or not go: continue
        k = lcp(go, ro); d.append(k); L.append(len(ro)); frac.append(k / len(ro)); full += k >= min(len(go), len(ro))
    d = np.array(d); L = np.array(L); frac = np.array(frac); pooled.extend(frac.tolist())
    print(f"{arm}\n  n={len(d)} out len p50 {np.median(L):.0f} | d p10/25/50/75/90 {np.percentile(d,[10,25,50,75,90]).round(0)} | d/len p50 {np.median(frac):.2f} "
          f"| identical {100*full/len(d):.0f}% | d==0 {100*(d==0).mean():.0f}% | tail recomputed/turn mean {np.mean(L-d):.0f}")
if out_json:
    import json
    json.dump(dict(stream=args[0], arms=args[1:], n=len(pooled), frac_quantiles=np.percentile(pooled, np.arange(101)).tolist()), open(out_json, "w"), indent=1)
    print("wrote", out_json)
