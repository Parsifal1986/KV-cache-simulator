"""Identical recorded conversations (tau2 banking: 34 tasks x 6 copies) -- do their copies generate the same tokens on
the GPU? Per (copy group, turn): LCP(copy output, the group's modal output) / len, for copies still on the group's track.
usage: copy_divergence.py STREAM.pkl ARM_DIR [--json OUT]   (run from DATA_ROOT)"""
import sys, os, re, gzip, tempfile, hashlib, collections, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import offline_replay as OR
rhs = OR.rhs


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i


args = sys.argv[1:]; out_json = None
if "--json" in args: i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
byp = collections.defaultdict(list)
for r in OR.load_stream(args[0]): byp[r.program_id or r.rid].append(r)
group = {}
for pid, lst in byp.items():
    lst.sort(key=lambda r: r.arrival); h = hashlib.sha1()
    for r in lst: h.update(np.asarray(r.prompt_ids, np.int64).tobytes()); h.update(b"|"); h.update(np.asarray(r.output_ids, np.int64).tobytes())
    group[pid] = h.hexdigest()
sizes = collections.Counter(collections.Counter(group.values()).values())
print("copy-group sizes:", dict(sizes))
arm = args[1]
log = next(os.path.join(arm, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm, x)))
fd, src = tempfile.mkstemp(suffix=".log", dir=os.environ.get("TMPDIR"))
with (gzip.open(log, "rb") if log.endswith(".gz") else open(log, "rb")) as f, os.fdopen(fd, "wb") as g:
    for line in f:
        if b'"event": "request.' in line: g.write(line)
G = rhs.parse_stream(src); os.unlink(src)
gen = {}
for q in G:
    m = re.match(r"(.*)#t(\d+)$", q.rid)
    if m: gen[(m.group(1), int(m.group(2)))] = list(q.output_ids)
members = collections.defaultdict(list)
for pid, g in group.items(): members[g].append(pid)
fr = []; on_track_turns = []
for g, pids in members.items():
    if len(pids) < 2: continue
    track = set(pids); t = 0
    while len(track) >= 2:
        outs = {p: gen.get((p, t)) for p in track}; outs = {p: o for p, o in outs.items() if o}
        if len(outs) < 2: break
        modal = collections.Counter(tuple(o) for o in outs.values()).most_common(1)[0][0]
        nxt = set()
        for p, o in outs.items():
            f = lcp(o, modal) / max(1, len(modal)); fr.append(f)
            if f >= 1.0 and len(o) == len(modal): nxt.add(p)
        on_track_turns.append(len(nxt) / len(outs)); track = nxt; t += 1
fr = np.array(fr)
print(f"{arm}: {len(fr)} copy-turns | identical to group mode {100*(fr>=1).mean():.0f}% | frac p10/25/50 {np.percentile(fr,[10,25,50]).round(2)} | turns followed per group mean {len(on_track_turns)/max(1,len(members)):.1f}")
if out_json:
    json.dump(dict(stream=args[0], arm=arm, n=len(fr), frac_quantiles=np.percentile(fr, np.arange(101)).tolist()), open(out_json, "w"), indent=1); print("wrote", out_json)
