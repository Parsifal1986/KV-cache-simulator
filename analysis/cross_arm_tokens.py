"""Do the token sequences depend on the policy? For GPU arms of one work point, per (program, turn) present in both:
identical prompt, identical output, output LCP / len, prompt length difference.
usage: cross_arm_tokens.py ARM_A ARM_B [ARM_A2 ARM_B2 ...]   (pairs; run from DATA_ROOT)"""
import sys, os, re, gzip, tempfile, gc
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import offline_replay as OR
rhs = OR.rhs


def tokens(arm):
    log = next(os.path.join(arm, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm, x)))
    fd, src = tempfile.mkstemp(suffix=".log", dir=os.environ.get("TMPDIR"))
    with (gzip.open(log, "rb") if log.endswith(".gz") else open(log, "rb")) as f, os.fdopen(fd, "wb") as g:
        for line in f:
            if b'"event": "request.' in line: g.write(line)
    G = rhs.parse_stream(src); os.unlink(src); out = {}
    for q in G:
        m = re.match(r"(.*)#t(\d+)$", q.rid)
        if m: out[(m.group(1), int(m.group(2)))] = (np.asarray(q.prompt_ids, np.int32), np.asarray(q.output_ids, np.int32))
    return out


def lcp(a, b):
    n = min(len(a), len(b)); d = np.nonzero(a[:n] != b[:n])[0]
    return int(d[0]) if len(d) else n


a = sys.argv[1:]
for A, B in zip(a[0::2], a[1::2]):
    TA = tokens(A); TB = tokens(B); keys = sorted(set(TA) & set(TB))
    sp = so = 0; fr = []; dl = []; pl = []
    for k in keys:
        (pa, oa), (pb, ob) = TA[k], TB[k]
        sp += len(pa) == len(pb) and bool((pa == pb).all()); so += len(oa) == len(ob) and bool((oa == ob).all())
        fr.append(lcp(oa, ob) / max(1, max(len(oa), len(ob)))); dl.append(abs(len(pa) - len(pb))); pl.append(lcp(pa, pb) / max(1, len(pa)))
    n = len(keys); fr = np.array(fr); dl = np.array(dl); pl = np.array(pl)
    print(f"{os.path.basename(os.path.dirname(A))}/{os.path.basename(A)} vs {os.path.basename(B)}: n={n} (only A {len(TA)-n}, only B {len(TB)-n}) | "
          f"prompt identical {100*sp/n:.0f}% (common-prefix p50 {100*np.median(pl):.0f}% p10 {100*np.percentile(pl,10):.0f}%) | output identical {100*so/n:.0f}% "
          f"(LCP/len p50 {np.median(fr):.2f} p10 {np.percentile(fr,10):.2f}) | |dprompt| p50/p90 {np.median(dl):.0f}/{np.percentile(dl,90):.0f}", flush=True)
    del TA, TB; gc.collect()
