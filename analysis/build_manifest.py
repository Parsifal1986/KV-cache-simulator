import re, os, glob, numpy as np
C = []
C += sorted(glob.glob("results/terminus_recordings/spider2_terminus/x2/setup_*_750k_s*/*/server.log.gz")) + sorted(glob.glob("results/terminus_recordings/spider2_terminus/x2_local*/setup_*_750k_s*/*/server.log.gz"))
C += sorted(glob.glob("results/terminus_recordings/mldev_terminus/x2/setup_*_750k_s*/*/server.log.gz"))
C += sorted(glob.glob("results/terminus_recordings/mldev_terminus/x3_fcfs*/setup_*_750k_s*/*/server.log.gz"))
C += sorted(glob.glob("results/tau2_banking/x3/setup_*_750k_s*/*/server.log.gz"))
C += sorted(glob.glob("results/terminus_recordings/*/x*_lpm/setup_*_750k_s*/*/server.log.gz"))
STREAM = {"spider2": ("results/terminus_recordings/spider2_terminus/trace_x/spider2_terminus_x2.pkl", 4, 9765.1),
          "mldev_x2": ("results/terminus_recordings/mldev_terminus/trace_x/mldev_terminus_x2.pkl", 4, 18244.0),
          "mldev_x3": ("results/terminus_recordings/mldev_terminus/trace_x/mldev_terminus_x3.pkl", 4, 27366.0),
          "banking": ("results/tau2_banking/trace_x/banking_x3.pkl", 20, 11871.6)}
bad = re.compile(r"polluted|aborted|attempt|cotenant|PARTIAL|killed")
for L in C:
    d = os.path.dirname(L)
    if bad.search(d) or not os.path.exists(d + "/DONE"): continue
    rl = open(d + "/run.log", errors="replace").read()
    m = re.search(r"\[([\w.]+)\] GPUS=([\d,]+) TP=(\d) POOL=(\d+) CONC=(\d+) SEED=(\d+) flags: (.*)", rl)
    if not m: continue
    arm, gpus, tp, pool, conc, seed, flags = m.groups()
    tg = re.search(r"trial_gap=([\d.]+)s", rl); tg = tg.group(1) if tg else ""
    key = "spider2" if "spider2" in d else "banking" if "banking" in d else ("mldev_x2" if "/x2/" in d else "mldev_x3")
    st, rc, rs = STREAM[key]
    sched = "lpm" if "_lpm" in d else "fcfs"
    ld = [float(x) for x in re.findall(r"load=([\d.]+)", open(d + "/cotenant.log", errors="replace").read())] if os.path.exists(d + "/cotenant.log") else []
    # foreign GPU activity on OTHER cards is irrelevant; host load is the CPU-side co-tenancy measure
    var = re.search(r"/x\d(?:_[a-z]+)?(?:_(l\d+|local\d+))?/", d); var = var.group(1) if var and var.group(1) else ""
    tag = f"{key.split('_')[0][:2]}{conc}s{seed}{'L' if sched == 'lpm' else ''}{var}_{os.path.basename(d)}"
    info = f"{np.median(ld):.0f} [{min(ld):.0f}-{max(ld):.0f}]" if ld else "?"
    print("\t".join([tag, d, st, conc, seed, str(rc), str(rs), tg, os.path.basename(d).split("_")[0] if os.path.basename(d).startswith(("lru", "rr")) else os.path.basename(d), sched, gpus, info]))
