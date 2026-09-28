#!/usr/bin/env python3
"""Event-driven (discrete-event) simulator of one aggregated SGLang server replaying a recorded agent stream.

The clock only jumps: it moves from one scheduler step's end to the next, or to the next request arrival when the
server is idle. Every step is built with the same rules as python/sglang/srt/managers/scheduler.py, so queueing,
head-of-line blocking and eviction timing come out of the mechanics instead of being fitted:

  get_next_batch_to_run  prefill-first: a prefill batch runs whenever one can be formed, else one decode step.
  PrefillAdder           open chunked request first (add_chunked_req), then waiting requests in policy order (fcfs =
                         arrival, lpm = longest cached prefix, fcfs above 128 waiting). Budgets exactly as in
                         schedule_policy.py: rem_total = available + evictable - sum_running min(max_new-out, 4096) *
                         new_token_ratio - (extend + min(max_new, 4096) + 1 per request added this step);
                         rem_chunk = chunked_prefill_size, rem_input = max_prefill_tokens. NO_TOKEN sets batch_is_full.
                         The prefix is matched when the request is tried, so material evicted while it waited is lost.
  batch_is_full          reset only when the running batch shrinks (a request finishes or is retracted).
  prefill result         each extended request inserts its filled prefix into the tree and moves its lock there
                         (cache_unfinished_req); a request whose prompt is complete emits its first token at step end.
  decode step            check_decode_mem: bs new tokens, evicting from the tree if needed; if still short, retract
                         (fewest output tokens first, as retract_decode) and reset new_token_ratio; otherwise decay
                         new_token_ratio 0.7 -> 0.098 over 600 steps (reset to 0.7 when the server goes idle).
  finish                 cache_finished_req: insert prompt+output, unlock; next turn of the program arrives after its
                         recorded tool gap (or the trial ends and the next trial takes the slot).

Step costs are regressions on the servers' own step lines (steplog_parse.py; cloud H100 TP2, Qwen3-Coder-30B-A3B):
  prefill ms = p0 + p_tok*Knew + p_attn*sum(new_i*(prefix_i+new_i/2))/1e9 + p_open*[step leaves a chunk open]
  decode  ms = d0 + d_tok*Ktokens_in_running + d_req*bs            (bs <= cuda_graph_max_bs, CUDA graph)
             = n0 + n_tok*Ktokens_in_running + n_req*bs            (bs >  cuda_graph_max_bs, eager)
Pure decode runs are advanced in macro-steps (k identical-rule steps at once) up to the next finish or arrival.

usage: des_sim.py STREAM.pkl --kv 750000 --conc 46 --seed 36 --arms rrdeclband0.001,rrdeclchunk1024 --out-dir DIR
"""
import os
# one simulation = one core: torch/OpenMP would otherwise spawn a pool per process (128 threads each measured on this
# box), and a 48-way ensemble then oversubscribes the CPU and ~5 GB x N of RSS pushes the container's memory cap.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import argparse, csv, heapq, json, math, random, sys, types
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_replay as OR
rhs = OR.rhs
import torch
torch.set_num_threads(1)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey

CLIP = 4096          # SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION
INIT_RATIO, MIN_RATIO_FACTOR, DECAY_STEPS, RETRACT_STEPS = 0.7, 0.14, 600, 20


class Req:
    __slots__ = ("pid", "turn", "rec", "ids", "base_ids", "out_base", "enq", "out_len", "arrival", "first_sched", "ttft_t", "finish_t", "prefix", "filled", "out", "node",
                 "uncommitted", "chunks", "retracts", "cached_at_admit", "tried")

    def __init__(s, pid, turn, rec, arrival):
        s.pid, s.turn, s.rec, s.arrival = pid, turn, rec, arrival
        s.ids = s.base_ids = rec.prompt_ids; s.out_base = 0; s.enq = arrival; s.out_len = max(1, len(rec.output_ids))
        s.first_sched = s.ttft_t = s.finish_t = None; s.prefix = 0; s.filled = 0; s.out = 0; s.node = None
        s.uncommitted = 0; s.chunks = 0; s.retracts = 0; s.cached_at_admit = None; s.tried = 0


def stitch_like_replay(reqs):
    """Emulate gpu_replay_harbor's prompt stitching (default ON since 2026-09-20). On the GPU every turn's output is the
    server's OWN generation, and the next prompt is rebuilt as own_history[:k] + recorded_next_prompt[k:], with
    k = LCP(recorded prompt + recorded output, recorded next prompt). Outputs therefore never coincide across programs
    even when the recording has identical conversations (tau2 banking: repeated trials of one task share their first
    turns verbatim, 5% of prompt tokens). Here each program's outputs get program-unique token ids (the server's
    generation is not bit-identical across batches) and prompts are stitched the same way; own-history reuse is kept."""
    import collections as _co
    from dataclasses import replace as _replace
    byp = _co.defaultdict(list)
    for r in reqs: byp[r.program_id or r.rid].append(r)
    nxt = [10 ** 9]; out = []
    def lcp(a, b):
        n = min(len(a), len(b)); i = 0
        while i + 256 <= n and a[i:i + 256] == b[i:i + 256]: i += 256
        while i < n and a[i] == b[i]: i += 1
        return i
    for pid, lst in byp.items():
        lst.sort(key=lambda r: r.arrival); hist = None; prev = None
        for r in lst:
            if prev is None: prompt = list(r.prompt_ids)
            else:
                k = lcp(list(prev.prompt_ids) + list(prev.output_ids), r.prompt_ids)
                prompt = hist[:k] + list(r.prompt_ids[k:])
            gen = list(range(nxt[0], nxt[0] + len(r.output_ids))); nxt[0] += len(r.output_ids)
            out.append(_replace(r, prompt_ids=prompt, output_ids=gen)); hist = prompt + gen; prev = r
    return sorted(out, key=lambda r: r.arrival)


def ids_from_gpu(reqs, arm_dir):
    """Replace every turn's prompt/output ids with what the GPU arm actually sent (stitched) and generated, keyed by
    rid '<program_id>#t<turn>'. A program is cut at its first turn the arm never finished (end-of-run or dropped)."""
    import gzip as _gz, re as _re, tempfile as _tf, collections as _co
    from dataclasses import replace as _replace
    log = next(os.path.join(arm_dir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm_dir, x)))
    src = log
    if log.endswith(".gz"):
        fd, src = _tf.mkstemp(suffix=".log", dir=os.environ.get("TMPDIR"))
        with _gz.open(log, "rb") as f, os.fdopen(fd, "wb") as g:
            for line in f:
                if b'"event": "request.' in line: g.write(line)
    try:
        G = rhs.parse_stream(src)
    finally:
        if src != log: os.unlink(src)
    ids = {}; clipped = 0; last_fin = _co.defaultdict(float)
    run_end = max(q.finish for q in G)
    for q in G:
        m0 = _re.match(r"(.*)#t(\d+)$", q.rid)
        if m0: last_fin[m0.group(1)] = max(last_fin[m0.group(1)], q.finish)
    for q in G:
        m = _re.match(r"(.*)#t(\d+)$", q.rid)
        if not m: continue
        if len(q.prompt_ids) != q.prompt_tokens or len(q.output_ids) != q.completion_tokens: clipped += 1
        ids[(m.group(1), int(m.group(2)))] = (list(q.prompt_ids), list(q.output_ids))
    byp = _co.defaultdict(list)
    for r in reqs: byp[r.program_id or r.rid].append(r)
    def lcp(a, b):
        n = min(len(a), len(b)); i = 0
        while i + 256 <= n and a[i:i + 256] == b[i:i + 256]: i += 256
        while i < n and a[i] == b[i]: i += 1
        return i
    out = []; cut = 0; cont = 0; nxt = [2 * 10 ** 9]
    for pid, lst in byp.items():
        lst.sort(key=lambda r: r.arrival); hist = None
        for t, r in enumerate(lst):
            if (pid, t) in ids:
                pr, ou = ids[(pid, t)]; out.append(_replace(r, prompt_ids=pr, output_ids=ou)); hist = pr + ou; continue
            next_gap = max(0.0, r.arrival - lst[t - 1].finish) if t else 0.0
            if t == 0 or last_fin[pid] + next_gap + 60.0 < run_end:
                cut += len(lst) - t; break                       # the client should have sent this turn before the run ended: dropped
            # the GPU run was stopped while this program was still alive: continue it the way the replay client would
            k = lcp(list(lst[t - 1].prompt_ids) + list(lst[t - 1].output_ids), r.prompt_ids)
            pr = hist[:k] + list(r.prompt_ids[k:]); ou = list(range(nxt[0], nxt[0] + len(r.output_ids))); nxt[0] += len(r.output_ids)
            out.append(_replace(r, prompt_ids=pr, output_ids=ou)); hist = pr + ou; cont += 1
    print(f"ids-from {arm_dir}: {len(ids)} GPU turns ({clipped} with clipped ids); {cut} turns cut (mid-run drops), {cont} turns continued past the end of the GPU run (stitched)", flush=True)
    return sorted(out, key=lambda r: r.arrival)


def simulate(reqs, spec, kv, conc, seed, trial_gap, P, occ_dt=5.0):
    clock = rhs.SimClock()
    import sglang.srt.mem_cache.evict_policy as ep_mod, sglang.srt.mem_cache.radix_cache as rc_mod, sglang.srt.mem_cache.rr_radix_cache as rr_mod
    saved = (rc_mod.time.monotonic, rr_mod.time.monotonic, ep_mod.time.monotonic); saved_time = rc_mod.time.time
    for m in (rc_mod, rr_mod, ep_mod): m.time.monotonic = clock.monotonic
    rc_mod.time.time = clock.monotonic
    try:
        return _simulate(reqs, spec, kv, conc, seed, trial_gap, P, occ_dt, clock)
    finally:
        for m, fn in zip((rc_mod, rr_mod, ep_mod), saved): m.time.monotonic = fn
        rc_mod.time.time = saved_time


def _simulate(reqs, spec, kv, conc, seed, trial_gap, P, occ_dt, clock):
    cache = spec.build(); declared = spec.declared
    jr = np.random.default_rng(P["jitter_seed"]) if P["jitter"] > 0 else None
    # machine noise model (measured on a pod's own step lines, fit_step_costs --joint residuals):
    #   decode speed factor resampled every dec_period s (sd dec_noise, ~independent between resamples),
    #   times a slow AR(1) drift (sd drift_noise, time constant drift_tau s); every prefill step iid (sd pre_noise);
    #   replay-client overhead per turn lognormal around turn_overhead (sd ovh_sigma).
    nr = np.random.default_rng(P["noise_seed"]) if P["noise_seed"] >= 0 else None
    nz = dict(next=0.0, fac=1.0, drift=0.0)
    ovh_run = float(nr.standard_normal()) * P["ovh_run_sd"] if nr is not None else 0.0   # this run's client-overhead offset
    ftd_rng = np.random.default_rng(1000 + max(P["noise_seed"], 0))
    ftd_run = float(ftd_rng.standard_normal()) * P.get("ftd_run_sigma", 0.0) if P["noise_seed"] >= 0 else 0.0   # this run's first-token offset
    def dec_factor():
        if nr is None: return 1.0
        if t >= nz["next"]:
            phi = math.exp(-P["dec_period"] / P["drift_tau"])
            nz["drift"] = phi * nz["drift"] + math.sqrt(1 - phi * phi) * P["drift_noise"] * nr.standard_normal()
            sg = P["dec_noise"]; nz["fac"] = math.exp(sg * nr.standard_normal() - sg * sg / 2 + nz["drift"]); nz["next"] = t + P["dec_period"]
        return nz["fac"]
    def pre_factor():
        if nr is None: return 1.0
        sg = P["pre_noise"]; return math.exp(sg * nr.standard_normal() - sg * sg / 2)
    def overhead(prompt_len=0, out_len=0):
        # replay client between stream end and the next send: fixed + per-token cost of stitching / JSON-encoding the
        # next prompt (measured: ~30 ms + 2.8 ms per 1K prompt tokens + 5.8 ms per 1K previous output tokens)
        base = max(0.0, P["turn_overhead"] + ovh_run + (P["ovh_tok"] * prompt_len + P["ovh_out"] * out_len) / 1e6)
        if nr is None: return base
        return base * math.exp(P["ovh_sigma"] * nr.standard_normal())
    def ingress(prompt_len):
        # server side between request_received and the scheduler's waiting queue (input_ids validation / IPC): per token
        return max(0.0, P["ingress_base"] + P["ingress_tok"] * prompt_len / 1e6)
    def jit(dt, scale=1.0):
        # mean-preserving lognormal step-time noise (GPU step times scatter ~7% decode / ~15% prefill around the fit)
        if jr is None: return dt
        sg = P["jitter"] * scale; return dt * float(jr.lognormal(-sg * sg / 2, sg))
    # ---------------- workload: trials -> programs -> turns with recorded tool gaps ----------------
    trials = rhs._trials(reqs)
    if seed is not None: random.Random(seed).shuffle(trials)
    progs = {}; trial_progs = defaultdict(list)
    for tid, rs, base, dur in trials:
        byp = defaultdict(list)
        for r in rs: byp[r.program_id or r.rid].append(r)
        for pid, lst in byp.items():
            lst.sort(key=lambda r: r.arrival)
            turns = [(r, max(0.0, (r.arrival - lst[i - 1].finish) if i else (r.arrival - base))) for i, r in enumerate(lst)]
            if P.get("drops") and pid in P["drops"]: turns = turns[:P["drops"][pid]]      # conversation the GPU client dropped mid-run
            progs[pid] = dict(turns=turns, k=0, trial=tid); trial_progs[tid].append(pid)
    trial_left = {tid: len(ps) for tid, ps in trial_progs.items()}; order = [tid for tid, *_ in trials]
    ev = []; seq = [0]; next_trial = [0]
    def push(t, kind, payload): seq[0] += 1; heapq.heappush(ev, (t, seq[0], kind, payload))
    def start_trial(t):
        if next_trial[0] >= len(order): return
        tid = order[next_trial[0]]; next_trial[0] += 1
        for pid in trial_progs[tid]:
            r0, g0 = progs[pid]["turns"][0]; send = t + g0; push(send + ingress(len(r0.prompt_ids)), "arr", (pid, send))
    open_loop = P.get("arrivals") is not None
    if open_loop:
        # open-loop validation: every (program, turn) arrives at the GPU arm's measured time, independent of the sim's own
        # finish times -> isolates scheduler/cost mechanics from closed-loop timing drift
        for pid, pg in progs.items():
            last = max((k for k in range(len(pg["turns"])) if (pid, k) in P["arrivals"]), default=-1); pg["last"] = last
            for k in range(len(pg["turns"])):
                if (pid, k) in P["arrivals"]:
                    ra = P["arrivals"][(pid, k)]
                    if k > 0 and 0 < P["open_until"] <= ra: continue          # hybrid: closed-loop from open_until on (pushed by finish)
                    push(ra + ingress(len(pg["turns"][k][0].prompt_ids)), "arr", (pid, k, ra))
    else:
        for _ in range(conc): start_trial(0.0)
    # ---------------- scheduler state ----------------
    waiting = []; running = []; chunked = [None]; full = [False]
    ratio = [INIT_RATIO * P["conservativeness"]]; min_ratio = ratio[0] * MIN_RATIO_FACTOR; decay = (ratio[0] - min_ratio) / DECAY_STEPS
    extra_cpu = [0.0]; probe = []; probe_next = [0.0]; finished_leaf = {}                                                        # scheduler CPU time not hidden by overlap (added to next step)
    counter = [1]; t = 0.0; last_start = [0.0]; records = []; occ = []; steps = []; next_occ = 0.0
    st = defaultdict(float)
    def tree_size(): return cache.protected_size() + cache.evictable_size()
    def uncommitted(): return sum(r.uncommitted for r in running) + (chunked[0].uncommitted if chunked[0] else 0)
    def available(): return kv - tree_size() - uncommitted()
    def fresh(n):
        v = torch.arange(counter[0], counter[0] + n, dtype=torch.int64); counter[0] += n; return v
    import time as _time
    cpu_log = defaultdict(float)                                           # sim-minute -> host CPU seconds in cache ops
    def evict(n):
        if n <= 0: return 0
        c0 = _time.perf_counter()
        res = cache.evict(EvictParams(num_tokens=n)); got = res.num_tokens_evicted or 0; st["evicted"] += got; st["evict_calls"] += 1
        dc = _time.perf_counter() - c0; cpu_log[("evict", int(t // 60))] += dc; cpu_log[("evict_n", int(t // 60))] += 1
        if P["cpu_evict_scale"] > 0: extra_cpu[0] += dc * P["cpu_evict_scale"]
        return got
    def match(r):
        mp = MatchPrefixParams(key=RadixKey(r.ids, None))
        # the band clock (_wall) advances with the request's CREATION time (time_stats.created_time = arrival), as on the server
        mp.req = types.SimpleNamespace(program_id=r.rec.program_id, program_state=r.rec.program_state or "active", time_stats=types.SimpleNamespace(created_time=getattr(r, "arrival", t)))
        m = cache.match_prefix(mp); return int(m.device_indices.numel()), m.last_device_node
    def commit_prefix(r, upto):
        """cache_unfinished_req: insert ids[:upto] into the tree, move the lock to the new last node."""
        key = r.ids[:upto]
        cache.insert(InsertParams(key=RadixKey(key, None), value=fresh(len(key))))
        _, node = match(types.SimpleNamespace(ids=key, rec=r.rec, arrival=r.arrival))
        cache.inc_lock_ref(node)
        if r.node is not None: cache.dec_lock_ref(r.node)
        r.node = node
    def finish(r):
        full_ids = r.base_ids + list(r.rec.output_ids[:r.out_base + r.out])
        cache.insert(InsertParams(key=RadixKey(full_ids, None), value=fresh(len(full_ids))))
        cache.dec_lock_ref(r.node); r.node = None; r.uncommitted = 0; r.finish_t = t
        # first-token response path; ftd_run_median (optional): per-run offset, this run's random effect sd ftd_run_sigma
        ftd_m = P["ftd_run_median"] * math.exp(ftd_run) if P.get("ftd_run_median") else P["ftd_median"]
        ftd = ftd_m * math.exp(P["ftd_sigma"] * ftd_rng.standard_normal()) if ftd_m > 0 else 0.0
        records.append(dict(program=r.pid, turn=r.turn, arrival=r.arrival, first_sched=r.first_sched, ttft=r.ttft_t - r.arrival + P["ttft_floor"] + ftd, ttft_raw=r.ttft_t - r.arrival, e2e=t - r.arrival + P["ttft_floor"],
                            prompt_len=len(r.base_ids), cached=r.cached_at_admit, output_len=r.out_base + r.out_len, chunks=r.chunks, retracts=r.retracts, tries=r.tried))
        pg = progs[r.pid]
        if open_loop:
            if P.get("probe_dead") and r.turn == pg["last"]: finished_leaf[r.pid] = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None))).last_device_node
            if declared and r.turn == pg["last"]:
                leaf = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None))).last_device_node
                cache._record_program_leaf(r.pid, leaf); cache.release_program([r.pid])
            k1 = r.turn + 1
            if 0 < P["open_until"] <= P["arrivals"].get((r.pid, k1), -1.0) and k1 <= pg["last"]:
                # hybrid open/closed loop: turns the GPU received after open_until follow the sim's own finish time
                nr_ = pg["turns"][k1][0]; send = t + pg["turns"][k1][1] + overhead(len(nr_.prompt_ids), r.out_base + r.out_len)
                push(send + ingress(len(nr_.prompt_ids)), "arr", (r.pid, k1, send))
            return
        pg["k"] += 1
        if pg["k"] < len(pg["turns"]):
            nr_ = pg["turns"][pg["k"]][0]; send = t + pg["turns"][pg["k"]][1] + overhead(len(nr_.prompt_ids), r.out_base + r.out_len)
            push(send + ingress(len(nr_.prompt_ids)), "arr", (r.pid, send))    # client send time; enters the scheduler after ingress
        else:
            if P.get("probe_dead"): finished_leaf[r.pid] = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None))).last_device_node
            if declared:
                leaf = cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None))).last_device_node
                cache._record_program_leaf(r.pid, leaf); cache.release_program([r.pid])
            tid = pg["trial"]; trial_left[tid] -= 1
            if trial_left[tid] == 0: push(t + trial_gap + overhead(0, r.out_base + r.out_len), "slot", tid)

    def prefill_cost(items, open_after):
        new = sum(n for n, _ in items); attn = sum(n * (p + n / 2) for n, p in items) / 1e9
        return (P["p0"] + P["p_tok"] * new / 1e3 + P["p_attn"] * attn + (P["p_open"] if open_after else 0.0)) / 1e3
    def decode_cost(bs, tokens, maxctx=0.0):
        # FA3 decode is paced by the LONGEST running sequence (fitted on step lines + request-log context, MAD 0.2-0.4 ms)
        if bs <= P["cg_max_bs"]: ms = P["d0"] + P["d_tok"] * tokens / 1e3 + P["d_req"] * bs + P.get("d_max", 0.0) * maxctx / 1e3
        else: ms = P["n0"] + P["n_tok"] * tokens / 1e3 + P["n_req"] * bs + P.get("n_max", 0.0) * maxctx / 1e3
        return max(ms, P["d_min"]) / 1e3

    def get_prefill_batch():
        """_get_new_batch_prefill_raw + PrefillAdder. Returns list of (req, extend_len, prefix_len) or None."""
        cut = last_start[0] if P["overlap"] else t
        elig = [r for r in waiting if r.enq <= cut + 1e-9]
        if (full[0] or not elig) and chunked[0] is None: return None
        # rem_total_tokens / cur_rem_tokens are LIVE properties: locking a matched prefix removes it from evictable
        def avail_evict(): return available() + cache.evictable_size()
        rem_total_off = sum(min(r.out_len - r.out, CLIP) * ratio[0] for r in running); cur_off = 0.0
        rem_input = P["max_prefill"]; rem_chunk = P["chunk"]; batch = []; new_chunked = None
        def upd(ext, max_new):
            nonlocal rem_total_off, cur_off, rem_input, rem_chunk
            rem_total_off += ext + max_new + 1; cur_off += ext + 1; rem_input -= ext; rem_chunk -= ext
        if chunked[0] is not None:
            r = chunked[0]; remaining = len(r.ids) - r.filled
            rem = min(rem_chunk, int(avail_evict() - rem_total_off))
            if rem <= 0: rem = rem_chunk
            ext = min(remaining, rem); trunc = ext < remaining
            batch.append((r, ext, r.filled)); upd(ext, 0 if trunc else min(r.out_len, CLIP))
            new_chunked = r if trunc else None
        if P["sched"] == "lpm" and len(elig) <= 128 and len(elig) > 1:
            elig.sort(key=lambda r: (-match(r)[0], r.arrival))
        for r in elig:
            if len(batch) - (1 if chunked[0] is not None else 0) >= P["max_running"] - len(running): full[0] = True
            if full[0]: break
            r.tried += 1
            prefix, node = match(r)
            if prefix >= len(r.ids): prefix = len(r.ids) - 1          # at least one token is always extended
            ext = len(r.ids) - prefix; total = ext + min(r.out_len - r.out, CLIP)
            if total >= avail_evict() - rem_total_off: full[0] = True; break          # NO_TOKEN
            if ext >= rem_input and batch: break                                        # OTHER
            cache.inc_lock_ref(node)                                                    # _lock_node, then re-check
            if total >= avail_evict() - rem_total_off:
                cache.dec_lock_ref(node); full[0] = True; break
            if ext <= rem_chunk:
                batch.append((r, ext, prefix)); upd(ext, min(r.out_len, CLIP))
            else:
                trunc = rem_chunk
                if trunc <= 0: cache.dec_lock_ref(node); break
                batch.append((r, trunc, prefix)); upd(trunc, 0); new_chunked = r
            if r.node is not None: cache.dec_lock_ref(r.node)
            r.node = node; r.prefix = prefix; r.filled = prefix
            if r.cached_at_admit is None: r.cached_at_admit = prefix
            if r.first_sched is None: r.first_sched = t
            if avail_evict() - rem_total_off <= 0 or avail_evict() - cur_off <= 0: full[0] = True; break
            if rem_input <= 0 or rem_chunk <= 0: break
        if not batch: return None
        for r, _, _ in batch[1 if chunked[0] is not None else 0:]: waiting.remove(r)
        chunked[0] = new_chunked
        return batch

    while True:
        clock.t = t
        while ev and ev[0][0] <= t:
            te, _, kind, payload = heapq.heappop(ev)
            if kind == "arr":
                if open_loop:
                    pid, k, ra = payload; q = Req(pid, k, progs[pid]["turns"][k][0], ra)
                else:
                    pid, send = payload; pg = progs[pid]; q = Req(pid, pg["k"], pg["turns"][pg["k"]][0], send)
                q.enq = te; waiting.append(q)
            else: start_trial(t)
        if P.get("probe_dead") and t >= probe_next[0]:
            # tree composition: evictable tokens by (conversation finished?, tier)
            done_leaves = {id(n) for n in finished_leaf.values()}
            acc = defaultdict(int); stack = [cache.root_node]
            while stack:
                n = stack.pop(); stack.extend(n.children.values())
                if n is cache.root_node or n.value is None or n.lock_ref > 0: continue
                tier = getattr(n, "_rr_tier", 1); acc[("tier", tier)] += len(n.value)
            owners = defaultdict(int)
            for pid, leaf in finished_leaf.items():
                x = leaf
                while x is not None and x is not cache.root_node and x.value is not None and x.lock_ref == 0 and len(x.children) <= 1:
                    owners["finished_private"] += len(x.value); x = x.parent
            probe.append((t, acc[("tier", 0)], acc[("tier", 1)], owners["finished_private"])); probe_next[0] = t + 30.0
        if t >= next_occ:
            infl = cache.protected_size() + uncommitted()
            occ.append((t, infl, cache.evictable_size(), len(running), len(waiting), int(full[0] and bool(waiting)), ratio[0])); next_occ = t + occ_dt
        batch = get_prefill_batch()
        if batch:
            need = sum(ext for _, ext, _ in batch)
            if available() < need:                                              # evict_from_tree_cache: evicts num_tokens, not the shortfall
                evict(need)
                if available() < need: st["alloc_fail"] += 1
            for r, ext, _ in batch: r.uncommitted += ext
            dt = jit(prefill_cost([(ext, p) for _, ext, p in batch], chunked[0] is not None), 2.0) * pre_factor() * (P["slow_factor"] if P["slow_t0"] <= t < P["slow_t1"] else 1.0)
            steps.append(("P", t, len(batch), need, sum(p for _, _, p in batch), len(running), len(waiting), dt, cache.protected_size() + uncommitted(), int(full[0])))
            last_start[0] = t; t += dt; clock.t = t; st["prefill_time"] += dt; st["n_prefill"] += 1
            for r, ext, _ in batch:
                r.filled += ext; r.chunks += 1; r.uncommitted -= ext
                commit_prefix(r, r.filled)
                if r.filled >= len(r.ids):
                    if r.ttft_t is None: r.ttft_t = t
                    r.out = 1; r.uncommitted = 1
                    if r.out >= r.out_len: finish(r); full[0] = False
                    else: running.append(r)
            continue
        if running:
            bs = len(running); tokens = cache.protected_size() + uncommitted(); mctx = max(len(r.ids) for r in running); dstep = decode_cost(bs, tokens, mctx)   # tokens = the log's #token (in-flight KV)
            # identical-rule decode steps that can run before anything else could change the schedule
            k = min(r.out_len - r.out for r in running)                       # next finish
            if waiting and not full[0]: k = 1                                   # a prefill may be formable next step
            elif ev and not full[0]: k = max(1, min(k, math.ceil((ev[0][0] - t) / dstep)))   # an arrival could start a prefill
            if nr is not None: k = max(1, min(k, math.ceil(P["dec_period"] / dstep)))          # one noise draw per resample period
            if available() < k * bs:
                evict(k * bs if k == 1 else k * bs - available() + bs)       # per step SGLang evicts bs when short
                if available() < k * bs: k = max(1, available() // bs)
            if available() < bs:
                # check_decode_mem failed -> retract_decode: fewest output tokens first (ties: longest input)
                order_r = sorted(running, key=lambda r: (r.out, -len(r.ids)), reverse=True)
                while len(order_r) > 1 and available() < len(order_r):
                    v = order_r.pop(); running.remove(v); v.retracts += 1; st["retracted"] += 1
                    cache.dec_lock_ref(v.node); v.node = None; v.uncommitted = 0
                    v.out_base += v.out; v.ids = v.base_ids + list(v.rec.output_ids[:v.out_base]); v.out_len -= v.out; v.out = 0; v.filled = 0
                    waiting.insert(0, v)
                    evict(len(order_r) - available())
                tot_dec = sum(r.out for r in running); tot_max = sum(r.out_len for r in running)
                ratio[0] = min(1.0, (tot_dec + RETRACT_STEPS * len(running)) / (tot_max + 1)); full[0] = False
                bs = len(running); tokens = cache.protected_size() + uncommitted(); mctx = max(len(r.ids) for r in running); k = 1
            else:
                ratio[0] = max(ratio[0] - decay * k, min_ratio)
            dt = k * jit(decode_cost(bs, tokens + bs * (k - 1) / 2, mctx), 1.0 / math.sqrt(k)) * dec_factor() * (P["slow_factor"] if P["slow_t0"] <= t < P["slow_t1"] else 1.0)
            steps.append(("D", t, bs, k, 0, bs, len(waiting), dt, cache.protected_size() + uncommitted(), int(full[0])))
            last_start[0] = t + (k - 1) * dt / k; t += dt; clock.t = t; st["decode_time"] += dt; st["n_decode"] += k
            done = []
            for r in running:
                r.out += k; r.uncommitted += k
                if r.out >= r.out_len: done.append(r)
            for r in done: running.remove(r); finish(r)
            if done: full[0] = False
            continue
        if chunked[0] is not None: continue
        if waiting and last_start[0] < t:
            last_start[0] = t; continue                                          # idle server: a new arrival is scheduled at once
        if ev:
            ratio[0] = INIT_RATIO * P["conservativeness"]                       # self_check_during_idle
            t = max(t, ev[0][0]); last_start[0] = t; continue
        if waiting:
            st["stuck"] += 1; full[0] = False
            if st["stuck"] > 5: raise RuntimeError(f"deadlock: {len(waiting)} waiting, nothing running, pool avail {available()}")
            continue
        break
    st["makespan"] = t
    st["cpu_log"] = {f"{k}@{m}": v for (k, m), v in cpu_log.items()}
    if probe: st["probe"] = probe
    return records, occ, steps, dict(st)


def tm(x, p): x = np.sort(np.asarray(x)); return float(x[int(len(x) * p / 100):].mean()) if len(x) else float("nan")


def summarize(records, occ, stats, kv, cut0=300.0, cut1=60.0, win=300.0):
    R = sorted(records, key=lambda r: r["arrival"]); t0 = R[0]["arrival"]; t1 = R[-1]["arrival"]
    sel = [r for r in R if r["turn"] > 0 and cut0 <= r["arrival"] - t0 <= t1 - t0 - cut1]
    pt = np.array([r["prompt_len"] for r in sel], float); u = np.array([r["prompt_len"] - r["cached"] for r in sel], float)
    s = np.array([r["ttft"] for r in sel]); q = np.array([r["first_sched"] - r["arrival"] for r in sel]); e2e = np.array([r["e2e"] for r in sel])
    o = np.array(occ); mid = (o[:, 0] > cut0) & (o[:, 0] < o[:, 0].max() - cut1)
    out = dict(n=len(sel), makespan_min=stats["makespan"] / 60, miss=100 * u.sum() / pt.sum(), U_p90=np.percentile(u, 90), U_p99=np.percentile(u, 99), U_TM99=tm(u, 99), U_CVaR95=tm(u, 95),
               n16k=int((u > 16000).sum()), sTTFT_p50=np.percentile(s, 50), sTTFT_p90=np.percentile(s, 90), sTTFT_p99=np.percentile(s, 99), sTTFT_TM99=tm(s, 99),
               queue_p99=np.percentile(q, 99), queue_share=100 * q.sum() / s.sum(), queued50=100 * (q > 0.05).mean(), dequeued_p99=np.percentile(s[q <= 0.05], 99),
               e2e_p50=np.percentile(e2e, 50), infl_max=o[:, 1].max(), infl_p90=np.percentile(o[mid, 1], 90), kv_blocked=100 * o[mid, 5].mean(), waiting_max=int(o[:, 4].max()),
               evicted_M=stats.get("evicted", 0) / 1e6, retracted=int(stats.get("retracted", 0)), n_prefill=int(stats["n_prefill"]), n_decode=int(stats["n_decode"]),
               decode_step_ms=1e3 * stats["decode_time"] / max(stats["n_decode"], 1), prefill_share=100 * stats["prefill_time"] / (stats["prefill_time"] + stats["decode_time"]))
    wins = []
    for w0 in np.arange(0, stats["makespan"], win):
        m = [r for r in R if w0 <= r["arrival"] - t0 < w0 + win and r["turn"] > 0]
        if len(m) < 5: continue
        pw = np.array([r["prompt_len"] for r in m], float); uw = np.array([r["prompt_len"] - r["cached"] for r in m], float); sw = np.array([r["ttft"] for r in m]); qw = np.array([r["first_sched"] - r["arrival"] for r in m])
        mm = (o[:, 0] >= w0) & (o[:, 0] < w0 + win)
        wins.append(dict(w=int(w0 // 60), n=len(m), miss=100 * uw.sum() / pw.sum(), U_p99=float(np.percentile(uw, 99)), sTTFT_p99=float(np.percentile(sw, 99)), queue_p99=float(np.percentile(qw, 99)),
                         infl_p50=float(np.percentile(o[mm, 1], 50)) if mm.any() else 0, infl_max=float(o[mm, 1].max()) if mm.any() else 0, running_p50=float(np.percentile(o[mm, 3], 50)) if mm.any() else 0,
                         waiting_max=int(o[mm, 4].max()) if mm.any() else 0, kv_blocked=100 * float(o[mm, 5].mean()) if mm.any() else 0))
    return out, wins


DEFAULTS = dict(p0=51.1, p_tok=3.28, p_attn=543.2, p_open=88.2, d0=22.4, d_tok=0.0443, d_req=-0.469, n0=-44.8, n_tok=0.043, n_req=2.20, d_min=12.0, cg_max_bs=32,
                ttft_floor=0.02, ftd_median=0.0, ftd_sigma=0.7, turn_overhead=0.10, ovh_tok=0.0, ovh_out=0.0, ingress_base=0.0, ingress_tok=0.0, overlap=1, jitter=0.0, jitter_seed=0, noise_seed=-1, dec_noise=0.053, dec_period=1.3, drift_noise=0.012, drift_tau=60.0, pre_noise=0.21, ovh_sigma=0.46, cpu_evict_scale=0.0, slow_factor=1.0, slow_t0=0.0, slow_t1=0.0, open_until=0.0, ovh_run_sd=0.0, chunk=8192, max_prefill=16384, max_running=4096, conservativeness=1.0, sched="fcfs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dead", action="store_true", help="every 30 s record evictable tokens by tier and the finished conversations' private chains")
    ap.add_argument("--ids-from", default=None, help="GPU arm dir: use the prompt/output token ids that arm ACTUALLY sent/generated (stitched prompts + real generations); programs end where the GPU arm's turns end; timing still from the recording")
    ap.add_argument("--no-stitch", action="store_true", help="use recorded ids verbatim (default: emulate the replay client's prompt stitching)")
    ap.add_argument("stream"); ap.add_argument("--kv", type=int, required=True); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="rrdeclband0.001,rrdeclchunk1024"); ap.add_argument("--rec-conc", type=int, default=4); ap.add_argument("--rec-span", type=float, default=None)
    ap.add_argument("--drops-from", default=None, help="GPU arm dir: truncate conversations the GPU replay client dropped mid-run (last turn >300 s before run end) at the same turn"); ap.add_argument("--arrivals-from", default=None, help="GPU arm dir: open-loop mode, inject every turn at its measured arrival time"); ap.add_argument("--params-json", default=None, help="step-cost overrides fitted by fit_step_costs.py"); ap.add_argument("--trial-gap", type=float, default=None); ap.add_argument("--out-dir", required=True); ap.add_argument("--csv", action="store_true")
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    P = {k: getattr(a, k) for k in DEFAULTS}; P["probe_dead"] = a.probe_dead
    if a.params_json:
        _pj = {k: v for k, v in json.load(open(a.params_json)).items() if not k.startswith("_")}; P.update(_pj)
        print("machine profile", a.params_json, {k: round(v, 4) for k, v in _pj.items()}, flush=True)
    reqs = OR.load_stream(a.stream)
    if a.ids_from:
        reqs = ids_from_gpu(reqs, a.ids_from)
    elif not a.no_stitch: reqs = stitch_like_replay(reqs); print("prompt stitching emulated (program-unique outputs)", flush=True)
    if a.drops_from and not a.ids_from:
        from gpu_windows import requests as _gpu_reqs
        import collections as _co
        _log = next(os.path.join(a.drops_from, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(a.drops_from, x)))
        _G = _gpu_reqs(_log); _kept = _co.defaultdict(int); _fin = _co.defaultdict(float)
        for (_p, _k), _v in _G.items(): _kept[_p] = max(_kept[_p], _k + 1); _fin[_p] = max(_fin[_p], _v[5] or _v[2])
        _end = max(_fin.values()); _n = _co.Counter(r.program_id for r in reqs)
        _byp = _co.defaultdict(list)
        for r in reqs: _byp[r.program_id].append(r)
        for _l in _byp.values(): _l.sort(key=lambda r: r.arrival)
        def _next_gap(_p):
            _l = _byp[_p]; _k = _kept[_p]
            return max(0.0, _l[_k].arrival - _l[_k - 1].finish) if 0 < _k < len(_l) else 0.0
        P["drops"] = {_p: _kept[_p] for _p in _n if 0 < _kept[_p] < _n[_p] and _fin[_p] + _next_gap(_p) + 60.0 < _end}
        print(f"drops: {len(P['drops'])} conversations dropped mid-run by the GPU client: " + ", ".join(f"{_p.split(':')[1][-22:]}@t{_k}" for _p, _k in P["drops"].items()), flush=True)
    if a.arrivals_from:
        from gpu_windows import requests as _gpu_reqs
        _log = next(os.path.join(a.arrivals_from, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(a.arrivals_from, x)))
        _R = {k: v[0] for k, v in _gpu_reqs(_log).items()}; _t0 = min(_R.values())
        P["arrivals"] = {k: v - _t0 for k, v in _R.items()}; print(f"open-loop: {len(P['arrivals'])} GPU arrival times from {a.arrivals_from}", flush=True)
    trial_gap = a.trial_gap if a.trial_gap is not None else rhs.calibrate_trial_gap(reqs, a.rec_conc, a.rec_span)
    print(f"stream {len(reqs)} reqs, trial_gap {trial_gap:.0f}s, conc {a.conc} seed {a.seed} kv {a.kv} sched {a.sched}", flush=True)
    out = dict(kv=a.kv, conc=a.conc, seed=a.seed, trial_gap=trial_gap, params={k: v for k, v in P.items() if k not in ("arrivals", "drops")}, arms={})
    for name in a.arms.split(","):
        rec, occ, steps, st = simulate(reqs, OR.ArmSpec.parse(name), a.kv, a.conc, a.seed, trial_gap, P)
        s, wins = summarize(rec, occ, st, a.kv); out["arms"][name] = dict(summary=s, windows=wins, stats=st)
        print(f"[{name}] makespan {s['makespan_min']:.1f} min n={s['n']} | miss {s['miss']:.2f}% U p90/p99/TM99 {s['U_p90']:.0f}/{s['U_p99']:.0f}/{s['U_TM99']:.0f} n>16k {s['n16k']} | sTTFT p50/p90/p99/TM99 {s['sTTFT_p50']:.3f}/{s['sTTFT_p90']:.3f}/{s['sTTFT_p99']:.2f}/{s['sTTFT_TM99']:.2f}s queue p99 {s['queue_p99']:.2f}s share {s['queue_share']:.0f}% dq p99 {s['dequeued_p99']:.2f} | in-flight max/p90 {s['infl_max']/1e3:.0f}/{s['infl_p90']/1e3:.0f}K kv-blocked {s['kv_blocked']:.1f}% wait max {s['waiting_max']} | decode {s['decode_step_ms']:.0f}ms prefill share {s['prefill_share']:.0f}% evicted {s['evicted_M']:.2f}M retracted {s['retracted']}", flush=True)
        for w in wins: print(f"    {w['w']:3d}-{w['w']+5:3d} n={w['n']:3d} miss {w['miss']:5.2f}% U p99 {w['U_p99']:6.0f} sTTFT p99 {w['sTTFT_p99']:5.2f}s q99 {w['queue_p99']:5.2f}s | in-flight p50/max {w['infl_p50']/1e3:4.0f}/{w['infl_max']/1e3:4.0f}K running {w['running_p50']:.0f} waiting max {w['waiting_max']} kv-blocked {w['kv_blocked']:.0f}%")
        if a.csv:
            with open(os.path.join(a.out_dir, f"req_{name}.csv"), "w") as f:
                wr = csv.DictWriter(f, fieldnames=list(rec[0].keys())); wr.writeheader(); wr.writerows(rec)
            with open(os.path.join(a.out_dir, f"occ_{name}.csv"), "w") as f:
                f.write("t,inflight,evictable,running,waiting,kv_blocked,ratio\n"); [f.write(",".join(str(v) for v in row) + "\n") for row in occ]
            with open(os.path.join(a.out_dir, f"steps_{name}.csv"), "w") as f:
                f.write("kind,t,n,new_or_k,cached,running,waiting,dur,inflight,full\n"); [f.write(",".join(str(v) for v in row) + "\n") for row in steps]
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
