#!/usr/bin/env python3
"""Closed-loop OFFLINE replay: the server is simulated step by step instead of replaying recorded turn durations.

What differs from offline_replay.py (which keeps every turn's recorded service time):
  * a turn's service time is COMPUTED: prefill of its UNCACHED tokens (chunked, prefill-first, one batch per step),
    then decode of its recorded output length; decode steps are shared by all running requests and are paused
    while a prefill step runs -> more misses => more prefill steps => every running request decodes slower =>
    holds its KV longer => more in-flight => more eviction (closed-loop amplification);
  * a request waits in a FCFS queue until the pool can hold (uncached prompt + output); prefix matching happens
    at ADMISSION, so material evicted while it waited is lost (queue effect);
  * tool gaps (arrival_k - finish_{k-1} of the same program) and trial admission order / trial_gap are taken from
    the recording exactly like retime_for_concurrency; second-level parked-band turnover therefore emerges when
    in-flight ~ pool instead of being assumed.
Calibrated defaults come from the 2026-09-20 cloud c42 LRU+D log (H100 TP2, Qwen3-Coder-30B): cold prefill
chunk 8192 -> 307 ms (26.7K tok/s), decode step 29 ms @225K running tokens -> 39 ms @548K.
usage: closed_loop_sim.py STREAM.pkl --kv 750000 --conc 42 --seed 0 --arms lru,rrdeclband0.001,rrdeclchunk1024 \
         --rec-conc 4 --rec-span 27366 --out-dir DIR
"""
import os as _os
# one process = one core (user limit: all simulation jobs together <= 30 threads); torch/OpenMP would otherwise
# start a 128-thread pool per process on this box
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): _os.environ.setdefault(_v, "1")
import argparse, csv, heapq, json, math, os, random, sys, types
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_replay as OR
rhs = OR.rhs
import torch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey


def simulate(reqs, spec, kv, conc, seed, trial_gap, P, occ_dt=5.0):
    """P: dict of server parameters."""
    clock = rhs.SimClock()
    import sglang.srt.mem_cache.evict_policy as ep_mod, sglang.srt.mem_cache.radix_cache as rc_mod, sglang.srt.mem_cache.rr_radix_cache as rr_mod
    saved = (rc_mod.time.monotonic, rr_mod.time.monotonic, ep_mod.time.monotonic); saved_time = rc_mod.time.time
    for m in (rc_mod, rr_mod, ep_mod): m.time.monotonic = clock.monotonic
    rc_mod.time.time = clock.monotonic
    cache = spec.build(); declared = spec.declared
    trials = rhs._trials(reqs)
    if P.get("order_file"):
        want = [l.strip() for l in open(P["order_file"]) if l.strip()]; pos = {t: i for i, t in enumerate(want)}
        trials.sort(key=lambda tr: pos.get(tr[0], 10**9)); print(f"admission order from {P['order_file']}: {sum(tr[0] in pos for tr in trials)} of {len(trials)} trials matched", flush=True)
    elif seed is not None: random.Random(seed).shuffle(trials)
    # per program: ordered turns with tool gap before each (recorded arrival_k - finish_{k-1}); first turn offset from trial base
    progs = {}
    trial_progs = defaultdict(list)
    for tid, rs, base, dur in trials:
        byp = defaultdict(list)
        for r in rs: byp[r.program_id or r.rid].append(r)
        for pid, lst in byp.items():
            lst.sort(key=lambda r: r.arrival)
            turns = []
            for i, r in enumerate(lst):
                gap = (r.arrival - lst[i - 1].finish) if i > 0 else (r.arrival - base)
                if i > 0: gap = gap * P["gap_scale"] + P["gap_add"]
                turns.append((r, max(0.0, gap)))
            progs[pid] = dict(turns=turns, k=0, trial=tid, cnt=len(turns))
            trial_progs[tid].append(pid)
    trial_left = {tid: len(ps) for tid, ps in trial_progs.items()}
    order = [tid for tid, *_ in trials]
    # events: (t, seq, kind, payload)   kinds: 'arr' (request enters waiting), 'slot' (a slot frees -> start next trial)
    ev = []; seq = [0]
    def push(t, kind, payload):
        seq[0] += 1; heapq.heappush(ev, (t, seq[0], kind, payload))
    next_trial = [0]
    def start_trial(t):
        if next_trial[0] >= len(order): return
        tid = order[next_trial[0]]; next_trial[0] += 1
        for pid in trial_progs[tid]:
            r, gap = progs[pid]["turns"][0]; push(t + gap, "arr", pid)
    for _ in range(conc): start_trial(0.0)
    waiting = []            # FIFO of (pid) waiting for admission
    running = {}            # pid -> dict(node, new_tokens, remaining_out, prompt_len, cached, arrival, admit, ctx)
    prefilling = None       # pid currently being chunk-prefilled: dict(pid, remaining_new, done_new)
    counter = 1; records = []; occ = []; t = 0.0; tokens_evicted = 0; evict_calls = 0
    n_decode_steps = 0; n_prefill_steps = 0; decode_time = 0.0; prefill_time = 0.0; next_occ = 0.0
    def occupancy(): return cache.protected_size() + cache.evictable_size() + sum(x["new_tokens"] for x in running.values())
    def make_room():
        nonlocal tokens_evicted, evict_calls
        while occupancy() > kv and cache.evictable_size() > 0:
            res = cache.evict(EvictParams(num_tokens=occupancy() - kv)); evict_calls += 1
            n = res.num_tokens_evicted or 0; tokens_evicted += n
            if n == 0: break
    blocked = [False]   # real scheduler: after NO_TOKEN the batch is marked full until some request finishes
    def try_admit(pid):
        """match prefix now; admit if the pool can take uncached+output after evicting. Returns True if admitted."""
        if blocked[0] and P["hol_retry"] == "finish": return False
        r, _ = progs[pid]["turns"][progs[pid]["k"]]
        mp = MatchPrefixParams(key=RadixKey(r.prompt_ids, None))
        mp.req = types.SimpleNamespace(program_id=r.program_id, program_state=r.program_state or "active", time_stats=types.SimpleNamespace(created_time=t))
        m = cache.match_prefix(mp); cached = int(m.device_indices.numel())
        need = (len(r.prompt_ids) - cached) + len(r.output_ids)
        free_after = kv - (occupancy() - cache.evictable_size())   # what remains if everything evictable is evicted
        if need > free_after: blocked[0] = True; head_kv_blocked[0] = True; return False
        # admission control: cap the LOCKED footprint (in-flight tokens) and/or the number of running requests
        infl_now = sum(x["new_tokens"] for x in running.values()) + cache.protected_size()
        if (P["max_inflight"] and infl_now + len(r.prompt_ids) + len(r.output_ids) > P["max_inflight"]) or (P["max_running"] and len(running) >= P["max_running"]):
            blocked[0] = True; head_kv_blocked[0] = True; return False
        head_kv_blocked[0] = False
        cache.inc_lock_ref(m.last_device_node)
        running[pid] = dict(node=m.last_device_node, new_tokens=(len(r.prompt_ids) - cached), remaining_out=len(r.output_ids), prompt_len=len(r.prompt_ids), cached=cached, arrival=arrivals[pid], admit=t, ctx=len(r.prompt_ids), req=r, ttft=None)
        make_room()
        return True
    def finish(pid):
        nonlocal counter
        blocked[0] = False
        x = running.pop(pid); r = x["req"]; full = r.prompt_ids + r.output_ids
        val = torch.arange(counter, counter + len(full), dtype=torch.int64); counter += len(full)
        cache.insert(InsertParams(key=RadixKey(full, None), value=val)); cache.dec_lock_ref(x["node"])
        records.append(dict(program=pid, turn=progs[pid]["k"], arrival=x["arrival"], admit=x["admit"], ttft=x["ttft"], e2e=t - x["arrival"], prompt_len=x["prompt_len"], cached=x["cached"], output_len=len(r.output_ids), wait_kv=wait_kv.pop(pid, 0.0), wait_cpu=wait_cpu.pop(pid, 0.0)))
        pg = progs[pid]; pg["k"] += 1
        if pg["k"] < pg["cnt"]:
            push(t + pg["turns"][pg["k"]][1], "arr", pid)
        else:
            if declared:
                leaf = cache.match_prefix(MatchPrefixParams(key=RadixKey(full, None))).last_device_node
                cache._record_program_leaf(pid, leaf); cache.release_program([pid])
            tid = pg["trial"]; trial_left[tid] -= 1
            if trial_left[tid] == 0: push(t + trial_gap, "slot", tid)
    arrivals = {}; wait_kv = {}; wait_cpu = {}; head_kv_blocked = [False]
    def accrue(dt):
        for pid in waiting:
            if head_kv_blocked[0]: wait_kv[pid] = wait_kv.get(pid, 0.0) + dt
            else: wait_cpu[pid] = wait_cpu.get(pid, 0.0) + dt
    # ---- main loop ----
    while ev or waiting or running or prefilling:
        clock.t = t
        while ev and ev[0][0] <= t:
            te, _, kind, payload = heapq.heappop(ev)
            if kind == "arr": arrivals[payload] = te; waiting.append(payload)
            else: start_trial(t)
        if t >= next_occ:
            occ.append((t, sum(x["new_tokens"] for x in running.values()) + cache.protected_size(), cache.evictable_size(), len(running), len(waiting), int(head_kv_blocked[0] and len(waiting) > 0))); next_occ = t + occ_dt
        # admission (FCFS, head-of-line blocking) -> prefill batch of up to max_prefill_tokens
        budget = P["max_prefill"]; batch = []   # list of (pid, new_tokens_this_step, ctx)
        prefilling_done = None
        if prefilling is not None:
            n = min(P["chunk"], prefilling["remaining_new"], budget); batch.append((prefilling["pid"], n, running[prefilling["pid"]]["ctx"])); budget -= n
            prefilling["remaining_new"] -= n
            if prefilling["remaining_new"] <= 0: prefilling_done = prefilling["pid"]; prefilling = None
        if P["sched"] == "lpm" and len(waiting) > 1:
            # longest-prefix-match: order the waiting queue by cached prefix length (desc), arrival asc
            def _cached(pid):
                rr, _ = progs[pid]["turns"][progs[pid]["k"]]
                mm = cache.match_prefix(MatchPrefixParams(key=RadixKey(rr.prompt_ids, None))); return int(mm.device_indices.numel())
            waiting.sort(key=lambda pid: (-_cached(pid), arrivals[pid]))
        while waiting and budget > 0:
            pid = waiting[0]
            r, _ = progs[pid]["turns"][progs[pid]["k"]]
            if pid in running and running[pid].get("_pending"):
                running[pid]["_pending"] = False
            elif not try_admit(pid): break
            waiting.pop(0); x = running[pid]; unc = x["prompt_len"] - x["cached"]
            if unc <= budget:
                batch.append((pid, max(unc, 1), x["ctx"])); budget -= max(unc, 1); continue
            if prefilling is None and prefilling_done is None:
                n = min(P["chunk"], budget); prefilling = dict(pid=pid, remaining_new=unc - n); batch.append((pid, n, x["ctx"])); budget -= n
            else:
                # a second chunked request cannot start this step: it holds its admission (locked) and prefills next step
                waiting.insert(0, pid); running[pid]["_pending"] = True; break
        if batch:
            dt = P["step_ms"] / 1e3 + sum(n for _, n, _ in batch) / P["prefill_tps"] + sum(c for _, _, c in batch) * P["attn_us"] / 1e6
            if t < P["cold_secs"]: dt *= P["cold_factor"]   # cold-container JIT: every step slower for the first cold_secs
            accrue(dt); t += dt; prefill_time += dt; n_prefill_steps += 1
            if P["decode_overlap"] > 0 and running:
                # decode of already-prefilled requests advances during a fraction of the prefill step (mixed / overlapped execution)
                tot = sum(x["ctx"] + (len(x["req"].output_ids) - x["remaining_out"]) for x in running.values())
                dstep = (P["dec_ms0"] + P["dec_ms_per_ktok"] * tot / 1e3 + P["dec_ms_per_req"] * len(running)) / 1e3; k = int(dt * P["decode_overlap"] / dstep)
                for pid in list(running):
                    x = running[pid]
                    if x["ttft"] is None or pid in [b[0] for b in batch]: continue
                    n = min(k, x["remaining_out"]); x["remaining_out"] -= n; x["new_tokens"] += n
                    if x["remaining_out"] <= 0: finish(pid)
            done = [pid for pid, *_ in batch if pid != (prefilling["pid"] if prefilling else None)]
            for pid in done:
                x = running[pid]
                if x["ttft"] is None: x["ttft"] = t - x["arrival"] + P["ttft_floor"]
                if x["remaining_out"] <= 0: finish(pid)
            continue
        if running:
            tot = sum(x["ctx"] + (len(x["req"].output_ids) - x["remaining_out"]) for x in running.values())
            dt = (P["dec_ms0"] + P["dec_ms_per_ktok"] * tot / 1e3 + P["dec_ms_per_req"] * len(running)) / 1e3
            if t < P["cold_secs"]: dt *= P["cold_factor"]
            accrue(dt); t += dt; decode_time += dt; n_decode_steps += 1
            for pid in list(running):
                x = running[pid]
                if x["ttft"] is None: continue   # not yet prefilled (cannot happen: prefill-first)
                x["remaining_out"] -= 1; x["new_tokens"] += 1
                if x["remaining_out"] <= 0: finish(pid)
            continue
        if ev: t = max(t, ev[0][0]); continue
        break
    for m, fn in zip((rc_mod, rr_mod, ep_mod), saved): m.time.monotonic = fn
    rc_mod.time.time = saved_time
    stats = dict(tokens_evicted=tokens_evicted, evict_calls=evict_calls, n_decode_steps=n_decode_steps, n_prefill_steps=n_prefill_steps, decode_time=decode_time, prefill_time=prefill_time, makespan=t)
    return records, occ, stats


def summarize(records, occ, stats, kv, win=300.0):
    R = sorted(records, key=lambda r: r["arrival"]); t0 = R[0]["arrival"]
    pt = np.array([r["prompt_len"] for r in R], float); unc = np.array([r["prompt_len"] - r["cached"] for r in R], float)
    ttft = np.array([r["ttft"] for r in R]); q = np.array([r["admit"] - r["arrival"] for r in R]); warm = np.array([r["turn"] > 0 for r in R]); e2e = np.array([r["e2e"] for r in R]); ol = np.array([r["output_len"] for r in R])
    def tm(x, p): x = np.sort(x); return x[int(len(x) * p / 100):].mean() if len(x) else float("nan")
    o = np.array(occ); infl = o[:, 1]; mid = (o[:, 0] > 300) & (o[:, 0] < o[:, 0].max() - 180)
    s = dict(n=len(R), makespan_min=stats["makespan"] / 60, miss=100 * unc.sum() / pt.sum(), miss_warm=100 * unc[warm].sum() / pt[warm].sum(),
             unc_p90=float(np.percentile(unc[warm], 90)), unc_p99=float(np.percentile(unc[warm], 99)), unc_tm99=tm(unc[warm], 99), unc_cvar95=tm(unc[warm], 95), n4k=int((unc[warm] > 4000).sum()), n16k=int((unc[warm] > 16000).sum()),
             e2e_p50=float(np.percentile(e2e, 50)), e2e_p90=float(np.percentile(e2e, 90)), e2e_mean=float(e2e.mean()), out_p50=float(np.median(ol)), ttft_p50=float(np.percentile(ttft, 50)), ttft_p90=float(np.percentile(ttft, 90)), ttft_p99=float(np.percentile(ttft, 99)), ttft_tm99=tm(ttft, 99), queue_p99=float(np.percentile(q, 99)), queue_share=100 * q.sum() / ttft.sum(), queued50ms=100 * (q > 0.05).mean(),
             infl_max=float(infl.max()), infl_p90=float(np.percentile(infl[mid], 90)), t_infl_wall=100 * float(np.mean(infl[mid] > 0.85 * kv)), waiting_max=int(o[:, 4].max()),
             decode_step_ms=1e3 * stats["decode_time"] / max(stats["n_decode_steps"], 1), prefill_share=100 * stats["prefill_time"] / (stats["prefill_time"] + stats["decode_time"]), **{k: stats[k] for k in ("tokens_evicted", "n_prefill_steps", "n_decode_steps")})
    wins = []
    for w0 in np.arange(0, stats["makespan"], win):
        m = np.array([(w0 <= r["arrival"] - t0 < w0 + win) for r in R])
        if m.sum() < 5: continue
        mm = (o[:, 0] >= w0) & (o[:, 0] < w0 + win)
        wk = np.array([r["wait_kv"] for r in R])[m]; wc = np.array([r["wait_cpu"] for r in R])[m]
        wins.append(dict(w=int(w0 // 60), n=int(m.sum()), miss=100 * unc[m].sum() / pt[m].sum(), wait_kv_mean=float(wk.mean()), wait_cpu_mean=float(wc.mean()), wait_kv_p99=float(np.percentile(wk, 99)), wait_cpu_p99=float(np.percentile(wc, 99)), t_kv_blocked=100 * float(np.mean(o[mm, 5])) if mm.any() else 0, unc_p99=float(np.percentile(unc[m], 99)), tm99=tm(unc[m], 99), n4k=int((unc[m] > 4000).sum()), ttft_p99=float(np.percentile(ttft[m], 99)), q_p99=float(np.percentile(q[m], 99)),
                         infl_p50=float(np.percentile(infl[mm], 50)) if mm.any() else 0, infl_max=float(infl[mm].max()) if mm.any() else 0, evictable_p50=float(np.percentile(o[mm, 2], 50)) if mm.any() else 0, wall=100 * float(np.mean(infl[mm] > 0.85 * kv)) if mm.any() else 0))
    return s, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream"); ap.add_argument("--kv", type=int, required=True); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="lru,rrdeclband0.001,rrdeclchunk1024"); ap.add_argument("--rec-conc", type=int, default=4); ap.add_argument("--rec-span", type=float, default=None); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefill-tps", type=float, default=52000); ap.add_argument("--step-ms", type=float, default=55); ap.add_argument("--attn-us", type=float, default=1.8, help="prefill attention cost per context token per chunk (us)"); ap.add_argument("--ttft-floor", type=float, default=0.02, help="client/HTTP floor added to TTFT (s), not server time"); ap.add_argument("--hol-retry", default="finish", choices=["finish", "step"]); ap.add_argument("--decode-overlap", type=float, default=0.5, help="fraction of each prefill step during which decode of running requests also advances (0 = strict prefill-first)")
    ap.add_argument("--dec-ms0", type=float, default=18); ap.add_argument("--dec-ms-per-ktok", type=float, default=0.022); ap.add_argument("--dec-ms-per-req", type=float, default=0.19, help="decode step cost per running request (ms); fitted on mldev c42 pure-decode pairs"); ap.add_argument("--chunk", type=int, default=8192); ap.add_argument("--max-prefill", type=int, default=16384)
    ap.add_argument("--csv", action="store_true"); ap.add_argument("--sched", default="fcfs", choices=["fcfs", "lpm"]); ap.add_argument("--order-file", default=None, help="explicit trial admission order (one trial id per line); overrides --seed shuffling"); ap.add_argument("--gap-scale", type=float, default=1.0, help="multiply every tool gap (turn k>0)"); ap.add_argument("--gap-add", type=float, default=0.0, help="add seconds to every tool gap"); ap.add_argument("--max-inflight", type=int, default=0, help="admission cap on locked in-flight tokens (0 = none)"); ap.add_argument("--max-running", type=int, default=0, help="admission cap on running requests (0 = none)"); ap.add_argument("--cold-secs", type=float, default=0.0); ap.add_argument("--cold-factor", type=float, default=1.0, help="multiply every step cost by this factor for the first --cold-secs (cold-container JIT shock)"); ap.add_argument("--replica-diverge", action="store_true", help="make x-replicas of a conversation share only their turn-0 prompt (stitched-protocol divergence)"); ap.add_argument("--trial-gap", type=float, default=None, help="override the calibrated per-trial overhead (s), e.g. 0 for spider2 as in the GPU runs")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    reqs = OR.load_stream(a.stream)
    if a.replica_diverge:
        # stitched GPU protocol: each replica's generated text diverges, so replicas share only their turn-0 prompt.
        import re as _re, collections as _c
        from dataclasses import replace as _replace
        byp = _c.defaultdict(list)
        for r in reqs: byp[r.program_id].append(r)
        new = []
        for pid, lst in byp.items():
            m = _re.search(r"#(\d+)", pid or ""); k = int(m.group(1)) if m else 0
            lst.sort(key=lambda r: r.arrival); L0 = len(lst[0].prompt_ids)
            for r in lst:
                if k == 0: new.append(r); continue
                off = k * 10_000_000
                pr = r.prompt_ids[:L0] + [t + off for t in r.prompt_ids[L0:]]; ou = [t + off for t in r.output_ids]
                new.append(_replace(r, prompt_ids=pr, output_ids=ou))
        reqs = sorted(new, key=lambda r: r.arrival); print(f"replica-diverge: replicas share only their turn-0 prompt", flush=True)
    trial_gap = a.trial_gap if a.trial_gap is not None else rhs.calibrate_trial_gap(reqs, a.rec_conc, a.rec_span)
    P = dict(prefill_tps=a.prefill_tps, step_ms=a.step_ms, attn_us=a.attn_us, dec_ms0=a.dec_ms0, dec_ms_per_ktok=a.dec_ms_per_ktok, dec_ms_per_req=a.dec_ms_per_req, cold_secs=a.cold_secs, cold_factor=a.cold_factor, max_inflight=a.max_inflight, max_running=a.max_running, gap_scale=a.gap_scale, gap_add=a.gap_add, order_file=a.order_file, sched=a.sched, chunk=a.chunk, max_prefill=a.max_prefill, ttft_floor=a.ttft_floor, hol_retry=a.hol_retry, decode_overlap=a.decode_overlap)
    out = dict(kv=a.kv, conc=a.conc, seed=a.seed, trial_gap=trial_gap, params=P, arms={})
    print(f"stream {len(reqs)} reqs, trial_gap {trial_gap:.0f}s, conc {a.conc} seed {a.seed} kv {a.kv}", flush=True)
    for name in a.arms.split(","):
        spec = OR.ArmSpec.parse(name); rec, occ, st = simulate(reqs, spec, a.kv, a.conc, a.seed, trial_gap, P)
        s, wins = summarize(rec, occ, st, a.kv); out["arms"][name] = dict(summary=s, windows=wins)
        print(f"[{name}] makespan {s['makespan_min']:.1f} min | miss {s['miss']:.2f}% (warm {s['miss_warm']:.2f}) | unc p90/p99/TM99/CVaR95 {s['unc_p90']:.0f}/{s['unc_p99']:.0f}/{s['unc_tm99']:.0f}/{s['unc_cvar95']:.0f} n>4k {s['n4k']} n>16k {s['n16k']} | TTFT p50/p90/p99 {s['ttft_p50']:.2f}/{s['ttft_p90']:.2f}/{s['ttft_p99']:.2f}s queue p99 {s['queue_p99']:.2f}s share {s['queue_share']:.0f}% | in-flight max/p90 {s['infl_max']/1e3:.0f}/{s['infl_p90']/1e3:.0f}K wall {s['t_infl_wall']:.0f}% | e2e p50/p90/mean {s['e2e_p50']:.1f}/{s['e2e_p90']:.1f}/{s['e2e_mean']:.1f}s out p50 {s['out_p50']:.0f} | decode step {s['decode_step_ms']:.0f}ms prefill share {s['prefill_share']:.0f}% evicted {s['tokens_evicted']/1e6:.2f}M", flush=True)
        for w in wins: print(f"    {w['w']:3d}-{w['w']+5:3d} min n={w['n']:3d} miss {w['miss']:5.2f}% unc p99 {w['unc_p99']:6.0f} TM99 {w['tm99']:6.0f} n>4k {w['n4k']:3d} | TTFT p99 {w['ttft_p99']:5.2f}s q99 {w['q_p99']:5.2f}s [KV-wait mean/p99 {w['wait_kv_mean']:.2f}/{w['wait_kv_p99']:.1f}s cpu-wait {w['wait_cpu_mean']:.2f}/{w['wait_cpu_p99']:.1f}s, %t KV-blocked {w['t_kv_blocked']:.0f}%] | in-flight p50/max {w['infl_p50']/1e3:4.0f}/{w['infl_max']/1e3:4.0f}K evictable p50 {w['evictable_p50']/1e3:4.0f}K wall {w['wall']:3.0f}%")
        if a.csv:
            with open(os.path.join(a.out_dir, f"req_{name}.csv"), "w") as f:
                wr = csv.DictWriter(f, fieldnames=list(rec[0].keys())); wr.writeheader(); wr.writerows(rec)
            with open(os.path.join(a.out_dir, f"occ_{name}.csv"), "w") as f:
                f.write("t,inflight,evictable,running,waiting,kv_blocked\n"); [f.write(",".join(str(v) for v in row) + "\n") for row in occ]
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
