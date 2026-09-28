#!/usr/bin/env python3
"""Offline radix-cache replay: drive the REAL SGLang cache classes with ONE
recorded harbor request stream, under each eviction policy, at a fixed KV budget.

Why this exists
---------------
In the live harbor sweep the four policies each ran a *different* workload
(LRU 162 reqs, pure-RR 75, ...) because real terminal output is non-deterministic
and slow TTFT truncates trajectories. So the live miss-rate confounds "policy
effect" with "which trajectories the run happened to draw". This replays the
*identical* recorded stream (same arrivals, same tokens, same think-gaps) against
every policy, so the ONLY variable is the eviction decision. Two consequences:

  1. Correctness/transparency: eviction never changes generated tokens — a hit
     returns exactly what a miss would recompute. Feeding the same stream to
     every policy makes that explicit: outputs are identical by construction;
     only *which* tokens were cache-resident at each arrival differs. This is the
     clean apples-to-apples the live runs cannot give.
  2. A trustworthy miss-rate ranking the live non-determinism otherwise hides.

Fidelity
--------
Uses the real RadixCache / RoundRobinRadixCache + the real RoundRobinStrategy /
LRUStrategy via a no-op mock KV allocator (page_size=1). The radix tree, the
prefix matching, the lock/unlock accounting, and the per-policy eviction *order*
(get_priority) are all the production code. time.monotonic() is monkeypatched to
a simulated clock driven by the recorded timestamps, so recency bands / declared
tiers see the real inter-turn gaps.

Modeling choices (applied IDENTICALLY to every policy, so cross-policy deltas are
robust):
  * Each request reserves its full footprint ((prompt-cached)+output tokens) from
    arrival to finish; decode-time incremental allocation is collapsed to one
    reservation. Slightly conservative on *when* pressure peaks; miss is recorded
    at match time exactly like the server's prefill #new/#cached split.
  * The tree is fed the request's real prompt_ids ++ output_ids at finish, so a
    resuming turn matches the full prior context (prompt AND assistant output) —
    exact, no synthetic tokens.
  * declared release: a program is demoted to the DEAD tier just after its LAST
    recorded request finishes (the conversation's final answer). Models the
    agent's /release_program POST, whose program_id isn't in the server log.

Usage:
  replay_harbor_stream.py <server_log> --kv 40000 \
      --policies lru,rrpure,rrband,rrdeclared,rrsynth [--band 5] [--window-s 10]
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time as _time
import types
import heapq
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.rr_radix_cache import RoundRobinRadixCache


# -----------------------------------------------------------------------------
# Recorded-stream parsing
# -----------------------------------------------------------------------------
@dataclass
class Req:
    rid: str
    arrival: float
    finish: float
    prompt_ids: List[int]
    output_ids: List[int]
    program_id: Optional[str]
    program_state: Optional[str]
    live_cached: int  # cached_tokens the live server reported (sanity baseline)
    prompt_tokens: int  # true prompt length (meta_info) — to detect log clipping
    completion_tokens: int


# _norm_rid: see rid normalization below
def parse_stream(path: str) -> List[Req]:
    """Pair request.received (prompt input_ids + program tags) with
    request.finished (output_ids + precise epoch timestamps) by rid."""
    recv: Dict[str, dict] = {}
    fin: Dict[str, dict] = {}
    for line in open(path):
        line = line.strip()
        if '"event": "request.received"' in line:
            if '"input_ids": [' not in line:
                continue  # warmup probe / openai pre-tokenize variant
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj = e.get("obj", {})
            if obj.get("input_ids"):
                # PD-disaggregated recordings occasionally emit rid as a
                # single-element list (batched submission path), which is
                # unhashable. Normalize so those requests are not dropped.
                _rid = e["rid"]
                if isinstance(_rid, list):
                    if not _rid:
                        continue
                    _rid = _rid[0]
                    e["rid"] = _rid
                recv[_rid] = e
        elif '"event": "request.finished"' in line:
            if "capital city of France" in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            fin[e["rid"]] = e

    reqs: List[Req] = []
    for rid, re_ in recv.items():
        fe = fin.get(rid)
        if fe is None:
            continue  # in-flight at shutdown; no finish -> skip
        obj = re_["obj"]
        mi = fe["out"]["meta_info"]
        arrival = mi.get("request_received_ts")
        finish = mi.get("request_finished_ts")
        if arrival is None or finish is None:
            continue
        reqs.append(
            Req(
                rid=rid,
                arrival=float(arrival),
                finish=float(finish),
                prompt_ids=list(obj["input_ids"]),
                output_ids=list(fe["out"].get("output_ids") or []),
                program_id=obj.get("program_id"),
                program_state=obj.get("program_state"),
                live_cached=int(mi.get("cached_tokens") or 0),
                prompt_tokens=int(mi.get("prompt_tokens") or 0),
                completion_tokens=int(mi.get("completion_tokens") or 0),
            )
        )
    reqs.sort(key=lambda r: r.arrival)
    return reqs


def check_not_clipped(reqs: List[Req]) -> None:
    """The request logger truncates input_ids/output_ids to max_length at
    --log-requests-level 2 (cap 2048). Token-exact replay needs level 3
    (untruncated). Abort loudly rather than silently replay clipped tokens."""
    n_clip_in = sum(1 for r in reqs if len(r.prompt_ids) < r.prompt_tokens)
    n_clip_out = sum(1 for r in reqs if len(r.output_ids) < r.completion_tokens)
    if n_clip_in or n_clip_out:
        raise SystemExit(
            f"ERROR: stream is CLIPPED — {n_clip_in}/{len(reqs)} prompts and "
            f"{n_clip_out}/{len(reqs)} outputs are shorter than their true token "
            f"count (logged at --log-requests-level 2, cap 2048). Re-record the "
            f"stream at LOG_LEVEL=3 for token-exact replay."
        )


def trial_of(program_id: Optional[str]) -> str:
    """The scheduling unit = what the harness's `-n` semaphore actually holds.

    Program ids are ``<prefix>:<unit>:<role...>`` in every harness we tag
    (harbor `harbor:<trial>:main` / `harbor:<trial>:general-purpose:0`,
    toolsandbox `toolsandbox:<scenario>:<role>`, tau2 likewise), so the first two
    fields name the unit and everything after it is a role INSIDE that unit.
    """
    if not program_id:
        return ""
    # Replica-aware (fix 2026-09-09). gpu_replay_harbor._replicate() suffixes the
    # id with "#r<j>", and that suffix sits INSIDE the third field
    # ("...:agent#r1"), so splitting on ":" and keeping the first two fields drops
    # it: every replica of a conversation collapsed into ONE trial and was packed
    # into a single semaphore slot, so a rep-K stream modelled K conversations
    # running simultaneously and reported ~K x the true in-flight KV. The GPU-side
    # replayer (gpu_replay_harbor._trial_key) always did this correctly, so only
    # OFFLINE shape/pool analysis of REPLICATED streams was affected.
    base, _, rep = program_id.partition("#")
    parts = base.split(":")
    key = ":".join(parts[:2]) if len(parts) >= 3 else base
    return f"{key}#{rep}" if rep else key


def _trials(reqs: List[Req]) -> List[Tuple[str, List[Req], float, float]]:
    byt: Dict[str, List[Req]] = defaultdict(list)
    for r in reqs:
        byt[trial_of(r.program_id) or r.rid].append(r)
    out = []
    for tid, rs in byt.items():
        rs = sorted(rs, key=lambda r: r.arrival)
        t0 = rs[0].arrival
        out.append((tid, rs, t0, max(r.finish for r in rs) - t0))
    out.sort(key=lambda c: c[2])
    return out


def calibrate_trial_gap(reqs: List[Req], rec_conc: int,
                        rec_span: Optional[float] = None) -> float:
    """Per-trial dead time that naive re-timing would delete.

    A harness slot is occupied by container build, agent setup and verification
    as well as by LLM turns, and none of that reaches the server log. Packing
    trials nose-to-tail therefore runs the workload denser than the recording
    did. Charge every trial the residual so re-timing at the RECORDED
    concurrency reproduces the recorded span, and other concurrencies inherit
    the same overhead. Validate by re-timing at rec_conc and comparing NEED.
    """
    ts = _trials(reqs)
    # rec_span overrides the wall-clock span. Required when a record was taken
    # in parts and concatenated: the seam between two runs is dead air that
    # belongs to neither, and charging it to the trials would invent overhead
    # that never happened (the 120-task record has a 1775s hole in a 5880s span).
    span = rec_span if rec_span else (
        max(r.finish for r in reqs) - min(r.arrival for r in reqs))
    return max(0.0, (span * rec_conc - sum(d for *_, d in ts)) / max(len(ts), 1))


def retime_for_concurrency(
    reqs: List[Req],
    n_concurrent: int,
    seed: Optional[int] = None,
    trial_gap: float = 0.0,
) -> List[Req]:
    """Re-schedule the SAME recorded workload under a different concurrency.

    A trial keeps its intrinsic internal timing — tool gaps and per-turn service
    times are properties of the workload, not the concurrency. Only trial START
    times change, as if ``n_concurrent`` trials ran behind a semaphore (harbor's
    `-n`). This reaches concurrencies the LIVE harness cannot: harbor is capped
    near 6 by the docker daemon's network address pool.

    Packs TRIALS, not program_ids. A subagent is not an independent conversation
    — it runs inside its parent trial and its KV overlaps the parent's. Grouping
    by program_id would deal subagents into their own slots and destroy exactly
    the overlap a subagent workload is about.

    ``seed`` shuffles the admission order. Recorded order is a single draw, and
    block-ordered admission has already manufactured a fake RR win on harbor
    once — average over several seeds rather than trusting one.

    Caveat that re-timing cannot fix: per-request service times are held at their
    recorded values, so a real run at higher concurrency would decode slower,
    stretch its turns, raise the duty cycle and hold MORE in-flight KV than this
    models. In-flight figures from a re-timed stream are lower bounds.
    """
    trials = _trials(reqs)
    if seed is not None:
        random.Random(seed).shuffle(trials)

    slots = [0.0] * max(1, n_concurrent)  # per-slot free-time
    heapq.heapify(slots)
    out: List[Req] = []
    for _tid, rs, base, dur in trials:
        start = heapq.heappop(slots)  # next trial starts when a slot frees
        for r in rs:
            out.append(replace(r, arrival=start + (r.arrival - base),
                               finish=start + (r.finish - base)))
        heapq.heappush(slots, start + dur + trial_gap)
    out.sort(key=lambda r: r.arrival)
    return out


# -----------------------------------------------------------------------------
# Mock KV allocator: the cache only needs .device + .free() for replay.
# -----------------------------------------------------------------------------
class MockAllocator:
    def __init__(self):
        self.device = torch.device("cpu")

    def free(self, indices):  # KV slots are abstract here; nothing to reclaim.
        pass


# -----------------------------------------------------------------------------
# Simulated clock: patch time.monotonic so recency bands see recorded gaps.
# -----------------------------------------------------------------------------
class SimClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def build_cache(policy: str, band: float):
    """Instantiate the real cache for a policy name."""
    kw = dict(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=MockAllocator(),
        page_size=1,
    )
    if policy == "lru":
        return RadixCache(CacheInitParams(eviction_policy="lru", **kw))
    if policy == "belady":
        # oracle ceiling: leaves stamped with their program's next-turn arrival
        return RadixCache(CacheInitParams(eviction_policy="belady", **kw))
    # all round_robin variants
    rr_kw = dict(eviction_policy="round_robin")
    if policy == "rrpure":
        pass
    elif policy == "rrband":
        rr_kw["rr_recency_band_secs"] = band
    elif policy == "rrdeclared":
        rr_kw["rr_liveness_source"] = "declared"
    elif policy == "rrsynth":
        rr_kw["rr_liveness_source"] = "declared"
        rr_kw["rr_adaptive_band_target_pctl"] = 85.0
        rr_kw["rr_adaptive_band_identity"] = "program_id"
    elif policy == "rradaptive":
        # adaptive recency band alone (no declared liveness)
        rr_kw["rr_adaptive_band_target_pctl"] = 85.0
        rr_kw["rr_adaptive_band_identity"] = "program_id"
    else:
        raise ValueError(f"unknown policy {policy}")
    return RoundRobinRadixCache(CacheInitParams(**rr_kw, **kw))


@dataclass
class Running:
    node: object        # locked last_device_node from match
    new_tokens: int     # footprint outside the tree ((prompt-cached)+output)
    full_tokens: int = 0  # prompt+output, i.e. the NEED-atlas per-request charge


@dataclass
class ReqResult:
    arrival: float
    prompt_len: int
    cached: int
    pressured: bool = False   # pool was at its budget when this request arrived

    @property
    def miss(self) -> int:
        return self.prompt_len - self.cached


def replay(reqs: List[Req], policy: str, kv_budget: int, band: float):
    """Discrete-event replay of one stream under one policy. Returns per-request
    (arrival, prompt_len, cached) + eviction stats."""
    clock = SimClock()
    # Start the sim clock at the first arrival so any early tracer event (e.g.
    # tracer_open) is stamped on the recorded timeline, not at t=0.
    clock.t = reqs[0].arrival if reqs else 0.0
    # Patch the module-level time refs the cache reads.
    import sglang.srt.mem_cache.radix_cache as rc_mod
    import sglang.srt.mem_cache.rr_radix_cache as rr_mod
    import sglang.srt.mem_cache.evict_policy as ep_mod

    saved = (rc_mod.time.monotonic, rr_mod.time.monotonic, ep_mod.time.monotonic)
    for m in (rc_mod, rr_mod, ep_mod):
        m.time.monotonic = clock.monotonic
    # The radix tracer stamps events with time.time() (wall clock). Drive it from
    # the sim clock too, so an offline replay's trace spans the recorded timeline
    # (~4661s) instead of the ~0.6s real replay duration — animate_radix_trace.py
    # lays out its timeline cursor from these timestamps. (rc_mod.time IS the
    # global time module, so this patches time.time() process-wide for the run.)
    saved_time = rc_mod.time.time
    rc_mod.time.time = clock.monotonic

    declared = policy in ("rrdeclared", "rrsynth")
    belady = policy == "belady"
    # Belady oracle: each request's leaf is reused by its program's NEXT turn.
    # Precompute that next-arrival per rid so eviction can evict furthest-future.
    next_access: Dict[str, float] = {}
    if belady:
        _byp: Dict[str, List[Req]] = defaultdict(list)
        for _r in reqs:
            _byp[_r.program_id or _r.rid].append(_r)
        for _rs in _byp.values():
            _rs = sorted(_rs, key=lambda x: x.arrival)
            for _i, _r in enumerate(_rs):
                next_access[_r.rid] = (
                    _rs[_i + 1].arrival if _i + 1 < len(_rs) else float("inf")
                )
    cache = build_cache(policy, band)

    # Optional full-occupancy log (SGLANG_REPLAY_OCC_LOG=<path prefix>): records
    # (t, protected, evictable, running_new, total) after every event. running_new
    # = tokens held by IN-FLIGHT requests that aren't in the radix tree yet — the
    # piece the radix tracer can't see. Proves the pool = cached tree + in-flight
    # working set is pinned at the budget under load, while evictable+protected
    # alone (what the GIF plots) dips when the in-flight set squeezes the tree.
    import os as _os
    _occ_path = _os.environ.get("SGLANG_REPLAY_OCC_LOG")
    _occ_log: List[tuple] = []

    # release schedule for declared: pid -> last finish ts among its requests
    last_finish: Dict[str, float] = {}
    if declared:
        for r in reqs:
            if r.program_id:
                last_finish[r.program_id] = max(
                    last_finish.get(r.program_id, 0.0), r.finish
                )

    # Build event list: (ts, order, kind, payload). order breaks ties:
    # finishes(0) before releases(1) before arrivals(2) at equal ts.
    EV_FIN, EV_REL, EV_ARR = 0, 1, 2
    events: List[Tuple[float, int, int, object]] = []
    for r in reqs:
        events.append((r.arrival, EV_ARR, EV_ARR, r))
        events.append((r.finish, EV_FIN, EV_FIN, r))
    if declared:
        for pid, ts in last_finish.items():
            events.append((ts + 1e-3, EV_REL, EV_REL, pid))
    events.sort(key=lambda e: (e[0], e[1]))

    counter = 1  # synthetic kv-index source
    running: Dict[str, Running] = {}
    results: List[ReqResult] = []
    tokens_evicted = 0
    evict_calls = 0

    def occupancy() -> int:
        return (
            cache.protected_size()
            + cache.evictable_size()
            + sum(rn.new_tokens for rn in running.values())
        )

    def _stamp_path(node, na: float, now: float) -> None:
        """Stamp a node and all its ancestors with the earliest FUTURE reuse.
        Expiring already-consumed stamps before taking the min is what keeps a
        node from being pinned by a prediction that has come and gone."""
        while node is not None:
            cur = getattr(node, "_belady_next_access", float("inf"))
            if cur <= now:
                cur = float("inf")
            node._belady_next_access = min(cur, na)
            node = node.parent

    def make_room(need_under: int):
        nonlocal tokens_evicted, evict_calls
        # Evict until occupancy <= budget or nothing evictable remains.
        while occupancy() > kv_budget and cache.evictable_size() > 0:
            over = occupancy() - kv_budget
            res = cache.evict(EvictParams(num_tokens=over))
            evict_calls += 1
            n = res.num_tokens_evicted or 0
            tokens_evicted += n
            if n == 0:
                break

    for ts, _o, kind, payload in events:
        clock.t = ts
        if kind == EV_ARR:
            r: Req = payload
            key = RadixKey(r.prompt_ids, None)
            mp = MatchPrefixParams(key=key)
            if declared:
                # adaptive(program_id) + tier re-alive want the req identity.
                mp.req = types.SimpleNamespace(
                    program_id=r.program_id, program_state=r.program_state
                )
            was_full = occupancy() >= kv_budget * 0.98
            m = cache.match_prefix(mp)
            cached = int(m.device_indices.numel())
            results.append(ReqResult(ts, len(r.prompt_ids), cached, was_full))
            cache.inc_lock_ref(m.last_device_node)
            new_tokens = (len(r.prompt_ids) - cached) + len(r.output_ids)
            running[r.rid] = Running(
                node=m.last_device_node, new_tokens=new_tokens,
                full_tokens=len(r.prompt_ids) + len(r.output_ids),
            )
            if belady:
                # Refresh the matched path's stamp. Stamping only at insert lets a
                # stamp go stale the moment the turn it predicted arrives, and a
                # stale stamp now reads as +inf (=evict first) — which would throw
                # away a hot SHARED prefix as soon as its earliest consumer moved
                # on. Re-stamping here keeps every path a program still owns
                # carrying that program's next-turn time.
                _stamp_path(m.last_device_node, next_access.get(r.rid, float("inf")), ts)
            make_room(kv_budget)
        elif kind == EV_FIN:
            r = payload
            rn = running.pop(r.rid, None)
            full = r.prompt_ids + r.output_ids
            val = torch.arange(counter, counter + len(full), dtype=torch.int64)
            counter += len(full)
            cache.insert(InsertParams(key=RadixKey(full, None), value=val))
            if rn is not None:
                cache.dec_lock_ref(rn.node)
            if declared and r.program_id:
                # Remember this program's leaf so release_program can demote it.
                leaf = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(full, None))
                ).last_device_node
                cache._record_program_leaf(r.program_id, leaf)
            if belady:
                # Stamp the WHOLE path (leaf->root) with the earliest time its
                # tokens are next reused (this program's next turn), taking the
                # min so a prefix shared by several programs carries its soonest
                # reuse. Stamping only the tip would leave exposed parents
                # unstamped (=never-reused=evict-first), cascading out good
                # shared prefixes. BeladyStrategy then evicts furthest-future.
                #
                # The min must only range over FUTURE accesses. A stamp whose
                # time has already passed was consumed by an earlier turn; left
                # in place it reads as "reused immediately" and pins the node
                # forever, which made the oracle score WORSE than LRU. Expire
                # any stamp at or before now before taking the min.
                leaf = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(full, None))
                ).last_device_node
                _stamp_path(leaf, next_access.get(r.rid, float("inf")), ts)
        elif kind == EV_REL:
            pid: str = payload
            cache.release_program([pid])

        if _occ_path is not None:
            prot = cache.protected_size()
            evic = cache.evictable_size()
            run = sum(rn.new_tokens for rn in running.values())
            # atlas_infl is the per-request accounting the NEED atlas uses: every
            # in-flight request charged its FULL prompt+output. The pool only
            # stores one copy of a prefix two concurrent requests share, so
            # atlas_infl >= prot+run by exactly the concurrent sharing. Logging
            # both makes that gap measurable instead of assumed.
            atlas_infl = sum(rn.full_tokens for rn in running.values())
            _occ_log.append((ts, prot, evic, run, prot + evic + run, atlas_infl))

    if _occ_path is not None:
        with open(f"{_occ_path}.{policy}.csv", "w") as fh:
            fh.write("t,protected,evictable,running_new,total,atlas_infl\n")
            t_base = _occ_log[0][0] if _occ_log else 0.0
            for ts_, p_, e_, r_, tot_, ai_ in _occ_log:
                fh.write(f"{ts_ - t_base:.3f},{p_},{e_},{r_},{tot_},{ai_}\n")

    # restore time
    for m, fn in zip((rc_mod, rr_mod, ep_mod), saved):
        m.time.monotonic = fn
    rc_mod.time.time = saved_time

    return results, dict(
        tokens_evicted=tokens_evicted,
        evict_calls=evict_calls,
        n_reqs=len(results),
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def summarize(results: List[ReqResult], window_s: float) -> dict:
    tot_new = sum(r.miss for r in results)
    tot_prompt = sum(r.prompt_len for r in results)
    overall = 100.0 * tot_new / max(tot_prompt, 1)

    # per-request miss fraction percentiles
    fr = [100.0 * r.miss / max(r.prompt_len, 1) for r in results]

    # windowed token-weighted miss; p90 over windows (burstiness)
    if results:
        t0 = min(r.arrival for r in results)
        buckets: Dict[int, List[ReqResult]] = {}
        for r in results:
            buckets.setdefault(int((r.arrival - t0) // window_s), []).append(r)
        win_miss = []
        for b in buckets.values():
            n = sum(x.miss for x in b)
            p = sum(x.prompt_len for x in b)
            if p > 0:
                win_miss.append(100.0 * n / p)
    else:
        win_miss = []

    # In-band = requests that arrived while the pool was saturated. A recording
    # with a long low-pressure ramp (start-up, drain-out) dilutes the overall
    # figure toward the compulsory floor and hides whatever the policy does when
    # it actually matters, so report both and never quote the overall alone.
    band = [r for r in results if r.pressured]
    b_new = sum(r.miss for r in band)
    b_prompt = sum(r.prompt_len for r in band)
    return dict(
        n=len(results),
        n_inband=len(band),
        inband_frac=100.0 * len(band) / max(len(results), 1),
        inband_miss=100.0 * b_new / max(b_prompt, 1),
        inband_new_tokens=b_new,
        inband_prompt_tokens=b_prompt,
        overall_miss=overall,
        req_miss_p50=_pct(fr, 50),
        req_miss_p90=_pct(fr, 90),
        win_miss_p90=_pct(win_miss, 90),
        new_tokens=tot_new,
        prompt_tokens=tot_prompt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("server_log")
    ap.add_argument("--kv", type=int, required=True, help="--max-total-tokens budget")
    ap.add_argument(
        "--policies", default="lru,rrpure,rrband,rrdeclared,rrsynth"
    )
    ap.add_argument("--band", type=float, default=5.0, help="rrband recency T (s)")
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument(
        "--concurrency", type=int, default=None,
        help="re-time the SAME trials under this concurrency (default: use "
        "recorded timing). Bypasses the live docker-network cap (~6).",
    )
    ap.add_argument(
        "--conc-seed", type=int, default=None,
        help="shuffle the admission order with this seed when re-timing. "
        "Average over several seeds — a single order is one draw and block "
        "order has faked an RR win before.",
    )
    ap.add_argument(
        "--rec-conc", type=int, default=None,
        help="concurrency the stream was RECORDED at. Enables the per-trial "
        "non-LLM overhead calibration, without which nose-to-tail packing runs "
        "the workload denser than the recording did.",
    )
    ap.add_argument(
        "--rec-span", type=float, default=None,
        help="true recording wall-time in seconds, overriding the stream's span. "
        "Use when the record was concatenated from several runs: the idle seam "
        "between them is not overhead any trial paid.",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.server_log.endswith(".pkl"):
        import pickle, sys as _sys
        _sys.modules.setdefault("rhs", _sys.modules[__name__])
        with open(args.server_log, "rb") as f:
            reqs = pickle.load(f)
    else:
        reqs = parse_stream(args.server_log)
    if not reqs:
        print("no paired requests parsed", file=sys.stderr)
        return 1
    check_not_clipped(reqs)
    n_prog = len({r.program_id for r in reqs if r.program_id})
    n_trial = len({trial_of(r.program_id) for r in reqs if r.program_id})
    if args.concurrency is not None:
        if n_trial < 3 * args.concurrency:
            print(
                f"  note: concurrency {args.concurrency} leaves only "
                f"{n_trial/args.concurrency:.1f} trials per slot; the pack barely "
                f"reaches steady state and the tail is start-up dominated "
                f"(record more trials to go higher).",
                file=sys.stderr,
            )
        gap = (calibrate_trial_gap(reqs, args.rec_conc, args.rec_span)
               if args.rec_conc else 0.0)
        if args.rec_conc:
            print(
                f"  re-timing {n_trial} trials to conc {args.concurrency} "
                f"(seed {args.conc_seed}, per-trial overhead {gap:.0f}s "
                f"calibrated at recorded conc {args.rec_conc})",
                file=sys.stderr,
            )
        reqs = retime_for_concurrency(reqs, args.concurrency, args.conc_seed, gap)
    span = reqs[-1].arrival - reqs[0].arrival
    live_overall = 100.0 * sum(len(r.prompt_ids) - r.live_cached for r in reqs) / max(
        sum(len(r.prompt_ids) for r in reqs), 1
    )
    print(
        f"stream: {len(reqs)} reqs over {span:.0f}s, {n_prog} programs, "
        f"KV={args.kv}, band={args.band}s\n"
        f"  live-server overall miss on THIS stream (sanity): {live_overall:.2f}%\n"
    )

    rows = []
    hdr = (
        f"{'policy':<12} {'reqs':>5} {'miss%':>7} {'inband%':>8} {'nBand':>6} "
        f"{'reqP50':>7} {'reqP90':>7} {'winP90':>7} {'evTok':>9} {'evCalls':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    out = {"stream": args.server_log, "kv": args.kv, "live_overall_miss": live_overall,
           "n_reqs": len(reqs), "policies": {}}
    for pol in args.policies.split(","):
        pol = pol.strip()
        results, stats = replay(reqs, pol, args.kv, args.band)
        s = summarize(results, args.window_s)
        print(
            f"{pol:<12} {s['n']:>5} {s['overall_miss']:>7.2f} {s['inband_miss']:>8.2f} "
            f"{s['n_inband']:>6} {s['req_miss_p50']:>7.2f} "
            f"{s['req_miss_p90']:>7.2f} {s['win_miss_p90']:>7.2f} "
            f"{stats['tokens_evicted']:>9,} {stats['evict_calls']:>8}"
        )
        out["policies"][pol] = {**s, **stats}
        rows.append((pol, s))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
