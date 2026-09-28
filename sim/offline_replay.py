#!/usr/bin/env python3
"""Unified OFFLINE eviction-policy replay for the Qwen3-Coder rerun (results/).

Drives the real SGLang RadixCache / RoundRobinRadixCache on CPU with a recorded
L3 request stream (server.log or a pickled ``rhs.Req`` list), re-timed to a
target concurrency, under every requested arm, and writes one summary.json plus
optional per-request CSVs.

Arms (parsed by name, so any combination works without touching this file):
  lru                      RadixCache LRU                          (baseline)
  belady                   per-program next-arrival oracle         (ceiling, offline only)
  rrpure                   round_robin, node rotation
  rrdecl                   round_robin + declared liveness
  rrchunk<K>               round_robin, token rotation, K tokens/step
  rrdeclchunk<K>           declared + chunk<K>                     (the rerun's arm)
  rr[decl][chunk<K>]band<T>  + recency band T seconds, last_access stamp

What the replay models (identically for every arm -- see replay_harbor_stream.py):
  * a request reserves (prompt-cached)+output tokens from arrival to finish;
  * the tree is fed prompt++output at finish;
  * declared release: a program is demoted right after its LAST request finishes;
  * the replicated band clock is driven from the recorded arrival stamps, so band
    arms are NOT inert here (they were in older offline scripts).

Metrics (per arm, per seed, and mean/ptp over seeds):
  miss overall (token-weighted), compulsory floor (unbounded-pool LRU), per-request
  miss-fraction p50/80/90/99, uncached tokens p50/90/99 + CVaR80/90/95, modelled
  latency (floor + per-token) p50/80/90/99 + CVaR80/90/95, 60 s windowed miss p90,
  in-band fraction, tokens evicted, evict calls, big-pool gate (evicted==0 and
  shortfall==0 at an unbounded pool).

Usage:
  offline_replay.py STREAM --kv 100000 --conc 12 --seeds 0,1,2 --rec-conc 4 \
      --arms lru,belady,rrchunk1024,rrdeclchunk1024 --out-dir results/<b>/setup_search/kv100k_c12
"""
from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import time as _time
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_harbor_stream.py")


def load_rhs():
    """Import bench/scripts/replay_harbor_stream.py under the module name 'rhs'
    (the name every pickled stream in this repo was written with)."""
    if "rhs" in sys.modules:
        return sys.modules["rhs"]
    spec = importlib.util.spec_from_file_location("rhs", RHS)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rhs"] = m
    spec.loader.exec_module(m)
    return m


rhs = load_rhs()
import torch  # noqa: E402  (after rhs so the sglang import path is set up)

torch.set_num_threads(1)

from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey  # noqa: E402
from sglang.srt.mem_cache.rr_radix_cache import RoundRobinRadixCache  # noqa: E402

# Latency model (ms): calibrated on the SWE PD prefill node, Qwen3-Coder-30B.
# Only used for a *modelled* latency column; GPU runs measure the real thing.
LAT_FLOOR_MS = 170.0
LAT_PERTOK_MS = 0.0657

ARM_RE = re.compile(r"^rr(?P<decl>decl)?(?:chunk(?P<chunk>\d+))?(?:band(?P<band>\d+(?:\.\d+)?))?$")


@dataclass
class ArmSpec:
    name: str
    policy: str  # lru | belady | round_robin
    declared: bool = False
    chunk: int = 0
    band: float = 0.0

    @staticmethod
    def parse(name: str) -> "ArmSpec":
        if name == "lru":
            return ArmSpec(name, "lru")
        if name == "belady":
            return ArmSpec(name, "belady")
        if name == "rrpure":
            return ArmSpec(name, "round_robin")
        m = ARM_RE.match(name)
        if not m:
            raise ValueError(f"unknown arm '{name}'")
        return ArmSpec(
            name,
            "round_robin",
            declared=bool(m.group("decl")),
            chunk=int(m.group("chunk") or 0),
            band=float(m.group("band") or 0.0),
        )

    def build(self):
        base = dict(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=rhs.MockAllocator(),
            page_size=1,
        )
        if self.policy in ("lru", "belady"):
            return RadixCache(CacheInitParams(eviction_policy=self.policy, **base))
        kw: dict = dict(eviction_policy="round_robin")
        if self.declared:
            kw["rr_liveness_source"] = "declared"
        if self.chunk:
            kw["rr_chunk_tokens"] = self.chunk
        if self.band:
            kw["rr_recency_band_secs"] = self.band
            kw["rr_band_stamp"] = "last_access"
        return RoundRobinRadixCache(CacheInitParams(**kw, **base))


@dataclass
class ReqRec:
    program_id: str
    turn: int
    arrival: float
    prompt_len: int
    cached: int
    output_len: int
    pressured: bool

    @property
    def uncached(self) -> int:
        return self.prompt_len - self.cached


def _turn_index(reqs) -> Dict[str, int]:
    """rid -> k-th request of its program (arrival order)."""
    byp: Dict[str, list] = defaultdict(list)
    for r in reqs:
        byp[r.program_id or r.rid].append(r)
    idx = {}
    for rs in byp.values():
        rs.sort(key=lambda r: r.arrival)
        for k, r in enumerate(rs):
            idx[r.rid] = k
    return idx


def replay(reqs, spec: ArmSpec, kv_budget: int, occ_log_path: Optional[str] = None):
    """Discrete-event replay of one re-timed stream under one arm.

    Returns (records, stats). Mirrors rhs.replay but (a) builds the cache from an
    ArmSpec, (b) always hands the cache a req carrying program tags AND a
    created_time stamp so the RR band clock advances offline, (c) keeps the
    program/turn identity per record, (d) times evict() calls."""
    clock = rhs.SimClock()
    clock.t = reqs[0].arrival if reqs else 0.0
    import sglang.srt.mem_cache.evict_policy as ep_mod
    import sglang.srt.mem_cache.radix_cache as rc_mod
    import sglang.srt.mem_cache.rr_radix_cache as rr_mod

    saved = (rc_mod.time.monotonic, rr_mod.time.monotonic, ep_mod.time.monotonic)
    for m in (rc_mod, rr_mod, ep_mod):
        m.time.monotonic = clock.monotonic
    saved_time = rc_mod.time.time
    rc_mod.time.time = clock.monotonic

    declared = spec.declared
    belady = spec.policy == "belady"
    turn_of = _turn_index(reqs)

    next_access: Dict[str, float] = {}
    if belady:
        byp: Dict[str, list] = defaultdict(list)
        for r in reqs:
            byp[r.program_id or r.rid].append(r)
        for rs in byp.values():
            rs = sorted(rs, key=lambda x: x.arrival)
            for i, r in enumerate(rs):
                next_access[r.rid] = rs[i + 1].arrival if i + 1 < len(rs) else math.inf

    cache = spec.build()

    last_finish: Dict[str, float] = {}
    if declared:
        for r in reqs:
            if r.program_id:
                last_finish[r.program_id] = max(last_finish.get(r.program_id, 0.0), r.finish)

    EV_FIN, EV_REL, EV_ARR = 0, 1, 2
    events: List[Tuple[float, int, int, object]] = []
    for r in reqs:
        events.append((r.arrival, EV_ARR, EV_ARR, r))
        events.append((r.finish, EV_FIN, EV_FIN, r))
    if declared:
        for pid, ts in last_finish.items():
            events.append((ts + 1e-3, EV_REL, EV_REL, pid))
    events.sort(key=lambda e: (e[0], e[1]))

    counter = 1
    running: Dict[str, rhs.Running] = {}
    records: List[ReqRec] = []
    tokens_evicted = 0
    evict_calls = 0
    evict_secs = 0.0
    occ_log: List[tuple] = []

    def occupancy() -> int:
        return (
            cache.protected_size()
            + cache.evictable_size()
            + sum(rn.new_tokens for rn in running.values())
        )

    def stamp_path(node, na: float, now: float) -> None:
        while node is not None:
            cur = getattr(node, "_belady_next_access", math.inf)
            if cur <= now:
                cur = math.inf
            node._belady_next_access = min(cur, na)
            node = node.parent

    def make_room():
        nonlocal tokens_evicted, evict_calls, evict_secs
        while occupancy() > kv_budget and cache.evictable_size() > 0:
            over = occupancy() - kv_budget
            t0 = _time.perf_counter()
            res = cache.evict(EvictParams(num_tokens=over))
            evict_secs += _time.perf_counter() - t0
            evict_calls += 1
            n = res.num_tokens_evicted or 0
            tokens_evicted += n
            if n == 0:
                break

    for ts, _o, kind, payload in events:
        clock.t = ts
        if kind == EV_ARR:
            r = payload
            mp = MatchPrefixParams(key=RadixKey(r.prompt_ids, None))
            # Always give the cache a request identity: declared tiers and the
            # adaptive controller key on program_id, and the replicated band
            # clock advances from time_stats.created_time.
            mp.req = types.SimpleNamespace(
                program_id=r.program_id,
                program_state=r.program_state or "active",
                time_stats=types.SimpleNamespace(created_time=ts),
            )
            was_full = occupancy() >= kv_budget * 0.98
            m = cache.match_prefix(mp)
            cached = int(m.device_indices.numel())
            records.append(
                ReqRec(
                    program_id=r.program_id or r.rid,
                    turn=turn_of[r.rid],
                    arrival=ts,
                    prompt_len=len(r.prompt_ids),
                    cached=cached,
                    output_len=len(r.output_ids),
                    pressured=was_full,
                )
            )
            cache.inc_lock_ref(m.last_device_node)
            new_tokens = (len(r.prompt_ids) - cached) + len(r.output_ids)
            running[r.rid] = rhs.Running(
                node=m.last_device_node,
                new_tokens=new_tokens,
                full_tokens=len(r.prompt_ids) + len(r.output_ids),
            )
            if belady:
                stamp_path(m.last_device_node, next_access.get(r.rid, math.inf), ts)
            make_room()
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
                leaf = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(full, None))
                ).last_device_node
                cache._record_program_leaf(r.program_id, leaf)
            if belady:
                leaf = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(full, None))
                ).last_device_node
                stamp_path(leaf, next_access.get(r.rid, math.inf), ts)
        else:
            cache.release_program([payload])

        if occ_log_path is not None:
            prot = cache.protected_size()
            evic = cache.evictable_size()
            run = sum(rn.new_tokens for rn in running.values())
            atlas = sum(rn.full_tokens for rn in running.values())
            occ_log.append((ts, prot, evic, run, prot + evic + run, atlas))

    if occ_log_path is not None and occ_log:
        t_base = occ_log[0][0]
        with open(occ_log_path, "w") as fh:
            fh.write("t,protected,evictable,running_new,total,atlas_infl\n")
            for ts_, p_, e_, r_, tot_, ai_ in occ_log:
                fh.write(f"{ts_ - t_base:.3f},{p_},{e_},{r_},{tot_},{ai_}\n")

    for m, fn in zip((rc_mod, rr_mod, ep_mod), saved):
        m.time.monotonic = fn
    rc_mod.time.time = saved_time

    return records, dict(
        tokens_evicted=tokens_evicted,
        evict_calls=evict_calls,
        evict_secs=evict_secs,
        n_reqs=len(records),
    )


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def cvar(a, alpha):
    """Mean of the worst (1-alpha) fraction: CVaR80 = mean of the top 20 %."""
    a = np.sort(np.asarray(a, dtype=float))
    if len(a) == 0:
        return float("nan")
    k = max(1, int(round(len(a) * (1.0 - alpha))))
    return float(a[-k:].mean())


def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    ne = np.nonzero(x != y)[0]
    return int(ne[0]) if len(ne) else n


def shortfall(records: List[ReqRec], reqs) -> float:
    """Big-pool ordering gate. At an UNBOUNDED pool a turn must match at least
    the common prefix between its prompt and its program's previous
    prompt++output (that context was inserted at the previous finish). A
    shortfall means the turn arrived BEFORE its predecessor finished (stream
    ordering violation) -- the failure mode that inflated CATraces 8x. Context
    resets do not count: they shorten the common prefix itself.
    Returns pp of total prompt tokens; must be ~0 for a well-formed stream."""
    by_rid = {}
    byp: Dict[str, list] = defaultdict(list)
    for r in reqs:
        byp[r.program_id or r.rid].append(r)
    for rs in byp.values():
        rs.sort(key=lambda r: r.arrival)
        for k, r in enumerate(rs):
            by_rid[(r.program_id or r.rid, k)] = (r, rs[k - 1] if k else None)
    short = 0
    tot = sum(r.prompt_len for r in records)
    for rec in records:
        r, prev = by_rid[(rec.program_id, rec.turn)]
        if prev is None:
            continue
        need = _lcp(prev.prompt_ids + prev.output_ids, r.prompt_ids)
        short += max(0, need - rec.cached)
    return 100.0 * short / max(tot, 1)


def metrics(records: List[ReqRec], stats: dict, window_s: float = 60.0,
            lat_floor=LAT_FLOOR_MS, lat_pertok=LAT_PERTOK_MS, reqs=None) -> dict:
    unc = np.array([r.uncached for r in records], dtype=float)
    pl = np.array([r.prompt_len for r in records], dtype=float)
    frac = 100.0 * unc / np.maximum(pl, 1)
    lat = lat_floor + lat_pertok * unc
    t0 = min(r.arrival for r in records) if records else 0.0
    buckets: Dict[int, List[ReqRec]] = defaultdict(list)
    for r in records:
        buckets[int((r.arrival - t0) // window_s)].append(r)
    win = []
    for b in buckets.values():
        p = sum(x.prompt_len for x in b)
        if p:
            win.append(100.0 * sum(x.uncached for x in b) / p)
    band = [r for r in records if r.pressured]
    out = dict(
        n=len(records),
        miss_overall=100.0 * unc.sum() / max(pl.sum(), 1),
        uncached_total=int(unc.sum()),
        prompt_total=int(pl.sum()),
        inband_frac=100.0 * len(band) / max(len(records), 1),
        inband_miss=(100.0 * sum(r.uncached for r in band) / max(sum(r.prompt_len for r in band), 1)),
        win_miss_p90=pct(win, 90) if win else float("nan"),
        shortfall_pp=(shortfall(records, reqs) if reqs is not None else float("nan")),
        **{k: stats[k] for k in ("tokens_evicted", "evict_calls", "evict_secs")},
    )
    for q in (50, 80, 90, 99):
        out[f"frac_p{q}"] = pct(frac, q)
        out[f"unc_p{q}"] = pct(unc, q)
        out[f"lat_p{q}"] = pct(lat, q)
    for a in (0.80, 0.90, 0.95):
        tag = f"cvar{int(a*100)}"
        out[f"frac_{tag}"] = cvar(frac, a)
        out[f"unc_{tag}"] = cvar(unc, a)
        out[f"lat_{tag}"] = cvar(lat, a)
    return out


def inflight_profile(timed) -> dict:
    """Time-weighted atlas in-flight (prompt+output per running request) of a
    re-timed stream: MAX / p50 / p90 / p99. Pool must exceed MAX (deadlock)."""
    evs = []
    for r in timed:
        w = len(r.prompt_ids) + len(r.output_ids)
        evs.append((r.arrival, w))
        evs.append((r.finish, -w))
    evs.sort()
    cur = 0
    prev_t = evs[0][0] if evs else 0.0
    levels, weights = [], []
    for t, d in evs:
        if t > prev_t:
            levels.append(cur)
            weights.append(t - prev_t)
        cur += d
        prev_t = t
    levels = np.array(levels, dtype=float)
    weights = np.array(weights, dtype=float)
    order = np.argsort(levels)
    cw = np.cumsum(weights[order]) / max(weights.sum(), 1e-9)

    def wq(q):
        return float(levels[order][np.searchsorted(cw, q / 100.0)]) if len(levels) else 0.0

    return dict(
        infl_max=int(levels.max()) if len(levels) else 0,
        infl_p50=wq(50), infl_p90=wq(90), infl_p99=wq(99),
        span_s=float(evs[-1][0] - evs[0][0]) if evs else 0.0,
    )


def p_infl_over(timed, pool: int) -> float:
    """Time-weighted P(in-flight > pool) in percent."""
    evs = []
    for r in timed:
        w = len(r.prompt_ids) + len(r.output_ids)
        evs.append((r.arrival, w))
        evs.append((r.finish, -w))
    evs.sort()
    cur = 0
    prev_t = evs[0][0] if evs else 0.0
    over = tot = 0.0
    for t, d in evs:
        if t > prev_t:
            tot += t - prev_t
            if cur > pool:
                over += t - prev_t
        cur += d
        prev_t = t
    return 100.0 * over / max(tot, 1e-9)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def load_stream(path: str):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            reqs = pickle.load(f)
    else:
        reqs = rhs.parse_stream(path)
    if not reqs:
        raise SystemExit("no paired requests parsed")
    rhs.check_not_clipped(reqs)
    return reqs


def retime(reqs, conc: int, seed: int, rec_conc: Optional[int], rec_span: Optional[float]):
    if conc <= 0:
        return list(reqs), 0.0
    gap = rhs.calibrate_trial_gap(reqs, rec_conc, rec_span) if rec_conc else 0.0
    return rhs.retime_for_concurrency(reqs, conc, seed, gap), gap


def run_setup(reqs, arms: List[str], kv: int, conc: int, seeds: List[int],
              rec_conc: Optional[int], rec_span: Optional[float], out_dir: str,
              write_csv: bool = False, gate: bool = True, window_s: float = 60.0,
              lat_floor=LAT_FLOOR_MS, lat_pertok=LAT_PERTOK_MS, occ_log=False,
              quiet=False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    specs = [ArmSpec.parse(a) for a in arms]
    result = dict(
        stream_n_reqs=len(reqs), kv=kv, conc=conc, seeds=seeds, rec_conc=rec_conc,
        rec_span=rec_span, arms=arms, lat_model_ms=dict(floor=lat_floor, pertok=lat_pertok),
        per_seed={}, gate={},
    )
    n_trial = len({rhs.trial_of(r.program_id) for r in reqs if r.program_id})
    result["n_trials"] = n_trial
    result["n_programs"] = len({r.program_id for r in reqs if r.program_id})

    for sd in seeds:
        timed, gap = retime(reqs, conc, sd, rec_conc, rec_span)
        prof = inflight_profile(timed)
        prof["p_infl_over_pool"] = p_infl_over(timed, kv)
        prof["trial_gap_s"] = gap
        seed_out = dict(profile=prof, arms={})
        if gate:
            # Big-pool gate: unbounded pool -> compulsory floor; evicted must be 0
            # and every turn must at least match its own previous prompt.
            rec0, st0 = replay(timed, ArmSpec.parse("lru"), 10**12)
            m0 = metrics(rec0, st0, window_s, lat_floor, lat_pertok, reqs=timed)
            seed_out["floor"] = dict(
                miss=m0["miss_overall"], tokens_evicted=st0["tokens_evicted"],
                shortfall_pp=m0["shortfall_pp"],
                ok=(st0["tokens_evicted"] == 0 and m0["shortfall_pp"] < 0.05),
            )
        for spec in specs:
            occ = os.path.join(out_dir, f"occ_{spec.name}_s{sd}.csv") if occ_log else None
            t0 = _time.perf_counter()
            recs, st = replay(timed, spec, kv, occ)
            m = metrics(recs, st, window_s, lat_floor, lat_pertok)
            m["wall_s"] = _time.perf_counter() - t0
            seed_out["arms"][spec.name] = m
            if write_csv:
                with open(os.path.join(out_dir, f"req_{spec.name}_s{sd}.csv"), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["program_id", "turn", "arrival", "prompt_len", "cached", "uncached", "output_len", "pressured"])
                    for r in recs:
                        w.writerow([r.program_id, r.turn, f"{r.arrival:.3f}", r.prompt_len, r.cached, r.uncached, r.output_len, int(r.pressured)])
            if not quiet:
                print(f"  seed {sd} {spec.name:<18} miss {m['miss_overall']:6.2f}  unc p99 {m['unc_p99']:7.0f}  "
                      f"cvar90 {m['unc_cvar90']:7.0f}  lat p90 {m['lat_p90']:6.0f}  ev {st['tokens_evicted']:>10,}  "
                      f"({m['wall_s']:.0f}s)", flush=True)
        result["per_seed"][str(sd)] = seed_out

    # aggregate over seeds
    agg = {}
    keys = None
    for spec in specs:
        rows = [result["per_seed"][str(sd)]["arms"][spec.name] for sd in seeds]
        keys = keys or [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        agg[spec.name] = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            agg[spec.name][k] = dict(mean=float(vals.mean()), ptp=float(vals.max() - vals.min()))
    result["agg"] = agg
    if gate:
        fl = [result["per_seed"][str(sd)]["floor"] for sd in seeds]
        result["gate"] = dict(
            floor_miss_mean=float(np.mean([f["miss"] for f in fl])),
            all_ok=all(f["ok"] for f in fl),
            tokens_evicted=[f["tokens_evicted"] for f in fl],
            shortfall_pp=[f["shortfall_pp"] for f in fl],
        )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def print_table(result: dict) -> str:
    lines = []
    seeds = result["seeds"]
    g = result.get("gate") or {}
    prof = [result["per_seed"][str(s)]["profile"] for s in seeds]
    lines.append(
        f"stream {result['stream_n_reqs']} req / {result['n_trials']} trials / {result['n_programs']} programs | "
        f"kv {result['kv']} conc {result['conc']} seeds {seeds} | trial_gap {prof[0]['trial_gap_s']:.0f}s"
    )
    lines.append(
        "in-flight (atlas) MAX " + "/".join(f"{p['infl_max']}" for p in prof)
        + f"  p99 {np.mean([p['infl_p99'] for p in prof]):.0f}  "
        + f"P(infl>pool) {np.mean([p['p_infl_over_pool'] for p in prof]):.1f}%"
    )
    if g:
        lines.append(f"big-pool gate: floor {g['floor_miss_mean']:.2f}%  evicted {g['tokens_evicted']}  "
                     f"shortfall {['%.2f' % s for s in g['shortfall_pp']]}pp  -> {'OK' if g['all_ok'] else 'VOID'}")
    hdr = (f"{'arm':<20}{'miss%':>8}{'±':>6} | {'unc p50':>8}{'p90':>7}{'p99':>8}{'cvar90':>8}{'cvar95':>8} | "
           f"{'lat p80':>8}{'p90':>6}{'p99':>7}{'cvar90':>8} | {'frac p90':>9}{'p99':>6} | {'win90':>6}{'evicted':>12}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for arm, a in result["agg"].items():
        lines.append(
            f"{arm:<20}{a['miss_overall']['mean']:8.2f}{a['miss_overall']['ptp']/2:6.2f} | "
            f"{a['unc_p50']['mean']:8.0f}{a['unc_p90']['mean']:7.0f}{a['unc_p99']['mean']:8.0f}"
            f"{a['unc_cvar90']['mean']:8.0f}{a['unc_cvar95']['mean']:8.0f} | "
            f"{a['lat_p80']['mean']:8.0f}{a['lat_p90']['mean']:6.0f}{a['lat_p99']['mean']:7.0f}{a['lat_cvar90']['mean']:8.0f} | "
            f"{a['frac_p90']['mean']:9.1f}{a['frac_p99']['mean']:6.1f} | "
            f"{a['win_miss_p90']['mean']:6.1f}{a['tokens_evicted']['mean']:12,.0f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", help="L3 server.log or pickled rhs.Req list")
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--conc", type=int, default=0, help="0 = recorded timing")
    ap.add_argument("--seeds", default="0", help="comma list; admission-order shuffle seeds")
    ap.add_argument("--rec-conc", type=int, default=None)
    ap.add_argument("--rec-span", type=float, default=None)
    ap.add_argument("--arms", default="lru,belady,rrchunk1024,rrdeclchunk1024")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv", action="store_true", help="write per-request CSV per arm/seed")
    ap.add_argument("--occ-log", action="store_true", help="write occupancy CSV per arm/seed")
    ap.add_argument("--no-gate", action="store_true", help="skip the unbounded-pool floor run")
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--lat-floor-ms", type=float, default=LAT_FLOOR_MS)
    ap.add_argument("--lat-pertok-ms", type=float, default=LAT_PERTOK_MS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    reqs = load_stream(args.stream)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    res = run_setup(
        reqs, arms, args.kv, args.conc, seeds, args.rec_conc, args.rec_span, args.out_dir,
        write_csv=args.csv, gate=not args.no_gate, window_s=args.window_s,
        lat_floor=args.lat_floor_ms, lat_pertok=args.lat_pertok_ms, occ_log=args.occ_log,
        quiet=args.quiet,
    )
    table = print_table(res)
    print(table)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
