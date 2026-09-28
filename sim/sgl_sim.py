#!/usr/bin/env python3
"""Closed-loop replay of a recorded agent stream through SGLang's REAL Scheduler (CPU, no model, no GPU), on a
virtual clock.

Only the GPU layer is replaced:
  model worker    FakeWorker: no weights; forward_batch_generation() prices the step with the machine profile
                  (same step-cost formulas and noise model as des_sim.py) and "samples" each request's recorded
                  output token, so the radix tree sees the recorded token ids.
  GPU timeline    batches execute back to back on one virtual GPU: start = max(launch, previous batch end). The
                  scheduler's clock only moves when it waits for a batch result, exactly where the real loop blocks.
                  By default the loop is event_loop_overlap (as the GPU servers): batch j+1 is formed and launched
                  before batch j's result is processed.
  KV pool         the real TokenToKVPoolAllocator / ReqToTokenPool on CPU; the KV storage itself is a stub.
  IPC             no ZMQ: requests are handed to Scheduler.handle_generate_request, outputs are captured from
                  send_to_detokenizer.
  time            time.time / perf_counter / monotonic are the virtual clock (patched before sglang is imported).
Everything else -- PrefillAdder, schedule policy, chunked prefill, retract, new_token_ratio, the overlap lag, radix
cache and the RR variants -- is SGLang's own code.

The replay client (trials -> programs -> turns, next turn = finish + recorded tool gap + client overhead, server
ingress, first-token response delay) is the same model as des_sim.py and reads the same machine profile keys.

usage: sgl_sim.py STREAM.pkl --kv 750000 --conc 46 --seed 36 --arm rrdeclchunk1024 --params-json M.json --out-dir DIR
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time as _time_mod


class VClock:
    """Virtual clock. The real functions stay available as VClock.real_*."""
    real_time, real_perf, real_mono = _time_mod.time, _time_mod.perf_counter, _time_mod.monotonic
    EPOCH = 1.7e9

    def __init__(self): self.t = 0.0
    def time(self): return self.EPOCH + self.t
    def mono(self): return self.t


CLOCK = VClock()
_time_mod.time = CLOCK.time
_time_mod.perf_counter = CLOCK.mono
_time_mod.monotonic = CLOCK.mono

import sys, types, logging, math, heapq, random
from collections import defaultdict, deque
import numpy as np
import torch
torch.set_num_threads(1)

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler, get_global_server_args
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput, ReleaseProgramReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_replay as OR

MODEL_PATH = "/extra/chuxi/Qwen3-Coder-30B-A3B-Instruct"   # only config.json is read
VOCAB = 151936


class _Group:
    """Stand-in for a GroupCoordinator of world size 1."""
    world_size = 1; rank = 0; rank_in_group = 0; local_rank = 0; is_first_rank = True; is_last_rank = True; first_rank = 0; last_rank = 0
    cpu_group = None; device_group = None; device = torch.device("cpu"); ranks = [0]


class FakeKVCache:
    """The allocator only keeps a handle; nothing reads KV data in the simulator."""
    def __init__(self, size): self.size = size


class FakeWorker:
    def __init__(self, sim, server_args, model_config, pool_reqs):
        self.sim = sim
        n = server_args.max_total_tokens
        self.req_to_token_pool = ReqToTokenPool(size=pool_reqs, max_context_len=model_config.context_len + 4, device="cpu", enable_memory_saver=False)
        self.allocator = TokenToKVPoolAllocator(n, dtype=torch.bfloat16, device="cpu", kvcache=FakeKVCache(n), need_sort=False)
        self.model_runner = types.SimpleNamespace(hybrid_gdn_config=None, mamba2_config=None, sliding_window_size=None, is_hybrid_swa=False,
                                                  model=None, forward_stream=None)
        self.model_config = model_config
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        # same limits as TpModelWorker / ModelRunner._resolve_max_num_reqs for this pool
        estimated = max(min(int(n / model_config.context_len * 512), 4096), 2048)
        max_running = min(estimated, n // 2) if server_args.max_running_requests is None else min(server_args.max_running_requests, estimated)
        max_req_len = min(model_config.context_len - 1, n - 1)
        self.info = (n, server_args.max_prefill_tokens, max_running, server_args.max_queued_requests, max_req_len, max_req_len - 5,
                     server_args.random_seed, "cpu", torch.cpu.Stream(), pool_reqs, model_config.context_len + 4, n)

    def get_worker_info(self): return self.info
    def get_memory_pool(self): return self.req_to_token_pool, self.allocator
    def get_pad_input_ids_func(self): return None
    def get_tokens_per_layer_info(self): return None, None
    def register_hicache_layer_transfer_counter(self, c): pass
    def set_hicache_consumer(self, *a): pass

    def forward_batch_generation(self, mwb, **kw):
        return self.sim.forward(mwb)


class SimScheduler(Scheduler):
    """Scheduler with the GPU/IPC-facing init steps replaced."""
    SIM = None  # set by the driver before construction

    def init_ipc_channels(self, port_args):
        sim = self.SIM
        class _Sink:
            def send_output(self, obj, *a, **k): sim.on_output(obj)
            def send_pyobj(self, obj, *a, **k): sim.on_output(obj)
        self.idle_sleeper = None
        self.recv_from_tokenizer = self.recv_from_rpc = None
        self.send_to_tokenizer = _Sink(); self.send_to_detokenizer = _Sink()

    def init_tokenizer(self):
        self.is_generation = self.model_config.is_generation
        self.tokenizer = self.processor = None; self._mm_processor = None; self._think_end_id = None

    def init_moe_gemm_config(self):
        self.require_mlp_sync = False

    def init_mamba_backend(self): pass

    def init_model_worker(self):
        self.tp_worker = FakeWorker(self.SIM, self.server_args, self.model_config, self.SIM.pool_reqs)
        self.draft_worker = None; self.model_worker = self.tp_worker
        (self.max_total_num_tokens, self.max_prefill_tokens, self.max_running_requests, self.max_queued_requests, self.max_req_len,
         self.max_req_input_len, self.random_seed, self.device, self.forward_stream, _, _, _) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(self.max_running_requests // self.pp_size, 1)
        g = _Group()
        self.tp_group = self.attn_tp_group = self.attn_cp_group = self.pp_group = self.world_group = self.dp_tp_group = g
        self.tp_cpu_group = self.attn_tp_cpu_group = self.attn_cp_cpu_group = self.dp_tp_cpu_group = None
        self.pad_input_ids_func = None

    def init_watch_dog_memory_saver_input_blocker(self):
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
        self.watchdog = None
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
        self.offload_tags = set(); self.recv_skipper = None; self.input_blocker = None

    def init_overlap(self):
        super().init_overlap()      # FutureMap etc. on the cpu device (SGLang's own CPU path)
        self.schedule_stream = torch.cpu.Stream()
        self.result_queue = deque()


class Sim:
    """Virtual-time driver around SimScheduler: one virtual GPU, SGLang's normal or overlap event loop."""

    def __init__(self, server_args, P, pool_reqs=256):
        self.P = P; self.pool_reqs = pool_reqs; self.clock = CLOCK
        self.overlap = not server_args.disable_overlap_schedule
        self.recorded_out = {}      # rid -> recorded output token ids
        self.on_finish = None       # callback(rid, t)
        self.first_token = {}; self.finish = {}; self.first_sched = {}; self.cached = {}
        self.pending = defaultdict(int)     # rid -> tokens emitted by launched batches whose result is not processed yet
        self.gpu_free = 0.0
        self.nsteps = [0, 0]; self.busy_time = [0.0, 0.0]
        # machine noise model (des_sim._simulate): decode speed factor resampled every dec_period s with a slow AR(1)
        # drift, iid prefill factor, optional mean-preserving lognormal jitter
        self.nr = np.random.default_rng(P["noise_seed"]) if P["noise_seed"] >= 0 else None
        self.jr = np.random.default_rng(P["jitter_seed"]) if P["jitter"] > 0 else None
        self.nz = dict(next=0.0, fac=1.0, drift=0.0)
        set_global_server_args_for_scheduler(server_args)
        SimScheduler.SIM = self
        self.sched = SimScheduler(server_args, types.SimpleNamespace(nccl_port=0), gpu_id=0, tp_rank=0, moe_ep_rank=0, pp_rank=0,
                                  attn_cp_rank=0, moe_dp_rank=0, dp_rank=None)
        self.sched.result_queue = deque()

    # ---- noise ----
    def dec_factor(self, t):
        P, nz = self.P, self.nz
        if self.nr is None: return 1.0
        if t >= nz["next"]:
            phi = math.exp(-P["dec_period"] / P["drift_tau"])
            nz["drift"] = phi * nz["drift"] + math.sqrt(1 - phi * phi) * P["drift_noise"] * self.nr.standard_normal()
            sg = P["dec_noise"]; nz["fac"] = math.exp(sg * self.nr.standard_normal() - sg * sg / 2 + nz["drift"]); nz["next"] = t + P["dec_period"]
        return nz["fac"]

    def pre_factor(self):
        if self.nr is None: return 1.0
        sg = self.P["pre_noise"]; return math.exp(sg * self.nr.standard_normal() - sg * sg / 2)

    def jit(self, dt, scale=1.0):
        if self.jr is None: return dt
        sg = self.P["jitter"] * scale; return dt * float(self.jr.lognormal(-sg * sg / 2, sg))

    # ---- step pricing (same formulas as des_sim.py) ----
    def prefill_cost(self, items, open_after):
        P = self.P; new = sum(n for n, _ in items); attn = sum(n * (p + n / 2) for n, p in items) / 1e9
        return (P["p0"] + P["p_tok"] * new / 1e3 + P["p_attn"] * attn + (P["p_open"] if open_after else 0.0)) / 1e3

    def decode_cost(self, bs, tokens, maxctx):
        P = self.P
        if bs <= P["cg_max_bs"]: ms = P["d0"] + P["d_tok"] * tokens / 1e3 + P["d_req"] * bs + P.get("d_max", 0.0) * maxctx / 1e3
        else: ms = P["n0"] + P["n_tok"] * tokens / 1e3 + P["n_req"] * bs + P.get("n_max", 0.0) * maxctx / 1e3
        return max(ms, P["d_min"]) / 1e3

    def forward(self, mwb):
        """The virtual GPU: price the batch, queue it behind the previous one, emit the recorded next tokens."""
        s = self.sched; batch = s.cur_batch
        start = max(self.clock.t, self.gpu_free)
        if mwb.forward_mode.is_extend():
            items = list(zip(mwb.extend_seq_lens, mwb.extend_prefix_lens))
            dt = self.jit(self.prefill_cost(items, s.chunked_req is not None), 2.0) * self.pre_factor(); self.nsteps[0] += 1; self.busy_time[0] += dt
            emit = [r is not s.chunked_req for r in batch.reqs]          # the request left open by this batch samples nothing
        else:
            used = s.max_total_num_tokens - s.token_to_kv_pool_allocator.available_size() - s.tree_cache.evictable_size()   # the log's #token
            maxctx = max(mwb.seq_lens_cpu.tolist()) if mwb.seq_lens_cpu is not None else 0
            dt = self.jit(self.decode_cost(len(batch.reqs), used, maxctx)) * self.dec_factor(start)
            self.nsteps[1] += 1; self.busy_time[1] += dt
            emit = [True] * len(batch.reqs)
        end = start + dt; self.gpu_free = end
        nxt = []; emitted = []
        for r, e in zip(batch.reqs, emit):
            self.first_sched.setdefault(r.rid, start)
            out = self.recorded_out.get(r.rid, ())
            k = len(r.output_ids) + self.pending[r.rid]
            nxt.append(out[k] if k < len(out) else 0)
            if e: self.pending[r.rid] += 1; emitted.append(r.rid)
        res = GenerationBatchResult(logits_output=LogitsProcessorOutput(next_token_logits=None), next_token_ids=torch.tensor(nxt, dtype=torch.int64),
                                    can_run_cuda_graph=len(batch.reqs) <= self.P["cg_max_bs"])
        res._sim_end = end; res._sim_emitted = emitted
        return res

    def _process(self, batch, result):
        """process_batch_result of a batch: the scheduler blocks until the virtual GPU has finished it."""
        self.clock.t = max(self.clock.t, result._sim_end)
        self.sched.process_batch_result(batch, result)
        for rid in result._sim_emitted:
            if rid in self.pending: self.pending[rid] -= 1
        for r in batch.reqs:
            if r.output_ids and r.rid not in self.first_token: self.first_token[r.rid] = self.clock.t

    def on_output(self, obj):
        rids = getattr(obj, "rids", None)
        if not rids: return
        fins = getattr(obj, "finished_reasons", [None] * len(rids))
        cts = getattr(obj, "cached_tokens", None) or [None] * len(rids)
        for rid, fin, ct in zip(rids, fins, cts):
            if fin is not None and rid not in self.finish:
                self.finish[rid] = self.clock.t; self.cached[rid] = ct
                if self.on_finish: self.on_finish(rid, self.clock.t)

    def submit(self, rid, prompt_ids, output_ids, program_id=None, program_state=None, created_time=None):
        self.recorded_out[rid] = list(output_ids)
        sp = SamplingParams(max_new_tokens=max(1, len(output_ids)), ignore_eos=True, temperature=0.0)
        sp.normalize(None)
        req = TokenizedGenerateReqInput(rid=rid, input_text="", input_ids=list(prompt_ids), mm_inputs=None, sampling_params=sp,
                                        return_logprob=False, logprob_start_len=-1, top_logprobs_num=0, token_ids_logprob=None, stream=False,
                                        program_id=program_id, program_state=program_state)
        self.sched.handle_generate_request(req)
        if created_time is not None:
            # the tokenizer manager stamps created_time upstream; RoundRobinRadixCache uses it as its band clock
            # (> 0: an unstamped 0.0 counts as missing and 32 in a row switch the cache to its time.monotonic() fallback)
            q = self.sched.waiting_queue
            if q and q[-1].rid == rid: q[-1].time_stats.created_time = max(created_time, 1e-6)

    def step(self):
        """One iteration of event_loop_normal / event_loop_overlap after recv. Returns True if a batch was launched."""
        s = self.sched
        batch = s.get_next_batch_to_run()
        s.cur_batch = batch
        if not self.overlap:
            if batch: self._process(batch, s.run_batch(batch))
            else: s.self_check_during_idle()
            s.last_batch = batch
            return batch is not None
        disable = s.is_disable_overlap_for_batch(batch)
        if disable: self._process(*s.result_queue.popleft())
        if batch: s.result_queue.append((batch.copy(), s.run_batch(batch)))
        if s.last_batch:
            if not disable: self._process(*s.result_queue.popleft())
        elif batch is None:
            s.self_check_during_idle()
        s.last_batch = batch
        return batch is not None

    def busy(self):
        s = self.sched
        return (bool(s.waiting_queue) or not s.running_batch.is_empty() or s.chunked_req is not None or bool(s.result_queue)
                or (s.last_batch is not None and not s.last_batch.is_empty()))


def fold_vocab(reqs, vocab):
    """stitch_like_replay gives every program unique output ids >= 1e9; the real Req treats an out-of-vocab sampled
    token as a stop (check_finished). Fold them into [vocab/2, vocab) with a multiplicative hash: sequences stay
    program-unique in practice (a collision would need consecutive matches), and the radix tree only compares ids."""
    from dataclasses import replace
    lo = vocab // 2; span = vocab - lo
    f = lambda x: x if x < vocab else lo + (x * 2654435761) % span
    return [replace(r, prompt_ids=[f(x) for x in r.prompt_ids], output_ids=[f(x) for x in r.output_ids]) for r in reqs]


# ServerArgs fields that change scheduling / caching; everything else (model loading, kernels, ports) is irrelevant here
SCHED_FIELDS = ("context_length", "max_total_tokens", "chunked_prefill_size", "max_prefill_tokens", "max_running_requests", "schedule_policy",
                "schedule_conservativeness", "page_size", "radix_eviction_policy", "rr_recency_band_secs", "rr_adaptive_band_target_pctl",
                "rr_adaptive_band_min_secs", "rr_adaptive_band_max_secs", "rr_adaptive_band_fp_skip", "rr_adaptive_band_identity",
                "rr_liveness_source", "rr_chunk_tokens", "rr_band_stamp", "disable_overlap_schedule", "enable_mixed_chunk", "disable_radix_cache",
                "cuda_graph_max_bs")


def gpu_server_args(arm_dir):
    """The scheduling-relevant ServerArgs a GPU run was launched with, read from its server.log."""
    import gzip, re, ast
    path = next(os.path.join(arm_dir, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(arm_dir, x)))
    with (gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")) as fh:
        line = next(l for l in fh if "server_args=ServerArgs(" in l)
    out = {}
    for k in SCHED_FIELDS:
        m = re.search(r"[(, ]" + k + r"=('[^']*'|None|True|False|[-\d.e]+)[,)]", line)
        if m: out[k] = ast.literal_eval(m.group(1))
    return out


def emulate_client(reqs, stitch=True, frac_quantiles=None, seed=0, copy_quantiles=None):
    """What the GPU replay client + server turn the recorded stream into.
    Generation: the server's own output follows the recorded output for its first d tokens, then leaves it
      (program-unique ids after that). d/len is drawn from the distribution measured on a GPU run of the same workload
      and client (analysis/gen_divergence.py --json); without one, d = 0 (every output fully program-unique, the old
      des_sim.stitch_like_replay assumption). Identical recorded conversations (tau2 banking: 6 copies) keep sharing
      exactly as long as their generations stay on the recording.
    Next prompt, stitching client (gpu_replay_harbor default since 2026-09-20): own_history[:k] + recorded_next[k:],
      k = LCP(recorded prompt + recorded output, recorded next prompt) -- gpu_replay_harbor.stitch_prompt.
    Next prompt, unstitched client (before 2026-09-20): the recorded prompt verbatim, so the cache hits only up to
      prompt + d and the generated tail stays behind as a dead branch.
    Copies: programs whose recorded conversation is token-identical form a copy group. Their prompts are identical, so
      the server generates the same tokens for all of them (banking ba80: 98% of copy-turns equal the group's modal
      output): the group shares one generation per turn; with copy_quantiles (analysis/copy_divergence.py --json) each
      copy follows it for a drawn fraction and, once it leaves, continues on program-unique tokens for good."""
    from dataclasses import replace
    rng = np.random.default_rng(seed); q = None if frac_quantiles is None else np.asarray(frac_quantiles)
    nxt = [10 ** 9]
    def lcp(a, b):
        n = min(len(a), len(b)); i = 0
        while i + 256 <= n and a[i:i + 256] == b[i:i + 256]: i += 256
        while i < n and a[i] == b[i]: i += 1
        return i
    def gen(o):
        d = 0 if q is None else min(len(o), int(round(float(np.interp(rng.random(), np.linspace(0, 1, len(q)), q)) * len(o))))
        g = list(o[:d]) + list(range(nxt[0], nxt[0] + len(o) - d)); nxt[0] += len(o) - d; return g
    import hashlib
    cq = None if copy_quantiles is None else np.asarray(copy_quantiles)
    byp = defaultdict(list)
    for r in reqs: byp[r.program_id or r.rid].append(r)
    grp = {}
    for pid, lst in byp.items():
        lst.sort(key=lambda r: r.arrival); h = hashlib.sha1()
        for r in lst: h.update(np.asarray(r.prompt_ids, np.int64).tobytes()); h.update(b"|"); h.update(np.asarray(r.output_ids, np.int64).tobytes())
        grp[pid] = h.hexdigest()
    shared = {}                                             # (group, turn) -> the group's generation for that turn
    out = []
    for pid in sorted(byp):
        lst = byp[pid]; hist = None; prev = None; on_track = True
        for t, r in enumerate(lst):
            if prev is None or not stitch: prompt = list(r.prompt_ids)
            else:
                k = lcp(list(prev.prompt_ids) + list(prev.output_ids), r.prompt_ids)
                prompt = hist[:k] + list(r.prompt_ids[k:])
            if on_track:
                key = (grp[pid], t)
                if key not in shared: shared[key] = gen(r.output_ids)
                g = list(shared[key])
                if cq is not None:
                    f = float(np.interp(rng.random(), np.linspace(0, 1, len(cq)), cq))
                    if f < 1.0:                              # this copy leaves the group here, for good
                        c = int(f * len(g)); g = g[:c] + list(range(nxt[0], nxt[0] + len(g) - c)); nxt[0] += len(g) - c; on_track = False
            else:
                g = gen(r.output_ids)
            out.append(replace(r, prompt_ids=prompt, output_ids=g)); hist = prompt + g; prev = r
    return sorted(out, key=lambda r: r.arrival)


def gpu_client_stitched(arm_dir):
    """Did that GPU run's replay client stitch prompts? (the client prints it at start since 2026-09-20)"""
    for f in ("run.log", "replay.log"):
        p = os.path.join(arm_dir, f)
        if os.path.exists(p) and "prompt stitching: ON" in open(p, errors="replace").read(): return True
    return False


def arm_server_args(spec, kv, sched, context_length, overlap=True, chunk=8192, max_prefill=16384, override=None):
    if override:
        o = dict(override); cg = o.pop("cuda_graph_max_bs", None)
        o.update(skip_tokenizer_init=True, device="cpu", disable_cuda_graph=True)
        if not overlap: o["disable_overlap_schedule"] = True
        return ServerArgs(model_path=MODEL_PATH, **o)
    kw = dict(radix_eviction_policy="lru" if spec.policy == "lru" else "round_robin")
    if spec.policy == "round_robin":
        if spec.declared: kw["rr_liveness_source"] = "declared"
        if spec.chunk: kw["rr_chunk_tokens"] = spec.chunk
        if spec.band: kw["rr_recency_band_secs"] = spec.band; kw["rr_band_stamp"] = "last_access"   # as the GPU arms (and ArmSpec.build)
    return ServerArgs(model_path=MODEL_PATH, context_length=context_length, max_total_tokens=kv, chunked_prefill_size=chunk, max_prefill_tokens=max_prefill,
                      schedule_policy=sched, page_size=1, skip_tokenizer_init=True, disable_overlap_schedule=not overlap, device="cpu",
                      disable_cuda_graph=True, **kw)


def simulate(reqs, spec, kv, conc, seed, trial_gap, P, sched="fcfs", context_length=73728, overlap=True, occ_dt=5.0, progress=0.0, server_override=None):
    """Closed-loop replay: same workload construction as des_sim._simulate (trials -> programs -> turns with recorded
    tool gaps, next turn = finish + gap + client overhead, + server ingress), server = SGLang's Scheduler."""
    rhs = OR.rhs
    CLOCK.t = 0.0
    sa = arm_server_args(spec, kv, sched, context_length, overlap, override=server_override)
    sim = Sim(sa, P, pool_reqs=max(64, 2 * conc + 16))
    s = sim.sched; nr = sim.nr
    ovh_run = float(nr.standard_normal()) * P["ovh_run_sd"] if nr is not None else 0.0   # this run's client-overhead offset
    ftd_rng = np.random.default_rng(1000 + max(P["noise_seed"], 0))
    def overhead(t, prompt_len=0, out_len=0):
        base = max(0.0, P["turn_overhead"] + ovh_run + (P["ovh_tok"] * prompt_len + P["ovh_out"] * out_len) / 1e6)
        return base if nr is None else base * math.exp(P["ovh_sigma"] * nr.standard_normal())
    def ingress(t, prompt_len):
        return max(0.0, P["ingress_base"] + P["ingress_tok"] * prompt_len / 1e6)
    ftd_run = float(ftd_rng.standard_normal()) * P.get("ftd_run_sigma", 0.0) if P["noise_seed"] >= 0 else 0.0
    def ftd(t):
        # first-token response path; ftd_run_median (optional): per-run offset, this run's random effect sd ftd_run_sigma
        m = P["ftd_run_median"] * math.exp(ftd_run) if P.get("ftd_run_median") else P["ftd_median"]
        return m * math.exp(P["ftd_sigma"] * ftd_rng.standard_normal()) if m > 0 else 0.0

    trials = rhs._trials(reqs)
    if seed is not None: random.Random(seed).shuffle(trials)
    progs = {}; trial_progs = defaultdict(list)
    for tid, rs, base, dur in trials:
        byp = defaultdict(list)
        for r in rs: byp[r.program_id or r.rid].append(r)
        for pid, lst in byp.items():
            lst.sort(key=lambda r: r.arrival)
            turns = [(r, max(0.0, (r.arrival - lst[i - 1].finish) if i else (r.arrival - base))) for i, r in enumerate(lst)]
            if P.get("drops") and pid in P["drops"]: turns = turns[:P["drops"][pid]]
            progs[pid] = dict(turns=turns, k=0, trial=tid); trial_progs[tid].append(pid)
    trial_left = {tid: len(ps) for tid, ps in trial_progs.items()}; order = [tid for tid, *_ in trials]
    ev = []; seq = [0]; next_trial = [0]; meta = {}; records = []; occ = []
    def push(t, kind, payload): seq[0] += 1; heapq.heappush(ev, (t, seq[0], kind, payload))
    def start_trial(t):
        if next_trial[0] >= len(order): return
        tid = order[next_trial[0]]; next_trial[0] += 1
        for pid in trial_progs[tid]:
            r0, g0 = progs[pid]["turns"][0]; send = t + g0; push(send + ingress(send, len(r0.prompt_ids)), "arr", (pid, send))
    open_loop = P.get("arrivals") is not None
    if open_loop:
        # diagnostic open loop: every (program, turn) the GPU arm served arrives at its measured time, independent of the
        # simulated finish times (isolates scheduler/cache/cost mechanics from closed-loop timing drift)
        last_k = {}
        for pid, pg in progs.items():
            for k in range(len(pg["turns"])):
                if (pid, k) in P["arrivals"]:
                    ra = P["arrivals"][(pid, k)]; push(ra + ingress(ra, len(pg["turns"][k][0].prompt_ids)), "arr", (pid, ra, k)); last_k[pid] = k
    else:
        for _ in range(conc): start_trial(0.0)

    def on_finish(rid, t):
        pid, k, send = meta[rid]; pg = progs[pid]; rec = pg["turns"][k][0]
        ft = sim.first_token.get(rid, t); d = ftd(t)
        records.append(dict(program=pid, turn=k, arrival=send, first_sched=sim.first_sched[rid], ttft=ft - send + P["ttft_floor"] + d, ttft_raw=ft - send,
                            e2e=t - send + P["ttft_floor"], prompt_len=len(rec.prompt_ids), cached=sim.cached[rid], output_len=len(rec.output_ids),
                            chunks=0, retracts=0, tries=0))
        sim.pending.pop(rid, None)
        if open_loop:
            if spec.declared and k == last_k.get(pid): s.release_program(ReleaseProgramReqInput(program_ids=[pid]))
            return
        pg["k"] += 1
        if pg["k"] < len(pg["turns"]):
            nr_ = pg["turns"][pg["k"]][0]; snd = t + pg["turns"][pg["k"]][1] + overhead(t, len(nr_.prompt_ids), len(rec.output_ids))
            push(snd + ingress(snd, len(nr_.prompt_ids)), "arr", (pid, snd))
        else:
            if spec.declared: s.release_program(ReleaseProgramReqInput(program_ids=[pid]))
            tid = pg["trial"]; trial_left[tid] -= 1
            if trial_left[tid] == 0: push(t + trial_gap + overhead(t, 0, len(rec.output_ids)), "slot", tid)
    sim.on_finish = on_finish

    next_occ = 0.0; idle = 0; wall0 = VClock.real_perf(); next_prog = progress
    while True:
        while ev and ev[0][0] <= CLOCK.t:
            _, _, kind, payload = heapq.heappop(ev)
            if kind == "slot": start_trial(CLOCK.t); continue
            if len(payload) == 3: pid, send, k = payload
            else: pid, send = payload; k = progs[pid]["k"]
            pg = progs[pid]; rec = pg["turns"][k][0]; rid = f"{pid}#t{k}"
            meta[rid] = (pid, k, send)
            sim.submit(rid, rec.prompt_ids, rec.output_ids, program_id=rec.program_id or pid, program_state=rec.program_state, created_time=send)
        if CLOCK.t >= next_occ:
            used = s.max_total_num_tokens - s.token_to_kv_pool_allocator.available_size() - s.tree_cache.evictable_size()
            occ.append((CLOCK.t, used, s.tree_cache.evictable_size(), len(s.running_batch.reqs), len(s.waiting_queue),
                        int(s.running_batch.batch_is_full and bool(s.waiting_queue)), s.new_token_ratio))
            next_occ = CLOCK.t + occ_dt
        if progress and CLOCK.t >= next_prog:
            print(f"  t={CLOCK.t/60:6.1f} min  done={len(records)}  steps P/D={sim.nsteps[0]}/{sim.nsteps[1]}  real={VClock.real_perf()-wall0:.0f}s", flush=True)
            next_prog += progress
        if sim.busy():
            if sim.step(): idle = 0; continue
            idle += 1
            if idle < 3: continue
        if not ev:
            if sim.busy(): raise RuntimeError(f"stuck: waiting={len(s.waiting_queue)} running={len(s.running_batch.reqs)} avail={s.token_to_kv_pool_allocator.available_size()}")
            break
        CLOCK.t = max(CLOCK.t, ev[0][0]); idle = 0
    stats = dict(makespan=CLOCK.t, n_prefill=sim.nsteps[0], n_decode=sim.nsteps[1], prefill_time=sim.busy_time[0], decode_time=sim.busy_time[1],
                 retracted=getattr(s, "num_retracted_reqs", 0), real_s=VClock.real_perf() - wall0)
    return records, occ, stats


def main():
    import argparse, json, csv
    import des_sim as DS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream"); ap.add_argument("--kv", type=int, required=True); ap.add_argument("--conc", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", required=True); ap.add_argument("--sched", default="fcfs"); ap.add_argument("--context-length", type=int, default=73728)
    ap.add_argument("--no-overlap", action="store_true", help="event_loop_normal instead of event_loop_overlap (the GPU servers ran overlap)")
    ap.add_argument("--rec-conc", type=int, default=4); ap.add_argument("--rec-span", type=float, default=None); ap.add_argument("--trial-gap", type=float, default=None)
    ap.add_argument("--params-json", default=None); ap.add_argument("--noise-seed", type=int, default=-1); ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--drops-from", default=None, help="GPU arm dir: truncate conversations the GPU client dropped mid-run (as des_sim)")
    ap.add_argument("--diverge-json", default=None, help="generation leaves the recorded output at d/len drawn from this distribution "
                    "(analysis/gen_divergence.py --json, measured on a GPU run of the same workload and client type); default: fully program-unique outputs")
    ap.add_argument("--copy-json", default=None, help="copy groups (identical recorded conversations): per-turn fraction a copy follows its group's generation "
                    "(analysis/copy_divergence.py --json); default: copies never diverge from each other")
    ap.add_argument("--client-from", default=None, help="GPU arm dir: emulate the replay client that run used (stitching or not)")
    ap.add_argument("--ids-from", default=None, help="GPU arm dir (diagnostic): use the prompt/output ids that arm actually sent/generated, as des_sim --ids-from")
    ap.add_argument("--arrivals-from", default=None, help="GPU arm dir (diagnostic): open loop, every turn arrives at that arm's measured request_received time")
    ap.add_argument("--server-args-from", default=None, help="GPU arm dir: take the scheduling/caching ServerArgs of that run from its server.log (overrides --kv/--sched/--context-length/--arm flags)")
    ap.add_argument("--set", action="append", help="override a profile key, e.g. --set d0=9.0 (repeatable)")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--csv", action="store_true"); ap.add_argument("--progress", type=float, default=0.0)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    P = dict(DS.DEFAULTS)
    if a.params_json: P.update({k: v for k, v in json.load(open(a.params_json)).items() if not k.startswith("_")})
    P["noise_seed"] = a.noise_seed
    for kv in a.set or []:
        k, v = kv.split("="); P[k] = float(v)
    reqs = OR.load_stream(a.stream)
    if a.ids_from: reqs = DS.ids_from_gpu(reqs, a.ids_from)
    else:
        stitch = gpu_client_stitched(a.client_from) if a.client_from else not a.no_stitch
        fq = json.load(open(a.diverge_json))["frac_quantiles"] if a.diverge_json else None
        cq = json.load(open(a.copy_json))["frac_quantiles"] if a.copy_json else None
        reqs = emulate_client(reqs, stitch=stitch, frac_quantiles=fq, seed=max(a.noise_seed, 0), copy_quantiles=cq)
        print(f"client: {'stitching' if stitch else 'unstitched (recorded prompts verbatim)'}; generation divergence {'from ' + a.diverge_json if fq else 'd=0 (all outputs unique)'}", flush=True)
    reqs = fold_vocab(reqs, VOCAB)
    if a.arrivals_from:
        from gpu_windows import requests as _gpu_reqs
        _log = next(os.path.join(a.arrivals_from, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(a.arrivals_from, x)))
        _R = {k: v[0] for k, v in _gpu_reqs(_log).items()}; _t0 = min(_R.values())
        P["arrivals"] = {k: v - _t0 for k, v in _R.items()}; print(f"open loop: {len(P['arrivals'])} GPU arrival times", flush=True)
    if a.drops_from:
        from gpu_windows import requests as _gpu_reqs
    if a.drops_from:
        _log = next(os.path.join(a.drops_from, x) for x in ("server.log.gz", "server.log") if os.path.exists(os.path.join(a.drops_from, x)))
        _G = _gpu_reqs(_log); _kept = defaultdict(int); _fin = defaultdict(float)
        for (_p, _k), _v in _G.items(): _kept[_p] = max(_kept[_p], _k + 1); _fin[_p] = max(_fin[_p], _v[5] or _v[2])
        _end = max(_fin.values()); _byp = defaultdict(list)
        for r in reqs: _byp[r.program_id].append(r)
        for _l in _byp.values(): _l.sort(key=lambda r: r.arrival)
        def _next_gap(_p):
            _l = _byp[_p]; _k = _kept[_p]
            return max(0.0, _l[_k].arrival - _l[_k - 1].finish) if 0 < _k < len(_l) else 0.0
        P["drops"] = {_p: _kept[_p] for _p in _byp if 0 < _kept[_p] < len(_byp[_p]) and _fin[_p] + _next_gap(_p) + 60.0 < _end}
        print(f"drops: {len(P['drops'])} conversations dropped mid-run by the GPU client", flush=True)
    override = None
    if a.server_args_from:
        override = gpu_server_args(a.server_args_from)
        a.kv = override.get("max_total_tokens", a.kv); a.sched = override.get("schedule_policy", a.sched); a.context_length = override.get("context_length", a.context_length)
        if "cuda_graph_max_bs" in override: P["cg_max_bs"] = override["cuda_graph_max_bs"]
        print("server args from GPU run:", {k: v for k, v in override.items() if v is not None}, flush=True)
    trial_gap = a.trial_gap if a.trial_gap is not None else OR.rhs.calibrate_trial_gap(reqs, a.rec_conc, a.rec_span)
    print(f"stream {len(reqs)} reqs, trial_gap {trial_gap:.0f}s, conc {a.conc} seed {a.seed} kv {a.kv} sched {a.sched} arm {a.arm} overlap {not a.no_overlap}", flush=True)
    rec, occ, st = simulate(reqs, OR.ArmSpec.parse(a.arm), a.kv, a.conc, a.seed, trial_gap, P, a.sched, a.context_length, overlap=not a.no_overlap,
                            progress=a.progress, server_override=override)
    sm, wins = DS.summarize(rec, occ, st, a.kv)
    print(f"[{a.arm}] real {st['real_s']:.0f}s makespan {sm['makespan_min']:.1f} min n={sm['n']} | miss {sm['miss']:.2f}% U p90/p99/TM99 {sm['U_p90']:.0f}/{sm['U_p99']:.0f}/{sm['U_TM99']:.0f} n>16k {sm['n16k']} | "
          f"sTTFT p50/p90/p99/TM99 {sm['sTTFT_p50']:.3f}/{sm['sTTFT_p90']:.3f}/{sm['sTTFT_p99']:.2f}/{sm['sTTFT_TM99']:.2f}s queue p99 {sm['queue_p99']:.2f}s dq p99 {sm['dequeued_p99']:.2f} | "
          f"in-flight max/p90 {sm['infl_max']/1e3:.0f}/{sm['infl_p90']/1e3:.0f}K | decode {sm['decode_step_ms']:.1f}ms prefill share {sm['prefill_share']:.0f}% steps P/D {st['n_prefill']}/{st['n_decode']} retracted {st['retracted']}", flush=True)
    for w in wins: print(f"    {w['w']:3d}-{w['w']+5:3d} n={w['n']:3d} miss {w['miss']:5.2f}% U p99 {w['U_p99']:6.0f} sTTFT p99 {w['sTTFT_p99']:5.2f}s q99 {w['queue_p99']:5.2f}s | in-flight p50/max {w['infl_p50']/1e3:4.0f}/{w['infl_max']/1e3:4.0f}K running {w['running_p50']:.0f} waiting max {w['waiting_max']}")
    json.dump(dict(arm=a.arm, kv=a.kv, conc=a.conc, seed=a.seed, trial_gap=trial_gap, summary=sm, windows=wins, stats=st), open(os.path.join(a.out_dir, f"summary_{a.arm}.json"), "w"), indent=1, default=float)
    if a.csv:
        with open(os.path.join(a.out_dir, f"req_{a.arm}.csv"), "w") as f:
            wr = csv.DictWriter(f, fieldnames=list(rec[0].keys())); wr.writeheader(); wr.writerows(rec)


if __name__ == "__main__":
    main()
