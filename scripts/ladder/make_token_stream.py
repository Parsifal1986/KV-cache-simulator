#!/usr/bin/env python3
"""Save the token ids a GPU arm actually sent/generated (des_sim.ids_from_gpu: stitched prompts + the server's own
generations; programs cut where the GPU client dropped them) as a stream pickle in the recorded stream's format
(list of rhs.Req), so every ladder method -- B1 offline_replay, B2, B3 closed_loop_sim, des_sim, sgl_sim -- replays the
SAME token input and only the timing / scheduling model differs.

usage: make_token_stream.py RECORDED_STREAM.pkl GPU_ARM_DIR OUT.pkl   (run from DATA_ROOT)"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, pickle
from dataclasses import replace
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
import des_sim as DS

rec, arm, out = sys.argv[1:4]
reqs = DS.ids_from_gpu(DS.OR.load_stream(rec), arm)
# token counts follow the new ids (check_not_clipped compares them)
reqs = [replace(r, prompt_tokens=len(r.prompt_ids), completion_tokens=len(r.output_ids)) for r in reqs]
with open(out, "wb") as f: pickle.dump(reqs, f)
print(f"wrote {out}: {len(reqs)} turns from {arm}")
