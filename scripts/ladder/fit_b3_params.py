#!/usr/bin/env python3
"""Give baseline B3 (sim/closed_loop_sim.py, the naive closed loop) the SAME machine calibration as des_sim.

closed_loop_sim prices steps with a narrower functional form:
  prefill step = step_ms + new_tokens / prefill_tps + sum(ctx) * attn_us     (no open-chunk term)
  decode step  = dec_ms0 + dec_ms_per_ktok * K_running + dec_ms_per_req * bs (no longest-sequence term, no eager cliff)
We project the machine profile onto that form: run des_sim (noise-free, profile-driven) at one point, take its step log,
and least-squares fit B3's parameters to the step durations the profile produced. The output JSON holds the
closed_loop_sim CLI values.

usage: fit_b3_params.py DES_STEPS.csv --out b3_local.json
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [l.rstrip("\n").split(",") for l in open(a.steps)][1:]
    P = np.array([[float(x) for x in r[2:]] for r in rows if r[0] == "P"])     # n,new,cached,running,waiting,dur,inflight,full
    D = np.array([[float(x) for x in r[2:]] for r in rows if r[0] == "D"])     # n=bs,k,0,running,waiting,dur,inflight,full
    # prefill: dur(s) = a + b*new + c*(cached+new)   [ctx of each item ~ its filled length]
    Xp = np.c_[np.ones(len(P)), P[:, 1], P[:, 2] + P[:, 1]]
    cp, *_ = np.linalg.lstsq(Xp, P[:, 5], rcond=None)
    # decode: per-step ms = a + b*Kinflight + c*bs, weighted by the k steps in each macro step
    k = D[:, 1]; per = D[:, 5] / k * 1e3; w = np.sqrt(k)
    Xd = np.c_[np.ones(len(D)), D[:, 6] / 1e3, D[:, 0]]
    cd, *_ = np.linalg.lstsq(Xd * w[:, None], per * w, rcond=None)
    rp = P[:, 5] - Xp @ cp; rd = per - Xd @ cd
    out = dict(step_ms=1e3 * cp[0], prefill_tps=1.0 / cp[1], attn_us=1e6 * cp[2], dec_ms0=cd[0], dec_ms_per_ktok=cd[1], dec_ms_per_req=cd[2],
               _fit=dict(n_prefill=len(P), n_decode_macro=len(D), prefill_resid_ms_mad=float(np.median(np.abs(rp)) * 1e3),
                         decode_resid_ms_mad=float(np.median(np.abs(rd))), source=a.steps))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
