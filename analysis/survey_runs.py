import sys, gzip, json, re, os, datetime, glob
rows = []
for L in sorted(glob.glob("results/**/server.log.gz", recursive=True)):
    try:
        args = {}; boot = None; first = None; nfin = 0
        with gzip.open(L, "rt", errors="replace") as fh:
            for i, line in enumerate(fh):
                if "server_args=ServerArgs(" in line and not args:
                    for k in ("model_path", "max_total_tokens", "tp_size", "cuda_graph_max_bs", "attention_backend", "log_requests_level", "schedule_policy", "chunked_prefill_size", "disaggregation_mode"):
                        m = re.search(k + r"=('?[^,)]*'?)", line); args[k] = m.group(1).strip("'") if m else "?"
                if boot is None and '"request.finished"' in line and '"inference_time"' in line:
                    m = json.loads(line)["out"]["meta_info"]
                    if m.get("inference_time"): boot = m["request_finished_ts"] - m["inference_time"]; first = m["request_finished_ts"]
                if boot is not None and args: break
                if i > 20000: break
        if not args or args.get("disaggregation_mode") not in ("null", "?"): continue
        rows.append((datetime.datetime.fromtimestamp(first).strftime("%m-%d %H:%M") if first else "?", f"{boot:.0f}" if boot else "?", os.path.basename(args["model_path"])[:22], args["max_total_tokens"], args["tp_size"], args["log_requests_level"], args["schedule_policy"], args["attention_backend"], args["cuda_graph_max_bs"], os.path.dirname(L)))
    except Exception as e:
        print("ERR", L, e, file=sys.stderr)
for r in sorted(rows): print(" | ".join(map(str, r)))
