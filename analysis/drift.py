import sys, pandas as pd, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "sim"))
from sim_vs_gpu_pair import load_gpu
sim,gpu=sys.argv[1],sys.argv[2]
sr=pd.read_csv(sim); G=load_gpu(gpu); t0=min(v["recv"] for v in G.values()); s0=sr.arrival.min()
sr["key"]=list(zip(sr.program,sr.turn)); sr=sr[sr.key.isin(G.keys())].copy()
sr["t_gpu"]=[(G[k]["recv"]-t0) for k in sr.key]; sr["t_sim"]=sr.arrival-s0; sr["e_gpu"]=[G[k]["e2e"] for k in sr.key]; sr["m"]=(sr.t_gpu//300*5).astype(int)
sr["u_sim"]=sr.prompt_len-sr.cached; sr["u_gpu"]=[G[k]["unc"] for k in sr.key]
for m,g in sr.groupby("m"):
    dr=(g.e2e-g.ttft)/g.output_len*1e3; gd=np.array([(G[k]["e2e"]-G[k]["srv"]) for k in g.key])/g.output_len.values*1e3
    print(f"  {m:3d}-{m+5:<3d} n={len(g):4d} drift {np.median(g.t_sim-g.t_gpu):+6.1f}s e2e ratio {np.median(g.e2e/g.e_gpu):.2f} ms/tok sim {np.median(dr):5.1f} gpu {np.median(gd):5.1f} | miss sim {100*g.u_sim.sum()/g.prompt_len.sum():5.2f} gpu {100*g.u_gpu.sum()/g.prompt_len.sum():5.2f}")
