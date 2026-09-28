import pandas as pd, numpy as np, sys, glob, os
S=os.path.dirname(os.path.abspath(__file__))
def ols(X,y):
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; return b, 1-r.var()/y.var(), np.sqrt((r**2).mean())
for grp in [["c38cA","c38cB","c42cA","c42cB","c44cA","c44cB","c46cA","c46cB"],["c46lpmA","c46lpmB"]]:
    P=pd.concat([pd.read_csv(f"{S}/steps/{k}/steps_prefill.csv").assign(arm=k) for k in grp]); D=pd.concat([pd.read_csv(f"{S}/steps/{k}/steps_decode.csv").assign(arm=k) for k in grp])
    P=P[P.sec>180]; D=D[D.sec>180]
    p=P[P.pure==True].copy()
    # step duration of THIS line's batch (completion-to-completion) vs previous batch composition: test both
    print("=== group",grp[0][:3], "pure prefill n=",len(p))
    for lab,newt,seq,cac in (("this",p.new_tok,p.new_seq,p.cached),("prev",p.prev_new_tok,p.prev_new_seq,p.prev_cached)):
        X=np.c_[np.ones(len(p)),newt/1e3,seq,cac/1e3]; b,r2,rm=ols(X,p.gap.values*1e3)
        print(f"  gap(ms) ~ batch[{lab}]: c0 {b[0]:.1f} + {b[1]:.2f}/Knew + {b[2]:.1f}/seq + {b[3]:.3f}/Kcached  R2 {r2:.3f} rmse {rm:.0f}ms")
    print("  gap ms pctl", np.percentile(p.gap*1e3,[10,50,90,99]).round(0), " new_tok", np.percentile(p.new_tok,[10,50,90]))
    d=D[(D.n_prefill_between==0)&D.step_ms.notna()].copy()
    for cg in (True,False):
        dd=d[d.cg==cg]
        if len(dd)<10: continue
        X=np.c_[np.ones(len(dd)),dd.tokens/1e3,dd.running]; b,r2,rm=ols(X,dd.step_ms.values)
        print(f"  decode cg={cg} n={len(dd)} step_ms ~ {b[0]:.1f} + {b[1]:.4f}/Ktok + {b[2]:.3f}/req  R2 {r2:.3f} rmse {rm:.1f}  running range {dd.running.min()}-{dd.running.max()} tok {dd.tokens.min()//1000}-{dd.tokens.max()//1000}K")
