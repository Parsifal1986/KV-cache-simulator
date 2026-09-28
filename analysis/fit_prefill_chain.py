import pandas as pd, numpy as np, os, sys
S=os.path.dirname(os.path.abspath(__file__))
def chain_rows(k):
    P=pd.read_csv(f"{S}/steps/{k}/steps_prefill.csv").to_dict("records"); rows=[]
    ctx=None   # prefix length of the open chunked request, if known
    prev=None
    for s in P:
        if prev is not None and prev["new_tok"]==8192 and ctx is not None and s["new_seq"]==1 and s["thr"]>0:
            pre=ctx; new=s["new_tok"]; gap=prev["new_tok"]/s["thr"]
            rows.append(dict(arm=k,sec=s["sec"],prefix=pre,new=new,gap=gap,running=s["running"],cont=1))
        # update chain state for next line
        if s["new_tok"]==8192 and s["new_seq"]==1:
            if prev is not None and prev["new_tok"]==8192 and ctx is not None: ctx=ctx+8192   # continuation chunk
            else: ctx=s["cached"]+8192                                                          # first chunk of a new chunked req
        else: ctx=None
        prev=s
    return rows
if __name__=='__main__':
    exec(compile('arms=sys.argv[1]'+'.split(",")\nR=pd.DataFrame([r for k in arms for r in chain_rows(k)]); R=R[(R.sec>180)&(R.gap<1.5)]\nR["attn"]=R.new*(R.prefix+R.new/2)/1e6  # M token^2\nX=np.c_[np.ones(len(R)),R.new/1e3,R.attn/1e3]; y=R.gap.values*1e3\nb,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b\nprint(f"{arms[0]}.. n={len(R)} (last chunks {int((R.new<8192).sum())})  gap_ms = {b[0]:.1f} + {b[1]:.2f}*Knew + {b[2]:.3f}*G(new*(prefix+new/2))/1e9  R2 {1-r.var()/y.var():.3f} rmse {np.sqrt((r**2).mean()):.0f}ms  prefix range {R.prefix.min()}-{R.prefix.max()}")\nfor lo,hi in ((0,16e3),(16e3,32e3),(32e3,48e3),(48e3,80e3)):\n    m=(R.prefix>=lo)&(R.prefix<hi)&(R.new==8192)\n    if m.sum(): print(f"   prefix {lo/1e3:.0f}-{hi/1e3:.0f}K full chunks n={m.sum()} gap p50 {np.median(R.gap[m])*1e3:.0f} ms  model {np.median((X@b)[m.values]):.0f}")\nF=R[R.new==8192]; b2=np.polyfit(F.prefix/1e3,F.gap*1e3,1); print(f"   full chunks only: gap_ms = {b2[1]:.0f} + {b2[0]:.2f}*Kprefix ; residual MAD {np.median(np.abs(F.gap*1e3-np.polyval(b2,F.prefix/1e3))):.0f} ms")\nL=R[R.new<8192]; X=np.c_[np.ones(len(L)),L.new/1e3,L.attn/1e3]; b3,*_=np.linalg.lstsq(X,L.gap.values*1e3,rcond=None); r=L.gap.values*1e3-X@b3\nprint(f"   last chunks only n={len(L)}: gap_ms = {b3[0]:.0f} + {b3[1]:.2f}*Knew + {b3[2]:.1f}*attnG ; R2 {1-r.var()/(L.gap*1e3).var():.2f}")\n# per-step fixed overhead: batches with tiny new tokens and no chain, from all pure rows\n','x','exec'))
