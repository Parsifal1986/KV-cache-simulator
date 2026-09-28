import pandas as pd, numpy as np, os, sys
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,S)
from fit_prefill_chain import chain_rows
arms=sys.argv[1].split(",")
# ---- prefill: single-seq chunk chains, with an indicator for "this batch leaves a chunked request open"
R=pd.DataFrame([r for k in arms for r in chain_rows(k)]); R=R[(R.sec>180)&(R.gap<1.5)]
attn=R.new*(R.prefix+R.new/2)/1e9; full=(R.new==8192).astype(float)
X=np.c_[np.ones(len(R)),R.new/1e3,attn,full]; y=R.gap.values*1e3
bp,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@bp
print(f"PREFILL n={len(R)}: ms = {bp[0]:.1f} + {bp[1]:.2f}*Knew + {bp[2]:.1f}*[G tok^2 attn] + {bp[3]:.1f}*[chunk open]  R2 {1-r.var()/y.var():.3f} MAD {np.median(np.abs(r)):.0f}ms")
for lo,hi in ((0,16e3),(16e3,32e3),(32e3,48e3),(48e3,80e3)):
    for f in (1,0):
        m=((R.prefix>=lo)&(R.prefix<hi)&(full==f)).values
        if m.sum()>5: print(f"   prefix {lo/1e3:.0f}-{hi/1e3:.0f}K {'full' if f else 'last'} n={m.sum()} obs p50 {np.median(y[m]):.0f} model p50 {np.median((X@bp)[m]):.0f}")
def pre_ms(new,cached,seq):
    # multi-seq approximation: every seq gets mean prefix; one open chunk if new==8192
    per=new/np.maximum(seq,1); a=seq*per*(cached/np.maximum(seq,1)+per/2)/1e9
    return bp[0]+bp[1]*new/1e3+bp[2]*a+bp[3]*(new==8192)
# ---- decode: intervals between consecutive decode lines; subtract modelled prefill steps in between
rows=[]
for k in arms:
    st=pd.read_csv(f"{S}/steps/{k}/steps_prefill.csv"); dc=pd.read_csv(f"{S}/steps/{k}/steps_decode.csv")
    # rebuild order: merge by sec is ambiguous inside a second; use n_prefill_between counts instead
    P=st.to_dict("records"); D=dc.to_dict("records"); ip=0
    # walk decode lines; the prefill lines between decode j-1 and j are the next n_prefill_between prefill lines
    for d in D:
        n=d["n_prefill_between"]
        if pd.isna(n): continue
        n=int(n); seg=P[ip:ip+n]; ip+=n
        if d["sec"]<180 or pd.isna(d["gap"]) or d["gap"]>20: continue
        pm=sum(pre_ms(s["new_tok"],s["cached"],s["new_seq"]) for s in seg)
        rows.append(dict(arm=k,gap_ms=d["gap"]*1e3,pre_ms=pm,npre=n,tokens=0.5*(d["tokens"]+(d["tokens"])),running=0.5*(d["running"]+d["prev_running"]),cg=d["cg"]))
Dd=pd.DataFrame(rows); Dd["dec_ms"]=(Dd.gap_ms-Dd.pre_ms)/40
for cg in (True,False):
    for pure in (True,False):
        m=(Dd.cg==cg)&((Dd.npre==0)==pure)
        if m.sum()<20: continue
        d=Dd[m]; X=np.c_[np.ones(len(d)),d.tokens/1e3,d.running]; b,*_=np.linalg.lstsq(X,d.dec_ms.values,rcond=None); r=d.dec_ms.values-X@b
        print(f"DECODE cg={cg} {'pure' if pure else 'w/prefill'} n={len(d)}: step ms = {b[0]:.1f} + {b[1]:.4f}*Ktok + {b[2]:.3f}*running  R2 {1-r.var()/d.dec_ms.var():.2f} MAD {np.median(np.abs(r)):.1f}  run {d.running.min():.0f}-{d.running.max():.0f} tok {d.tokens.min()/1e3:.0f}-{d.tokens.max()/1e3:.0f}K")
