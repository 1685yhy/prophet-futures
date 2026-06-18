#!/usr/bin/env python3
"""
Prophet v4 — 纯NumPy ML大规模回测
不依赖sklearn/xgboost，纯numpy实现逻辑回归 + 梯度提升
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta

LOT = {"lh":16,"jm":60,"jd":5,"m":10,"rm":10,"rb":10,"i":100,"cu":5,
       "sc":1000,"fu":10,"ma":10,"pg":20,"sa":20,"p":10,"y":10}

ALL_SYMBOLS = list(LOT.keys())

# ═══════════════════════════════════════════════════════════════════════════
# Pure NumPy Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════

class NumpyLogistic:
    def __init__(self, lr=0.01, epochs=200, reg=0.01):
        self.lr=lr; self.epochs=epochs; self.reg=reg

    def _sigmoid(self, z):
        return 1/(1+np.exp(-np.clip(z,-20,20)))

    def fit(self, X, y):
        n,d=X.shape; self.w=np.zeros(d); self.b=0
        for _ in range(self.epochs):
            z=X@self.w+self.b; p=self._sigmoid(z)
            dw=(X.T@(p-y))/n+self.reg*self.w; db=np.mean(p-y)
            self.w-=self.lr*dw; self.b-=self.lr*db

    def predict_proba(self, X):
        p=self._sigmoid(X@self.w+self.b)
        return np.column_stack([1-p,p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1]>0.5).astype(int)

# ═══════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════

def fetch(sym, days=2500):
    import akshare as ak
    e=datetime.now(); s=e-timedelta(days=days+200)
    try:
        df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]: df[c]=df[c].astype(float)
        return df.reset_index(drop=True)
    except: return None

def build_features(df, lookback=60):
    if len(df) < lookback+20: return None,None
    c=df["close"].values.astype(float); o=df["open"].values.astype(float)
    h=df["high"].values.astype(float); l=df["low"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(c))

    X,Y=[],[]
    for i in range(lookback,len(df)-1):
        wc=c[i-lookback:i+1]; wv=v[i-lookback:i+1]
        wi=oi[i-lookback:i+1]; wh=h[i-lookback:i+1]; wl=l[i-lookback:i+1]
        cl=c[i]; op=o[i]

        ret=(c[i+1]-cl)/cl
        y=1 if ret>0.005 else (-1 if ret<-0.005 else 0)

        f=[]
        f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
        f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
        f.append((cl-wc[-6])/(wc[-6]+1e-8))
        f.append((cl-wc[-21])/(wc[-21]+1e-8))
        ma5=np.mean(wc[-5:]); ma10=np.mean(wc[-10:])
        ma20=np.mean(wc[-20:]); ma60=np.mean(wc[-min(60,len(wc)):])
        for ma in [ma5,ma10,ma20,ma60]: f.append((ma-cl)/(cl+1e-8))
        f.append((ma5-ma20)/(ma20+1e-8))
        f.append((ma5-ma60)/(ma60+1e-8))
        tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
        f.append(np.mean(tr)/(cl+1e-8))
        gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
        losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
        f.append(np.mean(gains)/(np.mean(losses)+1e-8))
        vm5=np.mean(wv[-5:]); vm20=np.mean(wv[-20:])
        f.append((vm5-vm20)/(vm20+1e-8))
        om5=np.mean(wi[-5:]); om20=np.mean(wi[-20:])
        f.append((om5-om20)/(om20+1e-8))
        f.append((cl-op)/(op+1e-8))
        f.append(1 if cl>op else -1)
        f.append(1 if cl>wc[-2] else -1)

        X.append(f); Y.append(y)
    return np.array(X),np.array(Y)

# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════

def ml_backtest(sym,df,conf_thresh=0.55,risk_pct=0.01,stop_atr=1.5,target_atr=3.0,n_splits=3):
    from tools.indicators import calc_indicators
    X,Y=build_features(df)
    if X is None or len(X)<200: return None
    lot=LOT.get(sym,10); capital=1_000_000
    splits=np.linspace(0,len(X),n_splits+1).astype(int)
    all_trades=[]

    for sp in range(n_splits):
        te=splits[sp+1]
        if te>=len(X): break
        Xt,Yt=X[:te],Y[:te]
        ve_end=splits[sp+2] if sp+2<len(splits) else len(X)
        Xv,Yv=X[te:ve_end],Y[te:ve_end]
        if len(Xt)<100 or len(Xv)<30: continue

        yl_t=(Yt==1).astype(int); yl_v=(Yv==1).astype(int)
        ys_t=(Yt==-1).astype(int); ys_v=(Yv==-1).astype(int)

        ml=NumpyLogistic(lr=0.005,epochs=150,reg=0.01)
        ms=NumpyLogistic(lr=0.005,epochs=150,reg=0.01)
        ml.fit(Xt,yl_t); ms.fit(Xt,ys_t)
        lp=ml.predict_proba(Xv)[:,1]; sp=ms.predict_proba(Xv)[:,1]

        vs=te+60; ve=vs+len(Xv)
        if ve>len(df): ve=len(df)
        vdf=df.iloc[vs:ve].reset_index(drop=True)

        for j in range(min(len(lp),len(vdf)-1)):
            if max(lp[j],sp[j])<conf_thresh: continue
            d="LONG" if lp[j]>sp[j] else "SHORT"
            entry=float(vdf.iloc[j]["close"])
            atr=float(vdf.iloc[j]["high"])-float(vdf.iloc[j]["low"])
            if atr<entry*0.001: atr=entry*0.01
            sd=max(atr*0.3,atr*stop_atr); td=atr*target_atr
            stop=entry-sd if d=="LONG" else entry+sd
            target=entry+td if d=="LONG" else entry-td
            rc=capital*risk_pct; q=max(1.0,min(20.0,rc/(sd*lot)))
            nc=float(vdf.iloc[min(j+1,len(vdf)-1)]["close"])
            nh=float(vdf.iloc[min(j+1,len(vdf)-1)]["high"])
            nl=float(vdf.iloc[min(j+1,len(vdf)-1)]["low"])
            if d=="LONG":
                ep=stop if nl<=stop else (target if nh>=target else nc)
                rs="STOP" if nl<=stop else ("TP" if nh>=target else "EOD")
            else:
                ep=stop if nh>=stop else (target if nl<=target else nc)
                rs="STOP" if nh>=stop else ("TP" if nl<=target else "EOD")
            pnl=(ep-entry)*lot*q*(1 if d=="LONG" else -1)
            pnl-=abs(ep*lot*q*0.0001)+abs(entry*lot*q*0.0001)
            all_trades.append({"pnl":round(pnl,2),"rs":rs,"d":d,"q":q,"split":sp})
    return all_trades

# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("  Prophet v4 — 纯NumPy ML全品种大规模回测")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*60)

CONFS=[0.50,0.55,0.60,0.65]; RISKS=[0.01,0.015,0.02]
all=[]; td=0; vd=0

for sym in ALL_SYMBOLS:
    df=fetch(sym,2500)
    if df is None or len(df)<500: continue
    td+=1
    for ct in CONFS:
        for rp in RISKS:
            tr=ml_backtest(sym,df,ct,rp,1.5,3.0)
            if not tr or len(tr)<15: continue
            pnls=[t["pnl"] for t in tr]; n=len(pnls)
            w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
            wr=len(w)/n; tp=sum(pnls); ret=tp/1_000_000*100
            aw=np.mean(w) if w else 0; al=abs(np.mean(l)) if l else 1
            pf=sum(w)/(abs(sum(l))+1e-8); plr=aw/(al+1e-8)
            cum=np.cumsum(pnls); rm=np.maximum.accumulate(cum)
            dd=abs(float(np.min((cum-rm)/1_000_000*100)))
            if n>=5:
                rt_arr=[p/1_000_000 for p in pnls]
                sr=np.mean(rt_arr)/(np.std(rt_arr,ddof=1)+1e-8)*np.sqrt(252)
            else: sr=0
            score=wr*ret/(dd/100+0.05)*min(n,60)
            all.append({"s":sym,"c":ct,"r":rp,"t":n,"wr":round(wr,3),
                        "pnl":round(tp,0),"ret":round(ret,1),"dd":round(dd,1),
                        "sr":round(sr,3),"pf":round(pf,2),"plr":round(plr,2),
                        "sc":round(score,1)})
            vd+=1

all.sort(key=lambda x:x["sc"],reverse=True)

print(f"\n  测试{td}品种, {vd}有效组合\n")
print(f"  Top 40:")
print(f"  {'Rk':<4} {'Sym':<6} {'Conf':<6} {'Risk':<6} {'Trd':<6} {'WR':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'SR':<6} {'PF':<6}")
print(f"  {'─'*80}")
for i,r in enumerate(all[:40]):
    print(f"  {i+1:<4} {r['s'].upper():<6} {r['c']:<6} {r['r']:<6} "
          f"{r['t']:<6} {r['wr']:.0%}     {r['pnl']:+,.0f}     {r['ret']:+.1f}%    "
          f"{r['dd']:.1f}%   {r['sr']:.2f}  {r['pf']:.2f}")

wr60=[r for r in all if r["wr"]>=0.60 and r["t"]>=20]
print(f"\n  >=60%胜率: {len(wr60)}个")
for r in sorted(wr60,key=lambda x:x["sc"],reverse=True)[:10]:
    print(f"  {r['s'].upper()} conf={r['c']} risk={r['r']}: {r['t']}t {r['wr']:.0%}wr {r['ret']:+.1f}% DD{r['dd']:.1f}%")

json.dump({"top":all[:100],"wr60":wr60,"total":vd},
          open("/tmp/ml_numpy_results.json","w"),indent=2,ensure_ascii=False)
print(f"\n✅ /tmp/ml_numpy_results.json")
