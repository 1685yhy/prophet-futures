#!/usr/bin/env python3
"""
Prophet v4.1 — 规则+ML混合策略
规则生成候选信号 → ML确认过滤 → 只做双重确认的交易
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

LOT = {"lh":16,"jm":60,"jd":5,"m":10,"rm":10,"rb":10,"i":100,"cu":5,
       "sc":1000,"fu":10,"ma":10,"sa":20,"p":10,"y":10}

# Pure NumPy Logistic Regression (from ml_numpy.py)
class NumpyLogistic:
    def __init__(self,lr=0.005,epochs=150,reg=0.01):
        self.lr=lr;self.epochs=epochs;self.reg=reg
    def _sigmoid(self,z): return 1/(1+np.exp(-np.clip(z,-20,20)))
    def fit(self,X,y):
        n,d=X.shape;self.w=np.zeros(d);self.b=0
        for _ in range(self.epochs):
            z=X@self.w+self.b;p=self._sigmoid(z)
            self.w-=self.lr*(X.T@(p-y)/n+self.reg*self.w)
            self.b-=self.lr*np.mean(p-y)
    def predict_proba(self,X):
        p=self._sigmoid(X@self.w+self.b);return np.column_stack([1-p,p])

# ═══════════════════════════════════════════════════════════════════════════

def fetch(sym,days=2500):
    import akshare as ak
    e=datetime.now();s=e-timedelta(days=days+200)
    try:
        df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]:df[c]=df[c].astype(float)
        return df.reset_index(drop=True)
    except:return None

def build_ml_features(df,i):
    """Build feature vector at index i."""
    c=df["close"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(c))
    L=60;start=max(0,i-L);wc=c[start:i+1];wv=v[start:i+1];wi=oi[start:i+1]
    if len(wc)<L:return None
    cl=c[i];op=df["open"].values.astype(float)[i]
    f=[]
    f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
    f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
    f.append((cl-wc[-6])/(wc[-6]+1e-8))
    f.append((cl-wc[-21])/(wc[-21]+1e-8) if len(wc)>=21 else 0)
    for ma_len in [5,10,20,min(60,len(wc))]:
        ma=np.mean(wc[-ma_len:])
        f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]);ma20=np.mean(wc[-20:])
    ma60=np.mean(wc[-min(60,len(wc)):])
    f.append((ma5-ma20)/(ma20+1e-8))
    f.append((ma5-ma60)/(ma60+1e-8))
    f.append(np.mean([max(abs(wc[j]-wc[j-1]),0) for j in range(-14,0)])/(cl+1e-8))
    gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
    losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
    f.append(np.mean(gains)/(np.mean(losses)+1e-8))
    vm5=np.mean(wv[-5:]);vm20=np.mean(wv[-20:])
    f.append((vm5-vm20)/(vm20+1e-8))
    om5=np.mean(wi[-5:]);om20=np.mean(wi[-20:])
    f.append((om5-om20)/(om20+1e-8))
    f.append((cl-op)/(op+1e-8))
    return np.array(f)

def rule_signal(df_w,ind,mc=7):
    """Rule-based signal (same as before)."""
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"] not in ("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c)
    _,_,h1=_calc_macd(c[:-1]) if len(c)>1 else (0,0,0)
    _,_,h2=_calc_macd(c[:-2]) if len(c)>2 else (0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0)
    mh=ind.get("macd_hist",0)
    oic="oi" if "oi" in df_w.columns else None
    oi_=df_w[oic].values.astype(float) if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4]) if len(oi_)>=4 else 0
    o5=float(oi_[-1]-oi_[-6]) if len(oi_)>=6 else o3
    ot="ACCUMULATING" if (o3>0 and o5>0) else ("REDUCING" if (o3<0 and o5<0) else "FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return "SHORT"
    if lc>=mc:return "LONG"
    return None

def train_ml_filter(df):
    """Train ML to predict if a rule signal will be profitable."""
    features=[];labels=[]
    W=60
    for i in range(W,len(df)-1):
        window=df.iloc[i-W:i+1]
        ind=calc_indicators(window)
        sg=rule_signal(window,ind,7)
        if sg is None:continue
        f=build_ml_features(df,i)
        if f is None:continue
        # Label: 1 if next day was profitable in signal direction
        nc=float(df.iloc[i+1]["close"])
        c=float(df.iloc[i]["close"])
        ret=(nc-c)/c
        label=1 if (sg=="LONG" and ret>0) or (sg=="SHORT" and ret<0) else 0
        features.append(f);labels.append(label)
    if len(features)<50:return None,None
    X=np.array(features);y=np.array(labels)
    split=int(len(X)*0.7)
    model=NumpyLogistic(lr=0.005,epochs=200,reg=0.01)
    model.fit(X[:split],y[:split])
    pred=model.predict_proba(X[split:])[:,1]>0.5
    acc=(pred==y[split:]).mean()
    return model,acc

def hybrid_backtest(df,sym,risk_pct=0.01,stop_atr=1.5,target_atr=3.0):
    """Rule generates signal, ML confirms. Returns (stats_dict, trades_list)."""
    model,ml_acc=train_ml_filter(df)
    if model is None:
        return {"t":0}, []

    lot=LOT.get(sym,10);capital=1_000_000;trades=[];pos=None
    W=60;signals=0;confirmed=0

    for i in range(W,len(df)-1):
        window=df.iloc[i-W:i+1];ind=calc_indicators(window)
        sg=rule_signal(window,ind,7)
        if sg is None:continue
        signals+=1

        # ML confirmation
        f=build_ml_features(df,i)
        if f is None:continue
        ml_prob=model.predict_proba(f.reshape(1,-1))[0,1]
        if ml_prob<0.55:continue  # ML disagrees, skip
        confirmed+=1

        # Trade
        c=float(df.iloc[i]["close"]);atr=ind["atr14"]
        entry=c+0.0002*c*(1 if sg=="LONG" else -1)
        sd=max(atr*0.3,atr*stop_atr);td=atr*target_atr
        stop=entry-sd if sg=="LONG" else entry+sd
        target=entry+td if sg=="LONG" else entry-td
        rc=capital*risk_pct
        q=max(1.0,min(20.0,rc/(sd*lot)))
        nc=float(df.iloc[i+1]["close"])
        nh=float(df.iloc[i+1]["high"]);nl=float(df.iloc[i+1]["low"])
        if sg=="LONG":
            ep=stop if nl<=stop else (target if nh>=target else nc)
            rs="STOP" if nl<=stop else ("TP" if nh>=target else "EOD")
        else:
            ep=stop if nh>=stop else (target if nl<=target else nc)
            rs="STOP" if nh>=stop else ("TP" if nl<=target else "EOD")
        pnl=(ep-entry)*lot*q*(1 if sg=="LONG" else -1)
        pnl-=abs(ep*lot*q*0.0001)+abs(entry*lot*q*0.0001)
        trades.append({"pnl":round(pnl,2),"rs":rs,"dir":sg,"ml_conf":round(ml_prob,3)})

    if not trades:
        return {"t":0,"signals":signals,"confirmed":confirmed,"ml_acc":round(ml_acc,3)}, []
    pnls=[t["pnl"] for t in trades];n=len(pnls)
    w=[p for p in pnls if p>0]
    wr=len(w)/n;tp=sum(pnls);ret=tp/capital*100
    aw=np.mean(w) if w else 0;al=abs(np.mean([p for p in pnls if p<=0])) if n-len(w)>0 else 1
    cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
    dd=abs(float(np.min((cum-rm)/capital*100)))
    pf=sum(w)/(abs(sum([p for p in pnls if p<=0]))+1e-8)
    stats={"t":n,"wr":round(wr,3),"pnl":round(tp,0),"ret":round(ret,1),
           "dd":round(dd,1),"pf":round(pf,2),"plr":round(aw/al,2),
           "signals":signals,"confirmed":confirmed,"ml_acc":round(ml_acc,3) if ml_acc else 0}
    return stats,trades

# ═══════════════════════════════════════════════════════════════════════════

print("="*60)
print("  Prophet v4.1 — 规则+ML混合策略 (双重确认)")
print("="*60)

for sym in ["lh","jm","jd","rb","i","fu","sc","ma","sa"]:
    df=fetch(sym,2500)
    if df is None or len(df)<500:continue
    print(f"\n  {sym.upper()} ({len(df)}天, {df.iloc[0]['date']}→{df.iloc[-1]['date']})")

    for rp in [0.01,0.015,0.02]:
        for sa in [1.2,1.5,2.0]:
            result = hybrid_backtest(df,sym,rp,sa,3.0)
            if result is None: continue
            stats, trades = result
            if stats is None or stats["t"]<10:continue
            m="🔥" if stats["wr"]>=0.65 else ("⭐" if stats["wr"]>=0.60 else "  ")
            print(f"  {m} R={rp} S={sa} T=3.0 → {stats['t']}t {stats['wr']:.0%}wr "
                  f"{stats['pnl']:+,.0f} {stats['ret']:+.1f}% DD{stats['dd']:.1f}% "
                  f"MLacc={stats['ml_acc']:.0%} sigs={stats['signals']}→{stats['confirmed']}")
