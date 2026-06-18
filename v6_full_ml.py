#!/usr/bin/env python3
"""
Prophet v6 — 全量ML训练 (1300+样本)
ML直接预测次日方向 → 规则确认 → 双重过滤
"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

CAPITAL=1_000_000; RISK_PCT=0.015; STOP_ATR=1.5; TARGET_ATR=3.0; LOT={"lh":16}

def fetch(sym,days=2500):
    import akshare as ak
    e=datetime.now();s=e-timedelta(days=days+200)
    df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
    df.columns=["date","open","high","low","close","volume","oi","settle"]
    for c in["open","high","low","close","volume","oi"]:df[c]=df[c].astype(float)
    return df.reset_index(drop=True)

def build_features(df,i,L=60):
    if i<L:return None
    c=df["close"].values.astype(float);o=df["open"].values.astype(float)
    h=df["high"].values.astype(float);l=df["low"].values.astype(float)
    v=df["volume"].values.astype(float)
    oi=df["oi"].values.astype(float) if"oi" in df.columns else np.zeros(len(c))
    wc=c[i-L:i+1];wv=v[i-L:i+1];wi=oi[i-L:i+1];wh=h[i-L:i+1];wl=l[i-L:i+1];cl=c[i];op=o[i]
    f=[]
    # Returns (multiple horizons)
    for j in[1,3,5,10,21]:
        if len(wc)>j:f.append((cl-wc[-j-1])/(wc[-j-1]+1e-8))
        else:f.append(0)
    # Z-scores
    f.append((cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
    f.append((cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
    # Moving averages
    for Lv in[3,5,8,10,13,20,30,min(60,len(wc))]:
        ma=np.mean(wc[-Lv:]);f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]);ma8=np.mean(wc[-8:]);ma20=np.mean(wc[-20:]);ma60=np.mean(wc[-min(60,len(wc)):])
    f.extend([(ma5-ma8)/(ma8+1e-8),(ma5-ma20)/(ma20+1e-8),(ma20-ma60)/(ma60+1e-8)])
    # ATR + volatility
    tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
    atr=np.mean(tr);f.append(atr/(cl+1e-8))
    f.append(np.std(wc[-20:])/(np.mean(wc[-20:])+1e-8))
    # RSI + Stochastic
    g=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
    ls=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
    rs=np.mean(g)/(np.mean(ls)+1e-8);f.append(rs)
    f.append((cl-min(wc[-14:]))/(max(wc[-14:])-min(wc[-14:])+1e-8))
    # Volume
    vm3=np.mean(wv[-3:]);vm5=np.mean(wv[-5:]);vm10=np.mean(wv[-10:]);vm20=np.mean(wv[-20:])
    f.extend([(vm3-vm10)/(vm10+1e-8),(vm5-vm20)/(vm20+1e-8),(wv[-1]-vm20)/(vm20+1e-8)])
    # OI
    om3=np.mean(wi[-3:]);om5=np.mean(wi[-5:]);om20=np.mean(wi[-20:])
    f.extend([(om3-om5)/(om5+1e-8),(om5-om20)/(om20+1e-8)])
    # Price pattern
    f.extend([(cl-op)/(op+1e-8),1 if cl>op else-1,(h[-1]-l[-1])/(cl+1e-8)])
    f.extend([1 if cl>wc[-2] else-1,1 if cl>wc[-3] else-1])
    f.extend([1 if ma5>ma8 else-1,1 if ma5>ma20 else-1])
    # Directional movement
    pdm=[max(0,wh[j]-wh[j-1]) for j in range(-14,0)]
    mdm=[max(0,wl[j-1]-wl[j]) for j in range(-14,0)]
    dx=(np.mean(pdm)-np.mean(mdm))/(cl+1e-8);f.append(dx)
    return np.array(f,dtype=np.float64)

from tools.indicators import calc_indicators,_calc_macd
from tools.cycle_detector import detect_cycle,detect_rollover_noise

def rule_signal(df_w,ind,mc=7):
    c=df_w["close"].values.astype(float)
    cy=detect_cycle(df_w);ns=detect_rollover_noise(df_w)
    if cy["cycle"]not in("BULL","BEAR"):return None
    if ns["is_noise"]:return None
    _,_,h0=_calc_macd(c);_,_,h1=_calc_macd(c[:-1]) if len(c)>1 else(0,0,0);_,_,h2=_calc_macd(c[:-2]) if len(c)>2 else(0,0,0)
    mi=bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0);rsi=ind.get("rsi14",50);m5=ind.get("ma5",0);m20=ind.get("ma20",0);m60=ind.get("ma60",0);mh=ind.get("macd_hist",0)
    oic="oi"if"oi" in df_w.columns else None;oi_=df_w[oic].values.astype(float)if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4])if len(oi_)>=4 else 0;o5=float(oi_[-1]-oi_[-6])if len(oi_)>=6 else o3
    ot="ACCUMULATING"if(o3>0 and o5>0)else("REDUCING"if(o3<0 and o5<0)else"FLAT")
    mb=m5>m20>m60;mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc:return"SHORT"
    if lc>=mc:return"LONG"
    return None

# ═══════════════════════════════════════════════
# Train ML on ALL days (not just rule signals)
# ═══════════════════════════════════════════════

def train_full_ml(df):
    """Train ML on every trading day to predict next-day direction."""
    feats=[];labs=[];W=60
    for i in range(W,len(df)-1):
        f=build_features(df,i)
        if f is None:continue
        nc=float(df.iloc[i+1]["close"]);c=float(df.iloc[i]["close"])
        ret=(nc-c)/c
        # 3-class target
        if ret>0.005:lab=1     # UP >0.5%
        elif ret<-0.005:lab=-1 # DOWN >0.5%
        else:lab=0              # FLAT
        feats.append(f);labs.append(lab)
    return np.array(feats),np.array(labs)

# ═══════════════════════════════════════════════
# Walk-Forward Backtest
# ═══════════════════════════════════════════════

print("="*60)
print("  Prophet v6 — 全量ML训练 (1300+样本/品种)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*60)

for sym in["lh","jm"]:
    df=fetch(sym,2500)
    if df is None:continue
    X_all,Y_all=train_full_ml(df)
    n=len(X_all)
    print(f"\n  {sym.upper()}: {len(df)}天 → {n}训练样本")

    # Cross-val accuracy
    for model_name,model_cls in[("GB",GradientBoostingClassifier),("RF",RandomForestClassifier)]:
        if model_name=="GB":
            m=model_cls(n_estimators=100,max_depth=5,learning_rate=0.05,random_state=42)
        else:
            m=model_cls(n_estimators=100,max_depth=6,random_state=42,n_jobs=1)

        # Binary: UP vs not-UP
        y_bin=(Y_all==1).astype(int)
        scores=cross_val_score(m,X_all[:1000],y_bin[:1000],cv=5,scoring='accuracy')
        print(f"  {model_name} UP预测: {scores.mean():.0%} (±{scores.std():.0%})")

    # Walk-Forward backtest
    nsp=4;splits=np.linspace(0,n,nsp+1).astype(int)
    all_t=[];W=60

    for sp in range(nsp-1):
        te=splits[sp+1];ve=splits[sp+2]
        if ve>=n:break
        Xt,Yt=X_all[:te],Y_all[:te];Xv,Yv=X_all[te:ve],Y_all[te:ve]
        yb_t=(Yt==1).astype(int);ys_t=(Yt==-1).astype(int)

        ml=GradientBoostingClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42)
        ms=GradientBoostingClassifier(n_estimators=80,max_depth=4,learning_rate=0.05,random_state=42)
        ml.fit(Xt,yb_t);ms.fit(Xt,ys_t)
        lp=ml.predict_proba(Xv)[:,1];sp=ms.predict_proba(Xv)[:,1]

        # Map back to original df indices
        offset=te+W

        for j in range(min(len(lp),len(df)-offset-1)):
            idx=offset+j
            if idx>=len(df)-1:break

            # ML direction
            if lp[j]>0.55:ml_dir="LONG";conf=lp[j]
            elif sp[j]>0.55:ml_dir="SHORT";conf=sp[j]
            else:continue

            # Rule confirmation
            window=df.iloc[idx-W:idx+1];ind=calc_indicators(window)
            rs=rule_signal(window,ind,7)
            if rs is None or rs!=ml_dir:continue  # Must agree

            c=float(df.iloc[idx]["close"]);atr=ind["atr14"]
            entry=c+0.0002*c*(1 if ml_dir=="LONG" else-1)
            sd=max(atr*0.3,atr*1.5);td=atr*3.0
            stop=entry-sd if ml_dir=="LONG" else entry+sd
            target=entry+td if ml_dir=="LONG" else entry-td
            l=LOT.get(sym,10);q=max(1.0,min(20.0,CAPITAL*0.015/(sd*l)))
            nc=float(df.iloc[idx+1]["close"]);nh=float(df.iloc[idx+1]["high"]);nl=float(df.iloc[idx+1]["low"])
            if ml_dir=="LONG":ep=stop if nl<=stop else(target if nh>=target else nc)
            else:ep=stop if nh>=stop else(target if nl<=target else nc)
            pnl=(ep-entry)*l*q*(1 if ml_dir=="LONG" else-1)-abs(ep*l*q*0.0001)*2
            all_t.append({"pnl":pnl,"sp":sp,"sym":sym})

    if all_t:
        pnls=[t["pnl"] for t in all_t];n_=len(pnls);w=[p for p in pnls if p>0]
        wr=len(w)/n_;tp=sum(pnls);ret=tp/CAPITAL*100
        cum=np.cumsum(pnls);rm=np.maximum.accumulate(cum)
        dd=abs(float(np.min((cum-rm)/CAPITAL*100)))
        aw=np.mean(w)if w else 0;al=abs(np.mean([p for p in pnls if p<=0]))if n_-len(w)else 1
        print(f"  ✅ WF回测: {n_}笔 {wr:.0%}胜率 PnL{tp:+,.0f} {ret:+.1f}% DD{dd:.1f}% R:R{aw/al:.1f}")
    else:
        print(f"  ❌ 无交易")
