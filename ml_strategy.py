#!/usr/bin/env python3
"""
Prophet Futures v4.0 — ML增强策略
XGBoost预测方向 + 高置信度过滤 + 事件驱动 + 多品种轮动
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators

try:
    import xgboost as xgb; HAS_XGB = True
except:
    print("安装XGBoost..."); import subprocess
    subprocess.run([sys.executable,"-m","pip","install","xgboost","-q"])
    import xgboost as xgb; HAS_XGB = True

LOT = {"lh":16,"jm":60,"jd":5,"m":10,"rm":10,"p":10,"rb":10,"i":100,
       "cu":5,"sc":1000,"fu":10,"ma":10,"pg":20,"sa":20}

def fetch(sym, days=2000):
    import akshare as ak
    e=datetime.now(); s=e-timedelta(days=days+200)
    try:
        df=ak.futures_main_sina(sym.upper()+"0",s.strftime("%Y%m%d"),e.strftime("%Y%m%d"))
        df.columns=["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]: df[c]=df[c].astype(float)
        return df.reset_index(drop=True)
    except: return None

def build_features(df, lookback=60):
    if len(df) < lookback + 10: return None, None
    closes=df["close"].values.astype(float); opens=df["open"].values.astype(float)
    highs=df["high"].values.astype(float); lows=df["low"].values.astype(float)
    vols=df["volume"].values.astype(float)
    ois=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros_like(closes)

    X, Y = [], []
    for i in range(lookback, len(df)-1):
        wc=closes[i-lookback:i+1]; wo=opens[i-lookback:i+1]
        wh=highs[i-lookback:i+1]; wl=lows[i-lookback:i+1]
        wv=vols[i-lookback:i+1]; wi=ois[i-lookback:i+1]
        c=closes[i]; o=opens[i]

        ret=(closes[i+1]-c)/c
        if ret>0.005: y=1
        elif ret<-0.005: y=-1
        else: y=0

        f=[]
        # 1. Price z-scores
        f.append((c-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
        f.append((c-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
        # 2. Returns
        f.append((c-wc[-6])/(wc[-6]+1e-8))
        f.append((c-wc[-21])/(wc[-21]+1e-8))
        # 3. Moving averages
        ma5=np.mean(wc[-5:]); ma10=np.mean(wc[-10:])
        ma20=np.mean(wc[-20:]); ma60=np.mean(wc[-min(60,len(wc)):])
        for ma in [ma5,ma10,ma20,ma60]:
            f.append((ma-c)/(c+1e-8))
        f.append((ma5-ma20)/(ma20+1e-8))
        f.append((ma20-ma60)/(ma60+1e-8))
        # 4. ATR
        tr=[max(wh[j]-wl[j], abs(wh[j]-wc[j-1]), abs(wl[j]-wc[j-1])) for j in range(-14,0)]
        atr=np.mean(tr); f.append(atr/(c+1e-8))
        # 5. RSI
        gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
        losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
        rs=np.mean(gains)/(np.mean(losses)+1e-8); f.append(rs)
        # 6. Volume
        vm5=np.mean(wv[-5:]); vm20=np.mean(wv[-20:])
        f.append((vm5-vm20)/(vm20+1e-8))
        f.append((wv[-1]-vm20)/(vm20+1e-8))
        # 7. OI
        om5=np.mean(wi[-5:]); om20=np.mean(wi[-20:])
        f.append((om5-om20)/(om20+1e-8))
        # 8. Intraday
        f.append((c-o)/(o+1e-8))
        f.append(1 if c>o else -1)
        # 9. Trend
        f.append(1 if c>wc[-2] else -1)

        X.append(f); Y.append(y)
    return np.array(X), np.array(Y)

# ═══════════════════════════════════════════════════════════════════════════

print("="*60)
print("  Prophet v4.0 — ML方向预测 + 高置信度过滤")
print("="*60)

for sym in ["lh","jm","jd","rm","p","rb","fu","pg","sa"]:
    df=fetch(sym,2000)
    if df is None or len(df)<500: continue
    X,Y=build_features(df)
    if X is None: continue

    split=int(len(X)*0.7)
    Xt,Xv=X[:split],X[split:]; Yt,Yv=Y[:split],Y[split:]

    # Binary LONG/SHORT classifiers
    yl_t=(Yt==1).astype(int); yl_v=(Yv==1).astype(int)
    ys_t=(Yt==-1).astype(int); ys_v=(Yv==-1).astype(int)

    ml=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,random_state=42)
    ml.fit(Xt,yl_t)
    ms=xgb.XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,random_state=42)
    ms.fit(Xt,ys_t)

    la=(ml.predict(Xv)==yl_v).mean(); sa=(ms.predict(Xv)==ys_v).mean()

    # Simulate high-confidence trades on validation
    lp=ml.predict_proba(Xv)[:,1]; sp=ms.predict_proba(Xv)[:,1]
    vc=df["close"].values[60:][split:]
    pnls=[]
    for j in range(len(Xv)-1):
        if lp[j]>0.60:
            r=(vc[j+1]-vc[j])/vc[j]; pnls.append(r*LOT.get(sym,10)*vc[j])
        elif sp[j]>0.60:
            r=(vc[j]-vc[j+1])/vc[j]; pnls.append(r*LOT.get(sym,10)*vc[j])

    wr=sum(1 for p in pnls if p>0)/len(pnls) if pnls else 0
    tp=sum(pnls)
    nt=len(pnls)

    print(f"  {sym.upper():<6} LONG={la:.0%} SHORT={sa:.0%}  "
          f"信号{nt}笔 胜率{wr:.0%} PnL{tp:+,.0f}  "
          f"({df.iloc[0]['date']}→{df.iloc[-1]['date']})")

print(f"\n✅ ML模型揭示: 哪些品种的ML预测准确率最高，哪些最低")
print(f"   下一步: 只交易ML准确率>55%的品种 + 方向")
