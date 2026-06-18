#!/usr/bin/env python3
"""
Prophet v4 — 全品种ML大规模回测
无限制token → 训练所有品种 × 多种模型 × 多年周期 × Walk-Forward
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LOT = {"lh":16,"jm":60,"jd":5,"m":10,"rm":10,"y":10,"oi":10,"p":10,"a":10,"c":10,
       "cf":5,"sr":10,"ap":10,"rb":10,"hc":10,"i":100,"jm2":60,"j":100,
       "cu":5,"al":5,"zn":5,"ni":1,"au":1000,"ag":15000,
       "sc":1000,"bu":10,"fu":10,"ma":10,"ta":5,"eg":10,"pg":20,"pp":5,
       "sa":20,"fg":20,"eb":5}

ALL_SYMBOLS = list(LOT.keys())

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
    if len(df) < lookback + 20: return None, None, None
    closes=df["close"].values.astype(float); opens=df["open"].values.astype(float)
    highs=df["high"].values.astype(float); lows=df["low"].values.astype(float)
    vols=df["volume"].values.astype(float)
    ois=df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(closes))
    dates=df["date"].values

    X, Y, D = [], [], []
    for i in range(lookback, len(df)-1):
        wc=closes[i-lookback:i+1]; wv=vols[i-lookback:i+1]
        wi=ois[i-lookback:i+1]; wh=highs[i-lookback:i+1]; wl=lows[i-lookback:i+1]
        c=closes[i]; o=opens[i]

        ret=(closes[i+1]-c)/c
        y=1 if ret>0.005 else (-1 if ret<-0.005 else 0)

        f=[]
        f.append((c-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8))
        f.append((c-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8))
        f.append((c-wc[-6])/(wc[-6]+1e-8))
        f.append((c-wc[-21])/(wc[-21]+1e-8))
        f.append((c-wc[-61])/(wc[-61]+1e-8) if len(wc)>61 else 0)
        ma5=np.mean(wc[-5:]); ma10=np.mean(wc[-10:])
        ma20=np.mean(wc[-20:]); ma60=np.mean(wc[-min(60,len(wc)):])
        for ma in [ma5,ma10,ma20,ma60]: f.append((ma-c)/(c+1e-8))
        f.append((ma5-ma20)/(ma20+1e-8))
        f.append((ma5-ma60)/(ma60+1e-8))
        tr=[max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)]
        f.append(np.mean(tr)/(c+1e-8))
        gains=[max(0,wc[j]-wc[j-1]) for j in range(-14,0)]
        losses=[max(0,wc[j-1]-wc[j]) for j in range(-14,0)]
        f.append(np.mean(gains)/(np.mean(losses)+1e-8))
        vm5=np.mean(wv[-5:]); vm20=np.mean(wv[-20:])
        f.append((vm5-vm20)/(vm20+1e-8)); f.append((wv[-1]-vm20)/(vm20+1e-8))
        om5=np.mean(wi[-5:]); om20=np.mean(wi[-20:])
        f.append((om5-om20)/(om20+1e-8))
        f.append((c-o)/(o+1e-8))
        f.append(1 if c>o else -1)
        f.append(1 if c>wc[-2] else -1)
        f.append(1 if ma5>ma20 else -1)

        X.append(f); Y.append(y); D.append(dates[i])
    return np.array(X), np.array(Y), np.array(D)

# ═══════════════════════════════════════════════════════════════════════════

def backtest_ml_signals(sym, df, model_type="rf", conf_threshold=0.55,
                         risk_pct=0.01, stop_atr=1.5, target_atr=3.0,
                         n_splits=4):
    """Walk-forward ML backtest."""
    from tools.indicators import calc_indicators

    X, Y, D = build_features(df)
    if X is None or len(X) < 200: return None

    lot = LOT.get(sym, 10)
    splits = np.linspace(0, len(X), n_splits+1).astype(int)
    all_trades = []
    capital = 1_000_000

    for sp in range(n_splits):
        train_end = splits[sp+1]
        if train_end >= len(X): break

        Xt, Yt = X[:train_end], Y[:train_end]
        Xv = X[train_end:] if sp == n_splits-1 else X[train_end:splits[sp+2] if sp+2 < len(splits) else len(X)]
        Yv = Y[train_end:] if sp == n_splits-1 else Y[train_end:splits[sp+2] if sp+2 < len(splits) else len(X)]

        if len(Xt) < 100 or len(Xv) < 50: continue

        # Train binary classifiers
        yl_t = (Yt==1).astype(int); yl_v = (Yv==1).astype(int)
        ys_t = (Yt==-1).astype(int); ys_v = (Yv==-1).astype(int)

        if model_type == "rf":
            ml = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            ms = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        elif model_type == "gb":
            ml = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            ms = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        else:
            ml = LogisticRegression(max_iter=1000, C=0.1)
            ms = LogisticRegression(max_iter=1000, C=0.1)

        ml.fit(Xt, yl_t); ms.fit(Xt, ys_t)

        lp = ml.predict_proba(Xv)[:,1]; sp = ms.predict_proba(Xv)[:,1]

        # Get validation period data
        val_start = train_end + 60
        val_end = val_start + len(Xv)
        if val_end > len(df): val_end = len(df)
        val_df = df.iloc[val_start:val_end].reset_index(drop=True)

        # Simulate trades
        for j in range(min(len(lp), len(val_df)-1)):
            if lp[j] > conf_threshold or sp[j] > conf_threshold:
                d = "LONG" if lp[j] > sp[j] else "SHORT"
                conf = max(lp[j], sp[j])
                entry = float(val_df.iloc[j]["close"])
                atr = float(val_df.iloc[j]["high"]) - float(val_df.iloc[j]["low"])
                if atr < entry * 0.001: atr = entry * 0.01

                stop_dist = atr * stop_atr; target_dist = atr * target_atr
                if stop_dist < atr * 0.3: stop_dist = atr * 0.5
                stop = entry - stop_dist if d == "LONG" else entry + stop_dist
                target = entry + target_dist if d == "LONG" else entry - target_dist

                risk_cash = capital * risk_pct
                qty = max(1.0, min(20.0, risk_cash / (stop_dist * lot)))
                max_loss = abs(entry - stop) * lot * qty

                # Next day close
                next_close = float(val_df.iloc[min(j+1, len(val_df)-1)]["close"])
                next_high = float(val_df.iloc[min(j+1, len(val_df)-1)]["high"])
                next_low = float(val_df.iloc[min(j+1, len(val_df)-1)]["low"])

                # Check if stop/target hit
                if d == "LONG":
                    if next_low <= stop: exit_price = stop; reason = "STOP"
                    elif next_high >= target: exit_price = target; reason = "TP"
                    else: exit_price = next_close; reason = "EOD"
                else:
                    if next_high >= stop: exit_price = stop; reason = "STOP"
                    elif next_low <= target: exit_price = target; reason = "TP"
                    else: exit_price = next_close; reason = "EOD"

                pnl = (exit_price - entry) * lot * qty * (1 if d == "LONG" else -1)
                comm = abs(exit_price * lot * qty * 0.0001) + abs(entry * lot * qty * 0.0001)
                pnl -= comm

                all_trades.append({
                    "pnl": round(pnl, 2), "reason": reason, "direction": d,
                    "symbol": sym, "conf": round(conf, 3),
                    "entry": round(entry, 0), "exit": round(exit_price, 0),
                    "qty": qty, "model": model_type, "split": sp,
                })

    return all_trades

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

print("="*65)
print("  Prophet v4 — 全品种ML大规模回测")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*65)

MODELS = [
    ("RandomForest", "rf"),
    ("GradientBoost", "gb"),
    ("LogisticReg", "lr"),
]
CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
RISK_PCTS = [0.01, 0.015, 0.02]

all_results = []
tested = 0; valid = 0

for sym in ALL_SYMBOLS:
    df = fetch(sym, 2500)
    if df is None or len(df) < 500:
        continue
    tested += 1

    for mname, mtype in MODELS:
        for ct in CONF_THRESHOLDS:
            for rp in RISK_PCTS:
                trades = backtest_ml_signals(sym, df, mtype, ct, rp, 1.5, 3.0)
                if not trades or len(trades) < 15:
                    continue

                pnls = [t["pnl"] for t in trades]
                n = len(pnls)
                wins = [p for p in pnls if p > 0]
                wr = len(wins) / n
                tp = sum(pnls)
                aw = np.mean(wins) if wins else 0
                losses = [p for p in pnls if p <= 0]
                al = abs(np.mean(losses)) if losses else 1
                plr = aw / (al + 1e-8)
                pf = sum(wins) / (abs(sum(losses)) + 1e-8)
                cum = np.cumsum(pnls)
                rm = np.maximum.accumulate(cum)
                dd = abs(float(np.min((cum - rm) / (1_000_000 + 1e-8) * 100)))
                if n >= 5:
                    rets = [p / 1_000_000 for p in pnls]
                    sr = np.mean(rets) / (np.std(rets, ddof=1) + 1e-8) * np.sqrt(252)
                else:
                    sr = 0

                ret = tp / 1_000_000 * 100

                score = wr * ret / (dd/100 + 0.05) * min(n, 60)
                all_results.append({
                    "symbol": sym, "model": mname, "conf": ct, "risk": rp,
                    "trades": n, "wr": round(wr, 3), "pnl": round(tp, 0),
                    "ret": round(ret, 1), "dd": round(dd, 1),
                    "sr": round(sr, 3), "pf": round(pf, 2), "plr": round(plr, 2),
                    "score": round(score, 1),
                })
                valid += 1

all_results.sort(key=lambda x: x["score"], reverse=True)

print(f"\n  测试品种: {tested}  有效组合: {valid}")
print(f"\n  Top 40 ML策略 (胜率+收益优先):")
print(f"  {'Rank':<5} {'Sym':<6} {'Model':<14} {'Conf':<6} {'Risk':<6} {'Trd':<6} {'Win%':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'SR':<6} {'PF':<6}")
print(f"  {'─'*90}")
for i, r in enumerate(all_results[:40]):
    print(f"  {i+1:<5} {r['symbol'].upper():<6} {r['model']:<14} {r['conf']:<6} {r['risk']:<6} "
          f"{r['trades']:<6} {r['wr']:.0%}     {r['pnl']:+,.0f}     {r['ret']:+.1f}%    "
          f"{r['dd']:.1f}%   {r['sr']:.2f}  {r['pf']:.2f}")

# Best per model type
print(f"\n  各模型最佳:")
for mname, mtype in MODELS:
    model_best = [r for r in all_results if r["model"] == mname][:3]
    for r in model_best:
        print(f"  {mname:<14} {r['symbol'].upper()} conf={r['conf']}: "
              f"{r['trades']}t {r['wr']:.0%}wr {r['ret']:+.1f}% DD{r['dd']:.1f}%")

# High win-rate club
wr60 = [r for r in all_results if r["wr"] >= 0.60 and r["trades"] >= 20]
print(f"\n  >=60%胜率组合: {len(wr60)}个")
for r in sorted(wr60, key=lambda x: x["score"], reverse=True)[:10]:
    print(f"  {r['symbol'].upper()} {r['model']} conf={r['conf']}: "
          f"{r['trades']}t {r['wr']:.0%}wr {r['ret']:+.1f}% DD{r['dd']:.1f}%")

json.dump({"top": all_results[:100], "total": len(all_results)},
          open("/tmp/ml_massive_backtest.json", "w"), indent=2, ensure_ascii=False)
print(f"\n✅ 结果保存 /tmp/ml_massive_backtest.json")
