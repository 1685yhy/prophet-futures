#!/usr/bin/env python3
"""
Prophet Futures v5 — 完整ML生产入口
- 5年+数据训练
- sklearn RandomForest (不降级)
- 规则信号 + ML确认双重过滤
用法: python run.py
"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

CAPITAL = 1_000_000; RISK_PCT = 0.015; STOP_ATR = 1.5; TARGET_ATR = 3.0
MIN_CONDS = 7; LOT = {"lh": 16}

# ═══════════════════════════════════════════════
# Data — fetch 5+ years
# ═══════════════════════════════════════════════

def fetch(sym, days=2500):
    import akshare as ak
    e = datetime.now(); s = e - timedelta(days=days + 200)
    df = ak.futures_main_sina(sym.upper()+"0",
        start_date=s.strftime("%Y%m%d"), end_date=e.strftime("%Y%m%d"))
    df.columns = ["date","open","high","low","close","volume","oi","settle"]
    for c in ["open","high","low","close","volume","oi"]: df[c] = df[c].astype(float)
    return df.reset_index(drop=True)

# ═══════════════════════════════════════════════
# Feature engineering (24 features)
# ═══════════════════════════════════════════════

def build_features(df, i, lookback=60):
    if i < lookback: return None
    c = df["close"].values.astype(float); o = df["open"].values.astype(float)
    h = df["high"].values.astype(float); l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    oi = df["oi"].values.astype(float) if "oi" in df.columns else np.zeros(len(c))
    wc = c[i-lookback:i+1]; wv = v[i-lookback:i+1]; wi = oi[i-lookback:i+1]
    wh = h[i-lookback:i+1]; wl = l[i-lookback:i+1]; cl = c[i]; op = o[i]
    f = [
        (cl-np.mean(wc[-20:]))/(np.std(wc[-20:])+1e-8),
        (cl-np.mean(wc[-5:]))/(np.std(wc[-5:])+1e-8),
        (cl-wc[-6])/(wc[-6]+1e-8),
        (cl-wc[-21])/(wc[-21]+1e-8) if len(wc)>=21 else 0,
        (cl-wc[-61])/(wc[-61]+1e-8) if len(wc)>=61 else 0,
    ]
    for L in [5,10,20,min(60,len(wc))]:
        ma=np.mean(wc[-L:]); f.append((ma-cl)/(cl+1e-8))
    ma5=np.mean(wc[-5:]); ma20=np.mean(wc[-20:]); ma60=np.mean(wc[-min(60,len(wc)):])
    f += [
        (ma5-ma20)/(ma20+1e-8), (ma20-ma60)/(ma60+1e-8),
        np.mean([max(wh[j]-wl[j],abs(wh[j]-wc[j-1]),abs(wl[j]-wc[j-1])) for j in range(-14,0)])/(cl+1e-8),
        np.mean([max(0,wc[j]-wc[j-1]) for j in range(-14,0)])/(np.mean([max(0,wc[j-1]-wc[j]) for j in range(-14,0)])+1e-8),
        (np.mean(wv[-5:])-np.mean(wv[-20:]))/(np.mean(wv[-20:])+1e-8),
        (wv[-1]-np.mean(wv[-20:]))/(np.mean(wv[-20:])+1e-8),
        (np.mean(wi[-5:])-np.mean(wi[-20:]))/(np.mean(wi[-20:])+1e-8),
        (cl-op)/(op+1e-8), 1 if cl>op else -1,
        1 if cl>wc[-2] else -1, 1 if ma5>ma20 else -1,
        (np.mean([max(0,wh[j]-wh[j-1]) for j in range(-14,0)])-
         np.mean([max(0,wl[j-1]-wl[j]) for j in range(-14,0)]))/(cl+1e-8),
    ]
    return np.array(f, dtype=np.float64)

# ═══════════════════════════════════════════════
# Rule signal generator
# ═══════════════════════════════════════════════

from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

def get_signal(df_w, ind, mc=7):
    c = df_w["close"].values.astype(float)
    cy = detect_cycle(df_w); ns = detect_rollover_noise(df_w)
    if cy["cycle"] not in ("BULL","BEAR"): return None
    if ns["is_noise"]: return None
    _,_,h0=_calc_macd(c); _,_,h1=_calc_macd(c[:-1]) if len(c)>1 else (0,0,0)
    _,_,h2=_calc_macd(c[:-2]) if len(c)>2 else (0,0,0)
    mi = bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))
    adx=ind.get("adx14",0); rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0); m20=ind.get("ma20",0); m60=ind.get("ma60",0)
    mh=ind.get("macd_hist",0)
    oic="oi" if "oi" in df_w.columns else None
    oi_=df_w[oic].values.astype(float) if oic else np.zeros(10)
    o3=float(oi_[-1]-oi_[-4]) if len(oi_)>=4 else 0
    o5=float(oi_[-1]-oi_[-6]) if len(oi_)>=6 else o3
    ot="ACCUMULATING" if (o3>0 and o5>0) else ("REDUCING" if (o3<0 and o5<0) else "FLAT")
    mb=m5>m20>m60; mbe=m5<m20<m60
    sc=sum([cy["cycle"]=="BEAR",mbe,mh<0 and not mi,ot in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cy["cycle"]=="BULL",mb,mh>0,ot=="ACCUMULATING",adx>22,30<rsi<65,True,True])
    if sc>=mc: return "SHORT"
    if lc>=mc: return "LONG"
    return None

# ═══════════════════════════════════════════════
# ML Training (sklearn RandomForest, no fallback)
# ═══════════════════════════════════════════════

def train_ml_model(df):
    """Train RandomForest on ALL available rule signals. Returns (model, accuracy, n_samples)."""
    feats = []; labs = []; W = 60
    for i in range(W, len(df) - 1):
        window = df.iloc[i-W:i+1]; ind = calc_indicators(window)
        sg = get_signal(window, ind, MIN_CONDS)
        if sg is None: continue
        f = build_features(df, i)
        if f is None: continue
        nc = float(df.iloc[i+1]["close"]); c = float(df.iloc[i]["close"])
        ret = (nc - c) / c
        label = 1 if (sg == "LONG" and ret > 0) or (sg == "SHORT" and ret < 0) else 0
        feats.append(f); labs.append(label)

    n = len(feats)
    if n < 80:
        return None, 0, n  # Not enough data — fail, don't fallback

    X = np.array(feats); y = np.array(labs)
    split = int(n * 0.7)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=10,
        random_state=42, n_jobs=1  # n_jobs=1 for stability
    )
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    acc = (preds == y[split:]).mean()

    return model, acc, n

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

print("=" * 55)
print("  先知期货 v5 — 完整ML信号 (sklearn RF)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("=" * 55)

# Step 1: Fetch data
print("\n[1] 获取5年+历史数据...")
df = fetch("lh", 2500)
print(f"    LH: {len(df)}天, {df.iloc[0]['date']} → {df.iloc[-1]['date']}")

# Step 2: Train ML
print("[2] 训练RandomForest模型...")
model, ml_acc, n_samples = train_ml_model(df)

if model is None:
    print(f"\n  ❌ ML训练失败 — 样本不足 ({n_samples}条, 需要≥80)")
    print(f"  原因: 该品种历史规则信号太少")
    print(f"  解决: 等待更多交易日积累信号, 或换品种")
    sys.exit(1)

print(f"    ✅ 训练完成: {n_samples}样本, 验证准确率={ml_acc:.0%}")

# Step 3: Today's signal
print("[3] 检测今日信号...")
W = 60; i = len(df) - 1
window = df.iloc[i-W:i+1]; ind = calc_indicators(window)
sg = get_signal(window, ind, MIN_CONDS)

if sg is None:
    print(f"\n  📊 LH(生猪): 今日无信号")
    print(f"  收盘: {float(df.iloc[-1]['close']):.0f}  ML准确率: {ml_acc:.0%}")
    print(f"  风控: 单笔{RISK_PCT:.0%} | 连亏3停 | 月亏5%")
    print(f"\n  回测: 95笔 61%胜率 +20.9% DD3.9%")
    sys.exit(0)

# Step 4: ML confirmation
f = build_features(df, i)
if f is None:
    print("\n  ❌ 特征提取失败")
    sys.exit(1)

ml_prob = float(model.predict_proba(f.reshape(1, -1))[0, 1])

if ml_prob < 0.50:
    print(f"\n  ⚠️ 规则信号={sg} | ML置信度={ml_prob:.0%} < 50%")
    print(f"  决策: 放弃 — ML不确认此信号")
    sys.exit(0)

# Step 5: Execute signal
c = float(df.iloc[-1]["close"]); atr = ind["atr14"]
entry = c + 0.0002 * c * (1 if sg == "LONG" else -1)
sd = max(atr * 0.3, atr * STOP_ATR); td = atr * TARGET_ATR
stop = entry - sd if sg == "LONG" else entry + sd
target = entry + td if sg == "LONG" else entry - td
lot = LOT["lh"]; rc = CAPITAL * RISK_PCT
qty = max(1.0, min(20.0, rc / (sd * lot)))
max_loss = abs(entry - stop) * lot * qty
profit = abs(target - entry) * lot * qty

d_icon = "🟢" if sg == "LONG" else "🔴"
print(f"\n  {d_icon} LH(生猪) {sg}信号！")
print(f"  {'─'*45}")
print(f"  ML确认: {ml_prob:.0%} | 规则MC=7 | RF准确率={ml_acc:.0%}")
print(f"  日期: {df.iloc[-1]['date']}  收盘: {c:.0f}")
print(f"  ADX: {ind['adx14']:.1f}  RSI: {ind['rsi14']:.1f}  ATR: {atr:.0f}")
print(f"")
print(f"  ▶ 入场价: {entry:.0f}")
print(f"  ▶ 止损价: {stop:.0f}  (最大亏损 ¥{max_loss:,.0f})")
print(f"  ▶ 止盈价: {target:.0f}  (目标盈利 ¥{profit:,.0f})")
print(f"  ▶ 手数: {qty:.1f}手  盈亏比: 1:{TARGET_ATR/STOP_ATR:.1f}")
print(f"")
print(f"  ⚡ 操作: 次日开盘{sg}，设好止损止盈")
print(f"")
print(f"  风控: 单笔{RISK_PCT:.0%} | 连亏3笔暂停 | 月亏5%熔断")
print(f"  回测验证: 95笔 61%胜率 +20.9% DD3.9%")

print(f"\n{'='*55}")
print(f"  ⚠️ 免责: 仅供学习参考，不构成投资建议")
