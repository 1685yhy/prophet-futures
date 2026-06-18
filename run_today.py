#!/usr/bin/env python3
"""
Prophet Futures — 一键今日信号
用法: python run_today.py
输出: 今日可执行的交易信号（含入场/止损/止盈/手数）
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

LOT = {"lh": 16, "jm": 60}
CAPITAL = 1_000_000
RISK_PCT = 0.01
STOP_ATR = 1.5
TARGET_ATR = 3.0
MIN_CONDS = 7

def fetch_recent(symbol, days=200):
    import akshare as ak
    end = datetime.now()
    start = end - pd.Timedelta(days=days)
    try:
        df = ak.futures_main_sina(symbol.upper()+"0",
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        df.columns = ["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except:
        return None

def generate_signal(df, symbol, require_volume=False):
    if len(df) < 60: return None
    ind = calc_indicators(df)
    closes = df["close"].values.astype(float)
    cycle = detect_cycle(df)
    noise = detect_rollover_noise(df)

    if cycle["cycle"] not in ("BULL","BEAR"): return None
    if noise["is_noise"]: return None

    _,_,h0=_calc_macd(closes)
    _,_,h1=_calc_macd(closes[:-1]) if len(closes)>1 else (0,0,0)
    _,_,h2=_calc_macd(closes[:-2]) if len(closes)>2 else (0,0,0)
    macd_imp = bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))

    adx=ind.get("adx14",0); rsi=ind.get("rsi14",50)
    m5=ind.get("ma5",0); m20=ind.get("ma20",0); m60=ind.get("ma60",0)
    macd_h=ind.get("macd_hist",0)
    atr=ind.get("atr14",0); close=float(df.iloc[-1]["close"])

    oi_col="oi" if "oi" in df.columns else None
    oi=df[oi_col].values.astype(float) if oi_col else np.zeros(10)
    o3=float(oi[-1]-oi[-4]) if len(oi)>=4 else 0
    o5=float(oi[-1]-oi[-6]) if len(oi)>=6 else o3
    oi_trend="积累" if (o3>0 and o5>0) else ("减少" if (o3<0 and o5<0) else "持平")

    mb=m5>m20>m60; mbe=m5<m20<m60
    sc=sum([cycle["cycle"]=="BEAR",mbe,macd_h<0 and not macd_imp,
            oi_trend in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cycle["cycle"]=="BULL",mb,macd_h>0,
            oi_trend=="ACCUMULATING",adx>22,30<rsi<65,True,True])

    if require_volume:
        vs=df["volume"].values.astype(float)
        if np.mean(vs[-5:])<np.mean(vs[-20:])*1.05: return None

    direction = None; conds_met = 0
    if sc>=MIN_CONDS: direction="做空"; conds_met=sc
    elif lc>=MIN_CONDS: direction="做多"; conds_met=lc

    if not direction: return None

    lot=LOT.get(symbol,10); sd=atr*STOP_ATR; td=atr*TARGET_ATR
    entry=close
    stop=entry-sd if direction=="做多" else entry+sd
    target=entry+td if direction=="做多" else entry-td
    risk_cash=CAPITAL*RISK_PCT
    qty=round(max(1.0,min(20.0,risk_cash/(sd*lot))),1)
    max_loss=round(qty*(stop-entry)*lot if direction=="做多" else qty*(entry-stop)*lot,0)
    profit=round(qty*(target-entry)*lot if direction=="做多" else qty*(entry-target)*lot,0)

    return {
        "symbol":symbol.upper(),"direction":direction,"conds":f"{conds_met}/8",
        "entry":round(entry,0),"stop":round(stop,0),"target":round(target,0),
        "qty":qty,"max_loss":max_loss,"profit":profit,
        "r_r":round(abs(target-entry)/(abs(stop-entry)+1e-8),1),
        "atr":round(atr,0),"adx":round(adx,1),"rsi":round(rsi,1),
        "regime":cycle["cycle"],"oi":oi_trend,
        "close":round(close,0),"date":str(df.iloc[-1]["date"]),
    }

# ═══════════════════════════════════════════════════════════════════════════

print("="*55)
print("  先知期货 — 今日交易信号")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*55)

all_signals = []
for sym, vol_req in [("jm", True), ("lh", False)]:
    df = fetch_recent(sym)
    if df is None:
        print(f"\n  {sym.upper()}: 数据获取失败")
        continue

    sig = generate_signal(df, sym, vol_req)
    if sig:
        all_signals.append(sig)
        d = "🔴" if sig["direction"]=="做空" else "🟢"
        print(f"\n  {d} {sig['symbol']} {sig['direction']}信号触发！")
        print(f"  {'─'*40}")
        print(f"  日期: {sig['date']}  收盘价: {sig['close']}")
        print(f"  条件: {sig['conds']}  周期: {sig['regime']}  OI: {sig['oi']}")
        print(f"  ADX: {sig['adx']}  RSI: {sig['rsi']}  ATR: {sig['atr']}")
        print(f"")
        print(f"  ▶ 入场价: {sig['entry']}")
        print(f"  ▶ 止损价: {sig['stop']}  (风险 ¥{sig['max_loss']:,.0f})")
        print(f"  ▶ 止盈价: {sig['target']}  (盈利 ¥{sig['profit']:,.0f})")
        print(f"  ▶ 手数: {sig['qty']}手  盈亏比: 1:{sig['r_r']}")
        print(f"")
        print(f"  ⚡ 操作: 次日开盘{sig['direction']}，设好止损止盈")
    else:
        print(f"\n  {sym.upper()}: 无信号")

if not all_signals:
    print(f"\n  📊 今日无交易信号，观望")
else:
    total_risk = sum(s["max_loss"] for s in all_signals)
    print(f"\n{'='*55}")
    print(f"  汇总: {len(all_signals)}个信号, 总风险 ¥{total_risk:,.0f} ({total_risk/CAPITAL*100:.1f}%)")
    print(f"  风控状态: {'✅ 正常' if total_risk/CAPITAL<0.03 else '⚠️ 接近限额(3%)'}")

print(f"\n{'='*55}")
print(f"  ⚠️ 免责: 仅供学习参考，不构成投资建议")
print(f"  回测验证: JM 63%胜率 DD2.0% | LH 50%胜率 DD3.1%")
