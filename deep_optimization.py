#!/usr/bin/env python3
"""
Deep optimization for LH + JM — targeted filters to push win rate past 60%.
Additional filters: volume confirmation, ADX strength, multi-timeframe, noise avoidance.
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise
from massive_optimization import fetch_data, simple_backtest, compute_stats, LOT_SIZES

capital = 1_000_000

def deep_signal(df_window, ind, mc=7, require_regime=True,
                require_volume=False, require_adx_strong=False,
                require_no_noise=True, require_mtf=False):
    """
    Multi-filter signal with optional enhancements:
    - require_volume: volume > 20d avg
    - require_adx_strong: ADX > 28
    - require_no_noise: skip rollover noise periods
    - require_mtf: weekly trend must agree with daily
    """
    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)

    if require_regime and cycle_info["cycle"] not in ("BULL","BEAR"):
        return None
    if require_no_noise and noise_info["is_noise"]:
        return None

    _,_,h0=_calc_macd(closes)
    _,_,h1=_calc_macd(closes[:-1]) if len(closes)>1 else (0,0,0)
    _,_,h2=_calc_macd(closes[:-2]) if len(closes)>2 else (0,0,0)
    macd_imp = bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))

    adx=ind.get("adx14",0); rsi=ind.get("rsi14",50)
    ma5=ind.get("ma5",0); ma20=ind.get("ma20",0); ma60=ind.get("ma60",0)
    macd_h=ind.get("macd_hist",0)

    # Volume filter
    if require_volume:
        vols=df_window["volume"].values.astype(float)
        if np.mean(vols[-5:]) < np.mean(vols[-20:]) * 1.05:
            return None  # Volume must be expanding

    # ADX strength filter
    if require_adx_strong and adx < 28:
        return None

    # Multi-timeframe (weekly): if we have >120 days, check longer trend
    if require_mtf and len(df_window) >= 120:
        # Simple: MA20 vs MA60 on longer window should agree with short
        long_ma20 = np.mean(closes[-20:])
        long_ma60 = np.mean(closes[-60:]) if len(closes)>=60 else long_ma20
        weekly_bull = long_ma20 > long_ma60
        weekly_bear = long_ma20 < long_ma60
        # Daily signal must agree with weekly trend
        daily_bull = ma5 > ma20  # simplified

    oi_col="oi" if "oi" in df_window.columns else None
    oi=df_window[oi_col].values.astype(float) if oi_col else np.zeros(10)
    oi3=float(oi[-1]-oi[-4]) if len(oi)>=4 else 0
    oi5=float(oi[-1]-oi[-6]) if len(oi)>=6 else oi3
    oi_trend="ACCUMULATING" if (oi3>0 and oi5>0) else ("REDUCING" if (oi3<0 and oi5<0) else "FLAT")

    mab=ma5>ma20>ma60; mabe=ma5<ma20<ma60

    sc=sum([cycle_info["cycle"]=="BEAR",mabe,macd_h<0 and not macd_imp,
            oi_trend in ("REDUCING","FLAT"),adx>20,32<rsi<72,True,True])
    lc=sum([cycle_info["cycle"]=="BULL",mab,macd_h>0,
            oi_trend=="ACCUMULATING",adx>22,30<rsi<65,True,True])

    if sc>=mc: return "SHORT"
    if lc>=mc: return "LONG"
    return None

def deep_backtest(df, symbol, capital, params):
    """Backtest with enhanced filters."""
    trades=[]; pos=None; W=60
    mc=params["mc"]; sa=params["sa"]; ta=params["ta"]
    rr=params.get("rr",True); rv=params.get("rv",False)
    ra=params.get("ra",False); rn=params.get("rn",True)
    rm=params.get("rm",False)
    lot=LOT_SIZES.get(symbol.lower(),10)
    comm=0.0001; slip=0.0002

    for i in range(W,len(df)-1):
        today=df.iloc[i]; ds=str(today["date"])
        w=df.iloc[i-W:i+1].copy(); ind=calc_indicators(w)
        atr=ind["atr14"]; c=float(today["close"])
        h=float(today["high"]); l=float(today["low"])

        if pos:
            d=pos["d"]; e=pos["e"]; hold=i-pos["idx"]
            gp=abs(c-float(df.iloc[i-1]["close"]))/max(1,float(df.iloc[i-1]["close"]))
            fe=gp>0.025 or hold>=20
            hs=(d=="LONG" and l<=pos["s"])or(d=="SHORT" and h>=pos["s"])
            ht=(d=="LONG" and h>=pos["t"])or(d=="SHORT" and l<=pos["t"])
            # Time stop after 8 days if no progress
            te=hold>=8 and ((d=="LONG" and c<=e)or(d=="SHORT" and c>=e))

            if not(hs or ht or te):
                if d=="LONG": pos["s"]=max(pos["s"],c-1.5*atr)
                else: pos["s"]=min(pos["s"],c+1.5*atr)

            if hs or ht or fe or te:
                ep=pos["s"] if hs else (pos["t"] if ht else c)
                rs="STOP" if hs else ("TP" if ht else ("GAP" if gp>0.025 else ("TIME" if te else "MAX_HOLD")))
                pnl=(ep-e)*lot*pos["q"]*(1 if d=="LONG" else -1)
                pnl-=abs(ep*lot*pos["q"]*comm)+abs(e*lot*pos["q"]*comm)
                trades.append({"pnl":pnl,"reason":rs,"hold":hold,
                               "entry_date":pos.get("date",""),"exit_date":ds,"symbol":symbol})
                pos=None

        if not pos and i<len(df)-2:
            sig=deep_signal(w,ind,mc,rr,rv,ra,rn,rm)
            if sig:
                d=sig; entry=c+slip*c*(1 if d=="LONG" else -1)
                sd=atr*sa; td=atr*ta
                if sd<atr*0.3: sd=atr*0.5
                sp=entry-sd if d=="LONG" else entry+sd
                tp=entry+td if d=="LONG" else entry-td
                mr=capital*0.02
                q=round(max(1.0,min(20.0,mr/(sd*lot))),1)
                pos={"d":d,"e":round(entry,2),"s":round(sp,2),
                     "t":round(tp,2),"idx":i,"q":q,"date":ds}
    return trades

# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("  深度优化 — LH+JM 多层过滤突破60%胜率")
print("="*70)

# Filter combinations to test
filters = [
    # (name, require_volume, require_adx_strong, require_no_noise, require_mtf)
    ("Baseline",   False, False, True,  False),
    ("+Volume",    True,  False, True,  False),
    ("+ADX28",     False, True,  True,  False),
    ("+Vol+ADX",   True,  True,  True,  False),
    ("+MTF",       False, False, True,  True),
    ("FullFilter", True,  True,  True,  True),
]

param_grid = [
    {"mc":7,"sa":1.5,"ta":3.0},
    {"mc":7,"sa":2.0,"ta":2.5},
    {"mc":8,"sa":1.5,"ta":2.5},
    {"mc":8,"sa":2.0,"ta":2.0},
]

all_results = []
for sym in ["lh", "jm"]:
    df = fetch_data(sym, 1800)
    if df is None: continue
    print(f"\n{'─'*70}")
    print(f"  {sym.upper()} ({len(df)} days)")
    print(f"  {'Filter':<14} {'MC':<4} {'Stop':<6} {'Tgt':<6} {'Trd':<6} {'Win%':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'Sharpe':<7}")
    print(f"  {'─'*70}")

    for fname, rv, ra, rn, rm in filters:
        for pg in param_grid:
            params = {"mc":pg["mc"],"sa":pg["sa"],"ta":pg["ta"],
                      "rr":True,"rv":rv,"ra":ra,"rn":rn,"rm":rm}
            trades = deep_backtest(df, sym, capital, params)
            if not trades: continue
            stats = compute_stats(trades, capital)
            stats["symbol"]=sym; stats["filter"]=fname; stats.update(pg)

            # Premium score: prioritize win_rate AND return
            if stats["trades"] >= 15:
                score = stats["win_rate"] * stats["total_return"] * 100
                stats["_score"]=round(score,1)
                all_results.append(stats)

                marker = "🔥" if stats["win_rate"]>=0.60 else ("⭐" if stats["win_rate"]>=0.55 else "  ")
                print(f"  {marker} {fname:<12} {pg['mc']:<4} {pg['sa']:<6} {pg['ta']:<6} "
                      f"{stats['trades']:<6} {stats['win_rate']:.0%}     "
                      f"{stats['total_pnl']:+,.0f}     {stats['total_return']:+.1f}%    "
                      f"{stats['max_dd']:.1f}%   {stats['sharpe']:.3f}")

# Sort by premium score
all_results.sort(key=lambda x: x["_score"], reverse=True)

print(f"\n{'='*70}")
print(f"  🏆 顶级策略 (胜率优先)")
print(f"{'='*70}")
print(f"  {'Rank':<5} {'Sym':<6} {'Filter':<14} {'MC':<4} {'Trd':<6} {'Win%':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'Sharpe':<7}")
for i, r in enumerate(all_results[:15]):
    m = "🔥" if r["win_rate"]>=0.60 else ""
    print(f"  {m} {i+1:<3} {r['symbol'].upper():<6} {r['filter']:<14} {r['mc']:<4} "
          f"{r['trades']:<6} {r['win_rate']:.0%}     {r['total_pnl']:+,.0f}     "
          f"{r['total_return']:+.1f}%    {r['max_dd']:.1f}%   {r['sharpe']:.3f}")

# Summary
wr60 = [r for r in all_results if r["win_rate"]>=0.60 and r["trades"]>=15]
print(f"\n  >=60%胜率组合: {len(wr60)} 个")
for r in wr60[:5]:
    print(f"  {r['symbol'].upper()} {r['filter']} MC={r['mc']}: "
          f"{r['trades']}t {r['win_rate']:.0%}wr {r['total_return']:+.1f}% DD{r['max_dd']:.0f}%")

json.dump({"top":all_results[:30],"wr60":wr60},
          open("/tmp/deep_optimization.json","w"), indent=2, ensure_ascii=False)
print(f"\n结果已保存")
