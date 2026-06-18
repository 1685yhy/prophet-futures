#!/usr/bin/env python3
"""
Massive Multi-Symbol Optimization — all Chinese futures, 5-year data.
Finds the best symbols + parameters for high win-rate strategies.
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise
from advanced_strategy import advanced_backtest, compute_stats, DynamicSizer

# All major Chinese futures (liquid, diverse sectors)
ALL_SYMBOLS = [
    # Agriculture
    "lh", "jd",   # live hog, eggs
    "m", "rm",    # soybean meal, rapeseed meal
    "y", "oi",    # soybean oil, rapeseed oil
    "p",          # palm oil
    "a", "c",     # soybean, corn
    "cf", "sr",   # cotton, sugar
    "ap",         # apple
    # Black metals
    "rb", "hc",   # rebar, hot-rolled coil
    "i", "jm", "j", # iron ore, coking coal, coke
    # Non-ferrous
    "cu", "al", "zn", "ni", # copper, aluminum, zinc, nickel
    "au", "ag",   # gold, silver
    # Energy/Chemical
    "sc", "bu", "fu", # crude oil, bitumen, fuel oil
    "ma", "ta", "eg", "pg", "pp", # methanol, PTA, EG, LPG, PP
    "sa", "fg",   # soda ash, glass
    "eb",         # benzene
]

LOT_SIZES = {
    "lh":16,"jd":5,"m":10,"rm":10,"y":10,"oi":10,"p":10,"a":10,"c":10,
    "cf":5,"sr":10,"ap":10,"rb":10,"hc":10,"i":100,"jm":60,"j":100,
    "cu":5,"al":5,"zn":5,"ni":1,"au":1000,"ag":15000,
    "sc":1000,"bu":10,"fu":10,"ma":10,"ta":5,"eg":10,"pg":20,"pp":5,
    "sa":20,"fg":20,"eb":5,
}

def fetch_data(symbol, days_back=1800):
    """Fetch ~5 years of data."""
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days_back + 200)
    try:
        df = ak.futures_main_sina(
            symbol=symbol.upper() + "0",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        df.columns = ["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except:
        return None

def quick_signal(df_window, ind, min_conds=7, require_regime=True):
    """Fast signal generator."""
    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)

    if require_regime and cycle_info["cycle"] not in ("BULL","BEAR"):
        return None

    _,_,h0=_calc_macd(closes)
    _,_,h1=_calc_macd(closes[:-1]) if len(closes)>1 else (0,0,0)
    _,_,h2=_calc_macd(closes[:-2]) if len(closes)>2 else (0,0,0)
    macd_imp = bool(h0<0 and h1<0 and abs(h0)<abs(h1)<abs(h2))

    adx=ind.get("adx14",0); rsi=ind.get("rsi14",50)
    ma5=ind.get("ma5",0); ma20=ind.get("ma20",0); ma60=ind.get("ma60",0)
    macd_h=ind.get("macd_hist",0)

    oi_col="oi" if "oi" in df_window.columns else None
    oi=df_window[oi_col].values.astype(float) if oi_col else np.zeros(10)
    oi3=float(oi[-1]-oi[-4]) if len(oi)>=4 else 0
    oi5=float(oi[-1]-oi[-6]) if len(oi)>=6 else oi3
    oi_trend="ACCUMULATING" if (oi3>0 and oi5>0) else ("REDUCING" if (oi3<0 and oi5<0) else "FLAT")

    mab=ma5>ma20>ma60; mabe=ma5<ma20<ma60

    sc=sum([cycle_info["cycle"]=="BEAR",mabe,macd_h<0 and not macd_imp,
            oi_trend in ("REDUCING","FLAT"),adx>20,32<rsi<72,
            not noise_info["is_noise"],True])
    lc=sum([cycle_info["cycle"]=="BULL",mab,macd_h>0,
            oi_trend=="ACCUMULATING",adx>22,30<rsi<65,
            not noise_info["is_noise"],True])

    if sc>=min_conds: return "SHORT"
    if lc>=min_conds: return "LONG"
    return None

def simple_backtest(df, symbol, capital, params):
    """Lightweight backtest for speed."""
    trades=[]; pos=None; W=60
    mc=params["mc"]; sa=params["sa"]; ta=params["ta"]
    rr=params.get("rr",True)
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

            if not(hs or ht):
                if d=="LONG": pos["s"]=max(pos["s"],c-1.5*atr)
                else: pos["s"]=min(pos["s"],c+1.5*atr)

            if hs or ht or fe:
                ep=pos["s"] if hs else (pos["t"] if ht else c)
                rs="STOP" if hs else ("TP" if ht else ("GAP" if gp>0.025 else "MAX_HOLD"))
                pnl=(ep-e)*lot*pos["q"]*(1 if d=="LONG" else -1)
                pnl-=abs(ep*lot*pos["q"]*comm)+abs(e*lot*pos["q"]*comm)
                trades.append({"pnl":pnl,"reason":rs,"hold":hold,
                               "entry_date":pos.get("date",""),"exit_date":ds,
                               "symbol":symbol})
                pos=None

        if not pos and i<len(df)-2:
            sig=quick_signal(w,ind,mc,rr)
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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

print("="*70)
print("  大规模多品种优化 — 全期货品种 × 5年数据")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*70)

capital = 1_000_000
all_results = []

# Parameter grid
param_grid = [
    {"mc":7,"sa":1.5,"ta":3.0,"rr":True},   # Conservative
    {"mc":6,"sa":1.2,"ta":2.5,"rr":True},   # Moderate
    {"mc":6,"sa":1.0,"ta":2.0,"rr":True},   # Aggressive
    {"mc":7,"sa":2.0,"ta":2.5,"rr":True},   # Safe
    {"mc":6,"sa":1.5,"ta":2.0,"rr":False},  # No regime filter
    {"mc":7,"sa":1.0,"ta":2.5,"rr":True},   # Tight stop
]

print(f"\nTesting {len(ALL_SYMBOLS)} symbols × {len(param_grid)} configs = "
      f"{len(ALL_SYMBOLS)*len(param_grid)} combinations...\n")

tested = 0; passed = 0
for sym in ALL_SYMBOLS:
    df = fetch_data(sym, 1800)
    if df is None or len(df) < 200:
        continue
    tested += 1
    best_for_symbol = None
    best_score = -999

    for i, pg in enumerate(param_grid):
        trades = simple_backtest(df, sym, capital, pg)
        if not trades: continue
        stats = compute_stats(trades, capital)
        # Score: win_rate * return / drawdown, require min 10 trades for validity
        if stats["trades"] >= 10:
            score = stats["win_rate"] * stats["total_return"] * min(stats["trades"], 60) / (stats["max_dd"]/100 + 0.1)
            stats["symbol"] = sym; stats.update(pg)
            stats["_score"] = round(score, 1)
            all_results.append(stats)
            passed += 1
            if score > best_score:
                best_score = score
                best_for_symbol = stats

    if best_for_symbol:
        print(f"  {sym.upper():<6} {len(df):>4}d  best: {best_for_symbol['trades']:>3}t "
              f"{best_for_symbol['win_rate']:.0%}wr {best_for_symbol['total_pnl']:+,.0f} "
              f"DD{best_for_symbol['max_dd']:.0f}%")

# Sort and display
all_results.sort(key=lambda x: x["_score"], reverse=True)

print(f"\n{'='*70}")
print(f"  Top 30 Strategies (of {passed})")
print(f"{'='*70}")
print(f"  {'Rank':<5} {'Sym':<6} {'MC':<4} {'Stop':<6} {'Tgt':<6} {'Reg':<5} {'Trd':<6} {'Win%':<7} {'PnL':<12} {'Ret%':<8} {'DD%':<6} {'Sharpe':<7}")
print(f"  {'─'*85}")
for i, r in enumerate(all_results[:30]):
    print(f"  {i+1:<5} {r['symbol'].upper():<6} {r['mc']:<4} {r['sa']:<6} {r['ta']:<6} "
          f"{str(r['rr']):<5} {r['trades']:<6} {r['win_rate']:.0%}     "
          f"{r['total_pnl']:+,.0f}     {r['total_return']:+.1f}%    "
          f"{r['max_dd']:.1f}%   {r['sharpe']:.3f}")

# Portfolio summary
if all_results:
    top_symbols = list(set(r["symbol"] for r in all_results[:20]))
    print(f"\n  高潜力品种 (Top 20中出现的): {', '.join(s.upper() for s in top_symbols)}")

    # Best per sector
    sectors = {
        "Agriculture": ["lh","jd","m","rm","y","oi","p","a","c","cf","sr","ap"],
        "Metals": ["rb","hc","i","jm","j","cu","al","zn","ni","au","ag"],
        "Energy/Chem": ["sc","bu","fu","ma","ta","eg","pg","pp","sa","fg","eb"],
    }
    print(f"\n  分板块最佳:")
    for sector, syms in sectors.items():
        sector_results = [r for r in all_results if r["symbol"] in syms]
        if sector_results:
            best = sector_results[0]
            print(f"  {sector}: {best['symbol'].upper()} {best['trades']}t "
                  f"{best['win_rate']:.0%}wr {best['total_return']:+.1f}% DD{best['max_dd']:.0f}%")

# Save
json.dump({"top": all_results[:50], "all": all_results},
          open("/tmp/massive_optimization.json","w"), indent=2, ensure_ascii=False)
print(f"\n结果保存到 /tmp/massive_optimization.json")
print(f"测试 {tested} 品种, {passed} 有效组合")
