#!/usr/bin/env python3
"""
Advanced Strategy Engine v2 — Multi-dimensional optimization.
Features: regime filter, dynamic sizing, multi-layer confirmation,
          pyramiding, adaptive stops, time-based exits.
"""

import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def fetch_history(symbol, days_back):
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days_back + 200)
    try:
        df = ak.futures_main_sina(symbol.upper()+"0",
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        df.columns = ["date","open","high","low","close","volume","oi","settle"]
        for c in ["open","high","low","close","volume","oi"]:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  Failed {symbol}: {e}"); return None

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Signal Generator
# ═══════════════════════════════════════════════════════════════════════════

def advanced_signal(df_window, ind, min_conditions=7,
                    require_regime=True, require_volume=False):
    """
    Multi-layer signal with regime filter and volume confirmation.

    Returns: dict with signal, confidence, and filter details.
    """
    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)
    regime = cycle_info["cycle"]

    # Regime filter: only trade with trend
    if require_regime:
        if regime not in ("BULL", "BEAR"):
            return {"signal": None, "reason": f"Regime={regime}, skip"}

    # MACD
    _, _, h0 = _calc_macd(closes)
    _, _, h1 = _calc_macd(closes[:-1]) if len(closes) > 1 else (0, 0, 0)
    _, _, h2 = _calc_macd(closes[:-2]) if len(closes) > 2 else (0, 0, 0)
    macd_improving = bool(h0 < 0 and h1 < 0 and abs(h0) < abs(h1) < abs(h2))

    adx = ind.get("adx14", 0); rsi = ind.get("rsi14", 50)
    ma5 = ind.get("ma5", 0); ma20 = ind.get("ma20", 0); ma60 = ind.get("ma60", 0)
    macd_h = ind.get("macd_hist", 0)

    # OI
    oi_col = "oi" if "oi" in df_window.columns else None
    oi = df_window[oi_col].values.astype(float) if oi_col else np.zeros(10)
    oi_3d = float(oi[-1]-oi[-4]) if len(oi)>=4 else 0
    oi_5d = float(oi[-1]-oi[-6]) if len(oi)>=6 else oi_3d
    if oi_3d>0 and oi_5d>0: oi_trend = "ACCUMULATING"
    elif oi_3d<0 and oi_5d<0: oi_trend = "REDUCING"
    else: oi_trend = "FLAT"

    ma_bull = ma5 > ma20 > ma60
    ma_bear = ma5 < ma20 < ma60

    # Volume confirmation
    vols = df_window["volume"].values.astype(float)
    vol_ma5 = np.mean(vols[-5:])
    vol_ma20 = np.mean(vols[-20:])
    vol_expanding = vol_ma5 > vol_ma20 * 1.1

    # Divergence check (simple: RSI vs price)
    price_5d_ago = closes[-6] if len(closes) > 5 else closes[0]
    rsi_now = rsi
    # We'd need RSI 5 days ago; approximate with indicator
    divergence = False  # Simplified — real impl would compare

    # ── Condition scoring ──
    short_conds = sum([
        regime == "BEAR",
        ma_bear,
        macd_h < 0 and not macd_improving,
        oi_trend in ("REDUCING", "FLAT"),
        adx > 20,
        32 < rsi < 72,
        not noise_info["is_noise"],
        True,
    ])
    long_conds = sum([
        regime == "BULL",
        ma_bull,
        macd_h > 0,
        oi_trend == "ACCUMULATING",
        adx > 22,
        30 < rsi < 65,
        not noise_info["is_noise"],
        True,
    ])

    # ── Signal with confidence ──
    # Base confidence from conditions met
    # Boost if volume expands (>10% above avg)
    # Boost if ADX is strong (>30)
    # Penalize if near noise period

    if short_conds >= min_conditions:
        base_conf = 0.50 + short_conds * 0.04
        if vol_expanding and require_volume: base_conf += 0.05
        if adx > 30: base_conf += 0.05
        if noise_info["is_noise"]: base_conf -= 0.10
        conf = min(0.90, max(0.30, base_conf))
        return {"signal": "SHORT", "confidence": conf, "conds": short_conds,
                "regime": regime, "vol_expanding": vol_expanding,
                "adx": adx, "noise": noise_info["is_noise"]}

    if long_conds >= min_conditions:
        base_conf = 0.50 + long_conds * 0.04
        if vol_expanding and require_volume: base_conf += 0.05
        if adx > 30: base_conf += 0.05
        if noise_info["is_noise"]: base_conf -= 0.10
        conf = min(0.85, max(0.30, base_conf))
        return {"signal": "LONG", "confidence": conf, "conds": long_conds,
                "regime": regime, "vol_expanding": vol_expanding,
                "adx": adx, "noise": noise_info["is_noise"]}

    return {"signal": None, "reason": f"Conditions: S={short_conds}/L={long_conds}/{min_conditions}"}

# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Position Sizer (Kelly-based)
# ═══════════════════════════════════════════════════════════════════════════

class DynamicSizer:
    """Kelly criterion with half-Kelly safety + streak adjustment."""

    def __init__(self, capital, lot_size, base_risk=0.02, max_risk=0.04):
        self.capital = capital
        self.equity = capital
        self.lot_size = lot_size
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.recent_trades = []  # last 20 PnLs
        self.streak = 0

    def update(self, pnl):
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)
        self.equity += pnl
        if pnl > 0:
            self.streak = max(0, self.streak + 1)
        else:
            self.streak = min(0, self.streak - 1)

    def get_risk_pct(self, confidence):
        """Dynamic risk: Kelly fraction adjusted by recent performance."""
        if len(self.recent_trades) < 5:
            return self.base_risk * confidence

        wins = [p for p in self.recent_trades if p > 0]
        losses = [p for p in self.recent_trades if p <= 0]
        win_rate = len(wins) / len(self.recent_trades)
        avg_win = np.mean(wins) if wins else 1
        avg_loss = abs(np.mean(losses)) if losses else 1
        rr = avg_win / (avg_loss + 1e-8)

        # Kelly fraction: f* = p - (1-p)/R
        kelly = win_rate - (1 - win_rate) / (rr + 1e-8)

        # Half-Kelly for safety
        risk = max(0.005, min(self.max_risk, kelly * 0.5))

        # Streak adjustment: reduce after 2+ losses, increase after 3+ wins
        if self.streak <= -2:
            risk *= 0.5
        elif self.streak >= 3:
            risk *= min(1.3, 1.0 + self.streak * 0.1)

        # Confidence scaling
        risk *= (0.5 + confidence * 0.5)

        return max(0.005, min(self.max_risk, risk))

    def calc_qty(self, stop_dist, confidence):
        risk_pct = self.get_risk_pct(confidence)
        risk_cash = self.equity * risk_pct
        qty = risk_cash / (stop_dist * self.lot_size + 1e-8)
        return round(max(1.0, min(20.0, qty)), 1), risk_pct

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Backtest
# ═══════════════════════════════════════════════════════════════════════════

def advanced_backtest(df, symbol, capital, params):
    """
    params: {min_conds, stop_atr, target_atr, require_regime, require_volume,
             use_dynamic_size, use_pyramiding, use_time_stop, time_stop_days}
    """
    trades = []
    pos = None
    WINDOW = 60
    lot_size = 16.0 if symbol.lower() == "lh" else 5.0
    commission = 0.0001
    slippage_bps = 2
    sizer = DynamicSizer(capital, lot_size)

    mc = params["min_conds"]
    sa = params["stop_atr"]
    ta = params["target_atr"]
    req_regime = params.get("require_regime", True)
    req_vol = params.get("require_volume", False)
    dyn_size = params.get("use_dynamic_size", True)
    pyramiding = params.get("use_pyramiding", False)
    time_stop = params.get("use_time_stop", True)
    ts_days = params.get("time_stop_days", 8)

    for i in range(WINDOW, len(df) - 1):
        today = df.iloc[i]
        date_str = str(today["date"])
        window = df.iloc[i - WINDOW: i + 1].copy()
        ind = calc_indicators(window)
        atr = ind["atr14"]
        close = float(today["close"])
        high_ = float(today["high"])
        low_ = float(today["low"])

        # ── Position Management ──
        if pos is not None:
            d = pos["direction"]; entry = pos["entry"]
            hold = i - pos["entry_idx"]
            gap_pct = abs(close - float(df.iloc[i-1]["close"])) / max(1, float(df.iloc[i-1]["close"]))
            force_exit = gap_pct > 0.025 or hold >= 20

            # Time stop: exit if no progress after N days
            time_exit = False
            if time_stop and hold >= ts_days:
                if d == "LONG" and close <= entry:
                    time_exit = True
                elif d == "SHORT" and close >= entry:
                    time_exit = True

            hit_stop = (d == "LONG" and low_ <= pos["stop"]) or \
                       (d == "SHORT" and high_ >= pos["stop"])
            hit_target = (d == "LONG" and high_ >= pos["target"]) or \
                         (d == "SHORT" and low_ <= pos["target"])

            # Adaptive trailing stop based on ATR
            if not (hit_stop or hit_target or time_exit):
                atr_now = atr
                if d == "LONG":
                    pos["stop"] = max(pos["stop"], close - 1.5 * atr_now)
                else:
                    pos["stop"] = min(pos["stop"], close + 1.5 * atr_now)

            if hit_stop or hit_target or force_exit or time_exit:
                exit_price = (pos["stop"] if hit_stop else
                             (pos["target"] if hit_target else close))
                reason = ("STOP" if hit_stop else ("TP" if hit_target else
                         ("GAP" if gap_pct>0.025 else
                          ("TIME" if time_exit else "MAX_HOLD"))))

                pnl = (exit_price - entry) * lot_size * pos["qty"] * \
                      (1 if d == "LONG" else -1)
                pnl -= abs(exit_price * lot_size * pos["qty"] * commission)
                pnl -= abs(entry * lot_size * pos["qty"] * commission)

                trades.append({
                    "symbol": symbol, "direction": d, "pnl": round(pnl, 2),
                    "reason": reason, "hold_days": hold,
                    "entry_date": pos["entry_date"], "exit_date": date_str,
                    "entry_price": round(entry, 2),
                    "exit_price": round(exit_price, 2),
                    "qty": pos["qty"], "risk_pct": pos.get("risk_pct", 0),
                })
                sizer.update(pnl)
                pos = None

                # Pyramiding: if TP hit, immediately look for re-entry
                if pyramiding and reason == "TP":
                    pass  # Continue to signal check below

        # ── Signal Generation ──
        if pos is None and i < len(df) - 2:
            sig = advanced_signal(window, ind, mc, req_regime, req_vol)
            if sig["signal"]:
                d = sig["signal"]
                conf = sig["confidence"]
                slippage = close * slippage_bps / 10000
                entry = close + slippage * (1 if d == "LONG" else -1)
                stop_dist = atr * sa
                target_dist = atr * ta

                if stop_dist < atr * 0.3:
                    stop_dist = atr * 0.5

                stop_price = entry - stop_dist if d == "LONG" else entry + stop_dist
                target_price = entry + target_dist if d == "LONG" else entry - target_dist

                if dyn_size:
                    qty, risk_pct = sizer.calc_qty(stop_dist, conf)
                else:
                    max_risk = capital * 0.02
                    qty = round(max(1.0, min(20.0, max_risk/(stop_dist*lot_size))), 1)
                    risk_pct = 0.02

                pos = {
                    "direction": d, "entry": round(entry, 2),
                    "stop": round(stop_price, 2), "target": round(target_price, 2),
                    "entry_date": date_str, "entry_idx": i, "qty": qty,
                    "risk_pct": risk_pct, "confidence": conf,
                }

    return trades

# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(trades, capital):
    if not trades: return {"trades":0,"win_rate":0,"total_pnl":0,"total_return":0,
        "sharpe":0,"max_dd":0,"pl_ratio":0,"profit_factor":0,"avg_hold":0,
        "monthly_trades":0,"calmar":0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    wr = len(wins)/n if n else 0
    aw = np.mean(wins) if wins else 0
    al = abs(np.mean(losses)) if losses else 1
    plr = aw/(al+1e-8)
    tp = sum(pnls)
    tr = tp/capital*100
    pf = sum(wins)/(abs(sum(losses))+1e-8) if wins else 0
    avg_hold = np.mean([t.get("hold_days",0) for t in trades]) if trades else 0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    dd = abs(float(np.min((cumulative-running_max)/(capital+1e-8)*100)))

    if n>=3:
        rets = [pnls[i]/capital for i in range(n)]
        sr = np.mean(rets)/(np.std(rets,ddof=1)+1e-8)*np.sqrt(252)
    else: sr=0

    calmar = tr/(dd+1e-8) if dd>0 else 0
    days = (datetime.strptime(trades[-1]["exit_date"],"%Y-%m-%d")-
            datetime.strptime(trades[0]["entry_date"],"%Y-%m-%d")).days if trades else 365
    monthly = n/(days/30.44) if days>0 else 0

    return {"trades":n,"win_rate":round(wr,3),"avg_win":round(aw,0),
        "avg_loss":round(al,0),"pl_ratio":round(plr,2),
        "total_pnl":round(tp,0),"total_return":round(tr,2),
        "profit_factor":round(pf,2),"max_dd":round(dd,2),
        "sharpe":round(sr,3),"calmar":round(calmar,2),
        "avg_hold":round(avg_hold,1),"monthly_trades":round(monthly,1)}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

print("="*70)
print("  先知期货 — 高级策略引擎 v2")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("="*70)

capital = 1_000_000

print("\n[1] 获取数据...")
lh = fetch_history("lh", 1200)
jd = fetch_history("jd", 1200)
for n,d in [("LH",lh),("JD",jd)]:
    if d is not None: print(f"  {n}: {len(d)}条, {d.iloc[0]['date']}→{d.iloc[-1]['date']}")

# Strategy configurations to test
configs = [
    # (name, min_conds, stop, target, regime, volume, dynamic, pyramiding, time_stop, ts_days)
    # Baseline (previous best)
    ("Baseline", 7, 1.5, 2.5, False, False, False, False, False, 0),
    # Regime filter only
    ("Regime", 7, 1.5, 2.5, True, False, False, False, False, 0),
    # Regime + Dynamic sizing
    ("Reg+Dyn", 7, 1.5, 2.5, True, False, True, False, False, 0),
    # Regime + Dynamic + Time stop
    ("Reg+Dyn+Time", 7, 1.5, 2.5, True, False, True, False, True, 8),
    # Regime + Dynamic + Tighter stop
    ("Reg+Dyn+Tight", 7, 1.0, 2.5, True, False, True, False, True, 7),
    # All features
    ("Full Stack", 7, 1.5, 2.5, True, True, True, True, True, 7),
    # Aggressive regime filter
    ("Aggressive", 6, 1.0, 2.5, True, True, True, False, True, 6),
    # Conservative regime filter
    ("Conservative", 8, 2.0, 2.0, True, False, True, False, True, 10),
    # Multi-symbol: LH+JD combined
]

print("\n[2] 策略对比测试 (LH 3年数据)...")
print(f"  {'策略':<18} {'交易':<6} {'月均':<6} {'胜率':<7} {'盈亏比':<7} {'PnL':<12} {'收益%':<8} {'回撤%':<7} {'夏普':<7} {'Calmar':<7}")
print(f"  {'─'*85}")

best_score = -999
best_config = None
all_results = []

for name, mc, sa, ta, rr, rv, ds, py, ts, tsd in configs:
    params = {"min_conds":mc, "stop_atr":sa, "target_atr":ta,
              "require_regime":rr, "require_volume":rv,
              "use_dynamic_size":ds, "use_pyramiding":py,
              "use_time_stop":ts, "time_stop_days":tsd}
    trades = advanced_backtest(lh, "lh", capital, params)
    stats = compute_stats(trades, capital)

    # Composite: prefer high return + high win rate, penalize drawdown
    score = stats["total_return"] * stats["win_rate"] / (stats["max_dd"]/100 + 0.1)
    stats["_score"] = score
    all_results.append((name, params, stats))

    print(f"  {name:<18} {stats['trades']:<6} {stats['monthly_trades']:<6.1f} "
          f"{stats['win_rate']:.0%}     {stats['pl_ratio']:<7.2f} "
          f"{stats['total_pnl']:+,.0f}     {stats['total_return']:+.1f}%    "
          f"{stats['max_dd']:.1f}%    {stats['sharpe']:.3f}    {stats['calmar']:.2f}")

    if score > best_score:
        best_score = score
        best_config = (name, params, stats, trades)

# Also run on JD
print(f"\n[3] JD(鸡蛋) 最优策略验证...")
if jd is not None:
    _, bparams, _, _ = best_config
    jd_trades = advanced_backtest(jd, "jd", capital, bparams)
    jd_stats = compute_stats(jd_trades, capital)
    print(f"  JD: {jd_stats['trades']}笔, 胜率{jd_stats['win_rate']:.0%}, "
          f"PnL {jd_stats['total_pnl']:+,.0f}, 收益{jd_stats['total_return']:+.1f}%")

    # Combined LH+JD
    combined = lh_trades = best_config[3]
    combined_stats = compute_stats(combined + jd_trades, capital)
    print(f"  LH+JD合计: {combined_stats['trades']}笔, 胜率{combined_stats['win_rate']:.0%}, "
          f"PnL {combined_stats['total_pnl']:+,.0f}, 收益{combined_stats['total_return']:+.1f}%")

# Summary
print(f"\n{'='*70}")
print(f"  最优策略: {best_config[0]}")
print(f"{'='*70}")
bs = best_config[2]
print(f"  3年交易: {bs['trades']}笔 (月均{bs['monthly_trades']:.1f}笔)")
print(f"  胜率: {bs['win_rate']:.0%}  盈亏比: {bs['pl_ratio']:.2f}")
print(f"  总收益: {bs['total_pnl']:+,.0f}元 ({bs['total_return']:+.1f}%)")
print(f"  年化: ~{bs['total_return']/3:.1f}%  最大回撤: {bs['max_dd']:.1f}%")
print(f"  夏普: {bs['sharpe']:.3f}  Calmar: {bs['calmar']:.2f}")
print(f"  平均持仓: {bs['avg_hold']:.1f}天  盈利因子: {bs['profit_factor']:.2f}")

# Save
json.dump({"best": best_config[0], "stats": bs,
           "all": [{"name":n, "stats":s} for n,p,s in all_results]},
          open("/tmp/advanced_backtest.json","w"), indent=2, ensure_ascii=False)
print(f"\n结果已保存")
