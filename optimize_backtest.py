#!/usr/bin/env python3
"""Backtest optimizer — grid search over signal strictness + risk params for LH."""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
from datetime import datetime, timedelta
from tools.backtest import run_backtest as bt
from tools.cycle_detector import get_lh_signal_conditions, detect_cycle, detect_rollover_noise
from tools.indicators import calc_indicators, _calc_macd
import pandas as pd
import akshare as ak

def fetch_data(symbol, days_back):
    end = datetime.now()
    start = end - timedelta(days=days_back + 130)
    try:
        df = ak.futures_main_sina(
            symbol=symbol.upper() + "0",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        df.columns = ["date", "open", "high", "low", "close", "volume", "oi", "settle"]
        for c in ["open", "high", "low", "close", "volume", "oi"]:
            df[c] = df[c].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Data fetch failed: {e}")
        return None

def relaxed_signal(df_window, ind, min_conditions=6):
    """Relaxed version: only need min_conditions/8 met."""
    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)
    cycle = cycle_info["cycle"]

    _, _, h0 = _calc_macd(closes)
    _, _, h1 = _calc_macd(closes[:-1]) if len(closes) > 1 else (0, 0, 0)
    _, _, h2 = _calc_macd(closes[:-2]) if len(closes) > 2 else (0, 0, 0)
    macd_improving = bool(h0 < 0 and h1 < 0 and abs(h0) < abs(h1) < abs(h2))

    adx = ind.get("adx14", 0); rsi = ind.get("rsi14", 50)
    ma5 = ind.get("ma5", 0); ma20 = ind.get("ma20", 0); ma60 = ind.get("ma60", 0)
    macd_h = ind.get("macd_hist", 0)

    oi_col = "oi" if "oi" in df_window.columns else None
    oi = df_window[oi_col].values.astype(float) if oi_col else np.zeros(10)
    oi_3d = float(oi[-1] - oi[-4]) if len(oi) >= 4 else 0
    oi_5d = float(oi[-1] - oi[-6]) if len(oi) >= 6 else oi_3d
    if oi_3d > 0 and oi_5d > 0: oi_trend = "ACCUMULATING"
    elif oi_3d < 0 and oi_5d < 0: oi_trend = "REDUCING"
    else: oi_trend = "FLAT"

    ma_bull = ma5 > ma20 > ma60
    ma_bear = ma5 < ma20 < ma60

    # Count conditions met for SHORT
    short_conds = sum([
        cycle == "BEAR",
        ma_bear,
        macd_h < 0 and not macd_improving,
        oi_trend in ("REDUCING", "FLAT"),
        adx > 20,
        32 < rsi < 72,
        not noise_info["is_noise"],
        True,  # fundamental condition (always pass in relaxed mode)
    ])

    # Count conditions met for LONG
    long_conds = sum([
        cycle == "BULL",
        ma_bull,
        macd_h > 0,
        oi_trend == "ACCUMULATING",
        adx > 22,
        30 < rsi < 65,
        not noise_info["is_noise"],
        True,  # fundamental condition
    ])

    if short_conds >= min_conditions:
        conf = min(0.85, 0.50 + short_conds * 0.05)
        return {"signal": "SHORT", "confidence": conf, "stop_atr_mult": 1.5,
                "target_atr_mult": 2.5, "hold_days": 5, "conditions_met": short_conds,
                "reasoning": f"Relaxed SHORT ({short_conds}/8)"}
    elif long_conds >= min_conditions:
        conf = min(0.80, 0.50 + long_conds * 0.05)
        return {"signal": "LONG", "confidence": conf, "stop_atr_mult": 1.5,
                "target_atr_mult": 2.5, "hold_days": 5, "conditions_met": long_conds,
                "reasoning": f"Relaxed LONG ({long_conds}/8)"}
    return None

def run_optimized_backtest(df, symbol, capital, stop_atr, target_atr, min_conds):
    """Single backtest run with given params."""
    from tools.backtest import _backtest_symbol as orig_bt
    # Use original backtest but with relaxed signal
    trades = []
    pos = None
    WINDOW = 60
    lot_size = 16.0  # LH
    
    for i in range(WINDOW, len(df) - 1):
        today = df.iloc[i]
        date_str = str(today["date"])
        window = df.iloc[i - WINDOW: i + 1].copy()
        ind = calc_indicators(window)
        atr = ind["atr14"]
        close = float(today["close"])
        high_ = float(today["high"])
        low_ = float(today["low"])

        # Position management (simplified)
        if pos is not None:
            d = pos["direction"]
            entry = pos["entry"]
            hold = i - pos["entry_idx"]
            gap_pct = abs(close - float(df.iloc[i - 1]["close"])) / float(df.iloc[i - 1]["close"])
            force_exit = gap_pct > 0.025 or hold >= 20
            hit_stop = (d == "LONG" and low_ <= pos["stop"]) or (d == "SHORT" and high_ >= pos["stop"])
            hit_target = (d == "LONG" and high_ >= pos["target"]) or (d == "SHORT" and low_ <= pos["target"])

            if hit_stop or hit_target or force_exit:
                exit_price = pos["stop"] if hit_stop else (pos["target"] if hit_target else close)
                reason = "STOP" if hit_stop else ("TP" if hit_target else "GAP" if gap_pct > 0.025 else "MAX_HOLD")
                pnl = (exit_price - entry) * lot_size * pos["qty"] * (1 if d == "LONG" else -1)
                pnl -= abs(exit_price * lot_size * pos["qty"] * 0.0001)
                trades.append({"pnl": pnl, "reason": reason, "direction": d,
                               "entry_date": pos["entry_date"], "exit_date": date_str,
                               "symbol": symbol, "hold_days": hold})
                pos = None

        # Signal generation
        if pos is None and i < len(df) - 2:
            sig = relaxed_signal(window, ind, min_conds)
            if sig:
                d = sig["signal"]
                slippage = close * 0.0002
                entry = close + slippage * (1 if d == "LONG" else -1)
                stop_dist = atr * stop_atr
                target_dist = atr * target_atr
                stop_price = entry - stop_dist if d == "LONG" else entry + stop_dist
                target_price = entry + target_dist if d == "LONG" else entry - target_dist
                max_risk = capital * 0.02
                qty = round(max(1.0, min(20.0, max_risk / (stop_dist * lot_size))), 1)
                pos = {"direction": d, "entry": round(entry, 2), "stop": round(stop_price, 2),
                       "target": round(target_price, 2), "entry_date": date_str, "entry_idx": i,
                       "qty": qty}

    return trades

# Grid search
print("Running optimization grid search for LH...")
print("=" * 60)

df = fetch_data("lh", 365)
if df is None:
    print("Failed to fetch data")
    sys.exit(1)

capital = 1_000_000
results = []

for min_conds in [5, 6, 7]:
    for stop_atr in [1.0, 1.5, 2.0]:
        for target_atr in [2.0, 2.5, 3.0]:
            trades = run_optimized_backtest(df, "lh", capital, stop_atr, target_atr, min_conds)
            if not trades:
                continue
            pnls = [t["pnl"] for t in trades]
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls)
            total_pnl = sum(pnls)
            total_return = total_pnl / capital * 100
            avg_win = np.mean(wins) if wins else 0
            losses = [p for p in pnls if p <= 0]
            avg_loss = abs(np.mean(losses)) if losses else 1
            pl_ratio = avg_win / (avg_loss + 1e-8)
            
            # Composite score: prefer high return + high win rate + more trades
            score = total_return * win_rate * min(len(trades), 10)
            
            results.append({
                "min_conds": min_conds, "stop_atr": stop_atr, "target_atr": target_atr,
                "trades": len(trades), "win_rate": win_rate, "total_pnl": total_pnl,
                "total_return": total_return, "pl_ratio": pl_ratio, "score": score,
            })

# Sort by score descending
results.sort(key=lambda x: x["score"], reverse=True)

print(f"\nTop 10 configurations by composite score:")
print(f"{'Rank':<5} {'MinConds':<10} {'Stop':<8} {'Target':<10} {'Trades':<8} {'Win%':<8} {'PnL':<12} {'Return%':<10} {'PL Ratio':<10}")
print("-" * 85)
for i, r in enumerate(results[:10]):
    print(f"{i+1:<5} {r['min_conds']:<10} {r['stop_atr']:<8} {r['target_atr']:<10} "
          f"{r['trades']:<8} {r['win_rate']:.0%}     {r['total_pnl']:+,.0f}     {r['total_return']:+.2f}%     {r['pl_ratio']:.2f}")

# Print best config details
best = results[0]
print(f"\nBest config: min_conditions={best['min_conds']}, stop={best['stop_atr']}xATR, target={best['target_atr']}xATR")
print(f"Trades: {best['trades']}, Win rate: {best['win_rate']:.0%}, Total PnL: {best['total_pnl']:+,.0f} CNY")
print(f"Return: {best['total_return']:+.2f}%, PL Ratio: {best['pl_ratio']:.2f}")

# Save all results
with open("/tmp/backtest_optimization.json", "w") as f:
    json.dump(results[:30], f, indent=2, ensure_ascii=False)
print("\nAll results saved to /tmp/backtest_optimization.json")
