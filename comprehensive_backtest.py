#!/usr/bin/env python3
"""
Comprehensive backtest + walk-forward optimization for LH futures.
3-year data, parameter sweep, statistical validation.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tools.indicators import calc_indicators, _calc_macd
from tools.cycle_detector import detect_cycle, detect_rollover_noise

# ── Data Fetching ──────────────────────────────────────────────────────────

def fetch_history(symbol, days_back):
    """Fetch futures main contract history."""
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=days_back + 150)
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
        print(f"  Failed to fetch {symbol}: {e}")
        return None

# ── Signal Generator ──────────────────────────────────────────────────────

def generate_signal(df_window, ind, min_conditions=7):
    """Relaxed LH signal with configurable strictness."""
    closes = df_window["close"].values.astype(float)
    cycle_info = detect_cycle(df_window)
    noise_info = detect_rollover_noise(df_window)

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
    oi_trend = "ACCUMULATING" if (oi_3d > 0 and oi_5d > 0) else \
               ("REDUCING" if (oi_3d < 0 and oi_5d < 0) else "FLAT")

    ma_bull = ma5 > ma20 > ma60
    ma_bear = ma5 < ma20 < ma60

    # Count conditions met
    short_conds = sum([
        cycle_info["cycle"] == "BEAR",
        ma_bear,
        macd_h < 0 and not macd_improving,
        oi_trend in ("REDUCING", "FLAT"),
        adx > 20,
        32 < rsi < 72,
        not noise_info["is_noise"],
        True,  # fundamental (always pass in relaxed)
    ])
    long_conds = sum([
        cycle_info["cycle"] == "BULL",
        ma_bull,
        macd_h > 0,
        oi_trend == "ACCUMULATING",
        adx > 22,
        30 < rsi < 65,
        not noise_info["is_noise"],
        True,
    ])

    if short_conds >= min_conditions:
        return {"signal": "SHORT", "confidence": min(0.85, 0.50 + short_conds * 0.05),
                "conds_met": short_conds, "cycle": cycle_info["cycle"]}
    if long_conds >= min_conditions:
        return {"signal": "LONG", "confidence": min(0.80, 0.50 + long_conds * 0.05),
                "conds_met": long_conds, "cycle": cycle_info["cycle"]}
    return None

# ── Single Backtest Run ───────────────────────────────────────────────────

def backtest_symbol(df, symbol, capital, stop_atr, target_atr, min_conds):
    """Run backtest with given parameters. Returns trades list."""
    trades = []
    pos = None
    WINDOW = 60
    lot_size = 16.0 if symbol.lower() == "lh" else 5.0
    commission = 0.0001  # 0.01% per side
    slippage_bps = 2  # 2 bps

    for i in range(WINDOW, len(df) - 1):
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]
        date_str = str(today["date"])
        window = df.iloc[i - WINDOW: i + 1].copy()
        ind = calc_indicators(window)
        atr = ind["atr14"]
        close = float(today["close"])
        high_ = float(today["high"])
        low_ = float(today["low"])

        # ── Position management ──
        if pos is not None:
            d = pos["direction"]
            entry = pos["entry"]
            hold = i - pos["entry_idx"]
            gap_pct = abs(close - float(df.iloc[i - 1]["close"])) / max(1, float(df.iloc[i - 1]["close"]))
            force_exit = gap_pct > 0.025 or hold >= 20

            hit_stop = (d == "LONG" and low_ <= pos["stop"]) or \
                       (d == "SHORT" and high_ >= pos["stop"])
            hit_target = (d == "LONG" and high_ >= pos["target"]) or \
                         (d == "SHORT" and low_ <= pos["target"])

            # Trailing stop
            if not (hit_stop or hit_target):
                if d == "LONG":
                    pos["stop"] = max(pos["stop"], close - 2.0 * atr)
                else:
                    pos["stop"] = min(pos["stop"], close + 2.0 * atr)

            if hit_stop or hit_target or force_exit:
                exit_price = pos["stop"] if hit_stop else \
                             (pos["target"] if hit_target else close)
                reason = "STOP" if hit_stop else ("TP" if hit_target else
                         ("GAP" if gap_pct > 0.025 else "MAX_HOLD"))

                pnl = (exit_price - entry) * lot_size * pos["qty"] * \
                      (1 if d == "LONG" else -1)
                pnl -= abs(exit_price * lot_size * pos["qty"] * commission)
                pnl -= abs(entry * lot_size * pos["qty"] * commission)  # entry commission

                trades.append({
                    "symbol": symbol, "direction": d,
                    "entry_date": pos["entry_date"], "exit_date": date_str,
                    "entry_price": round(entry, 2), "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2), "reason": reason, "hold_days": hold,
                    "pnl_pct": round(pnl / capital * 100, 2),
                })
                pos = None

        # ── Signal generation ──
        if pos is None and i < len(df) - 2:
            sig = generate_signal(window, ind, min_conds)
            if sig:
                d = sig["signal"]
                slippage = close * slippage_bps / 10000
                entry = close + slippage * (1 if d == "LONG" else -1)
                stop_dist = atr * stop_atr
                target_dist = atr * target_atr

                if stop_dist < atr * 0.5:
                    stop_dist = atr

                stop_price = entry - stop_dist if d == "LONG" else entry + stop_dist
                target_price = entry + target_dist if d == "LONG" else entry - target_dist

                max_risk = capital * 0.02
                qty = round(max(1.0, min(20.0, max_risk / (stop_dist * lot_size))), 1)

                pos = {
                    "direction": d, "entry": round(entry, 2),
                    "stop": round(stop_price, 2), "target": round(target_price, 2),
                    "entry_date": date_str, "entry_idx": i, "qty": qty,
                }

    return trades

# ── Stats ─────────────────────────────────────────────────────────────────

def compute_stats(trades, capital):
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0, "sharpe": 0,
                "max_dd": 0, "pl_ratio": 0, "profit_factor": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(pnls)
    win_rate = len(wins) / n
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    pl_ratio = avg_win / (avg_loss + 1e-8)
    total_pnl = sum(pnls)
    total_return = total_pnl / capital * 100

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / (gross_loss + 1e-8)

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / (capital + 1e-8) * 100
    max_dd = abs(float(np.min(drawdowns)))

    # Sharpe (simplified)
    if n >= 3:
        returns = [pnls[i] / capital for i in range(n)]
        std_ret = np.std(returns, ddof=1)
        sharpe = (np.mean(returns) / (std_ret + 1e-8)) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        "trades": n, "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0),
        "pl_ratio": round(pl_ratio, 2),
        "total_pnl": round(total_pnl, 0),
        "total_return": round(total_return, 2),
        "profit_factor": round(profit_factor, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "monthly_return": round(total_return / (n * 5 / 252) if n > 0 else 0, 2),
    }

# ── Walk-Forward ──────────────────────────────────────────────────────────

def walk_forward(df, symbol, capital, n_splits=3):
    """Walk-forward: train on first N splits, test on next."""
    total_len = len(df)
    split_size = total_len // (n_splits + 1)

    print(f"\n  Walk-Forward ({n_splits} splits, ~{split_size} days each):")
    print(f"  {'Split':<8} {'Period':<25} {'Trades':<8} {'Win%':<8} {'PnL':<12} {'Return%':<10} {'MaxDD%':<10}")

    all_test_trades = []
    for split in range(n_splits):
        train_end = split_size * (split + 1)
        test_start = train_end
        test_end = min(test_start + split_size, total_len)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(test_df) < 60:
            continue

        # Optimize on training data (simplified: test a few combos)
        best_score = -999
        best_params = (7, 1.0, 2.5)
        for mc in [5, 6, 7]:
            for sa in [1.0, 1.5, 2.0]:
                for ta in [2.0, 2.5, 3.0]:
                    tr = backtest_symbol(train_df, symbol, capital, sa, ta, mc)
                    if tr:
                        s = compute_stats(tr, capital)
                        score = s["total_return"] * s["win_rate"]
                        if score > best_score:
                            best_score = score
                            best_params = (mc, sa, ta)

        # Test on out-of-sample
        mc, sa, ta = best_params
        test_trades = backtest_symbol(test_df, symbol, capital, sa, ta, mc)
        stats = compute_stats(test_trades, capital)
        all_test_trades.extend(test_trades)

        period = f"{test_df.iloc[0]['date']}→{test_df.iloc[-1]['date']}"
        print(f"  {split+1:<8} {period:<25} {stats['trades']:<8} "
              f"{stats['win_rate']:.0%}     {stats['total_pnl']:+,.0f}     "
              f"{stats['total_return']:+.2f}%     {stats['max_dd']:.1f}%")

    # Aggregate all test trades
    if all_test_trades:
        agg = compute_stats(all_test_trades, capital)
        print(f"  {'TOTAL':<8} {'All test periods':<25} {agg['trades']:<8} "
              f"{agg['win_rate']:.0%}     {agg['total_pnl']:+,.0f}     "
              f"{agg['total_return']:+.2f}%     {agg['max_dd']:.1f}%")
        return agg
    return compute_stats([], capital)

# ── MAIN ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("  先知期货认知交易系统 — 综合回测优化")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

capital = 1_000_000

# 1. Fetch 3+ years of data
print("\n[1] 获取数据...")
lh_df = fetch_history("lh", 1200)  # ~3.3 years
jd_df = fetch_history("jd", 1200)

for name, df in [("LH(生猪)", lh_df), ("JD(鸡蛋)", jd_df)]:
    if df is not None:
        print(f"  {name}: {len(df)} 条, {df.iloc[0]['date']} → {df.iloc[-1]['date']}")

# 2. Full parameter sweep on LH (3-year)
print("\n[2] 参数网格搜索（LH, 3年数据）...")
results = []
for mc in [5, 6, 7, 8]:
    for sa in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for ta in [1.5, 2.0, 2.5, 3.0]:
            trades = backtest_symbol(lh_df, "lh", capital, sa, ta, mc)
            stats = compute_stats(trades, capital)
            # Composite score: reward return, win rate, and trade count
            score = stats["total_return"] * stats["win_rate"] * min(stats["trades"], 30)
            stats["min_conds"] = mc; stats["stop_atr"] = sa; stats["target_atr"] = ta
            stats["score"] = round(score, 1)
            results.append(stats)

results.sort(key=lambda x: x["score"], reverse=True)

print(f"\n  Top 15 configurations (of {len(results)}):")
print(f"  {'Rank':<5} {'MC':<4} {'Stop':<7} {'Tgt':<6} {'Trades':<7} {'Win%':<7} {'PnL':<12} {'Ret%':<8} {'MaxDD%':<8} {'Sharpe':<8}")
print(f"  {'─'*75}")
for i, r in enumerate(results[:15]):
    print(f"  {i+1:<5} {r['min_conds']:<4} {r['stop_atr']:<7} {r['target_atr']:<6} "
          f"{r['trades']:<7} {r['win_rate']:.0%}     {r['total_pnl']:+,.0f}     "
          f"{r['total_return']:+.2f}%    {r['max_dd']:.1f}%     {r['sharpe']:.3f}")

# 3. Walk-forward validation
print("\n[3] Walk-Forward 步进验证...")
wf_lh = walk_forward(lh_df, "lh", capital, n_splits=3)
if jd_df is not None:
    wf_jd = walk_forward(jd_df, "jd", capital, n_splits=3)

# 4. Best config summary
best = results[0]
print(f"\n{'='*70}")
print(f"  最优配置")
print(f"{'='*70}")
print(f"  品种: LH(生猪)")
print(f"  条件严格度: {best['min_conds']}/8")
print(f"  止损: {best['stop_atr']}xATR  止盈: {best['target_atr']}xATR")
print(f"  3年交易: {best['trades']} 笔 (月均 {best['trades']/36:.1f} 笔)")
print(f"  胜率: {best['win_rate']:.0%}")
print(f"  盈亏比: {best['pl_ratio']:.2f}")
print(f"  盈利因子: {best['profit_factor']:.2f}")
print(f"  总收益: {best['total_pnl']:+,.0f} 元 ({best['total_return']:+.2f}%)")
print(f"  年化收益: ~{best['total_return']/3:.1f}%")
print(f"  最大回撤: {best['max_dd']:.1f}%")
print(f"  夏普比率: {best['sharpe']:.3f}")

# 5. Walk-forward aggregate
print(f"\n  Walk-Forward 汇总:")
wf_trades = wf_lh.get("trades", 0)
print(f"  步进验证交易: {wf_trades} 笔")
print(f"  步进胜率: {wf_lh.get('win_rate', 0):.0%}")
print(f"  步进收益: {wf_lh.get('total_return', 0):+.2f}%")

# Save
with open("/tmp/backtest_final.json", "w") as f:
    json.dump({"top_configs": results[:20], "walk_forward_lh": wf_lh}, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 /tmp/backtest_final.json")
