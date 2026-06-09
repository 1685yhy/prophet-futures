"""
事件驱动回测引擎 — 基于真实历史行情逐日运行规则信号并模拟交易。

设计原则：
- 不调用 LLM（速度快，结果可复现）
- 信号来自指标规则 + DCS 融合（与实盘 Commander 逻辑一致）
- 每日收盘生成信号，次日开盘按滑点入场
- 记录每笔交易明细，输出完整绩效统计
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from tools.indicators import calc_indicators, adx_regime
from agents.commander import _compute_dcs, DCS_THRESHOLD, DIR_SCORE

logger = logging.getLogger(__name__)

SLIPPAGE_BPS = 2       # 滑点，基点
COMMISSION   = 3.0     # 手续费，元/手
ATR_STOP_MULT= 1.5     # 止损 ATR 倍数
ATR_TARGET_MULT= 3.0   # 目标 ATR 倍数
MAX_POSITIONS= 1       # 每品种同时最多持仓数


# ── 真实数据获取 ─────────────────────────────────────────────────────────────

def _fetch_history(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        import akshare as ak
        df = ak.futures_main_sina(
            symbol=symbol.upper() + "0",
            start_date=start_date,
            end_date=end_date,
        )
        df.columns = ["date", "open", "high", "low", "close", "volume", "oi", "settle"]
        for col in ["open", "high", "low", "close", "volume", "oi"]:
            df[col] = df[col].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("Failed to fetch history for %s: %s", symbol, e)
        return None


# ── 轻量信号生成（不调 LLM）──────────────────────────────────────────────────

def _generate_signal(window_df: pd.DataFrame) -> Dict[str, Any]:
    """
    纯规则信号，与 Commander DCS 逻辑对齐：
    Technician: MACD方向 + RSI + 均线
    Fund:       OI变化方向
    Macro:      NEUTRAL（回测中无实时宏观，保守处理）
    Vision:     NEUTRAL（无图像，保守处理）
    """
    ind = calc_indicators(window_df)

    # ── Technician 信号 ──
    macd_dir = 1 if ind["macd_hist"] > 0 else -1
    rsi      = ind["rsi14"]
    ma_bull  = ind["ma5"] > ind["ma20"] > ind["ma60"]
    ma_bear  = ind["ma5"] < ind["ma20"] < ind["ma60"]

    if macd_dir > 0 and rsi < 70 and ma_bull:
        tech_dir, tech_conf = 1, 0.65
    elif macd_dir < 0 and rsi > 30 and ma_bear:
        tech_dir, tech_conf = -1, 0.65
    elif macd_dir > 0 and rsi < 65:
        tech_dir, tech_conf = 1, 0.45
    elif macd_dir < 0 and rsi > 35:
        tech_dir, tech_conf = -1, 0.45
    else:
        tech_dir, tech_conf = 0, 0.30

    # ── Fund 信号（OI 变化）──
    oi_vals = window_df["oi"].values if "oi" in window_df.columns else None
    if oi_vals is not None and len(oi_vals) >= 3:
        oi_chg = (oi_vals[-1] - oi_vals[-3]) / (oi_vals[-3] + 1e-8)
        if oi_chg > 0.03:
            fund_dir, fund_conf = tech_dir, 0.55   # 持仓增加，增强方向
        elif oi_chg < -0.03:
            fund_dir, fund_conf = -tech_dir, 0.50  # 持仓减少，弱化方向
        else:
            fund_dir, fund_conf = 0, 0.40
    else:
        fund_dir, fund_conf = 0, 0.40

    # ── 拥挤度简化检查 ──
    rsi_extreme = rsi > 78 or rsi < 22
    crowding_veto = rsi_extreme  # 极度超买超卖时不开新仓

    regime = adx_regime(ind["adx14"])
    w = ({"tech":0.40,"fund":0.30,"macro":0.25,"vision":0.05}
         if regime == "TRENDING"
         else {"tech":0.25,"fund":0.40,"macro":0.25,"vision":0.10})

    signals = [
        (float(tech_dir), tech_conf, w["tech"]),
        (float(fund_dir), fund_conf, w["fund"]),
        (0.0, 0.40, w["macro"]),
        (0.0, 0.40, w["vision"]),
    ]

    dcs, agreement, _ = _compute_dcs(signals)
    threshold = DCS_THRESHOLD.get("TRENDING" if regime == "TRENDING" else "RANGING", 0.25)

    if crowding_veto or abs(dcs) < threshold or agreement < 0.60:
        direction = "WAIT"
    elif dcs > 0:
        direction = "LONG"
    else:
        direction = "SHORT"

    return {
        "direction":  direction,
        "dcs":        round(dcs, 3),
        "agreement":  round(agreement, 2),
        "atr":        ind["atr14"],
        "close":      ind["current_close"],
        "adx":        ind["adx14"],
        "rsi":        rsi,
    }


# ── 主回测函数 ───────────────────────────────────────────────────────────────

def run_backtest(
    date: str,
    symbols: Optional[List[str]] = None,
    backtest_days: int = 180,
    capital: float = 1_000_000.0,
    system=None,
) -> Dict[str, Any]:
    """
    事件驱动回测。

    Args:
        date:          回测结束日期 'YYYY-MM-DD'
        symbols:       品种列表，默认 ['rb','i','sc','lh','jd']
        backtest_days: 回测天数
        capital:       初始资金
        system:        占位参数（保持接口兼容，当前未使用）

    Returns:
        包含每笔交易明细和绩效统计的字典
    """
    if symbols is None:
        symbols = ["rb", "i", "sc", "lh", "jd"]

    try:
        end_dt   = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: {date}, expected YYYY-MM-DD"}

    start_dt = end_dt - timedelta(days=backtest_days + 120)  # 额外 120 天作为指标预热
    start_str= start_dt.strftime("%Y%m%d")
    end_str  = end_dt.strftime("%Y%m%d")

    all_trades: List[Dict] = []
    daily_pnl: Dict[str, float] = {}

    for symbol in symbols:
        logger.info("Backtesting %s (%s → %s)...", symbol, start_str, end_str)
        df = _fetch_history(symbol, start_str, end_str)
        if df is None or len(df) < 70:
            logger.warning("Insufficient data for %s", symbol)
            continue

        trades = _backtest_symbol(df, symbol, capital / len(symbols))
        all_trades.extend(trades)

        for t in trades:
            d = t["exit_date"]
            daily_pnl[d] = daily_pnl.get(d, 0.0) + t["pnl"]

    return _compute_stats(all_trades, daily_pnl, capital, date, symbols)


def _backtest_symbol(df: pd.DataFrame, symbol: str, capital: float) -> List[Dict]:
    """逐日回测单个品种，返回成交记录列表。"""
    trades = []
    position = None  # {"direction", "entry", "stop", "target", "entry_date", "qty"}
    WINDOW   = 60

    for i in range(WINDOW, len(df) - 1):
        today   = df.iloc[i]
        tomorrow= df.iloc[i + 1]
        date_str= str(today["date"])

        # ── 检查持仓止损/止盈 ──
        if position is not None:
            open_p = float(tomorrow["open"])
            hit_stop   = (position["direction"] == "LONG"  and open_p <= position["stop"]) or \
                         (position["direction"] == "SHORT" and open_p >= position["stop"])
            hit_target = (position["direction"] == "LONG"  and open_p >= position["target"]) or \
                         (position["direction"] == "SHORT" and open_p <= position["target"])

            exit_reason = None
            exit_price  = open_p

            if hit_stop:
                exit_price  = position["stop"]
                exit_reason = "stop_loss"
            elif hit_target:
                exit_price  = position["target"]
                exit_reason = "take_profit"

            if exit_reason:
                qty  = position["qty"]
                pnl  = (exit_price - position["entry"]) * qty * (1 if position["direction"] == "LONG" else -1)
                pnl -= COMMISSION * qty * 2  # 开平各一次

                trades.append({
                    "symbol":     symbol,
                    "direction":  position["direction"],
                    "entry_date": position["entry_date"],
                    "exit_date":  str(tomorrow["date"]),
                    "entry_price":round(position["entry"], 2),
                    "exit_price": round(exit_price, 2),
                    "qty":        qty,
                    "pnl":        round(pnl, 2),
                    "reason":     exit_reason,
                    "dcs":        position.get("dcs", 0),
                })
                position = None

        # ── 无持仓时生成信号 ──
        if position is None:
            window_df = df.iloc[i - WINDOW: i].copy()
            sig = _generate_signal(window_df)

            if sig["direction"] in ("LONG", "SHORT"):
                atr        = sig["atr"]
                close      = sig["close"]
                slippage   = close * SLIPPAGE_BPS / 10000
                entry      = close + slippage * (1 if sig["direction"] == "LONG" else -1)
                stop       = entry - ATR_STOP_MULT * atr  if sig["direction"] == "LONG" \
                             else entry + ATR_STOP_MULT * atr
                target     = entry + ATR_TARGET_MULT * atr if sig["direction"] == "LONG" \
                             else entry - ATR_TARGET_MULT * atr
                stop_dist  = abs(entry - stop)
                max_risk   = capital * 0.02
                qty        = round(max(1.0, min(10.0, max_risk / (stop_dist + 1e-8))), 1)

                position = {
                    "direction":  sig["direction"],
                    "entry":      round(entry, 2),
                    "stop":       round(stop, 2),
                    "target":     round(target, 2),
                    "entry_date": date_str,
                    "qty":        qty,
                    "dcs":        sig["dcs"],
                }

    return trades


def _compute_stats(
    trades: List[Dict],
    daily_pnl: Dict[str, float],
    capital: float,
    date: str,
    symbols: List[str],
) -> Dict[str, Any]:
    """从交易记录计算完整绩效统计。"""
    if not trades:
        return {
            "date_range": date, "symbols": symbols,
            "total_trades": 0, "pnl": 0.0,
            "win_rate": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0, "total_pnl": 0.0,
            "equity_curve": [capital],
            "trades": [],
            "message": "No trades generated",
        }

    pnls      = [t["pnl"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    win_rate  = len(wins) / len(pnls)
    avg_win   = float(np.mean(wins))  if wins   else 0.0
    avg_loss  = abs(float(np.mean(losses))) if losses else 1.0
    pl_ratio  = avg_win / (avg_loss + 1e-8)

    # 日度收益率序列
    sorted_dates  = sorted(daily_pnl.keys())
    daily_returns = [daily_pnl[d] / capital for d in sorted_dates]
    sharpe        = calculate_sharpe(daily_returns)

    # 权益曲线
    equity = [capital]
    for ret in daily_returns:
        equity.append(equity[-1] * (1 + ret))
    dd_stats  = drawdown_analysis(equity)
    total_pnl = sum(pnls)

    return {
        "date_range":      date,
        "symbols":         symbols,
        "total_trades":    len(trades),
        "win_rate":        round(win_rate, 3),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "pl_ratio":        round(pl_ratio, 3),
        "sharpe_ratio":    round(sharpe, 3),
        "max_drawdown_pct":dd_stats["max_drawdown_pct"],
        "calmar_ratio":    dd_stats["calmar_ratio"],
        "total_pnl":       round(total_pnl, 2),
        "total_return_pct":round(total_pnl / capital * 100, 2),
        "equity_curve":    [round(e, 2) for e in equity],
        "trades":          trades,
    }


# ── 统计工具函数 ─────────────────────────────────────────────────────────────

def calculate_sharpe(returns: List[float], risk_free: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    arr      = np.array(returns)
    daily_rf = risk_free / 252
    excess   = arr - daily_rf
    std      = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0


def drawdown_analysis(equity_curve: List[float]) -> Dict[str, Any]:
    if len(equity_curve) < 2:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0,
                "current_drawdown": 0.0, "calmar_ratio": 0.0}
    equity      = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown    = (equity - running_max) / (running_max + 1e-8)
    max_dd      = float(drawdown.min())
    return {
        "max_drawdown":     round(max_dd, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "current_drawdown": round(float(drawdown[-1]), 4),
        "calmar_ratio":     round(-1 / (max_dd * 252) if max_dd < -1e-6 else 0, 3),
    }
