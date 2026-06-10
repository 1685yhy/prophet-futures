"""Futures fund flow and position data."""

import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_volume_oi(symbol: str) -> Dict[str, Any]:
    try:
        import akshare as ak
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
        df    = ak.futures_main_sina(symbol=symbol.upper() + "0", start_date=start, end_date=end)
        col_map = {"成交量": "volume", "持仓量": "open_interest"}
        df = df.rename(columns=col_map)
        if df.empty:
            raise ValueError("empty")
        latest   = df.tail(7)   # 取7日以支持5日趋势计算
        vol_avg  = float(latest["volume"].mean())          if "volume"        in latest.columns else 10000.0
        oi_last  = float(latest["open_interest"].iloc[-1]) if "open_interest" in latest.columns else 50000.0
        oi_prev  = float(latest["open_interest"].iloc[-2]) if ("open_interest" in latest.columns and len(latest) > 1) else oi_last
        vol_today= float(latest["volume"].iloc[-1])        if "volume"        in latest.columns else vol_avg

        # 多日OI趋势（排除单日噪音）
        oi_col = "open_interest"
        oi_3d = float(latest[oi_col].iloc[-1] - latest[oi_col].iloc[-4]) if len(latest) >= 4 else 0.0
        oi_5d = float(latest[oi_col].iloc[-1] - latest[oi_col].iloc[-6]) if len(latest) >= 6 else oi_3d
        if oi_3d > 0 and oi_5d > 0:
            oi_trend_direction = "ACCUMULATING"
        elif oi_3d < 0 and oi_5d < 0:
            oi_trend_direction = "REDUCING"
        else:
            oi_trend_direction = "FLAT"

        return {
            "volume_today":        vol_today,
            "volume_ma5":          vol_avg,
            "vol_ratio":           round(vol_today / (vol_avg + 1e-8), 2),
            "open_interest":       oi_last,
            "oi_change":           oi_last - oi_prev,
            "oi_change_pct":       round((oi_last - oi_prev) / (oi_prev + 1e-8) * 100, 2),
            # 多日趋势（优化新增）
            "oi_trend_3d":         round(oi_3d, 0),
            "oi_trend_5d":         round(oi_5d, 0),
            "oi_trend_direction":  oi_trend_direction,
        }
    except Exception as e:
        logger.warning("volume/oi data unavailable for %s: %s", symbol, e)
        return _synthetic_volume_oi(symbol)


def get_basis(symbol: str) -> Dict[str, Any]:
    try:
        import akshare as ak
        date_str = datetime.now().strftime("%Y%m%d")
        df       = ak.futures_spot_price(date=date_str)
        row      = df[df["symbol"].str.upper() == symbol.upper()]
        if row.empty:
            raise ValueError(f"No basis data for {symbol}")
        row = row.iloc[0]
        spot    = float(row.get("spot_price", 0) or 0)
        dom     = float(row.get("dominant_contract_price", 0) or row.get("near_contract_price", 0) or 0)
        if spot <= 0 or dom <= 0:
            raise ValueError("zero prices")
        basis = dom - spot
        return {
            "spot_price":    round(spot, 2),
            "futures_price": round(dom, 2),
            "basis":         round(basis, 2),
            "basis_pct":     round(basis / spot * 100, 3),
            "structure":     "CONTANGO" if basis > 0 else ("BACKWARDATION" if basis < 0 else "FLAT"),
        }
    except Exception as e:
        logger.warning("Basis data unavailable for %s: %s", symbol, e)
        np.random.seed(hash(symbol) % 2**32)
        spot  = round(5000 + np.random.normal(0, 200), 2)
        basis = round(np.random.normal(20, 30), 2)
        return {
            "spot_price":    spot,
            "futures_price": round(spot + basis, 2),
            "basis":         basis,
            "basis_pct":     round(basis / spot * 100, 3),
            "structure":     "CONTANGO" if basis > 0 else "BACKWARDATION",
        }


def get_member_positions(symbol: str) -> Dict[str, Any]:
    try:
        import akshare as ak
        date_str = datetime.now().strftime("%Y%m%d")
        df = ak.get_futures_daily(start_date=date_str, end_date=date_str, market="DCE")
        if df is None or df.empty:
            raise ValueError("empty")
        return {"data": df[df["symbol"] == symbol.lower()].head(10).to_dict("records"), "source": "akshare"}
    except Exception as e:
        logger.warning("Member positions unavailable for %s: %s", symbol, e)
        np.random.seed(hash(symbol + "member") % 2**32)
        return {
            "top_long_pct":      round(np.random.uniform(30, 60), 1),
            "top_short_pct":     round(np.random.uniform(25, 55), 1),
            "net_long_change":   int(np.random.normal(0, 500)),
            "concentration_index": round(np.random.uniform(0.3, 0.7), 2),
            "source": "synthetic",
        }


def get_cftc_like_report(symbol: str) -> Dict[str, Any]:
    np.random.seed(hash(symbol + "cot") % 2**32)
    comm_net  = int(np.random.normal(-5000, 3000))
    spec_net  = int(np.random.normal(5000, 4000))
    return {
        "commercial_long":     abs(comm_net) + 20000,
        "commercial_short":    20000,
        "commercial_net":      comm_net,
        "speculative_long":    abs(spec_net) + 15000,
        "speculative_short":   15000,
        "speculative_net":     spec_net,
        "small_trader_net":    -(comm_net + spec_net),
        "report_date":         datetime.now().strftime("%Y-%m-%d"),
    }


def get_intraday_oi_pattern(symbol: str) -> Dict[str, Any]:
    """
    分析今日分时OI变化模式，区分主力方向与日内换手。

    分三段统计：
    - morning (09:00-11:30)：主力建仓/减仓的最可靠时段
    - afternoon (13:00-14:30)：下午主力延续
    - eod (14:30-15:00)：日内交易者结构性平仓，不代表趋势

    pattern:
    - "ACCUMULATE"：上午+下午均净增仓，主力积累
    - "DISTRIBUTE"：上午+下午均净减仓，主力出货
    - "DAY_TRADE_CLOSE"：上午增仓+尾盘减仓，日内多头了结，趋势仍是积累
    - "MIXED"：信号混杂
    """
    try:
        import akshare as ak
        df_min = ak.futures_zh_minute_sina(symbol=symbol.upper() + "0", period="1")
        df_min.columns = ["datetime", "open", "high", "low", "close", "volume", "hold"]
        today_str = datetime.now().strftime("%Y-%m-%d")

        def _segment_oi(start_t: str, end_t: str) -> float:
            seg = df_min[
                (df_min["datetime"].astype(str) >= f"{today_str} {start_t}") &
                (df_min["datetime"].astype(str) <= f"{today_str} {end_t}")
            ]
            if len(seg) < 2:
                return 0.0
            return float(seg["hold"].iloc[-1] - seg["hold"].iloc[0])

        morning_chg   = _segment_oi("09:00", "11:30")
        afternoon_chg = _segment_oi("13:00", "14:30")
        eod_chg       = _segment_oi("14:30", "15:00")

        if morning_chg > 0 and afternoon_chg > 0:
            pattern = "ACCUMULATE"
        elif morning_chg < 0 and afternoon_chg < 0:
            pattern = "DISTRIBUTE"
        elif morning_chg > 0 and eod_chg < 0 and abs(eod_chg) < abs(morning_chg):
            pattern = "DAY_TRADE_CLOSE"   # 上午增仓，尾盘日内平仓，净仍为增
        else:
            pattern = "MIXED"

        return {
            "morning_oi_change":   round(morning_chg, 0),
            "afternoon_oi_change": round(afternoon_chg, 0),
            "eod_oi_change":       round(eod_chg, 0),
            "pattern":             pattern,
            "is_eod_window":       _is_eod_window(),
        }
    except Exception as e:
        logger.warning("Intraday OI pattern unavailable for %s: %s", symbol, e)
        return {
            "morning_oi_change":   0.0,
            "afternoon_oi_change": 0.0,
            "eod_oi_change":       0.0,
            "pattern":             "UNKNOWN",
            "is_eod_window":       _is_eod_window(),
        }


def _is_eod_window() -> bool:
    """当前是否处于尾盘平仓时间窗口（14:30-15:15）。"""
    now = datetime.now()
    return (now.hour == 14 and now.minute >= 30) or now.hour == 15


def _synthetic_volume_oi(symbol: str) -> Dict[str, Any]:
    np.random.seed(hash(symbol) % 2**32)
    vol = int(np.random.exponential(50000))
    oi  = int(np.random.exponential(200000))
    return {
        "volume_today":        float(vol),
        "volume_ma5":          float(vol * np.random.uniform(0.8, 1.2)),
        "vol_ratio":           round(np.random.uniform(0.7, 1.5), 2),
        "open_interest":       float(oi),
        "oi_change":           float(int(np.random.normal(0, 1000))),
        "oi_change_pct":       round(np.random.normal(0, 0.5), 2),
        "oi_trend_3d":         float(int(np.random.normal(0, 2000))),
        "oi_trend_5d":         float(int(np.random.normal(0, 3000))),
        "oi_trend_direction":  "FLAT",
    }
