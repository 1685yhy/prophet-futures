"""Technician Agent — full technical analysis with hardcoded indicator computation."""

import logging
import json
import pandas as pd
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.market_data import get_kline
from tools.indicators import calc_indicators, detect_divergence, adx_regime
from models.schemas import TechReport, TechSignal

logger = logging.getLogger(__name__)


def _get_kline_with_indicators(symbol: str) -> str:
    kline = get_kline(symbol, "daily", 120)
    df    = pd.DataFrame({
        "open": kline.opens, "high": kline.highs,
        "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
    })
    indicators = calc_indicators(df)
    divergence = detect_divergence(df)
    regime     = adx_regime(indicators["adx14"])
    return json.dumps({
        "symbol":          symbol,
        "latest_close":    kline.closes[-1],
        "indicators":      indicators,
        "divergence":      divergence,
        "regime":          regime,
        "recent_closes":   kline.closes[-10:],
        "recent_volumes":  kline.volumes[-10:],
    })


def run_technician(symbol: str) -> TechReport:
    result = invoke_structured(
        agent_name="technician",
        tools=[Tool(name="get_kline_with_indicators",
                    func=lambda sym: _get_kline_with_indicators(sym.strip() or symbol),
                    description="Get OHLCV kline with pre-computed indicators. Input: symbol")],
        input_text=(f"Perform complete technical analysis on {symbol}. "
                    "Call get_kline_with_indicators first — all numeric values must come from that tool."),
        schema=TechReport, temperature=0.1, max_iterations=4,
    )
    if result is not None:
        return result

    logger.warning("Technician fallback for %s", symbol)
    kline = get_kline(symbol, "daily", 60)
    df    = pd.DataFrame({
        "open": kline.opens, "high": kline.highs,
        "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
    })
    ind   = calc_indicators(df)
    close = ind["current_close"]
    atr   = ind["atr14"]
    direction = (
        "LONG"  if ind["macd_hist"] > 0 and ind["rsi14"] < 70 else
        "SHORT" if ind["macd_hist"] < 0 and ind["rsi14"] > 30 else "NEUTRAL"
    )
    return TechReport(
        symbol=symbol,
        signal=TechSignal(direction=direction, strength="MODERATE",
                          reasoning="Fallback: MACD+RSI heuristic"),
        key_support=   round(close - 2 * atr, 2),
        key_resistance=round(close + 2 * atr, 2),
        stop_loss=     round(close - 1.5 * atr, 2) if direction == "LONG" else round(close + 1.5 * atr, 2),
        target_price=  round(close + 3 * atr, 2)   if direction == "LONG" else round(close - 3 * atr, 2),
        confidence=0.5,
        indicators=ind,
    )
