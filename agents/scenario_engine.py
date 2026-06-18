"""Scenario Engine Agent — generates multi-path price scenarios."""

import logging
import json
import pandas as pd
from langchain_core.tools import tool

from tools.llm_utils import invoke_structured
from tools.market_data import get_kline
from tools.indicators import calc_indicators
from models.schemas import ScenarioReport, ScenarioPath

logger = logging.getLogger(__name__)


def _get_market_context(symbol: str) -> str:
    kline = get_kline(symbol, "daily", 60)
    df    = pd.DataFrame({
        "open": kline.opens, "high": kline.highs,
        "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
    })
    ind = calc_indicators(df)
    return json.dumps({
        "symbol":       symbol,
        "current_price":ind["current_close"],
        "atr14":        ind["atr14"],
        "ma20":         ind["ma20"],
        "ma60":         ind["ma60"],
        "rsi14":        ind["rsi14"],
        "adx14":        ind["adx14"],
        "recent_high":  max(kline.highs[-20:]),
        "recent_low":   min(kline.lows[-20:]),
    })


def run_scenario_engine(symbol: str) -> ScenarioReport:
    @tool
    def get_market_context_tool(sym: str = "") -> str:
        """Get market context with technical indicators for a symbol."""
        return _get_market_context(sym.strip() or symbol)

    result = invoke_structured(
        agent_name="scenario_engine",
        tools=[get_market_context_tool],
        input_text=f"Generate 3 price path scenarios (bullish/neutral/bearish) for {symbol}.",
        schema=ScenarioReport, temperature=0.3, max_iterations=3,
    )
    if result is not None:
        return result

    logger.warning("Scenario engine fallback for %s", symbol)
    ctx   = json.loads(_get_market_context(symbol))
    price = ctx["current_price"]
    atr   = ctx["atr14"]
    return ScenarioReport(
        paths=[
            ScenarioPath(probability=0.30, description="Bullish breakout",
                         target_price=round(price + 3 * atr, 2),
                         key_trigger="Volume expansion above resistance"),
            ScenarioPath(probability=0.50, description="Consolidation",
                         target_price=round(price + 0.5 * atr, 2),
                         key_trigger="No directional catalyst"),
            ScenarioPath(probability=0.20, description="Bearish breakdown",
                         target_price=round(price - 3 * atr, 2),
                         key_trigger="Support failure with volume"),
        ],
        worst_case_loss_pct=round(3 * atr / price * 100, 2),
        best_case_gain_pct= round(3 * atr / price * 100, 2),
    )
