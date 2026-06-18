"""Macro Analyst Agent — evaluates macroeconomic drivers."""

import logging
import json
from langchain_core.tools import tool

from tools.llm_utils import invoke_structured
from tools.macro_data import get_sector_performance, get_market_index, get_global_futures, get_news_sentiment
from models.schemas import MacroReport

logger = logging.getLogger(__name__)

SYMBOL_SECTOR_MAP = {
    "rb": "black", "i": "black", "j": "black", "jm": "black", "hc": "black",
    "sc": "energy", "bu": "energy", "pg": "energy", "eg": "energy",
    "cu": "metals", "al": "metals", "zn": "metals", "ni": "metals",
    "au": "metals", "ag": "metals",
    "lh": "agriculture", "jd": "agriculture",
}


def run_macro_analyst(symbol: str) -> MacroReport:
    sector = SYMBOL_SECTOR_MAP.get(symbol.lower(), "general")

    @tool
    def get_sector_performance_tool(sector_name: str = "") -> str:
        """Get sector performance data. Input: sector name (e.g. agriculture, metals, energy, black)."""
        return json.dumps(get_sector_performance(sector_name.strip() or sector))

    @tool
    def get_market_index_tool() -> str:
        """Get major market index data (SSE, SZSE, etc)."""
        return json.dumps(get_market_index())

    @tool
    def get_global_futures_tool() -> str:
        """Get global futures benchmarks (CBOT, LME, NYMEX, etc)."""
        return json.dumps(get_global_futures())

    @tool
    def get_news_sentiment_tool(keyword: str = "") -> str:
        """Get news sentiment for keywords. Input: comma-separated keywords."""
        return json.dumps(get_news_sentiment([keyword.strip(), symbol]))

    result = invoke_structured(
        agent_name="macro_analyst",
        tools=[get_sector_performance_tool, get_market_index_tool,
               get_global_futures_tool, get_news_sentiment_tool],
        input_text=f"Analyze macroeconomic drivers for {symbol} (sector: {sector}).",
        schema=MacroReport, temperature=0.1, max_iterations=5,
    )
    if result is not None:
        return result

    logger.warning("Macro analyst fallback for %s", symbol)
    idx    = get_market_index()
    change = idx.get("sse_change_pct", 0.0)
    trend  = "BULLISH" if change > 0.5 else ("BEARISH" if change < -0.5 else "NEUTRAL")
    return MacroReport(
        sector=sector, macro_trend=trend,
        key_drivers=["Market index direction", "Global commodity prices"],
        risk_events=["Policy announcement risk", "Macroeconomic data release"],
        confidence=0.4,
        reasoning=f"Fallback: SSE change={change:.2f}%",
    )
