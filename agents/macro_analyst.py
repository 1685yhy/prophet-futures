"""Macro Analyst Agent — evaluates macroeconomic drivers."""

import logging
import json
from langchain_core.tools import Tool

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
    result = invoke_structured(
        agent_name="macro_analyst",
        tools=[
            Tool(name="get_sector_performance",
                 func=lambda s: json.dumps(get_sector_performance(s.strip() or sector)),
                 description="Get sector performance. Input: sector name"),
            Tool(name="get_market_index",
                 func=lambda _: json.dumps(get_market_index()),
                 description="Get major market index data"),
            Tool(name="get_global_futures",
                 func=lambda _: json.dumps(get_global_futures()),
                 description="Get global futures benchmarks"),
            Tool(name="get_news_sentiment",
                 func=lambda kw: json.dumps(get_news_sentiment([kw.strip(), symbol])),
                 description="Get news sentiment. Input: keyword"),
        ],
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
