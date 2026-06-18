"""Memory Retriever Agent — historical analogue retrieval and win-rate analysis."""

import logging
import json
import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.llm_utils import invoke_structured
from tools.market_data import get_kline
from tools.indicators import calc_indicators
from tools.memory_store import init_vector_db, compute_embedding, search_similar_markets
from models.schemas import MemoryReport, MemoryCase

logger = logging.getLogger(__name__)


def _search_memory(symbol: str) -> str:
    kline = get_kline(symbol, "daily", 60)
    df    = pd.DataFrame({
        "open": kline.opens, "high": kline.highs,
        "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
    })
    ind   = calc_indicators(df)
    init_vector_db()
    vec   = compute_embedding(ind)
    cases = search_similar_markets(vec, top_k=5)
    return json.dumps([c.model_dump() for c in cases])


def run_memory_retriever(symbol: str) -> MemoryReport:
    @tool
    def search_memory_tool(sym: str = "") -> str:
        """Search historical similar market cases for a symbol."""
        return _search_memory(sym.strip() or symbol)

    result = invoke_structured(
        agent_name="memory_retriever",
        tools=[search_memory_tool],
        input_text=f"Retrieve historical market analogues for {symbol} and compute win rate statistics.",
        schema=MemoryReport, temperature=0.1, max_iterations=3,
    )
    if result is not None:
        return result

    logger.warning("Memory retriever fallback for %s", symbol)
    cases_json = json.loads(_search_memory(symbol))
    cases      = [MemoryCase(**c) for c in cases_json]
    if not cases:
        return MemoryReport(
            similar_cases=[], historical_win_rate=0.5,
            avg_profit_loss_ratio=1.0, conclusion="No historical analogues found",
        )
    returns  = [c.subsequent_5d_return for c in cases]
    wins     = [r for r in returns if r > 0]
    losses   = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns)
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 1.0
    return MemoryReport(
        similar_cases=cases,
        historical_win_rate=round(win_rate, 3),
        avg_profit_loss_ratio=round(avg_win / (avg_loss + 1e-8), 3),
        conclusion="Supports direction" if win_rate > 0.55 else "Caution — win rate below threshold",
    )
