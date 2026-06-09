"""Trap Detector Agent — identifies bull traps, bear traps, and shakeouts."""

import logging
import json
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.fund_data import get_volume_oi, get_member_positions
from tools.market_data import get_tick_data, get_realtime_quote
from models.schemas import TrapAnalysisReport, TrapType

logger = logging.getLogger(__name__)


def _build_tools(symbol: str) -> list:
    return [
        Tool(name="get_volume_oi",
             func=lambda sym: json.dumps(get_volume_oi(sym.strip() or symbol)),
             description="Get volume and OI data. Input: symbol"),
        Tool(name="get_member_positions",
             func=lambda sym: json.dumps(get_member_positions(sym.strip() or symbol)),
             description="Get top member positions. Input: symbol"),
        Tool(name="get_tick_data",
             func=lambda sym: json.dumps(get_tick_data(sym.strip() or symbol, 120)),
             description="Get recent tick data. Input: symbol"),
        Tool(name="get_realtime_quote",
             func=lambda sym: json.dumps(get_realtime_quote(sym.strip() or symbol).model_dump()),
             description="Get realtime quote. Input: symbol"),
    ]


def run_trap_detector(symbol: str) -> TrapAnalysisReport:
    result = invoke_structured(
        agent_name="trap_detector",
        tools=_build_tools(symbol),
        input_text=f"Analyze {symbol} for potential bull traps, bear traps, or shakeout patterns.",
        schema=TrapAnalysisReport, temperature=0.1, max_iterations=4,
    )
    if result is not None:
        return result

    logger.warning("Trap detector fallback for %s", symbol)
    oi_data       = get_volume_oi(symbol)
    vol_ratio     = oi_data.get("vol_ratio", 1.0)
    oi_change_pct = oi_data.get("oi_change_pct", 0.0)
    trap_type     = "BULL_TRAP" if vol_ratio < 0.8 and oi_change_pct < -2 else "NONE"
    return TrapAnalysisReport(
        symbol=symbol,
        trap=TrapType(
            type=trap_type,
            current_phase="Observation" if trap_type == "NONE" else "Potential setup forming",
            trigger_to_confirm="Price reversal with volume > 120% of average",
            confidence=0.55 if trap_type != "NONE" else 0.3,
        ),
        reasoning=f"Fallback: vol_ratio={vol_ratio:.2f}, oi_change={oi_change_pct:.1f}%",
    )
