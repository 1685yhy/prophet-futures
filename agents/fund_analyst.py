"""Fund Analyst Agent — analyzes capital flow and position structure."""

import logging
import json
from langchain_core.tools import tool

from tools.llm_utils import invoke_structured
from tools.fund_data import (get_volume_oi, get_basis, get_member_positions,
                              get_cftc_like_report, get_intraday_oi_pattern)
from models.schemas import FundReport

logger = logging.getLogger(__name__)


def run_fund_analyst(symbol: str) -> FundReport:
    @tool
    def get_volume_oi_tool(sym: str = "") -> str:
        """Get volume and open interest data for a symbol."""
        return json.dumps(get_volume_oi((sym or symbol).strip()))

    @tool
    def get_basis_tool(sym: str = "") -> str:
        """Get spot-futures basis data for a symbol."""
        return json.dumps(get_basis((sym or symbol).strip()))

    @tool
    def get_member_positions_tool(sym: str = "") -> str:
        """Get top member long/short positions for a symbol."""
        return json.dumps(get_member_positions((sym or symbol).strip()))

    @tool
    def get_cftc_like_report_tool(sym: str = "") -> str:
        """Get COT-style positioning breakdown for a symbol."""
        return json.dumps(get_cftc_like_report((sym or symbol).strip()))

    result = invoke_structured(
        agent_name="fund_analyst",
        tools=[get_volume_oi_tool, get_basis_tool, get_member_positions_tool, get_cftc_like_report_tool],
        input_text=(f"Analyze capital flow and position structure for {symbol}. "
                    "Call all four tools: get_volume_oi_tool, get_basis_tool, get_member_positions_tool, get_cftc_like_report_tool."),
        schema=FundReport, temperature=0.1, max_iterations=5,
    )
    if result is not None:
        return result

    logger.warning("Fund analyst fallback for %s", symbol)
    oi_data    = get_volume_oi(symbol)
    basis_data = get_basis(symbol)

    oi_trend   = oi_data.get("oi_trend_direction", "FLAT")
    oi_change  = oi_data.get("oi_change", 0)
    oi_3d      = oi_data.get("oi_trend_3d", 0)
    oi_5d      = oi_data.get("oi_trend_5d", 0)

    if oi_trend == "ACCUMULATING":
        net_flow   = "INFLOW"
        confidence = 0.60
        trend_note = f"3日趋势增仓{oi_3d:.0f}手/5日{oi_5d:.0f}手"
    elif oi_trend == "REDUCING":
        net_flow   = "OUTFLOW"
        confidence = 0.60
        trend_note = f"3日趋势减仓{oi_3d:.0f}手/5日{oi_5d:.0f}手"
    else:
        net_flow   = "INFLOW" if oi_change > 500 else ("OUTFLOW" if oi_change < -500 else "NEUTRAL")
        confidence = 0.40
        trend_note = f"趋势不明FLAT，单日变化{oi_change:.0f}手"

    try:
        intraday = get_intraday_oi_pattern(symbol)
        pattern  = intraday.get("pattern", "UNKNOWN")
        morning  = intraday.get("morning_oi_change", 0)
        if pattern == "ACCUMULATE" and morning > 1000:
            confidence = min(0.85, confidence + 0.20)
            trend_note += f" | 盘中早段增仓{morning:.0f}手(ACCUMULATE)"
        elif pattern == "DAY_TRADE_CLOSE" and morning > 500:
            confidence = min(0.75, confidence + 0.10)
            trend_note += f" | 早段增仓{morning:.0f}手+尾盘日内平仓(DAY_TRADE_CLOSE)"
        elif pattern == "DISTRIBUTE":
            confidence = max(0.35, confidence - 0.15)
            trend_note += f" | 盘中主动减仓(DISTRIBUTE)"
    except Exception:
        pass

    return FundReport(
        symbol=symbol,
        net_flow=net_flow,
        basis_status=basis_data.get("structure", "FLAT"),
        top_member_action=f"Fallback: {trend_note}",
        confidence=confidence,
        reasoning=f"Fallback: {trend_note}, basis={basis_data.get('basis', 0):.2f}",
    )
