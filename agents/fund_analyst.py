"""Fund Analyst Agent — analyzes capital flow and position structure."""

import logging
import json
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.fund_data import get_volume_oi, get_basis, get_member_positions, get_cftc_like_report
from models.schemas import FundReport

logger = logging.getLogger(__name__)


def run_fund_analyst(symbol: str) -> FundReport:
    result = invoke_structured(
        agent_name="fund_analyst",
        tools=[
            Tool(name="get_volume_oi",
                 func=lambda sym: json.dumps(get_volume_oi(sym.strip())),
                 description="Get volume and open interest data. Input: symbol"),
            Tool(name="get_basis",
                 func=lambda sym: json.dumps(get_basis(sym.strip())),
                 description="Get spot-futures basis data. Input: symbol"),
            Tool(name="get_member_positions",
                 func=lambda sym: json.dumps(get_member_positions(sym.strip())),
                 description="Get top member long/short positions. Input: symbol"),
            Tool(name="get_cftc_like_report",
                 func=lambda sym: json.dumps(get_cftc_like_report(sym.strip())),
                 description="Get COT-style positioning breakdown. Input: symbol"),
        ],
        input_text=(f"Analyze capital flow and position structure for {symbol}. "
                    "Call all four tools: get_volume_oi, get_basis, get_member_positions, get_cftc_like_report."),
        schema=FundReport, temperature=0.1, max_iterations=5,
    )
    if result is not None:
        return result

    logger.warning("Fund analyst fallback for %s", symbol)
    oi_data   = get_volume_oi(symbol)
    basis_data= get_basis(symbol)
    oi_change = oi_data.get("oi_change", 0)
    return FundReport(
        symbol=symbol,
        net_flow="INFLOW" if oi_change > 0 else ("OUTFLOW" if oi_change < 0 else "NEUTRAL"),
        basis_status=basis_data.get("structure", "FLAT"),
        top_member_action="Undetermined (fallback mode)",
        confidence=0.4,
        reasoning=f"Fallback: OI change={oi_change:.0f}, basis={basis_data.get('basis', 0):.2f}",
    )
