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
    from tools.fund_data import _is_eod_window
    oi_data       = get_volume_oi(symbol)
    vol_ratio     = oi_data.get("vol_ratio", 1.0)
    oi_change_pct = oi_data.get("oi_change_pct", 0.0)
    oi_trend      = oi_data.get("oi_trend_direction", "FLAT")
    is_eod        = _is_eod_window()

    trap_type  = "NONE"
    confidence = 0.20
    note       = ""

    if oi_change_pct < -2:
        # 过滤器A：尾盘时间窗口（14:30-15:15）
        # 日内交易者结构性平仓，不代表主力意图
        if is_eod:
            confidence = 0.15
            note       = "尾盘时间窗口OI减少属日内平仓，不判断陷阱"
        # 过滤器B：多日OI趋势交叉
        # 3日趋势仍在增仓时，当日减少是换手而非出货
        elif oi_trend == "ACCUMULATING":
            confidence = 0.20
            note       = f"3日趋势仍积累({oi_trend})，日内减仓为换手非出货"
        # 非尾盘 + 趋势也在减少 + 缩量 → 真实BULL_TRAP
        elif vol_ratio < 0.8 and oi_trend != "ACCUMULATING":
            trap_type  = "BULL_TRAP"
            confidence = 0.55
            note       = f"缩量({vol_ratio:.2f}x)+OI减+趋势{oi_trend}，疑似诱多"
        else:
            note = f"OI减少但缩量条件不满足(vol_ratio={vol_ratio:.2f})"

    return TrapAnalysisReport(
        symbol=symbol,
        trap=TrapType(
            type=trap_type,
            current_phase="Observation" if trap_type == "NONE" else "Potential setup forming",
            trigger_to_confirm="Price reversal with volume > 120% of average",
            confidence=confidence,
        ),
        reasoning=(f"Fallback: vol_ratio={vol_ratio:.2f}, oi_chg={oi_change_pct:.1f}%, "
                   f"trend={oi_trend}, eod={is_eod}. {note}"),
    )
