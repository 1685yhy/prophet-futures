"""
Scanner Agent — 优先筛选大趋势明确的品种。

核心逻辑（优化后）：
1. 用户指定的关注品种（LH为主，JD/BU/MA为辅）优先级最高
2. 用 cycle_detector 判断每个品种的大周期强度
3. 只推荐 BULL 或 BEAR 明确的品种，排除 NEUTRAL 震荡品种
4. 按趋势强度排序
"""

import logging
import json
from datetime import datetime
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.market_data import get_kline
from tools.cycle_detector import detect_cycle
from models.schemas import ScannerOutput

logger = logging.getLogger(__name__)

# 用户关注的品种（按优先级）
FOCUS_SYMBOLS = ["lh", "jd", "bu", "ma"]

# 扩展品种（次级关注）
EXTENDED_SYMBOLS = ["rb", "i", "sc", "cu", "au", "m", "y", "fu", "ta"]


def _scan_symbol_cycle(symbol: str) -> dict:
    """获取品种的大周期状态"""
    try:
        kline = get_kline(symbol, "daily", 130)
        import pandas as pd
        df = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low": kline.lows, "close": kline.closes,
            "volume": kline.volumes,
        })
        if kline.open_interests:
            df["oi"] = kline.open_interests
        info = detect_cycle(df)
        return {"symbol": symbol, **info}
    except Exception as e:
        logger.warning("Cycle detection failed for %s: %s", symbol, e)
        return {"symbol": symbol, "cycle": "NEUTRAL", "strength": 0.0, "reasoning": str(e)}


def run_scanner(focus: list = None) -> ScannerOutput:
    """
    扫描品种，优先返回大趋势明确的品种。

    Args:
        focus: 用户指定的关注品种列表，默认使用 FOCUS_SYMBOLS
    """
    symbols_to_scan = list(focus or FOCUS_SYMBOLS)
    # 补充扩展品种（去重）
    for s in EXTENDED_SYMBOLS:
        if s not in symbols_to_scan:
            symbols_to_scan.append(s)

    # 并行扫描周期（用线程池加速）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_scan_symbol_cycle, symbols_to_scan))

    # 过滤并排序：只保留趋势明确的品种
    trending = [r for r in results if r["cycle"] in ("BULL", "BEAR")]
    trending.sort(key=lambda x: x["strength"], reverse=True)

    # 关注品种中即使震荡也保留（用户指定的优先）
    focus_set    = set(focus or FOCUS_SYMBOLS)
    focus_result = [r for r in results if r["symbol"] in focus_set]
    focus_result.sort(key=lambda x: x["strength"], reverse=True)

    # 合并：关注品种在前，再加其他趋势明确的品种
    merged = list(focus_result)
    for r in trending:
        if r["symbol"] not in focus_set:
            merged.append(r)

    candidates = [r["symbol"] for r in merged[:8]]
    if not candidates:
        candidates = list(focus_set)

    trend_summary = "; ".join(
        f"{r['symbol']}={r['cycle']}({r['strength']:.2f})" for r in merged[:5]
    )

    logger.info("Scanner: %s", trend_summary)

    return ScannerOutput(
        candidates=candidates,
        scan_timestamp=datetime.now().isoformat(),
        total_screened=len(symbols_to_scan),
        selection_criteria=f"大周期筛选: {trend_summary}",
    )
