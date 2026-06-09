"""Vision Technical Agent — chart pattern recognition via multimodal LLM."""

import logging
import base64
from tools.llm_utils import get_llm, load_prompt, parse_json_output
from tools.market_data import plot_kline_chart
from models.schemas import VisionReport

logger = logging.getLogger(__name__)


def run_vision_tech(symbol: str) -> VisionReport:
    chart_bytes = plot_kline_chart(symbol, "daily", 60)
    if chart_bytes is None:
        logger.warning("Chart generation failed for %s", symbol)
        return _neutral_report(symbol)

    system_prompt = load_prompt("vision_tech")
    try:
        llm    = get_llm(temperature=0.1)
        img_b64= base64.b64encode(chart_bytes).decode("utf-8")
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content=[
            {"type": "text", "text": system_prompt + f"\n\nAnalyze the chart for {symbol}. "
             "Return ONLY valid JSON matching VisionReport schema."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ])
        response = llm.invoke([message])
        data     = parse_json_output(response.content if hasattr(response, "content") else str(response))
        if data and "visual_signal" in data:
            return VisionReport(**data)
    except Exception as e:
        logger.warning("Vision analysis failed for %s: %s", symbol, e)

    return _neutral_report(symbol)


def _neutral_report(symbol: str) -> VisionReport:
    return VisionReport(
        symbol=symbol,
        chart_pattern="Unable to analyze",
        visual_signal="NEUTRAL",
        pattern_completion_pct=0.0,
        reasoning="Chart analysis unavailable — using neutral default",
    )
