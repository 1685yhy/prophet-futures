"""Vision Technical Agent — chart pattern recognition via text-based analysis."""

import logging
import json
import pandas as pd
from tools.llm_utils import get_llm, load_prompt, parse_json_output
from tools.market_data import get_kline
from tools.indicators import calc_indicators
from models.schemas import VisionReport

logger = logging.getLogger(__name__)


def run_vision_tech(symbol: str) -> VisionReport:
    """Analyze chart patterns using text-based OHLCV data (DeepSeek doesn't support images)."""
    try:
        kline = get_kline(symbol, "daily", 60)
        df = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low": kline.lows, "close": kline.closes, "volume": kline.volumes,
        })
        ind = calc_indicators(df)

        # Build a text description of the chart
        closes = kline.closes[-20:]
        volumes = kline.volumes[-20:]
        highs = kline.highs[-20:]
        lows = kline.lows[-20:]

        chart_text = (
            f"Symbol: {symbol}\n"
            f"Latest close: {closes[-1]:.0f}\n"
            f"20-day high: {max(highs):.0f}, low: {min(lows):.0f}\n"
            f"MA5: {ind['ma5']:.0f}, MA20: {ind['ma20']:.0f}, MA60: {ind['ma60']:.0f}\n"
            f"RSI14: {ind['rsi14']:.1f}, ADX14: {ind['adx14']:.1f}\n"
            f"MACD histogram: {ind['macd_hist']:.1f}\n"
            f"Bollinger: upper={ind['bb_upper']:.0f}, lower={ind['bb_lower']:.0f}\n"
            f"Recent closes (last 5): {[f'{c:.0f}' for c in closes[-5:]]}\n"
            f"Recent volumes (last 5): {[f'{v:.0f}' for v in volumes[-5:]]}\n"
            f"Price trend (10d): {'UP' if closes[-1] > closes[-10] else 'DOWN'}\n"
            f"Volume trend (5d avg vs 20d avg): "
            f"{'INCREASING' if sum(volumes[-5:])/5 > sum(volumes)/20 else 'DECREASING'}\n"
        )

        system_prompt = load_prompt("vision_tech")
        llm = get_llm(temperature=0.1)
        response = llm.invoke(
            f"{system_prompt}\n\n"
            f"Analyze the following chart data for {symbol} and output ONLY valid JSON "
            f"matching VisionReport schema (chart_pattern, visual_signal, pattern_completion_pct, reasoning):\n\n"
            f"{chart_text}"
        )
        raw_text = response.content if hasattr(response, 'content') else str(response)
        data = parse_json_output(raw_text)
        if data and "visual_signal" in data:
            try:
                return VisionReport(**data)
            except Exception as ve:
                logger.warning("Vision analysis failed for %s: %s", symbol, ve)
                # Try with defaults for missing fields
                data.setdefault("chart_pattern", "Auto-detected")
                data.setdefault("pattern_completion_pct", 50.0)
                data.setdefault("reasoning", str(data.get("visual_signal", "NEUTRAL")))
                try:
                    return VisionReport(**data)
                except Exception:
                    pass
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
