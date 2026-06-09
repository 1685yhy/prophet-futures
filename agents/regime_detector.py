"""Regime Detector Agent — classifies market conditions and sets analysis weights."""

import logging
import pandas as pd
from tools.market_data import get_kline
from tools.indicators import calc_indicators, adx_regime
from models.schemas import RegimeOutput

logger = logging.getLogger(__name__)

REGIME_WEIGHTS = {
    "TRENDING_BULL":   {"tech": 0.40, "fund": 0.30, "macro": 0.20, "vision": 0.10},
    "TRENDING_BEAR":   {"tech": 0.40, "fund": 0.30, "macro": 0.20, "vision": 0.10},
    "RANGING":         {"tech": 0.25, "fund": 0.40, "macro": 0.25, "vision": 0.10},
    "HIGH_VOLATILITY": {"tech": 0.20, "fund": 0.20, "macro": 0.50, "vision": 0.10},
    "CRISIS":          {"tech": 0.15, "fund": 0.15, "macro": 0.60, "vision": 0.10},
}


def run_regime_detector(symbol: str = "rb") -> RegimeOutput:
    try:
        kline = get_kline(symbol, "daily", 60)
        df    = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
        })
        ind    = calc_indicators(df)
        adx    = ind["adx14"]
        atr    = ind["atr14"]
        close  = ind["current_close"]
        atr_pct= atr / close * 100

        if atr_pct > 5.0:
            regime = "CRISIS"
        elif atr_pct > 3.0:
            regime = "HIGH_VOLATILITY"
        elif adx >= 25:
            regime = "TRENDING_BULL" if ind["ma20"] > ind["ma60"] else "TRENDING_BEAR"
        else:
            regime = "RANGING"

        return RegimeOutput(
            regime=regime,
            adx_value=round(adx, 2),
            vix_equivalent=round(atr_pct * 10, 2),
            recommended_weights=REGIME_WEIGHTS[regime],
            reasoning=(f"ADX={adx:.1f} ({adx_regime(adx)}), ATR%={atr_pct:.2f}%, "
                       f"MA20={'above' if ind['ma20']>ind['ma60'] else 'below'} MA60"),
        )
    except Exception as e:
        logger.warning("Regime detector fallback: %s", e)
        return RegimeOutput(
            regime="RANGING", adx_value=20.0, vix_equivalent=15.0,
            recommended_weights=REGIME_WEIGHTS["RANGING"],
            reasoning="Fallback: default ranging regime",
        )
