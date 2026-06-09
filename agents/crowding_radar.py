"""Crowding Radar Agent — measures market crowdedness and stampede risk."""

import logging
import numpy as np
import pandas as pd
from tools.fund_data import get_volume_oi, get_member_positions, get_basis
from tools.market_data import get_kline
from tools.indicators import calc_indicators
from models.schemas import CrowdingLevel

logger = logging.getLogger(__name__)


def run_crowding_radar(symbol: str) -> CrowdingLevel:
    score   = 0
    details = []

    try:
        oi_data   = get_volume_oi(symbol)
        vol_ratio = oi_data.get("vol_ratio", 1.0)
        if vol_ratio > 1.5:
            score += 20; details.append(f"Volume spike {vol_ratio:.1f}x (+20)")
        elif vol_ratio > 1.2:
            score += 10; details.append(f"Above-avg volume {vol_ratio:.1f}x (+10)")
    except Exception as e:
        logger.warning("OI data failed in crowding: %s", e)

    try:
        member       = get_member_positions(symbol)
        concentration= member.get("concentration_index", 0.5)
        if concentration > 0.65:
            score += 25; details.append(f"High concentration {concentration:.2f} (+25)")
        elif concentration > 0.5:
            score += 15; details.append(f"Moderate concentration {concentration:.2f} (+15)")
    except Exception as e:
        logger.warning("Member data failed in crowding: %s", e)

    try:
        basis_data= get_basis(symbol)
        basis_pct = abs(basis_data.get("basis_pct", 0))
        if basis_pct > 2.0:
            score += 20; details.append(f"Extreme basis {basis_pct:.2f}% (+20)")
        elif basis_pct > 1.0:
            score += 10; details.append(f"Elevated basis {basis_pct:.2f}% (+10)")
    except Exception as e:
        logger.warning("Basis data failed in crowding: %s", e)

    try:
        kline = get_kline(symbol, "daily", 60)
        df    = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
        })
        ind       = calc_indicators(df)
        deviation = abs(ind["current_close"] - ind["ma60"]) / (ind["ma60"] + 1e-8) * 100
        if deviation > 8:
            score += 20; details.append(f"Large MA60 deviation {deviation:.1f}% (+20)")
        elif deviation > 4:
            score += 10; details.append(f"Moderate deviation {deviation:.1f}% (+10)")
        rsi = ind["rsi14"]
        if rsi > 75 or rsi < 25:
            score += 15; details.append(f"Extreme RSI {rsi:.1f} (+15)")
    except Exception as e:
        logger.warning("Kline failed in crowding: %s", e)

    score = min(100, score)
    return CrowdingLevel(
        symbol=symbol,
        score=score,
        warning=score > 80,
        similar_funds_pct=round(min(1.0, score / 100 * 1.2), 2),
        reasoning=" | ".join(details) if details else "Low crowding indicators",
    )
