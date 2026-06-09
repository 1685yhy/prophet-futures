"""Technical indicator calculations — all math in Python, never delegated to LLM."""

import numpy as np
import pandas as pd
from typing import Dict, Any


def calc_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all standard technical indicators from OHLCV dataframe."""
    result = {}
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    volumes= df["volume"].values.astype(float)

    result["ma5"]  = float(np.mean(closes[-5:]))  if len(closes) >= 5  else float(closes[-1])
    result["ma10"] = float(np.mean(closes[-10:])) if len(closes) >= 10 else float(closes[-1])
    result["ma20"] = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(closes[-1])
    result["ma60"] = float(np.mean(closes[-60:])) if len(closes) >= 60 else float(closes[-1])

    macd_line, signal_line, histogram = _calc_macd(closes)
    result["macd"]        = float(macd_line)
    result["macd_signal"] = float(signal_line)
    result["macd_hist"]   = float(histogram)

    result["rsi14"] = float(_calc_rsi(closes, 14))

    bb_upper, bb_mid, bb_lower = _calc_bollinger(closes, 20, 2.0)
    result["bb_upper"] = float(bb_upper)
    result["bb_mid"]   = float(bb_mid)
    result["bb_lower"] = float(bb_lower)

    result["atr14"] = float(_calc_atr(highs, lows, closes, 14))
    result["adx14"] = float(_calc_adx(highs, lows, closes, 14))

    result["vol_ma20"]  = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(volumes[-1])
    result["vol_ratio"] = float(volumes[-1] / result["vol_ma20"]) if result["vol_ma20"] > 0 else 1.0

    result["current_close"] = float(closes[-1])
    result["prev_close"]    = float(closes[-2]) if len(closes) >= 2 else float(closes[-1])
    result["change_pct"]    = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0.0

    return result


def detect_divergence(df: pd.DataFrame) -> str:
    closes = df["close"].values.astype(float)
    if len(closes) < 30:
        return "INSUFFICIENT_DATA"
    rsi_series = [_calc_rsi(closes[:i], 14) for i in range(20, len(closes) + 1)]
    if len(rsi_series) < 10:
        return "INSUFFICIENT_DATA"
    if min(closes[-10:]) < min(closes[-20:-10]) and min(rsi_series[-10:]) > min(rsi_series[-20:-10]):
        return "BULLISH_DIVERGENCE"
    if max(closes[-10:]) > max(closes[-20:-10]) and max(rsi_series[-10:]) < max(rsi_series[-20:-10]):
        return "BEARISH_DIVERGENCE"
    return "NO_DIVERGENCE"


def adx_regime(adx: float) -> str:
    return "TRENDING" if adx >= 25 else "RANGING"


# ─── Private helpers ────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    alpha  = 2.0 / (period + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _calc_macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= _ema(macd_line, signal)
    return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes[-(period + 1):])
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))


def _calc_bollinger(closes: np.ndarray, period: int = 20, multiplier: float = 2.0):
    if len(closes) < period:
        c = closes[-1]
        return c, c, c
    window = closes[-period:]
    mid    = np.mean(window)
    std    = np.std(window, ddof=1)
    return mid + multiplier * std, mid, mid - multiplier * std


def _calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float(np.mean(highs[-period:] - lows[-period:]))
    trs = []
    for i in range(1, period + 1):
        idx = -period - 1 + i
        trs.append(max(highs[idx] - lows[idx],
                       abs(highs[idx] - closes[idx - 1]),
                       abs(lows[idx]  - closes[idx - 1])))
    return float(np.mean(trs))


def _calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period * 2:
        return 20.0
    n        = len(closes)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr       = np.zeros(n)
    for i in range(1, n):
        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i]  = up   if up > down and up > 0   else 0
        minus_dm[i] = down if down > up and down > 0 else 0
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    atr      = _smooth(tr[1:],       period)
    plus_di  = 100 * _smooth(plus_dm[1:],  period) / (atr + 1e-10)
    minus_di = 100 * _smooth(minus_dm[1:], period) / (atr + 1e-10)
    dx       = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx      = _smooth(dx, period)
    return float(adx[-1]) if len(adx) > 0 else 20.0


def _smooth(values: np.ndarray, period: int) -> np.ndarray:
    result    = np.empty(len(values))
    result[0] = np.mean(values[:period]) if len(values) >= period else values[0]
    for i in range(1, len(values)):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period
    return result
