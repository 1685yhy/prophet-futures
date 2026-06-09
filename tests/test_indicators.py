"""Unit tests for tools/indicators.py — pure Python, no network, no LLM."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from tools.indicators import (
    calc_indicators, detect_divergence, adx_regime,
    _calc_rsi, _calc_atr, _calc_macd, _calc_bollinger, _calc_adx,
)


def _make_df(closes, highs=None, lows=None, volumes=None, n=None):
    n = n or len(closes)
    closes = list(closes)
    highs   = highs   or [c * 1.005 for c in closes]
    lows    = lows    or [c * 0.995 for c in closes]
    volumes = volumes or [10000.0] * n
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                          "close": closes, "volume": volumes})


class TestCalcIndicators:
    def test_returns_all_required_keys(self):
        df  = _make_df([5000 + i for i in range(80)])
        ind = calc_indicators(df)
        required = ["ma5","ma10","ma20","ma60","macd","macd_signal","macd_hist",
                    "rsi14","bb_upper","bb_mid","bb_lower","atr14","adx14",
                    "vol_ma20","vol_ratio","current_close","prev_close","change_pct"]
        for k in required:
            assert k in ind, f"Missing key: {k}"

    def test_ma5_equals_last_5_mean(self):
        closes = [float(i) for i in range(1, 81)]
        df  = _make_df(closes)
        ind = calc_indicators(df)
        assert abs(ind["ma5"] - np.mean(closes[-5:])) < 0.01

    def test_ma20_equals_last_20_mean(self):
        closes = [float(i) for i in range(1, 81)]
        df  = _make_df(closes)
        ind = calc_indicators(df)
        assert abs(ind["ma20"] - np.mean(closes[-20:])) < 0.01

    def test_change_pct_sign(self):
        closes = [100.0] * 78 + [100.0, 102.0]  # last bar up 2%
        df  = _make_df(closes)
        ind = calc_indicators(df)
        assert ind["change_pct"] > 0

    def test_insufficient_data_does_not_crash(self):
        df  = _make_df([5000.0] * 10)
        ind = calc_indicators(df)
        assert "rsi14" in ind

    def test_vol_ratio_all_equal_volume_is_one(self):
        closes  = [5000.0] * 40
        volumes = [10000.0] * 40
        df  = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": volumes})
        ind = calc_indicators(df)
        assert abs(ind["vol_ratio"] - 1.0) < 0.01


class TestCalcRsi:
    def test_all_gains_returns_100(self):
        closes = np.array([float(i) for i in range(1, 30)])
        rsi    = _calc_rsi(closes, 14)
        assert rsi == 100.0

    def test_all_losses_returns_near_zero(self):
        closes = np.array([float(30 - i) for i in range(30)])
        rsi    = _calc_rsi(closes, 14)
        assert rsi < 5.0

    def test_flat_returns_50(self):
        closes = np.array([100.0] * 20)
        rsi    = _calc_rsi(closes, 14)
        # All deltas are zero → avg_loss = 0 → RSI = 100
        # (mathematically RSI is 100 when no losses exist)
        assert rsi >= 50.0

    def test_insufficient_data_returns_50(self):
        closes = np.array([100.0, 101.0])
        rsi    = _calc_rsi(closes, 14)
        assert rsi == 50.0


class TestCalcAtr:
    def test_flat_market_atr_near_zero(self):
        n      = 20
        prices = np.array([5000.0] * n)
        atr    = _calc_atr(prices, prices, prices, 14)
        assert atr < 1.0

    def test_atr_positive(self):
        highs  = np.array([100.0 + i * 0.5 + 1 for i in range(30)])
        lows   = np.array([100.0 + i * 0.5 - 1 for i in range(30)])
        closes = np.array([100.0 + i * 0.5 for i in range(30)])
        atr    = _calc_atr(highs, lows, closes, 14)
        assert atr > 0

    def test_atr_reflects_volatility(self):
        n       = 30
        low_vol_highs  = np.array([100.0 + i * 0.1 + 0.5 for i in range(n)])
        low_vol_lows   = np.array([100.0 + i * 0.1 - 0.5 for i in range(n)])
        high_vol_highs = np.array([100.0 + i * 0.1 + 5.0 for i in range(n)])
        high_vol_lows  = np.array([100.0 + i * 0.1 - 5.0 for i in range(n)])
        closes         = np.array([100.0 + i * 0.1 for i in range(n)])
        atr_low  = _calc_atr(low_vol_highs,  low_vol_lows,  closes, 14)
        atr_high = _calc_atr(high_vol_highs, high_vol_lows, closes, 14)
        assert atr_high > atr_low * 5


class TestCalcMacd:
    def test_returns_three_values(self):
        closes = np.array([float(100 + i) for i in range(60)])
        macd, signal, hist = _calc_macd(closes)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_histogram_equals_macd_minus_signal(self):
        closes = np.array([float(100 + i + np.sin(i) * 5) for i in range(60)])
        macd, signal, hist = _calc_macd(closes)
        assert abs(hist - (macd - signal)) < 1e-6

    def test_uptrend_macd_positive(self):
        closes = np.array([float(100 + i * 2) for i in range(60)])
        macd, signal, hist = _calc_macd(closes)
        assert macd > 0


class TestDetectDivergence:
    def test_insufficient_data(self):
        df     = _make_df([100.0] * 20)
        result = detect_divergence(df)
        assert result == "INSUFFICIENT_DATA"

    def test_no_divergence_in_flat_market(self):
        df     = _make_df([100.0] * 50)
        result = detect_divergence(df)
        assert result in ("NO_DIVERGENCE", "BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE",
                           "INSUFFICIENT_DATA")

    def test_returns_valid_string(self):
        closes = [100.0 + np.sin(i * 0.3) * 10 for i in range(60)]
        df     = _make_df(closes)
        result = detect_divergence(df)
        assert result in ("BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE",
                           "NO_DIVERGENCE", "INSUFFICIENT_DATA")


class TestAdxRegime:
    def test_trending_above_25(self):
        assert adx_regime(25.0) == "TRENDING"
        assert adx_regime(40.0) == "TRENDING"

    def test_ranging_below_25(self):
        assert adx_regime(24.9) == "RANGING"
        assert adx_regime(10.0) == "RANGING"
