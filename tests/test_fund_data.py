"""Unit tests for fund_data.py — OI trend calculation and intraday pattern."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from tools.fund_data import _is_eod_window, _synthetic_volume_oi


class TestIsEodWindow:
    def test_eod_at_14_30(self):
        from datetime import datetime
        with patch("tools.fund_data.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 9, 14, 30)
            assert _is_eod_window() is True

    def test_eod_at_14_45(self):
        from datetime import datetime
        with patch("tools.fund_data.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 9, 14, 45)
            assert _is_eod_window() is True

    def test_not_eod_at_10_00(self):
        from datetime import datetime
        with patch("tools.fund_data.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 9, 10, 0)
            assert _is_eod_window() is False

    def test_not_eod_at_14_29(self):
        from datetime import datetime
        with patch("tools.fund_data.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 9, 14, 29)
            assert _is_eod_window() is False

    def test_eod_at_15_00(self):
        from datetime import datetime
        with patch("tools.fund_data.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 9, 15, 0)
            assert _is_eod_window() is True


class TestOiTrendDirection:
    """验证 get_volume_oi 中多日OI趋势计算正确性。
    akshare 在函数内部 import，用 patch 整个 futures_main_sina 路径。"""

    def _make_mock_df(self, oi_values: list) -> pd.DataFrame:
        n = len(oi_values)
        df = pd.DataFrame({
            "date":           [f"2026-06-{i:02d}" for i in range(1, n+1)],
            "open":           [5000.0] * n,
            "high":           [5010.0] * n,
            "low":            [4990.0] * n,
            "close":          [5005.0] * n,
            "成交量":          [10000.0] * n,
            "持仓量":          [float(v) for v in oi_values],
            "settle":         [5000.0] * n,
        })
        return df

    @patch("akshare.futures_main_sina")
    def test_accumulating_when_both_3d_5d_positive(self, mock_sina):
        oi = [100000, 101000, 102000, 103000, 104000, 105000, 106000]
        mock_sina.return_value = self._make_mock_df(oi)
        from tools.fund_data import get_volume_oi
        result = get_volume_oi("rb")
        assert result["oi_trend_direction"] == "ACCUMULATING"
        assert result["oi_trend_3d"] > 0
        assert result["oi_trend_5d"] > 0

    @patch("akshare.futures_main_sina")
    def test_reducing_when_both_3d_5d_negative(self, mock_sina):
        oi = [106000, 105000, 104000, 103000, 102000, 101000, 100000]
        mock_sina.return_value = self._make_mock_df(oi)
        from tools.fund_data import get_volume_oi
        result = get_volume_oi("rb")
        assert result["oi_trend_direction"] == "REDUCING"
        assert result["oi_trend_3d"] < 0

    @patch("akshare.futures_main_sina")
    def test_flat_when_mixed(self, mock_sina):
        # 3日为正（103→105→107），5日为负（110→...→107） → FLAT
        oi = [110000, 108000, 106000, 103000, 105000, 106000, 107000]
        mock_sina.return_value = self._make_mock_df(oi)
        from tools.fund_data import get_volume_oi
        result = get_volume_oi("rb")
        # oi_3d = 107000 - 103000 = +4000（正）
        # oi_5d = 107000 - 108000 = -1000（负）→ FLAT
        assert result["oi_trend_direction"] == "FLAT"


class TestSyntheticVolumeOi:
    def test_returns_all_required_keys(self):
        result = _synthetic_volume_oi("rb")
        required = ["volume_today", "volume_ma5", "vol_ratio", "open_interest",
                    "oi_change", "oi_change_pct", "oi_trend_3d", "oi_trend_5d",
                    "oi_trend_direction"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_oi_trend_direction_is_valid(self):
        for sym in ["rb", "cu", "lh", "sc"]:
            result = _synthetic_volume_oi(sym)
            assert result["oi_trend_direction"] in ("ACCUMULATING", "REDUCING", "FLAT")
