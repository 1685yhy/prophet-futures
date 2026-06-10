"""Unit tests for trap_detector fallback — verifying the two new filters."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from agents.trap_detector import run_trap_detector


def _mock_oi_data(vol_ratio=0.7, oi_change_pct=-3.0,
                  oi_trend_direction="FLAT"):
    return {
        "vol_ratio":          vol_ratio,
        "oi_change_pct":      oi_change_pct,
        "oi_change":          -1000,
        "oi_trend_direction": oi_trend_direction,
        "oi_trend_3d":        -3000,
        "oi_trend_5d":        -5000,
    }


class TestEodWindowFilter:
    """过滤器A：尾盘时间窗口内减仓不应判断为BULL_TRAP。"""

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=True)   # 模拟14:45
    def test_eod_oi_reduction_not_bull_trap(self, mock_eod, mock_oi, mock_invoke):
        mock_oi.return_value = _mock_oi_data(vol_ratio=0.7, oi_change_pct=-4.0,
                                              oi_trend_direction="FLAT")
        result = run_trap_detector("lh")
        assert result.trap.type == "NONE", "尾盘OI减少不应判断为BULL_TRAP"
        assert result.trap.confidence <= 0.20

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=False)  # 非尾盘
    def test_non_eod_oi_reduction_can_be_bull_trap(self, mock_eod, mock_oi, mock_invoke):
        mock_oi.return_value = _mock_oi_data(vol_ratio=0.7, oi_change_pct=-4.0,
                                              oi_trend_direction="REDUCING")
        result = run_trap_detector("lh")
        assert result.trap.type == "BULL_TRAP", "非尾盘缩量+OI减+趋势减 应为BULL_TRAP"
        assert result.trap.confidence >= 0.50


class TestTrendCrossFilter:
    """过滤器B：3日OI趋势积累时，当日减仓不应判断为BULL_TRAP。"""

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=False)
    def test_accumulating_trend_overrides_daily_reduction(self, mock_eod, mock_oi, mock_invoke):
        # 6-9的真实场景：日内减仓但3日趋势是积累的
        mock_oi.return_value = _mock_oi_data(vol_ratio=0.7, oi_change_pct=-3.0,
                                              oi_trend_direction="ACCUMULATING")
        result = run_trap_detector("lh")
        assert result.trap.type == "NONE", "3日趋势ACCUMULATING时日内减仓不应为BULL_TRAP"

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=False)
    def test_reducing_trend_allows_bull_trap(self, mock_eod, mock_oi, mock_invoke):
        mock_oi.return_value = _mock_oi_data(vol_ratio=0.7, oi_change_pct=-4.0,
                                              oi_trend_direction="REDUCING")
        result = run_trap_detector("lh")
        assert result.trap.type == "BULL_TRAP"

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=False)
    def test_no_oi_reduction_always_none(self, mock_eod, mock_oi, mock_invoke):
        mock_oi.return_value = _mock_oi_data(vol_ratio=1.2, oi_change_pct=+2.0,
                                              oi_trend_direction="ACCUMULATING")
        result = run_trap_detector("lh")
        assert result.trap.type == "NONE"


class TestCombinedFilters:
    """组合测试：验证优先级顺序（EOD > 趋势 > 缩量）。"""

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=True)
    def test_eod_takes_priority_over_trend(self, mock_eod, mock_oi, mock_invoke):
        # 尾盘 + 趋势也是REDUCING → 仍应NONE（尾盘优先）
        mock_oi.return_value = _mock_oi_data(vol_ratio=0.6, oi_change_pct=-5.0,
                                              oi_trend_direction="REDUCING")
        result = run_trap_detector("lh")
        assert result.trap.type == "NONE", "尾盘过滤器优先级高于趋势判断"

    @patch("agents.trap_detector.invoke_structured", return_value=None)
    @patch("agents.trap_detector.get_volume_oi")
    @patch("tools.fund_data._is_eod_window", return_value=False)
    def test_high_vol_ratio_not_bull_trap(self, mock_eod, mock_oi, mock_invoke):
        # 放量+OI减（不是缩量）→ 不判断BULL_TRAP
        mock_oi.return_value = _mock_oi_data(vol_ratio=1.5, oi_change_pct=-3.0,
                                              oi_trend_direction="REDUCING")
        result = run_trap_detector("lh")
        assert result.trap.type == "NONE", "放量时OI减少不是缩量诱多，不判断BULL_TRAP"
