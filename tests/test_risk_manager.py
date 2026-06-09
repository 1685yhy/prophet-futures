"""Unit tests for agents/risk_manager.py — position sizing and order generation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from agents.risk_manager import run_risk_manager
from models.schemas import CommanderDecision, IgniterSignal, RiskOrder


def _make_decision(action="SHORT", stop=12080.0, target=11400.0, symbol="lh"):
    return CommanderDecision(
        symbol=symbol, action=action, confidence=0.65,
        stop_loss=stop, target_price=target,
        position_size_pct=0.016, reasoning="test",
        posterior_probability=0.78,
    )


def _make_trigger(price=11870.0, triggered=True, symbol="lh"):
    return IgniterSignal(
        symbol=symbol, triggered=triggered,
        trigger_reason="test trigger", trigger_price=price,
        timestamp="2026-06-09T10:00:00",
    )


class TestRiskManagerWait:
    def test_wait_decision_returns_empty_orders(self):
        decision = _make_decision(action="WAIT")
        trigger  = _make_trigger()
        result   = run_risk_manager(decision, trigger, capital=1_000_000)
        assert result.orders == []
        assert result.max_loss == 0.0
        assert result.risk_pct == 0.0


class TestPositionSizing:
    @patch("agents.risk_manager.get_kline")
    @patch("agents.risk_manager.run_abm")
    @patch("agents.risk_manager.rl_executor")
    def test_quantity_within_bounds(self, mock_rl, mock_abm, mock_kline):
        import pandas as pd
        mock_kline.return_value = MagicMock(
            opens=[11800.0]*30, highs=[11900.0]*30,
            lows=[11700.0]*30, closes=[11850.0]*30, volumes=[100000.0]*30,
        )
        mock_abm.return_value = MagicMock(optimal_entry_offset=0.0)
        from models.schemas import Order
        mock_rl.return_value = [Order(symbol="lh", side="SELL",
                                       order_type="LIMIT", price=11870.0, quantity=5.0)]

        decision = _make_decision(action="SHORT", stop=12080.0, target=11400.0)
        trigger  = _make_trigger(price=11870.0)
        result   = run_risk_manager(decision, trigger, capital=1_000_000)

        if result.orders:
            total_qty = sum(o.quantity for o in result.orders)
            assert 1.0 <= total_qty <= 20.0

    @patch("agents.risk_manager.get_kline")
    @patch("agents.risk_manager.run_abm")
    @patch("agents.risk_manager.rl_executor")
    def test_max_loss_within_2pct_capital(self, mock_rl, mock_abm, mock_kline):
        mock_kline.return_value = MagicMock(
            opens=[11800.0]*30, highs=[11900.0]*30,
            lows=[11700.0]*30, closes=[11850.0]*30, volumes=[100000.0]*30,
        )
        mock_abm.return_value = MagicMock(optimal_entry_offset=0.0)
        from models.schemas import Order
        mock_rl.return_value = [Order(symbol="lh", side="SELL",
                                       order_type="LIMIT", price=11870.0, quantity=5.0)]

        capital  = 1_000_000
        decision = _make_decision(action="SHORT", stop=12080.0, target=11400.0)
        trigger  = _make_trigger(price=11870.0)
        result   = run_risk_manager(decision, trigger, capital=capital)
        assert result.risk_pct <= 0.02


class TestPLRatioGate:
    @patch("agents.risk_manager.get_kline")
    @patch("agents.risk_manager.run_abm")
    def test_low_pl_ratio_rejected(self, mock_abm, mock_kline):
        mock_kline.return_value = MagicMock(
            opens=[11800.0]*30, highs=[11900.0]*30,
            lows=[11700.0]*30, closes=[11850.0]*30, volumes=[100000.0]*30,
        )
        mock_abm.return_value = MagicMock(optimal_entry_offset=0.0)
        # stop=11900, target=11870 → stop_dist=30, reward=0 → pl_ratio≈0
        decision = _make_decision(action="SHORT", stop=11900.0, target=11870.0)
        trigger  = _make_trigger(price=11870.0)
        result   = run_risk_manager(decision, trigger, capital=1_000_000)
        assert result.orders == []
        assert "P/L" in result.execution_notes or "拒绝" in result.execution_notes or "Rejected" in result.execution_notes

    @patch("agents.risk_manager.get_kline")
    @patch("agents.risk_manager.run_abm")
    @patch("agents.risk_manager.rl_executor")
    def test_good_pl_ratio_accepted(self, mock_rl, mock_abm, mock_kline):
        mock_kline.return_value = MagicMock(
            opens=[11800.0]*30, highs=[11900.0]*30,
            lows=[11700.0]*30, closes=[11850.0]*30, volumes=[100000.0]*30,
        )
        mock_abm.return_value = MagicMock(optimal_entry_offset=0.0)
        from models.schemas import Order
        mock_rl.return_value = [Order(symbol="lh", side="SELL",
                                       order_type="LIMIT", price=11870.0, quantity=5.0)]
        # stop=12080 (+210), target=11400 (-470) → pl_ratio=2.24
        decision = _make_decision(action="SHORT", stop=12080.0, target=11400.0)
        trigger  = _make_trigger(price=11870.0)
        result   = run_risk_manager(decision, trigger, capital=1_000_000)
        assert len(result.orders) > 0


class TestZeroStopDistance:
    @patch("agents.risk_manager.get_kline")
    @patch("agents.risk_manager.run_abm")
    @patch("agents.risk_manager.rl_executor")
    def test_zero_stop_distance_uses_default(self, mock_rl, mock_abm, mock_kline):
        mock_kline.return_value = MagicMock(
            opens=[11870.0]*30, highs=[11870.0]*30,
            lows=[11870.0]*30, closes=[11870.0]*30, volumes=[100000.0]*30,
        )
        mock_abm.return_value = MagicMock(optimal_entry_offset=0.0)
        from models.schemas import Order
        mock_rl.return_value = [Order(symbol="lh", side="SELL",
                                       order_type="LIMIT", price=11870.0, quantity=1.0)]
        # stop == entry → stop_distance = 0 → should use 2% default
        decision = _make_decision(action="SHORT", stop=11870.0, target=11400.0)
        trigger  = _make_trigger(price=11870.0)
        result   = run_risk_manager(decision, trigger, capital=1_000_000)
        assert result.risk_pct <= 0.02
