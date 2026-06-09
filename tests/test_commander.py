"""Unit tests for agents/commander.py — DCS fusion and veto logic."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.commander import (
    _compute_dcs, _sigmoid, bayesian_fusion, run_commander,
    DCS_THRESHOLD, MIN_DIRECTION_AGREEMENT, DIR_SCORE,
)
from models.schemas import (
    TechReport, TechSignal, FundReport, MacroReport, VisionReport,
    RegimeOutput, ScenarioReport, ScenarioPath, CausalEffect,
    MemoryReport, MemoryCase, TrapAnalysisReport, TrapType, CrowdingLevel,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_reports(
    tech_dir="SHORT", tech_conf=0.65,
    fund_flow="OUTFLOW", fund_conf=0.55,
    macro_trend="BEARISH", macro_conf=0.60,
    vision_signal="BEARISH",
    regime="RANGING",
    worst_loss=2.0,
    hist_wr=0.58,
    trap_type="NONE", trap_conf=0.20,
    crowding_score=50,
    causal_dir="NEUTRAL", causal_str="WEAK",
):
    tech = TechReport(
        symbol="lh", signal=TechSignal(direction=tech_dir, strength="MODERATE",
                                        reasoning="test"),
        key_support=11400.0, key_resistance=12000.0,
        stop_loss=12080.0 if tech_dir == "SHORT" else 11300.0,
        target_price=11400.0 if tech_dir == "SHORT" else 12500.0,
        confidence=tech_conf,
    )
    fund = FundReport(
        symbol="lh", net_flow=fund_flow, basis_status="CONTANGO",
        top_member_action="test", confidence=fund_conf, reasoning="test",
    )
    macro = MacroReport(
        sector="agriculture", macro_trend=macro_trend,
        key_drivers=["test"], risk_events=["test"],
        confidence=macro_conf, reasoning="test",
    )
    vision = VisionReport(
        symbol="lh", chart_pattern="test", visual_signal=vision_signal,
        pattern_completion_pct=60.0, reasoning="test",
    )
    weights = {"RANGING": {"tech":0.25,"fund":0.40,"macro":0.25,"vision":0.10},
               "TRENDING_BULL": {"tech":0.40,"fund":0.30,"macro":0.20,"vision":0.10},
               "TRENDING_BEAR": {"tech":0.40,"fund":0.30,"macro":0.20,"vision":0.10},
               "HIGH_VOLATILITY":{"tech":0.20,"fund":0.20,"macro":0.50,"vision":0.10},
               "CRISIS":        {"tech":0.15,"fund":0.15,"macro":0.60,"vision":0.10}}
    regime_obj = RegimeOutput(
        regime=regime, adx_value=19.0, vix_equivalent=15.0,
        recommended_weights=weights[regime], reasoning="test",
    )
    scenario = ScenarioReport(
        paths=[ScenarioPath(probability=0.5, description="test",
                             target_price=11500.0, key_trigger="test")],
        worst_case_loss_pct=worst_loss,
    )
    causal = CausalEffect(direction=causal_dir, strength=causal_str,
                           chain="test chain", confidence=0.5)
    memory = MemoryReport(
        similar_cases=[],
        historical_win_rate=hist_wr, avg_profit_loss_ratio=1.8,
        conclusion="test",
    )
    trap = TrapAnalysisReport(
        symbol="lh",
        trap=TrapType(type=trap_type, current_phase="obs",
                      trigger_to_confirm="watch", confidence=trap_conf),
        reasoning="test",
    )
    crowding = CrowdingLevel(symbol="lh", score=crowding_score,
                              warning=crowding_score > 80,
                              similar_funds_pct=0.4, reasoning="test")
    return tech, fund, macro, vision, regime_obj, scenario, causal, memory, trap, crowding


# ── DCS core tests ────────────────────────────────────────────────────────────

class TestComputeDcs:
    def test_all_short_dcs_negative_exceeds_ranging_threshold(self):
        signals = [(-1, 0.65, 0.25), (-1, 0.55, 0.40), (-1, 0.60, 0.25), (-1, 0.55, 0.10)]
        dcs, agree, conf = _compute_dcs(signals)
        assert dcs < 0
        assert abs(dcs) >= DCS_THRESHOLD["RANGING"]
        assert agree == 1.0

    def test_all_long_dcs_positive(self):
        signals = [(1, 0.70, 0.25), (1, 0.65, 0.40), (1, 0.60, 0.25), (1, 0.55, 0.10)]
        dcs, agree, conf = _compute_dcs(signals)
        assert dcs > 0
        assert agree == 1.0

    def test_mixed_signals_low_agreement(self):
        signals = [(1, 0.7, 0.25), (-1, 0.6, 0.40), (1, 0.5, 0.25), (-1, 0.5, 0.10)]
        dcs, agree, conf = _compute_dcs(signals)
        assert agree < MIN_DIRECTION_AGREEMENT

    def test_all_neutral_returns_zero(self):
        signals = [(0, 0.5, 0.25), (0, 0.5, 0.40), (0, 0.5, 0.25), (0, 0.5, 0.10)]
        dcs, agree, conf = _compute_dcs(signals)
        assert dcs == 0.0
        assert agree == 0.0

    def test_dcs_range(self):
        signals = [(-1, 1.0, 0.25), (-1, 1.0, 0.40), (-1, 1.0, 0.25), (-1, 1.0, 0.10)]
        dcs, _, _ = _compute_dcs(signals)
        assert -1.0 <= dcs <= 1.0

    def test_higher_confidence_gives_larger_abs_dcs(self):
        low_conf  = [(-1, 0.30, 0.25), (-1, 0.30, 0.40), (-1, 0.30, 0.25), (-1, 0.30, 0.10)]
        high_conf = [(-1, 0.90, 0.25), (-1, 0.90, 0.40), (-1, 0.90, 0.25), (-1, 0.90, 0.10)]
        dcs_low,  _, _ = _compute_dcs(low_conf)
        dcs_high, _, _ = _compute_dcs(high_conf)
        assert abs(dcs_high) > abs(dcs_low)


# ── Veto tests ────────────────────────────────────────────────────────────────

class TestVetoRules:
    def test_crowding_above_85_vetoes(self):
        args = _make_reports(crowding_score=90)
        result = run_commander(*args)
        assert result.action == "WAIT"
        assert any("拥挤度" in r for r in result.veto_reasons)

    def test_low_win_rate_vetoes(self):
        args = _make_reports(hist_wr=0.40)
        result = run_commander(*args)
        assert result.action == "WAIT"
        assert any("历史胜率" in r for r in result.veto_reasons)

    def test_worst_case_loss_above_5pct_vetoes(self):
        args = _make_reports(worst_loss=6.0)
        result = run_commander(*args)
        assert result.action == "WAIT"
        assert any("最坏情景" in r for r in result.veto_reasons)

    def test_causal_veto_negative_strong_long(self):
        args = _make_reports(
            tech_dir="LONG", fund_flow="INFLOW", macro_trend="BULLISH",
            vision_signal="BULLISH",
            causal_dir="NEGATIVE", causal_str="STRONG",
        )
        result = run_commander(*args)
        assert result.action == "WAIT"
        assert any("因果否决" in r for r in result.veto_reasons)

    def test_causal_veto_does_not_fire_on_short(self):
        args = _make_reports(causal_dir="NEGATIVE", causal_str="STRONG")
        result = run_commander(*args)
        assert "因果否决" not in " ".join(result.veto_reasons)

    def test_bull_trap_vetoes_long(self):
        args = _make_reports(
            tech_dir="LONG", fund_flow="INFLOW", macro_trend="BULLISH",
            vision_signal="BULLISH", trap_type="BULL_TRAP", trap_conf=0.80,
        )
        result = run_commander(*args)
        assert result.action == "WAIT"
        assert any("陷阱否决" in r for r in result.veto_reasons)

    def test_trap_below_confidence_threshold_does_not_veto(self):
        args = _make_reports(trap_type="BULL_TRAP", trap_conf=0.50)
        result = run_commander(*args)
        assert not any("陷阱否决" in r for r in result.veto_reasons)


# ── Decision output tests ─────────────────────────────────────────────────────

class TestDecisionOutput:
    def test_all_short_signals_produce_short(self):
        args   = _make_reports()  # default is all-SHORT
        result = run_commander(*args)
        assert result.action == "SHORT"
        assert result.confidence > 0
        assert result.stop_loss is not None

    def test_all_long_signals_produce_long(self):
        args = _make_reports(
            tech_dir="LONG", fund_flow="INFLOW", macro_trend="BULLISH",
            vision_signal="BULLISH",
        )
        result = run_commander(*args)
        assert result.action == "LONG"

    def test_mixed_signals_produce_wait(self):
        # 2 long, 2 short → agreement < 60%
        args = _make_reports(
            tech_dir="LONG", fund_flow="OUTFLOW",
            macro_trend="BULLISH", vision_signal="BEARISH",
        )
        result = run_commander(*args)
        assert result.action == "WAIT"

    def test_position_size_within_bounds(self):
        args   = _make_reports()
        result = run_commander(*args)
        assert 0.0 <= result.position_size_pct <= 0.02

    def test_posterior_probability_in_range(self):
        args   = _make_reports()
        result = run_commander(*args)
        assert 0.0 <= result.posterior_probability <= 1.0
