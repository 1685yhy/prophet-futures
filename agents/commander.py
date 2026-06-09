"""Commander Agent — Directional Consensus Score (DCS) fusion decision engine."""

import logging
import math
from typing import Tuple, List

from models.schemas import (
    CommanderDecision, TechReport, FundReport, MacroReport, VisionReport,
    RegimeOutput, ScenarioReport, CausalEffect, MemoryReport,
    TrapAnalysisReport, CrowdingLevel,
)

logger = logging.getLogger(__name__)

DIR_SCORE = {
    "LONG": 1.0, "BULLISH": 1.0, "INFLOW": 1.0,
    "SHORT": -1.0, "BEARISH": -1.0, "OUTFLOW": -1.0,
    "NEUTRAL": 0.0, "WAIT": 0.0,
}

# DCS 阈值按 regime 动态调整（比原 sigmoid 阈值更直观）
DCS_THRESHOLD = {
    "TRENDING_BULL":   0.35,
    "TRENDING_BEAR":   0.35,
    "RANGING":         0.25,
    "HIGH_VOLATILITY": 0.40,
    "CRISIS":          0.50,
}

# 方向一致度最低要求：至少 60% 的信号同向
MIN_DIRECTION_AGREEMENT = 0.60


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _compute_dcs(
    signals: List[Tuple[float, float, float]]
) -> Tuple[float, float, float]:
    """
    Directional Consensus Score (DCS) — 替代 sigmoid 的融合公式。

    signals: List of (direction_score, confidence, weight)
             direction_score ∈ {-1, 0, +1}
             confidence ∈ [0, 1]
             weight ∈ [0, 1]

    Returns: (dcs, direction_agreement, weighted_conf)
      dcs               ∈ [-1, 1]  总体信号强度和方向
      direction_agreement ∈ [0, 1]  方向一致度（0=完全分歧，1=完全一致）
      weighted_conf     ∈ [0, 1]  加权平均置信度
    """
    non_neutral = [(s, c, w) for s, c, w in signals if s != 0]
    if not non_neutral:
        return 0.0, 0.0, 0.0

    directions = [s for s, c, w in non_neutral]
    max_abs    = max(abs(d) for d in directions)

    # consensus ∈ [-1, 1]：方向一致时为 ±1，完全分歧时为 0
    consensus = sum(directions) / (len(directions) * max_abs)

    # direction_agreement：同向信号占比
    dominant_sign      = 1 if consensus >= 0 else -1
    same_direction_cnt = sum(1 for d in directions if d * dominant_sign > 0)
    direction_agreement= same_direction_cnt / len(directions)

    # 加权平均置信度（只计非中性信号）
    total_w        = sum(w for _, _, w in non_neutral)
    weighted_conf  = sum(abs(s) * c * w for s, c, w in non_neutral) / (total_w + 1e-8)

    # DCS = 方向一致性 × 加权置信度
    dcs = consensus * weighted_conf

    return dcs, direction_agreement, weighted_conf


def bayesian_fusion(
    tech: TechReport, fund: FundReport, macro: MacroReport, vision: VisionReport,
    regime: RegimeOutput, scenario: ScenarioReport, causal: CausalEffect,
    memory: MemoryReport, trap: TrapAnalysisReport, crowding: CrowdingLevel,
) -> CommanderDecision:
    symbol       = tech.symbol
    veto_reasons = []

    # ── 否决规则 ────────────────────────────────────────────────────────────
    if crowding.score > 85:
        veto_reasons.append(f"拥挤度 {crowding.score} > 85")
    if memory.historical_win_rate < 0.45:
        veto_reasons.append(f"历史胜率 {memory.historical_win_rate:.1%} < 45%")
    if scenario.worst_case_loss_pct > 5.0:
        veto_reasons.append(f"最坏情景亏损 {scenario.worst_case_loss_pct:.1f}% > 5%")
    if causal.direction == "NEGATIVE" and causal.strength == "STRONG":
        if tech.signal.direction == "LONG":
            veto_reasons.append(f"因果否决：强烈负向效应（{causal.chain}）")
    if trap.trap.type != "NONE" and trap.trap.confidence > 0.70:
        if (trap.trap.type == "BULL_TRAP" and tech.signal.direction == "LONG") or \
           (trap.trap.type == "BEAR_TRAP" and tech.signal.direction == "SHORT"):
            veto_reasons.append(f"陷阱否决：{trap.trap.type} 置信度={trap.trap.confidence:.0%}")

    if veto_reasons:
        return CommanderDecision(
            symbol=symbol, action="WAIT", confidence=0.0,
            position_size_pct=0.0,
            reasoning="否决: " + "; ".join(veto_reasons),
            posterior_probability=0.0, veto_reasons=veto_reasons,
        )

    # ── DCS 融合 ─────────────────────────────────────────────────────────────
    w = regime.recommended_weights
    signals = [
        (DIR_SCORE.get(tech.signal.direction, 0),  tech.confidence,  w.get("tech",   0.35)),
        (DIR_SCORE.get(fund.net_flow, 0),           fund.confidence,  w.get("fund",   0.30)),
        (DIR_SCORE.get(macro.macro_trend, 0),       macro.confidence, w.get("macro",  0.25)),
        (DIR_SCORE.get(vision.visual_signal, 0),    0.55,             w.get("vision", 0.10)),
    ]

    dcs, direction_agreement, weighted_conf = _compute_dcs(signals)

    # 情景惩罚
    if scenario.worst_case_loss_pct > 3.0:
        dcs *= 0.8

    # 记忆修正（胜率偏离 50% 的线性调整，最大 ±0.10）
    memory_adj = (memory.historical_win_rate - 0.5) * 0.2
    dcs = max(-1.0, min(1.0, dcs + memory_adj))

    threshold = DCS_THRESHOLD.get(regime.regime, 0.35)

    # posterior_probability 用 sigmoid(dcs * 5) 保持向后兼容
    posterior = _sigmoid(dcs * 5)

    if abs(dcs) < threshold or direction_agreement < MIN_DIRECTION_AGREEMENT:
        return CommanderDecision(
            symbol=symbol, action="WAIT",
            confidence=round(abs(dcs), 3),
            position_size_pct=0.0,
            reasoning=(f"信号不足: DCS={dcs:.3f} (需≥{threshold}), "
                       f"一致度={direction_agreement:.0%} (需≥{MIN_DIRECTION_AGREEMENT:.0%})"),
            posterior_probability=round(posterior, 3), veto_reasons=[],
        )

    action = "LONG" if dcs > 0 else "SHORT"
    # 仓位按 DCS 强度线性缩放，最大 2%
    position_pct = round(min(0.02, 0.02 * abs(dcs) / threshold), 4)

    return CommanderDecision(
        symbol=symbol, action=action,
        confidence=round(min(0.95, abs(dcs)), 3),
        entry_price=None,
        stop_loss=tech.stop_loss,
        target_price=tech.target_price,
        position_size_pct=position_pct,
        reasoning=(f"DCS融合: DCS={dcs:.3f}, 一致度={direction_agreement:.0%}, "
                   f"加权置信={weighted_conf:.2f}, regime={regime.regime}"),
        posterior_probability=round(posterior, 3), veto_reasons=[],
    )


def run_commander(
    tech: TechReport, fund: FundReport, macro: MacroReport, vision: VisionReport,
    regime: RegimeOutput, scenario: ScenarioReport, causal: CausalEffect,
    memory: MemoryReport, trap: TrapAnalysisReport, crowding: CrowdingLevel,
) -> CommanderDecision:
    return bayesian_fusion(tech, fund, macro, vision, regime,
                           scenario, causal, memory, trap, crowding)
