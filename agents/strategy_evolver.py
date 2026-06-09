"""Strategy Evolver Agent — monthly hypothesis generation."""

import logging
import json
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from models.schemas import StrategyEvolverOutput, StrategyHypothesis

logger = logging.getLogger(__name__)


def run_strategy_evolver(performance_summary: dict) -> StrategyEvolverOutput:
    result = invoke_structured(
        agent_name="strategy_evolver",
        tools=[Tool(name="get_performance_summary",
                    func=lambda _: json.dumps(performance_summary),
                    description="Get recent strategy performance summary")],
        input_text="Generate 3 new, logically distinct trading strategy hypotheses.",
        schema=StrategyEvolverOutput, temperature=0.5, max_iterations=3,
    )
    if result is not None:
        return result

    logger.warning("Strategy evolver fallback")
    return StrategyEvolverOutput(
        analysis_period="Last 6 months",
        new_hypotheses=[
            StrategyHypothesis(
                name="Basis Mean Reversion",
                logic_description="When basis deviates >2SD from 60-day mean, trade convergence",
                pseudocode="if abs(basis - basis_ma60) > 2 * basis_std: trade_convergence()",
                robustness_score=0.72,
                market_regime_fit=["RANGING", "TRENDING_BULL"],
            ),
            StrategyHypothesis(
                name="Member Position Divergence",
                logic_description="Trade against excess retail crowding when institutions diverge",
                pseudocode="if retail_net > +3SD and inst_net < 0: go_short()",
                robustness_score=0.65,
                market_regime_fit=["TRENDING_BEAR", "HIGH_VOLATILITY"],
            ),
            StrategyHypothesis(
                name="Cross-Sector Spread Momentum",
                logic_description="Iron ore vs coke spread momentum following steel margin trends",
                pseudocode="if spread_momentum(i, j) > threshold and margin_trend > 0: long_i_short_j()",
                robustness_score=0.68,
                market_regime_fit=["TRENDING_BULL", "TRENDING_BEAR"],
            ),
        ],
        deprecated_strategies=[],
        reasoning="Fallback: well-known futures arbitrage principles",
    )
