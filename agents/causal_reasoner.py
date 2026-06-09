"""Causal Reasoner Agent — do-calculus based event impact assessment."""

import logging
import json
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.causal_graph import query_causal_graph, do_intervention, build_futures_causal_graph
from models.schemas import CausalEffect

logger = logging.getLogger(__name__)


def _build_tools() -> list:
    return [
        Tool(name="query_causal_graph",
             func=lambda args: json.dumps(
                 query_causal_graph(*[a.strip() for a in args.split(",", 1)])),
             description="Query causal effects. Input: 'event_type, target_symbol'"),
        Tool(name="do_intervention",
             func=lambda args: json.dumps(
                 do_intervention(build_futures_causal_graph(), json.loads(args))),
             description="Estimate intervention effects. Input: JSON dict of {variable: delta}"),
    ]


def run_causal_reasoner(event_description: str, symbol: str) -> CausalEffect:
    result = invoke_structured(
        agent_name="causal_reasoner",
        tools=_build_tools(),
        input_text=f"Assess the causal effect of '{event_description}' on futures symbol {symbol}.",
        schema=CausalEffect, temperature=0.1, max_iterations=4,
    )
    if result is not None:
        return result

    logger.warning("Causal reasoner fallback for %s / %s", event_description, symbol)
    graph_result = query_causal_graph(event_description.replace(" ", "_"), symbol)
    direction    = graph_result.get("direction", "NEUTRAL")
    weight       = abs(graph_result.get("net_causal_weight", 0))
    strength     = "STRONG" if weight > 0.7 else ("MODERATE" if weight > 0.3 else "WEAK")
    return CausalEffect(
        direction=direction, strength=strength,
        chain=f"{event_description} → {symbol} (graph-based fallback)",
        confidence=min(0.8, weight),
    )
