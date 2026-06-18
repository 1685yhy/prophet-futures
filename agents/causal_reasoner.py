"""Causal Reasoner Agent — do-calculus based event impact assessment."""

import logging
import json
from langchain_core.tools import Tool

from tools.llm_utils import invoke_structured
from tools.causal_graph import query_causal_graph, do_intervention, build_futures_causal_graph
from models.schemas import CausalEffect

logger = logging.getLogger(__name__)


def _build_tools() -> list:
    from langchain_core.tools import tool

    @tool
    def query_causal_graph_tool(event_type: str = "", target_symbol: str = "") -> str:
        """Query causal effects between an event type and a target symbol.
        event_type can be a string like 'market_policy_change' or a JSON dict."""
        # Handle case where LLM passes a dict as event_type
        import ast
        et = str(event_type).strip()
        if et.startswith('{'):
            try:
                d = ast.literal_eval(et) if isinstance(event_type, str) else event_type
                if isinstance(d, dict):
                    # Extract the VALUE (e.g., "market_policy_change") not the key
                    vals = list(d.values())
                    et = str(vals[0]) if vals else ""
            except Exception:
                pass
        ts = str(target_symbol).strip()
        return json.dumps(query_causal_graph(et, ts))

    @tool
    def do_intervention_tool(intervention: str = "") -> str:
        """Estimate do-calculus intervention effects. Input: JSON string like '{"variable": delta}'."""
        from tools.causal_graph import do_intervention as _do, build_futures_causal_graph
        data = intervention or "{}"
        return json.dumps(_do(build_futures_causal_graph(), data))

    return [query_causal_graph_tool, do_intervention_tool]


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
