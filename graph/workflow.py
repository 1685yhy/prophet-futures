"""LangGraph workflow — full multi-agent trading pipeline."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from graph.state import TradingState
from agents.scanner import run_scanner
from agents.technician import run_technician
from agents.fund_analyst import run_fund_analyst
from agents.macro_analyst import run_macro_analyst
from agents.vision_tech import run_vision_tech
from agents.regime_detector import run_regime_detector
from agents.scenario_engine import run_scenario_engine
from agents.causal_reasoner import run_causal_reasoner
from agents.memory_retriever import run_memory_retriever
from agents.trap_detector import run_trap_detector
from agents.crowding_radar import run_crowding_radar
from agents.commander import run_commander
from agents.igniter import run_igniter
from agents.risk_manager import run_risk_manager
from agents.meta_cognition import run_meta_cognition
from tools.llm_utils import load_config

logger = logging.getLogger(__name__)
cfg    = load_config()


def scanner_node(state: TradingState) -> TradingState:
    logger.info("=== SCANNER ===")
    try:
        scan = run_scanner()
        return {**state, "candidates": scan.candidates, "scan_output": scan,
                "errors": state.get("errors", [])}
    except Exception as e:
        logger.error("Scanner failed: %s", e)
        return {**state, "candidates": ["rb", "cu"], "errors": state.get("errors", []) + [str(e)]}


def analyze_symbol_node(state: TradingState) -> TradingState:
    candidates = state.get("candidates", [])
    if not candidates:
        return {**state, "final_output": "No candidates found", "active_symbol": None}
    symbol = candidates[0]
    logger.info("=== ANALYSIS: %s ===", symbol)
    with ThreadPoolExecutor(max_workers=4) as ex:
        tf = ex.submit(run_technician, symbol)
        ff = ex.submit(run_fund_analyst, symbol)
        mf = ex.submit(run_macro_analyst, symbol)
        vf = ex.submit(run_vision_tech, symbol)
        tech = tf.result(); fund = ff.result()
        macro = mf.result(); vision = vf.result()
    regime = run_regime_detector(symbol)
    return {**state, "active_symbol": symbol, "current_symbol": symbol,
            "technical_report": tech, "fund_report": fund,
            "macro_report": macro, "vision_report": vision, "regime": regime}


def advanced_cognition_node(state: TradingState) -> TradingState:
    symbol = state.get("active_symbol", "rb")
    logger.info("=== ADVANCED COGNITION: %s ===", symbol)
    with ThreadPoolExecutor(max_workers=5) as ex:
        sf = ex.submit(run_scenario_engine, symbol)
        cf = ex.submit(run_causal_reasoner, "market_policy_change", symbol)
        mf = ex.submit(run_memory_retriever, symbol)
        tf = ex.submit(run_trap_detector, symbol)
        crf= ex.submit(run_crowding_radar, symbol)
        scenario = sf.result(); causal = cf.result()
        memory = mf.result(); trap = tf.result(); crowding = crf.result()
    return {**state, "scenario_report": scenario, "causal_report": causal,
            "memory_report": memory, "trap_report": trap, "crowding": crowding}


def commander_node(state: TradingState) -> TradingState:
    logger.info("=== COMMANDER ===")
    try:
        decision = run_commander(
            tech=state["technical_report"], fund=state["fund_report"],
            macro=state["macro_report"],   vision=state["vision_report"],
            regime=state["regime"],         scenario=state["scenario_report"],
            causal=state["causal_report"],  memory=state["memory_report"],
            trap=state["trap_report"],      crowding=state["crowding"],
        )
        logger.info("Decision: %s (confidence=%.2f)", decision.action, decision.confidence)
        return {**state, "commander_decision": decision}
    except Exception as e:
        logger.error("Commander failed: %s", e)
        from models.schemas import CommanderDecision
        return {**state, "commander_decision": CommanderDecision(
            symbol=state.get("active_symbol", "unknown"), action="WAIT",
            confidence=0.0, position_size_pct=0.0,
            reasoning=f"Error: {e}", posterior_probability=0.0,
        )}


def should_trade(state: TradingState) -> str:
    decision = state.get("commander_decision")
    return "igniter" if (decision and decision.action in ("LONG", "SHORT")) else "meta_cognition"


def igniter_node(state: TradingState) -> TradingState:
    logger.info("=== IGNITER ===")
    try:
        trigger = run_igniter(state["commander_decision"])
        return {**state, "igniter_trigger": trigger}
    except Exception as e:
        logger.error("Igniter failed: %s", e)
        from models.schemas import IgniterSignal
        return {**state, "igniter_trigger": IgniterSignal(
            symbol=state.get("active_symbol", "unknown"), triggered=False,
            trigger_reason=f"Error: {e}", trigger_price=0.0,
            timestamp=datetime.now().isoformat(),
        )}


def should_execute(state: TradingState) -> str:
    trigger = state.get("igniter_trigger")
    return "risk_manager" if (trigger and trigger.triggered) else "meta_cognition"


def risk_manager_node(state: TradingState) -> TradingState:
    logger.info("=== RISK MANAGER ===")
    try:
        capital    = cfg.get("risk", {}).get("capital", 1_000_000)
        risk_order = run_risk_manager(state["commander_decision"], state["igniter_trigger"], capital)
        summary    = {
            "symbol": state.get("active_symbol"),
            "action": state["commander_decision"].action,
            "orders": [o.model_dump() for o in risk_order.orders],
            "max_loss": risk_order.max_loss,
            "risk_pct": risk_order.risk_pct,
            "notes":    risk_order.execution_notes,
        }
        return {**state, "risk_order": risk_order, "daily_summary": summary}
    except Exception as e:
        logger.error("Risk manager failed: %s", e)
        return {**state, "daily_summary": {"error": str(e)}}


def meta_cognition_node(state: TradingState) -> TradingState:
    logger.info("=== META-COGNITION ===")
    try:
        reflection  = run_meta_cognition(state.get("daily_summary", {}))
        final_output= _build_final_output(state)
        return {**state, "meta_reflection": reflection, "final_output": final_output}
    except Exception as e:
        logger.error("Meta-cognition failed: %s", e)
        return {**state, "final_output": _build_final_output(state)}


def _build_final_output(state: TradingState) -> str:
    decision = state.get("commander_decision")
    order    = state.get("risk_order")
    symbol   = state.get("active_symbol", "N/A")
    lines    = [
        f"=== Prophet Futures — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===",
        f"Symbol: {symbol}",
    ]
    if decision:
        lines += [f"Decision: {decision.action} (confidence={decision.confidence:.2f})",
                  f"Reasoning: {decision.reasoning}"]
        if decision.veto_reasons:
            lines.append(f"Vetoes: {', '.join(decision.veto_reasons)}")
    if order and order.orders:
        lines += [f"Orders: {len(order.orders)} slices",
                  f"Max Loss: {order.max_loss:.2f} | Risk: {order.risk_pct:.2%}",
                  f"Notes: {order.execution_notes}"]
    elif order:
        lines.append(f"No orders: {order.execution_notes}")
    return "\n".join(lines)


def build_workflow() -> StateGraph:
    workflow = StateGraph(TradingState)
    workflow.add_node("scanner",            scanner_node)
    workflow.add_node("analyze_symbol",     analyze_symbol_node)
    workflow.add_node("advanced_cognition", advanced_cognition_node)
    workflow.add_node("commander",          commander_node)
    workflow.add_node("igniter",            igniter_node)
    workflow.add_node("risk_manager",       risk_manager_node)
    workflow.add_node("meta_cognition",     meta_cognition_node)
    workflow.add_edge(START, "scanner")
    workflow.add_edge("scanner",            "analyze_symbol")
    workflow.add_edge("analyze_symbol",     "advanced_cognition")
    workflow.add_edge("advanced_cognition", "commander")
    workflow.add_conditional_edges("commander", should_trade,
                                   {"igniter": "igniter", "meta_cognition": "meta_cognition"})
    workflow.add_conditional_edges("igniter", should_execute,
                                   {"risk_manager": "risk_manager", "meta_cognition": "meta_cognition"})
    workflow.add_edge("risk_manager",   "meta_cognition")
    workflow.add_edge("meta_cognition", END)
    return workflow


def get_compiled_workflow():
    return build_workflow().compile()
