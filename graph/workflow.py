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
    # If user specified a single symbol, skip scanner and use it directly
    pre_set = state.get("candidates", [])
    if len(pre_set) == 1 and pre_set[0]:
        logger.info("User-specified symbol: %s", pre_set[0])
        return {**state, "candidates": pre_set, "errors": state.get("errors", [])}
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
    tech     = state.get("technical_report")
    fund     = state.get("fund_report")
    macro    = state.get("macro_report")
    scenario = state.get("scenario_report")
    crowding = state.get("crowding")
    trap     = state.get("trap_report")
    memory   = state.get("memory_report")
    causal   = state.get("causal_report")
    regime   = state.get("regime")

    lines = [
        f"{'='*55}",
        f"  先知期货认知交易系统 — 完整分析",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  品种: {symbol.upper()}",
        f"{'='*55}",
        "",
    ]

    # ── 各维度分析 ──
    if tech:
        lines.append("【技术面】")
        lines.append(f"  方向: {tech.signal.direction}  强度: {tech.signal.strength}  置信度: {tech.confidence:.0%}")
        lines.append(f"  支撑: {tech.key_support}  阻力: {tech.key_resistance}")
        lines.append(f"  止损: {tech.stop_loss}  目标: {tech.target_price}")
        lines.append(f"  理由: {tech.signal.reasoning[:150] if tech.signal.reasoning else 'N/A'}")
        lines.append("")
    if fund:
        lines.append(f"【资金面】净流: {fund.net_flow}  置信度: {fund.confidence:.0%}")
        lines.append(f"  {fund.reasoning[:120]}")
        lines.append("")
    if macro:
        lines.append(f"【宏观面】趋势: {macro.macro_trend}  置信度: {macro.confidence:.0%}")
        lines.append(f"  驱动力: {', '.join(macro.key_drivers[:3]) if macro.key_drivers else 'N/A'}")
        lines.append(f"  风险事件: {', '.join(macro.risk_events[:3]) if macro.risk_events else '无'}")
        lines.append("")
    if regime:
        lines.append(f"【市场气象】{regime.regime}  ADX: {regime.adx_value:.1f}  VIX当量: {regime.vix_equivalent:.1f}")
        lines.append(f"  建议权重: {regime.recommended_weights}")
        lines.append("")
    if crowding:
        lines.append(f"【拥挤度】评分: {crowding.score}/100  {'⚠ 拥挤' if crowding.score > 80 else '✓ 正常'}")
        lines.append(f"  同类资金占比: {crowding.similar_funds_pct:.0%}")
        lines.append("")
    if trap:
        lines.append(f"【陷阱检测】{trap.trap.type}  置信度: {trap.trap.confidence:.0%}")
        lines.append(f"  当前阶段: {trap.trap.current_phase}")
        lines.append("")
    if memory:
        lines.append(f"【历史记忆】相似案例: {len(memory.similar_cases)}个  历史胜率: {memory.historical_win_rate:.1%}")
        lines.append(f"  平均盈亏比: {memory.avg_profit_loss_ratio:.2f}")
        lines.append(f"  结论: {memory.conclusion[:120]}")
        lines.append("")
    if scenario:
        lines.append(f"【情景分析】最坏亏损: {scenario.worst_case_loss_pct:.1f}%  最好收益: {scenario.best_case_gain_pct:.1f}%")
        for i, p in enumerate(scenario.paths[:3]):
            lines.append(f"  情景{i+1} ({p.probability:.0%}): {p.description[:80]} → 目标{p.target_price}")
        lines.append("")
    if causal:
        lines.append(f"【因果引擎】方向: {causal.direction}  强度: {causal.strength}  置信度: {causal.confidence:.0%}")
        lines.append(f"  因果链: {causal.chain}")
        lines.append("")

    # ── 决策 ──
    lines.append(f"{'─'*55}")
    lines.append("【综合决策】")
    if decision:
        lines.append(f"  行动: {decision.action}")
        lines.append(f"  置信度: {decision.confidence:.0%}")
        lines.append(f"  仓位: {decision.position_size_pct:.2%}")
        if decision.entry_price:
            lines.append(f"  入场价: {decision.entry_price}")
        if decision.stop_loss:
            lines.append(f"  止损价: {decision.stop_loss}")
        if decision.target_price:
            lines.append(f"  目标价: {decision.target_price}")
        lines.append(f"  理由: {decision.reasoning}")
        if decision.veto_reasons:
            lines.append(f"  否决项: {', '.join(decision.veto_reasons)}")
    lines.append("")

    # ── 订单 ──
    if order and order.orders:
        lines.append("【风控订单】")
        lines.append(f"  笔数: {len(order.orders)}  最大亏损: {order.max_loss:.2f}  风险: {order.risk_pct:.2%}")
        lines.append(f"  备注: {order.execution_notes}")
        for i, o in enumerate(order.orders, 1):
            lines.append(f"  [{i}] {o.side} {o.quantity:.1f}手 @ {o.price or 'MARKET'}")
    elif decision and decision.action == "WAIT":
        lines.append("【操作建议】观望，不操作")
        lines.append("  等待信号明确后再入场")

    lines.append("")
    lines.append("⚠ 免责声明: 本系统仅供学习研究，不构成投资建议")
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
