#!/usr/bin/env python3
"""完整多维度分析脚本 — 逐个 Agent 调用并打印全部输出"""

import sys, json, os
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.logger import setup_logging
import logging
setup_logging(level="WARNING", log_to_file=False)

from concurrent.futures import ThreadPoolExecutor

SYMBOL = "lh"
CAPITAL = 500_000

print()
print("=" * 70)
print(f"  先知期货认知交易系统 — 完整多维度分析")
print(f"  品种: {SYMBOL.upper()}  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  账户: {CAPITAL:,}元")
print("=" * 70)

# ── 数据采集 ───────────────────────────────────────────────────────────
import pandas as pd
from tools.market_data import get_kline, get_realtime_quote
from tools.indicators import calc_indicators, detect_divergence, adx_regime
from tools.fund_data import get_volume_oi, get_basis, get_member_positions
from tools.cycle_detector import detect_cycle
from tools.causal_graph import query_causal_graph
from tools.memory_store import init_vector_db, compute_embedding, search_similar_markets

print("\n[数据采集] 拉取行情...")
kline = get_kline(SYMBOL, "daily", 120)
df = pd.DataFrame({
    "open": kline.opens, "high": kline.highs,
    "low": kline.lows, "close": kline.closes, "volume": kline.volumes,
})
if kline.open_interests:
    df["oi"] = kline.open_interests

ind = calc_indicators(df)
divergence = detect_divergence(df)
regime_type = adx_regime(ind["adx14"])

try:
    quote = get_realtime_quote(SYMBOL)
    cur_price = quote.last_price
except Exception:
    cur_price = ind["current_close"]

print(f"  当前价: {cur_price:.0f}  ATR: {ind['atr14']:.1f}")

# ── 1. SCANNER ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("① SCANNER（市场扫描器）")
print("=" * 70)
cycle_info = detect_cycle(df)
print(f"  大周期: {cycle_info.get('cycle', 'N/A')}")
print(f"  趋势强度: {cycle_info.get('strength', 0):.2f}")
print(f"  判断: {cycle_info.get('reasoning', 'N/A')}")

# ── 2. TECHNICIAN ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("② TECHNICIAN（技术分析师）")
print("=" * 70)
from agents.technician import run_technician
try:
    tech = run_technician(SYMBOL)
    print(f"  方向: {tech.signal.direction}")
    print(f"  强度: {tech.signal.strength}")
    print(f"  支撑: {tech.key_support:.0f}  阻力: {tech.key_resistance:.0f}")
    print(f"  止损: {tech.stop_loss:.0f}  目标: {tech.target_price:.0f}")
    print(f"  置信度: {tech.confidence:.0%}")
    print(f"  理由: {tech.signal.reasoning}")
    print(f"  指标: {json.dumps(tech.indicators, indent=4, default=str)}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 3. FUND ANALYST ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("③ FUND ANALYST（资金分析师）")
print("=" * 70)
from agents.fund_analyst import run_fund_analyst
try:
    fund = run_fund_analyst(SYMBOL)
    print(f"  净流向: {fund.net_flow}")
    print(f"  基差结构: {fund.basis_status}")
    print(f"  主力动作: {fund.top_member_action}")
    print(f"  置信度: {fund.confidence:.0%}")
    print(f"  理由: {fund.reasoning}")
except Exception as e:
    print(f"  [ERROR] {e}")

# 补充资金数据
print("\n  [补充数据]")
vol_oi = get_volume_oi(SYMBOL)
print(f"  今日成交量: {vol_oi.get('volume_today', 'N/A')}")
print(f"  持仓量: {vol_oi.get('open_interest', 'N/A')}")
print(f"  OI变化: {vol_oi.get('oi_change', 'N/A')} ({vol_oi.get('oi_change_pct', 'N/A')})")
basis = get_basis(SYMBOL)
print(f"  基差: {basis.get('basis', 'N/A')} ({basis.get('basis_pct', 'N/A')})")
print(f"  现货价: {basis.get('spot_price', 'N/A')}  期货价: {basis.get('futures_price', 'N/A')}")

# ── 4. MACRO ANALYST ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("④ MACRO ANALYST（宏观分析师）")
print("=" * 70)
from agents.macro_analyst import run_macro_analyst
try:
    macro = run_macro_analyst(SYMBOL)
    print(f"  宏观趋势: {macro.macro_trend}")
    print(f"  关键驱动: {macro.key_drivers}")
    print(f"  风险事件: {macro.risk_events}")
    print(f"  置信度: {macro.confidence:.0%}")
    print(f"  理由: {macro.reasoning}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 5. REGIME DETECTOR ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑤ REGIME DETECTOR（市场气象台）")
print("=" * 70)
from agents.regime_detector import run_regime_detector
try:
    regime = run_regime_detector(SYMBOL)
    print(f"  市场状态: {regime.regime}")
    print(f"  ADX: {regime.adx_value:.1f}")
    print(f"  VIX等效: {regime.vix_equivalent:.1f}")
    print(f"  建议权重: {regime.recommended_weights}")
    print(f"  理由: {regime.reasoning}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 6. SCENARIO ENGINE ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑥ SCENARIO ENGINE（情景规划）")
print("=" * 70)
from agents.scenario_engine import run_scenario_engine
try:
    scenario = run_scenario_engine(SYMBOL)
    print(f"  最坏情景亏损: {scenario.worst_case_loss_pct:.1%}")
    print(f"  最好情景收益: {scenario.best_case_gain_pct:.1%}")
    for i, path in enumerate(scenario.paths, 1):
        print(f"  路径{i}({path.probability:.0%}): {path.description}")
        print(f"    目标价: {path.target_price:.0f}  触发: {path.key_trigger}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 7. CAUSAL REASONER ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑦ CAUSAL REASONER（因果推断）")
print("=" * 70)
from agents.causal_reasoner import run_causal_reasoner
try:
    causal_policy = run_causal_reasoner("market_policy_change", SYMBOL)
    print(f"  政策因果: {causal_policy.direction} / {causal_policy.strength}")
    print(f"  因果链: {causal_policy.chain}")
    print(f"  置信度: {causal_policy.confidence:.0%}")
except Exception as e:
    print(f"  [ERROR] {e}")

try:
    causal_oil = run_causal_reasoner("crude_oil_price", SYMBOL)
    print(f"  原油因果: {causal_oil.direction} / {causal_oil.strength}")
    print(f"  因果链: {causal_oil.chain}")
except Exception as e:
    print(f"  原油因果: [ERROR] {e}")

# ── 8. MEMORY RETRIEVER ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑧ MEMORY RETRIEVER（历史记忆检索）")
print("=" * 70)
from agents.memory_retriever import run_memory_retriever
try:
    memory = run_memory_retriever(SYMBOL)
    print(f"  历史胜率: {memory.historical_win_rate:.1%}")
    print(f"  平均盈亏比: {memory.avg_profit_loss_ratio:.2f}")
    print(f"  结论: {memory.conclusion}")
    for c in memory.similar_cases[:3]:
        print(f"  [{c.date}] 相似度={c.similarity:.0%}  {c.description}  后续5日={c.subsequent_5d_return:+.1%}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 9. TRAP DETECTOR ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑨ TRAP DETECTOR（主力陷阱识别）")
print("=" * 70)
from agents.trap_detector import run_trap_detector
try:
    trap = run_trap_detector(SYMBOL)
    print(f"  陷阱类型: {trap.trap.type}")
    print(f"  当前阶段: {trap.trap.current_phase}")
    print(f"  确认条件: {trap.trap.trigger_to_confirm}")
    print(f"  置信度: {trap.trap.confidence:.0%}")
    print(f"  理由: {trap.reasoning}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 10. CROWDING RADAR ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑩ CROWDING RADAR（拥挤度雷达）")
print("=" * 70)
from agents.crowding_radar import run_crowding_radar
try:
    crowding = run_crowding_radar(SYMBOL)
    print(f"  拥挤度评分: {crowding.score}/100")
    print(f"  预警: {crowding.warning}")
    print(f"  同类资金占比: {crowding.similar_funds_pct:.1%}")
    print(f"  理由: {crowding.reasoning}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 11. COMMANDER ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑪ COMMANDER（综合决策官 — DCS贝叶斯融合）")
print("=" * 70)
from agents.commander import run_commander
try:
    decision = run_commander(
        tech=tech, fund=fund, macro=macro,
        vision=None, regime=regime,
        scenario=scenario, causal=causal_policy,
        memory=memory, trap=trap, crowding=crowding,
    )
    print(f"  决策: {decision.action}")
    print(f"  置信度: {decision.confidence:.0%}")
    print(f"  后验概率: {decision.posterior_probability:.3f}")
    print(f"  入场: {decision.entry_price or 'N/A'}")
    print(f"  止损: {decision.stop_loss or 'N/A'}")
    print(f"  目标: {decision.target_price or 'N/A'}")
    print(f"  仓位: {decision.position_size_pct:.1%}")
    print(f"  理由: {decision.reasoning}")
    if decision.veto_reasons:
        print(f"  否决: {', '.join(decision.veto_reasons)}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 12. IGNITER ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑫ IGNITER（微观点火器）")
print("=" * 70)
if decision.action in ("LONG", "SHORT"):
    from agents.igniter import run_igniter
    try:
        trigger = run_igniter(decision)
        print(f"  触发: {'是' if trigger.triggered else '否'}")
        print(f"  触发价: {trigger.trigger_price:.0f}")
        print(f"  理由: {trigger.trigger_reason}")
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print(f"  ⏭ 跳过（Commander={decision.action}）")

# ── 13. RISK MANAGER ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑬ RISK MANAGER（风控与订单生成）")
print("=" * 70)
if decision.action in ("LONG", "SHORT") and trigger.triggered:
    from agents.risk_manager import run_risk_manager
    try:
        risk_order = run_risk_manager(decision, trigger, CAPITAL)
        print(f"  订单数: {len(risk_order.orders)}")
        print(f"  最大亏损: {risk_order.max_loss:,.0f}元")
        print(f"  风险比例: {risk_order.risk_pct:.2%}")
        print(f"  备注: {risk_order.execution_notes}")
        for i, o in enumerate(risk_order.orders, 1):
            print(f"  [{i}] {o.side} {o.order_type} {o.quantity:.1f}手 @ {o.price or 'MARKET'}")
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print(f"  ⏭ 跳过（未触发交易信号）")

# ── 14. META-COGNITION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("⑭ META-COGNITION（元认知反思）")
print("=" * 70)
from agents.meta_cognition import run_meta_cognition
try:
    daily_summary = {
        "symbol": SYMBOL,
        "action": decision.action,
        "confidence": decision.confidence,
    }
    reflection = run_meta_cognition(daily_summary)
    print(f"  日期: {reflection.date}")
    print(f"  反思: {reflection.reflection}")
    print(f"  识别偏差: {reflection.identified_biases}")
    print(f"  参数建议: {reflection.parameter_adjustment_suggestion}")
except Exception as e:
    print(f"  [ERROR] {e}")

# ── 最终总结 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  📊 分析完成")
print("=" * 70)
print(f"  品种: {SYMBOL.upper()}  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  最终决策: {decision.action}")
print(f"  置信度: {decision.confidence:.0%}")
print()
