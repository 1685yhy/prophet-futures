"""Risk Manager Agent — position sizing, order generation, and risk validation."""

import logging
import pandas as pd
from tools.llm_utils import load_config
from tools.market_data import get_kline
from tools.indicators import calc_indicators
from tools.abm_engine import run_abm
from tools.rl_execution_env import rl_executor
from models.schemas import CommanderDecision, IgniterSignal, RiskOrder, Order

logger = logging.getLogger(__name__)


def run_risk_manager(
    decision: CommanderDecision,
    trigger: IgniterSignal,
    capital: float = 1_000_000.0,
) -> RiskOrder:
    cfg             = load_config().get("risk", {})
    max_single_risk = cfg.get("max_single_risk_pct", 0.02)

    if decision.action == "WAIT":
        return RiskOrder(decision=decision, orders=[], max_loss=0.0, risk_pct=0.0,
                         execution_notes="No trade — Commander decision is WAIT")

    symbol     = decision.symbol
    entry_price= trigger.trigger_price
    stop_loss  = decision.stop_loss or (
        entry_price * 0.97 if decision.action == "LONG" else entry_price * 1.03)
    target     = decision.target_price or (
        entry_price * 1.03 if decision.action == "LONG" else entry_price * 0.97)

    # ATR-based stop confirmation
    try:
        kline = get_kline(symbol, "daily", 30)
        df    = pd.DataFrame({
            "open": kline.opens, "high": kline.highs,
            "low":  kline.lows,  "close": kline.closes, "volume": kline.volumes,
        })
        atr = calc_indicators(df)["atr14"]
        atr_stop = (entry_price - 1.5 * atr if decision.action == "LONG"
                    else entry_price + 1.5 * atr)
        stop_loss = (max(stop_loss, atr_stop) if decision.action == "LONG"
                     else min(stop_loss, atr_stop))
    except Exception:
        atr = entry_price * 0.01

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        stop_distance = entry_price * 0.02

    reward   = abs(target - entry_price)
    pl_ratio = reward / (stop_distance + 1e-8)
    if pl_ratio < 1.5:
        return RiskOrder(decision=decision, orders=[], max_loss=0.0, risk_pct=0.0,
                         execution_notes=f"Rejected: P/L ratio {pl_ratio:.2f} < 1.5")

    quantity = round(max(1.0, min(20.0, capital * max_single_risk / stop_distance)), 1)

    # ABM price adjustment
    try:
        abm_result  = run_abm({"mid_price": entry_price, "bid_volume": 1000,
                                "ask_volume": 1000, "atr": atr}, n_agents=200, steps=50)
        entry_price = round(entry_price + abm_result.optimal_entry_offset * 0.1, 2)
    except Exception as e:
        logger.warning("ABM adjustment failed: %s", e)

    side = "BUY" if decision.action == "LONG" else "SELL"
    try:
        orders = rl_executor(symbol, side, quantity, time_horizon=10)
    except Exception:
        orders = [Order(symbol=symbol, side=side, order_type="LIMIT",
                        price=entry_price, quantity=quantity)]

    max_loss = round(quantity * stop_distance, 2)
    return RiskOrder(
        decision=decision, orders=orders,
        max_loss=max_loss,
        risk_pct=round(max_loss / capital, 4),
        execution_notes=(f"Entry={entry_price:.2f} Stop={stop_loss:.2f} "
                         f"Target={target:.2f} Qty={quantity} P/L={pl_ratio:.2f}"),
    )
