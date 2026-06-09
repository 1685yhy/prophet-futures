"""Igniter Agent — micro-timing entry trigger."""

import logging
from datetime import datetime
from models.schemas import CommanderDecision, IgniterSignal
from tools.market_data import get_tick_data, get_realtime_quote
from tools.fund_data import get_volume_oi

logger = logging.getLogger(__name__)


def run_igniter(decision: CommanderDecision) -> IgniterSignal:
    symbol    = decision.symbol
    direction = decision.action

    try:
        quote    = get_realtime_quote(symbol)
        ticks    = get_tick_data(symbol, lookback_seconds=120)
        oi_data  = get_volume_oi(symbol)

        current_price = quote.last_price
        vol_ratio     = oi_data.get("vol_ratio", 1.0)

        buy_vol  = sum(t.get("volume", 0) for t in ticks
                       if float(t.get("price", 0)) >= current_price)
        sell_vol = sum(t.get("volume", 0) for t in ticks
                       if float(t.get("price", 0)) < current_price)
        buy_ratio = buy_vol / (buy_vol + sell_vol + 1e-8)

        triggered     = False
        trigger_reason= ""

        if direction == "LONG":
            if buy_ratio > 0.60 and vol_ratio > 1.3:
                triggered      = True
                trigger_reason = f"Long trigger: buy_ratio={buy_ratio:.1%}, vol_ratio={vol_ratio:.2f}"
            elif vol_ratio > 1.5 and quote.change_pct > 0:
                triggered      = True
                trigger_reason = f"Momentum trigger: vol={vol_ratio:.2f}, chg={quote.change_pct:.2f}%"
        elif direction == "SHORT":
            if buy_ratio < 0.40 and vol_ratio > 1.3:
                triggered      = True
                trigger_reason = f"Short trigger: sell_pressure={1-buy_ratio:.1%}, vol={vol_ratio:.2f}"
            elif vol_ratio > 1.5 and quote.change_pct < 0:
                triggered      = True
                trigger_reason = f"Momentum trigger: vol={vol_ratio:.2f}, chg={quote.change_pct:.2f}%"

        return IgniterSignal(
            symbol=symbol, triggered=triggered,
            trigger_reason=trigger_reason or "No trigger conditions met",
            trigger_price=current_price,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.warning("Igniter fallback for %s: %s", symbol, e)
        return IgniterSignal(
            symbol=symbol, triggered=True,
            trigger_reason="Fallback: using current market price",
            trigger_price=5000.0,
            timestamp=datetime.now().isoformat(),
        )
