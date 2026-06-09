"""RL Execution Agent — wraps the RL executor tool as an agent-compatible interface."""

import logging
from models.schemas import Order
from tools.rl_execution_env import rl_executor, train_execution_policy

logger = logging.getLogger(__name__)

_policy = None


def get_policy():
    global _policy
    if _policy is None:
        _policy = train_execution_policy(episodes=100)
    return _policy


def execute_order(symbol: str, side: str, total_qty: float, time_horizon: int = 15):
    """Execute an order using RL-based optimal execution strategy."""
    try:
        return rl_executor(symbol, side, total_qty, time_horizon)
    except Exception as e:
        logger.warning("RL execution failed, using single market order: %s", e)
        return [Order(symbol=symbol, side=side, order_type="MARKET", quantity=total_qty)]
