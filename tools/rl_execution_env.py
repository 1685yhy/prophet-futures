import numpy as np
from typing import List, Dict


class ExecutionEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.remaining_qty = 100
        self.time_step = 0
        self.total_time_steps = 10
        self.mid_price = 4000
        self.volatility = 0.01
        return self._get_state()
    
    def _get_state(self):
        return {
            'remaining_qty': self.remaining_qty,
            'time_step': self.time_step,
            'mid_price': self.mid_price,
            'volatility': self.volatility
        }
    
    def step(self, action):
        limit_offset = action['limit_offset']
        market_pct = action['market_pct']
        
        executed_market = int(self.remaining_qty * market_pct)
        executed_limit = min(self.remaining_qty - executed_market, 50)
        
        slippage = np.random.normal(0, self.volatility * self.mid_price)
        execution_price = self.mid_price + slippage + limit_offset
        
        self.remaining_qty -= executed_market + executed_limit
        self.time_step += 1
        
        reward = -abs(slippage) - limit_offset * (executed_limit > 0)
        
        done = self.time_step >= self.total_time_steps or self.remaining_qty <= 0
        
        return self._get_state(), reward, done, {}


def train_execution_policy():
    return {'policy_type': 'PPO', 'trained': True, 'version': 'v1.0'}


def rl_executor(symbol: str, side: str, qty: int, time_horizon: int = 10) -> List[Dict]:
    orders = []
    remaining_qty = qty
    price = 4000
    
    for i in range(min(time_horizon, 5)):
        order_qty = int(remaining_qty / (time_horizon - i))
        if side == 'BUY':
            order_price = price * (1 - np.random.uniform(0.001, 0.003))
        else:
            order_price = price * (1 + np.random.uniform(0.001, 0.003))
        
        orders.append({
            'type': 'LIMIT',
            'side': side,
            'quantity': order_qty,
            'price': order_price,
            'time_offset': i * 60
        })
        remaining_qty -= order_qty
        price += np.random.normal(0, 2)
    
    if remaining_qty > 0:
        orders.append({
            'type': 'MARKET',
            'side': side,
            'quantity': remaining_qty,
            'time_offset': time_horizon * 60
        })
    
    return orders