import numpy as np
from typing import Dict, Any
from models.schemas import ABMResult


class MarketAgent:
    def __init__(self, agent_id: int, initial_balance: float = 100000):
        self.agent_id = agent_id
        self.balance = initial_balance
        self.position = 0
        self.risk_aversion = np.random.uniform(0.5, 2.0)
        self.time_horizon = np.random.choice([1, 5, 30])
        self.strategy = np.random.choice(['momentum', 'mean_reversion', 'noise'])
    
    def decide(self, market_price: float, volatility: float) -> Dict:
        decision = {'action': 'hold', 'quantity': 0, 'price': market_price}
        
        if self.strategy == 'momentum':
            if np.random.random() < 0.3:
                decision['action'] = 'buy' if np.random.random() > 0.5 else 'sell'
                decision['quantity'] = int(np.random.uniform(1, 10))
        
        elif self.strategy == 'mean_reversion':
            if np.random.random() < 0.25:
                decision['action'] = 'buy' if np.random.random() > 0.5 else 'sell'
                decision['quantity'] = int(np.random.uniform(1, 5))
        
        elif self.strategy == 'noise':
            if np.random.random() < 0.1:
                decision['action'] = 'buy' if np.random.random() > 0.5 else 'sell'
                decision['quantity'] = int(np.random.uniform(1, 3))
        
        return decision


def run_abm(order_book_snapshot: Dict[str, Any], n_agents: int = 500, steps: int = 100) -> ABMResult:
    agents = [MarketAgent(i) for i in range(n_agents)]
    
    price = order_book_snapshot.get('mid_price', 4000)
    volatility = order_book_snapshot.get('volatility', 0.01)
    
    buy_pressure = []
    sell_pressure = []
    
    for _ in range(steps):
        buy_volume = 0
        sell_volume = 0
        
        for agent in agents:
            decision = agent.decide(price, volatility)
            if decision['action'] == 'buy':
                buy_volume += decision['quantity']
            elif decision['action'] == 'sell':
                sell_volume += decision['quantity']
        
        buy_pressure.append(buy_volume)
        sell_pressure.append(sell_volume)
        
        price_change = (buy_volume - sell_volume) * 0.1 * volatility
        price += price_change
        volatility = max(0.001, volatility + np.random.normal(0, 0.0001))
    
    liquidity_vacuums = []
    for i in range(steps - 1):
        if abs(buy_pressure[i] - sell_pressure[i]) > np.mean(buy_pressure + sell_pressure) * 2:
            liquidity_vacuums.append({
                'price': price,
                'side': 'buy' if buy_pressure[i] > sell_pressure[i] else 'sell',
                'step': i
            })
    
    avg_buy = np.mean(buy_pressure)
    avg_sell = np.mean(sell_pressure)
    
    if avg_buy > avg_sell * 1.1:
        balance = 'BUYER_DOMINANT'
    elif avg_sell > avg_buy * 1.1:
        balance = 'SELLER_DOMINANT'
    else:
        balance = 'BALANCED'
    
    optimal_offset = np.random.uniform(-5, 5)
    
    return ABMResult(
        liquidity_vacuums=liquidity_vacuums[:3],
        projected_short_term_balance=balance,
        optimal_entry_offset=optimal_offset
    )