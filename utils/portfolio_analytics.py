import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_portfolio_metrics(returns: np.array) -> Dict[str, float]:
    if len(returns) == 0:
        return {}
    
    total_return = np.prod(1 + returns) - 1
    daily_return = np.mean(returns)
    volatility = np.std(returns)
    
    sharpe_ratio = np.sqrt(252) * daily_return / volatility if volatility > 0 else 0.0
    
    equity_curve = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    winning_trades = sum(1 for r in returns if r > 0)
    win_rate = winning_trades / len(returns) if len(returns) > 0 else 0.0
    
    positive_returns = [r for r in returns if r > 0]
    negative_returns = [r for r in returns if r < 0]
    avg_win = np.mean(positive_returns) if positive_returns else 0.0
    avg_loss = abs(np.mean(negative_returns)) if negative_returns else 1.0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    return {
        'total_return': total_return,
        'daily_return': daily_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': len(returns)
    }


def analyze_position(symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
    analysis = {
        'symbol': symbol,
        'current_price': position.get('current_price', 0),
        'entry_price': position.get('entry_price', 0),
        'quantity': position.get('quantity', 0),
        'side': position.get('side', 'LONG'),
        'pnl': 0.0,
        'pnl_pct': 0.0,
        'risk_reward': 0.0
    }
    
    if analysis['current_price'] > 0 and analysis['entry_price'] > 0:
        if analysis['side'] == 'LONG':
            analysis['pnl'] = (analysis['current_price'] - analysis['entry_price']) * analysis['quantity']
            analysis['pnl_pct'] = (analysis['current_price'] - analysis['entry_price']) / analysis['entry_price']
        else:
            analysis['pnl'] = (analysis['entry_price'] - analysis['current_price']) * analysis['quantity']
            analysis['pnl_pct'] = (analysis['entry_price'] - analysis['current_price']) / analysis['entry_price']
    
    stop_loss = position.get('stop_loss', 0)
    take_profit = position.get('take_profit', 0)
    if stop_loss > 0 and take_profit > 0 and analysis['entry_price'] > 0:
        if analysis['side'] == 'LONG':
            risk = analysis['entry_price'] - stop_loss
            reward = take_profit - analysis['entry_price']
        else:
            risk = stop_loss - analysis['entry_price']
            reward = analysis['entry_price'] - take_profit
        
        if risk > 0:
            analysis['risk_reward'] = reward / risk
    
    return analysis