
import pandas as pd
import numpy as np


def get_benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """60% SPY / 40% BND static allocation"""
    returns = prices.pct_change().dropna()
    return 0.6 * returns['SPY'] + 0.4 * returns['BND']


def simulate_strategy(prices: pd.DataFrame, weights: dict, rebalance_freq: str='M') -> pd.Series:
    """Simulate portfolio with periodic rebalancing"""
    returns = prices.pct_change().dropna()
    portfolio = pd.DataFrame(index=returns.index)
    
    for date in pd.date_range(start='2024-08-01', end='2025-07-31', freq=rebalance_freq):
        period_returns = returns.loc[date:date+pd.offsets.MonthEnd(1)]
        weighted_returns = period_returns * pd.Series(weights)
        portfolio[date] = weighted_returns.sum(axis=1)
    
    return portfolio.sum(axis=1)  # Combined returns

def calculate_performance(returns: pd.Series, risk_free_rate: float=0.02/252) -> dict:
    """Compute Sharpe Ratio and cumulative return"""
    excess_returns = returns - risk_free_rate
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    cumulative_return = (1 + returns).prod() - 1
    return {'Sharpe': sharpe, 'Cumulative_Return': cumulative_return}