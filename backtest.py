import pandas as pd
import numpy as np
from data_manager import DataManager
from portfolio_optimizer import optimize_portfolio
from config import UNIVERSES, ACTIVE_UNIVERSE, TRAIN_WINDOW, REBALANCE_FREQ, COMMISSION

def run_backtest():
    tickers = UNIVERSES[ACTIVE_UNIVERSE]
    dm = DataManager(tickers)
    returns = dm.returns
    dates = returns.index
    
    results = []  # list of dicts with date and ticker weights
    current_weights = None
    
    for i in range(TRAIN_WINDOW, len(dates), REBALANCE_FREQ):
        train_end = dates[i-1]
        train_start = dates[i - TRAIN_WINDOW]
        train_returns = returns.loc[train_start:train_end]
        
        if len(train_returns) < 100:
            continue
        
        posterior_mean, _, _ = optimize_portfolio(train_returns, initial_weights=current_weights)
        # Normalise to sum = 1 (ensures long-only, no leverage)
        new_weights = posterior_mean / np.sum(posterior_mean)
        
        # Build dictionary with ticker symbols as keys
        row = {'date': dates[i]}
        for ticker, w in zip(tickers, new_weights):
            row[ticker] = w
        results.append(row)
        current_weights = new_weights
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results).set_index('date')
    return df
