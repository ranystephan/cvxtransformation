# transformation_strategies.py
# author: Rany Stephan 
# Sunet ID: ranycs
# Date: 2025-07-10

# experiments/transformation_strategies.py (Complete Code)

import numpy as np
import cvxpy as cp
from backtest import OptimizationInput

def dynamic_uniform_strategy(
    inputs: OptimizationInput, **kwargs
) -> tuple[np.ndarray, float, cp.Problem]:
    """
    Trades a uniform fraction of the *remaining distance* to a target each day.
    This is a path-dependent strategy.
    """
    # Get extra parameters passed by the backtester
    target_weights = kwargs['target_weights']
    days_remaining = kwargs['days_remaining']

    # Calculate the current portfolio weights (w_prev)
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices
    
    # Handle the case of zero portfolio value at the start
    if portfolio_value < 1e-6:
        w_prev = np.zeros(inputs.n_assets)
    else:
        w_prev = (inputs.quantities * latest_prices) / portfolio_value

    if days_remaining <= 1:
        # On the last day, trade the whole remaining difference
        w_today = target_weights
    else:
        # Trade 1/N of the remaining difference
        weight_difference_to_trade = (target_weights - w_prev) / days_remaining
        w_today = w_prev + weight_difference_to_trade

    # The budget constraint must hold
    c_today = 1.0 - np.sum(w_today)

    # Return in the format the backtester expects (no CVXPY problem)
    return w_today, c_today, None


def front_loaded_strategy(
    inputs: OptimizationInput, **kwargs
) -> tuple[np.ndarray, float, cp.Problem]:
    """A strategy that trades a fixed percentage of the remaining amount."""
    
    target_weights = kwargs['target_weights']
    days_remaining = kwargs['days_remaining']
    trade_fraction = kwargs.get('trade_fraction', 0.20) # Get param, with a default

    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    if portfolio_value < 1e-6:
        w_prev = np.zeros(inputs.n_assets)
    else:
        w_prev = (inputs.quantities * latest_prices) / portfolio_value

    if days_remaining <= 1:
        w_today = target_weights
    else:
        # Trade a fixed fraction of the remaining difference
        weight_difference_to_trade = (target_weights - w_prev) * trade_fraction
        w_today = w_prev + weight_difference_to_trade

    c_today = 1.0 - np.sum(w_today)
    return w_today, c_today, None

# You can add many more strategies here. For example:
# def cost_aware_strategy(...): ...
# def volatility_targeting_transformation(...): ...