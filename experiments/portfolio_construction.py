# experiments/portfolio_construction.py

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

def get_minimum_variance_portfolio(
    prices_subset: pd.DataFrame, 
    risk_target: float | None = None
) -> pd.Series:
    """
    Calculates the minimum variance portfolio for a given subset of assets.
    Optionally scales the portfolio to a specific volatility target.
    """
    logger.info(f"Constructing min-variance portfolio for {len(prices_subset.columns)} assets...")
    
    returns = prices_subset.pct_change().dropna()
    mu = returns.mean()
    Sigma = returns.cov()
    
    w = cp.Variable(len(prices_subset.columns))
    
    # The core minimum variance objective
    risk = cp.quad_form(w, Sigma)
    objective = cp.Minimize(risk)
    
    # Standard constraints
    constraints = [
        cp.sum(w) == 1, # Fully invested
        w >= 0          # No shorting (a common choice for these)
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        logger.error("Min-variance optimization failed.")
        return None

    # Get the raw min-variance weights
    min_var_weights = pd.Series(w.value, index=prices_subset.columns)
    
    # If a risk target is specified, scale the portfolio
    if risk_target is not None:
        # Calculate the volatility of the unscaled min-var portfolio
        unscaled_vol = np.sqrt(min_var_weights.T @ Sigma @ min_var_weights)
        annual_unscaled_vol = unscaled_vol * np.sqrt(252)
        
        # Calculate the leverage needed to hit the target
        leverage = risk_target / annual_unscaled_vol
        logger.info(f"Scaling portfolio with leverage {leverage:.2f} to meet {risk_target:.1%} risk target.")
        
        scaled_weights = min_var_weights * leverage
        return scaled_weights
        
    return min_var_weights