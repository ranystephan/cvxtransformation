# portfolio_construction.py
# author: Rany Stephan 
# Sunet ID: ranycs
# Date: 2025-01-11

import numpy as np
import pandas as pd
import cvxpy as cp
from loguru import logger


def get_minimum_variance_portfolio(prices: pd.DataFrame, risk_target: float = 0.10) -> pd.Series:
    """
    Create a minimum variance portfolio with a target risk level.
    
    Args:
        prices: DataFrame of historical prices with DatetimeIndex and asset columns
        risk_target: Target annualized volatility (e.g., 0.10 for 10% annualized vol)
    
    Returns:
        pd.Series: Portfolio weights that sum to 1.0
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    if len(returns) < 60:  # Need at least 60 days of data
        logger.warning(f"Insufficient data for portfolio construction. Got {len(returns)} days, need at least 60.")
        # Return equal weights as fallback
        n_assets = len(prices.columns)
        return pd.Series(1.0 / n_assets, index=prices.columns)
    
    # Calculate covariance matrix using exponential weighted moving average
    # This matches the approach used in the backtesting framework
    halflife = 125
    covariance_df = returns.ewm(halflife=halflife).cov()
    
    # Get the most recent covariance matrix
    latest_date = returns.index[-1]
    covariance_matrix = covariance_df.loc[latest_date].values
    
    # Add small regularization to ensure positive definiteness
    n_assets = covariance_matrix.shape[0]
    regularization = 1e-6 * np.eye(n_assets)
    covariance_matrix += regularization
    
    # Create the optimization problem
    weights = cp.Variable(n_assets)
    
    # Objective: minimize portfolio variance
    portfolio_variance = cp.quad_form(weights, covariance_matrix)
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1.0,  # Budget constraint
        weights >= 0,  # Long-only constraint
        weights <= 0.25,  # Maximum 25% in any single asset
    ]
    
    # Add risk constraint if risk_target is specified
    if risk_target > 0:
        # Convert annualized risk target to daily (assuming 252 trading days)
        daily_risk_target = risk_target / np.sqrt(252)
        constraints.append(portfolio_variance <= daily_risk_target**2)
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    
    try:
        # Try with default solver first
        problem.solve()
        
        # If that fails, try with ECOS solver
        if problem.status != cp.OPTIMAL:
            problem.solve(solver=cp.ECOS)
        
        # If that fails, try with SCS solver
        if problem.status != cp.OPTIMAL:
            problem.solve(solver=cp.SCS)
        
        if problem.status == cp.OPTIMAL:
            # Convert to pandas Series with asset names
            optimal_weights = pd.Series(weights.value, index=prices.columns)
            
            # Normalize to ensure weights sum to 1.0 (handle numerical precision issues)
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            # Calculate actual portfolio risk
            actual_risk = np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights)
            annualized_risk = actual_risk * np.sqrt(252)
            
            logger.info(f"Minimum variance portfolio constructed with {annualized_risk:.2%} annualized volatility")
            
            return optimal_weights
        else:
            logger.warning(f"Portfolio optimization failed with status: {problem.status}")
            # Return equal weights as fallback
            n_assets = len(prices.columns)
            return pd.Series(1.0 / n_assets, index=prices.columns)
            
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        # Return equal weights as fallback
        n_assets = len(prices.columns)
        return pd.Series(1.0 / n_assets, index=prices.columns)


def get_equal_weight_portfolio(prices: pd.DataFrame) -> pd.Series:
    """
    Create an equal-weight portfolio.
    
    Args:
        prices: DataFrame of historical prices with DatetimeIndex and asset columns
    
    Returns:
        pd.Series: Equal weights that sum to 1.0
    """
    n_assets = len(prices.columns)
    return pd.Series(1.0 / n_assets, index=prices.columns)


def get_market_cap_weighted_portfolio(prices: pd.DataFrame, market_caps: pd.Series) -> pd.Series:
    """
    Create a market cap weighted portfolio.
    
    Args:
        prices: DataFrame of historical prices with DatetimeIndex and asset columns
        market_caps: Series of market capitalizations for each asset
    
    Returns:
        pd.Series: Market cap weights that sum to 1.0
    """
    # Align market caps with price columns
    aligned_market_caps = market_caps.reindex(prices.columns, fill_value=0)
    
    # Calculate weights
    total_market_cap = aligned_market_caps.sum()
    if total_market_cap > 0:
        weights = aligned_market_caps / total_market_cap
    else:
        # Fallback to equal weights if no market cap data
        n_assets = len(prices.columns)
        weights = pd.Series(1.0 / n_assets, index=prices.columns)
    
    return weights


if __name__ == "__main__":
    # Example usage
    from backtest import load_data
    
    # Load sample data
    prices, _, _, _ = load_data()
    
    # Create minimum variance portfolio
    min_var_weights = get_minimum_variance_portfolio(prices, risk_target=0.10)
    print("Minimum Variance Portfolio Weights:")
    print(min_var_weights.head())
    
    # Create equal weight portfolio
    equal_weights = get_equal_weight_portfolio(prices)
    print("\nEqual Weight Portfolio Weights:")
    print(equal_weights.head()) 