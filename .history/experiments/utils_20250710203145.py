import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

__folder = Path(__file__).parent


def checkpoints_path() -> Path:
    return __folder.parent / "checkpoints"


def figures_path() -> Path:
    return __folder.parent / "figures"


def data_path() -> Path:
    return __folder.parent / "data_ranycs"


def experiment_path() -> Path:
    return __folder.parent / "experiments"


def synthetic_returns(
    prices: pd.DataFrame, information_ratio: float, forward_smoothing: int
) -> pd.DataFrame:
    """
    prices: a DataFrame of prices
    information_ratio: the desired information ratio of the synthetic returns
    smoothing_len: the length of the smoothing window for the synthetic returns
    seed: random seed for reproducibility

    returns: a DataFrame of "synthetic return predictions" computed as
    alpha*(returns+noise), where alpha=var_r / (var_r + var_eps); this is the
    coefficient that minimize the variance of the prediction error under the
    above model.
    """
    rng = np.random.default_rng(1)

    returns = prices.pct_change()
    returns = returns.rolling(forward_smoothing).mean().shift(-(forward_smoothing - 1))
    var_r = returns.var()

    alpha = information_ratio**2
    var_eps = var_r * (1 - alpha) / alpha
    noise = rng.normal(0, np.sqrt(var_eps), size=returns.shape)
    synthetic_returns = alpha * (returns + noise)

    return synthetic_returns


def generate_random_inputs(
    n_assets: int, n_factors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    mean = rng.normal(0, 0.05, size=n_assets)

    A = rng.uniform(-1, 1, size=(n_factors, n_factors))
    covariance = A @ A.T

    loadings = rng.normal(0, 1, size=(n_assets, n_factors))

    return mean, loadings, covariance


def get_solver() -> str:
    if os.getenv("CI"):
        return cp.CLARABEL

    return cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL


if __name__ == "__main__":
    prices = pd.read_csv(data_path() / "prices_full.csv", index_col=0, parse_dates=True)
    synthetic_returns = synthetic_returns(prices, information_ratio=0.15, forward_smoothing=5)
    returns = prices.pct_change()
    print(
        (np.sign(synthetic_returns - returns.mean()) == np.sign(returns - returns.mean()))
        .mean(axis=0)
        .describe()
    )


def estimate_transition_period(
    initial_weights: pd.Series,
    target_weights: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    portfolio_value: float = 1_000_000,
    max_participation_rate: float = 0.10 # Don't be more than 10% of daily volume
) -> int:
    """
    Estimates a reasonable number of days for a portfolio transition
    based on the size of the trade and market liquidity.
    """
    trade_weights = target_weights.subtract(initial_weights, fill_value=0)
    trade_dollars = trade_weights.abs() * portfolio_value
    total_dollars_to_trade = trade_dollars.sum()

    # Get recent average daily dollar volume for the assets we are trading
    assets_in_trade = trade_dollars[trade_dollars > 0].index
    avg_daily_dollar_volume = (prices[assets_in_trade] * volumes[assets_in_trade]).rolling(60).mean().iloc[-1]
    
    # The total liquidity we can safely access per day
    daily_tradable_value = (avg_daily_dollar_volume * max_participation_rate).sum()
    
    if daily_tradable_value < 1: # Avoid division by zero
        return 252 # Default to a very long time if no liquidity data

    # Estimated days = (Total size of our trade) / (How much we can trade per day)
    estimated_days = int(np.ceil(total_dollars_to_trade / daily_tradable_value))
    
    logger.info(f"Estimated transition period: {estimated_days} days.")
    return estimated_days