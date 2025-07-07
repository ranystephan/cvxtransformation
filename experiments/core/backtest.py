from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger
from core.utils import data_path, synthetic_returns


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ranycs change starts here
    # Use cleaned files with consistent assets across all three files
    prices = pd.read_csv(data_path() / "prices_cleaned.csv", index_col=0, parse_dates=True)
    spread = pd.read_csv(data_path() / "spread_cleaned.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(data_path() / "volume_cleaned.csv", index_col=0, parse_dates=True)
    # ranycs change: Load risk-free rate data
    rf_data = pd.read_csv(data_path() / "rf.csv", index_col=0, parse_dates=True)
    rf = rf_data['rf']  # Use 'rf' column instead of 'DFF'
    
    # Load real shorting fee data (cleaned version)
    short_fee_data = pd.read_csv(data_path() / "short_fee_data_cleaned.csv", index_col=0, parse_dates=True)
    
    # ranycs change: Drop the first date from all dataframes
    # This removes the first row which always has NaNs in returns due to pct_change()
    prices = prices.iloc[1:]
    spread = spread.iloc[1:]
    volume = volume.iloc[1:]
    rf = rf.iloc[1:]  # Also drop first date from rf
    short_fee_data = short_fee_data.iloc[1:]  # Also drop first date from short fee data
    
    # NEW: Filter to intersection of assets with both price and shorting fee data
    # Extract asset names from shorting fee columns (remove 'fee_' prefix)
    fee_assets = [c[4:] for c in short_fee_data.columns if c.startswith('fee_')]
    price_assets = list(prices.columns)
    
    # Find intersection of assets
    common_assets = sorted(list(set(price_assets) & set(fee_assets)))
    
    print(f"Original data - Prices: {len(price_assets)} assets, Shorting fees: {len(fee_assets)} assets")
    print(f"Using intersection: {len(common_assets)} assets with both price and shorting fee data")
    print(f"Dropped {len(price_assets) - len(common_assets)} assets without shorting fee data")
    
    # Filter all dataframes to only include common assets
    prices = prices[common_assets]
    spread = spread[common_assets]
    volume = volume[common_assets]
    
    # Filter shorting fee data to only include common assets
    fee_columns = [f'fee_{asset}' for asset in common_assets]
    short_fee_data = short_fee_data[fee_columns]
    
    print(f"Final aligned data shapes - Prices: {prices.shape}, Spread: {spread.shape}, Volume: {volume.shape}, RF: {rf.shape}, Short Fees: {short_fee_data.shape}")
    # ranycs change ends here
    
    if os.getenv("CI"):
        prices = prices.tail(1800)
        spread = spread.tail(1800)
        volume = volume.tail(1800)
        rf = rf.tail(1800)
        short_fee_data = short_fee_data.tail(1800)
    return prices, spread, rf, volume, short_fee_data


@dataclass
class OptimizationInput:
    """
    At time t, we have data from t-lookback to t-1.
    """

    prices: pd.DataFrame
    mean: pd.Series
    chol: np.array
    volas: np.array
    spread: pd.DataFrame
    quantities: np.ndarray
    cash: float
    risk_target: float
    risk_free: float
    shorting_fees: np.ndarray

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]


def run_backtest(strategy: Callable, risk_target: float, verbose: bool = False) -> BacktestResult:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-lookback to t to compute the optimal portfolio
    weights and then execute the trades at time t.
    """

    prices, spread, rf, volume, short_fee_data = load_data()
    training_length = 1250
    prices, spread, rf, short_fee_data = (
        prices.iloc[training_length:],
        spread.iloc[training_length:],
        rf.iloc[training_length:],
        short_fee_data.iloc[training_length:],
    )

    n_assets = prices.shape[1]

    lookback = 500
    forward_smoothing = 5

    quantities = np.zeros(n_assets)
    cash = 1e6

    post_trade_cash = []
    post_trade_quantities = []
    timings = []
    
    # Extract shorting fees for the assets we're using
    def get_shorting_fees(short_fee_data, asset_names, date):
        """Extract shorting fees for specific assets on a specific date"""
        if date not in short_fee_data.index:
            return np.zeros(len(asset_names))
        
        fees = []
        for asset in asset_names:
            fee_col = f'fee_{asset}'
            # Since we filtered to intersection, all assets should have fee data
            assert fee_col in short_fee_data.columns, f"Missing shorting fee for {asset}"
            fee = short_fee_data.loc[date, fee_col]
            # Convert from annual basis points to daily decimal
            # e.g., 0.25% annual = 0.0025/360 daily
            fees.append(fee / 100 / 360 if not pd.isna(fee) else 0.0)
        return np.array(fees)

    # ranycs change starts here
    # Fix date alignment issue by ensuring returns and covariance use the same date range
    # This prevents KeyError when accessing covariance_df.loc[day] where day doesn't exist
    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(prices, information_ratio=0.15, forward_smoothing=forward_smoothing)
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    
    # ranycs change: Ensure covariance calculation uses the same date range as returns
    # This prevents the KeyError when accessing dates that don't exist in the covariance DataFrame
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    
    # ranycs change: Use the same date range for indices calculation to ensure alignment
    indices = range(lookback, len(prices) - forward_smoothing)
    days = [prices.index[t] for t in indices]
    
    # ranycs change: Verify that all days exist in the covariance DataFrame
    # This helps debug any remaining date alignment issues
    missing_days = [day for day in days if day not in covariance_df.index.get_level_values(0)]
    if missing_days:
        print(f"Warning: {len(missing_days)} days missing from covariance DataFrame")
        print(f"First few missing days: {missing_days[:5]}")
        # Filter out missing days
        days = [day for day in days if day in covariance_df.index.get_level_values(0)]
        indices = [i for i in indices if prices.index[i] in days]
    # ranycs change ends here
    
    covariances = {}
    cholesky_factorizations = {}
    valid_days = []
    for day in days:
        # ranycs change starts here
        # Check if covariance matrix has NaNs before processing
        cov_matrix = covariance_df.loc[day].values
        
        # Skip days with NaN covariance matrices
        if np.isnan(cov_matrix).any():
            print(f"Warning: Skipping day {day} due to NaN covariance matrix")
            continue
            
        valid_days.append(day)
        
        # Add regularization to ensure covariance matrix is positive definite
        # This prevents LinAlgError when the matrix is not positive definite
        n_assets = cov_matrix.shape[0]
        regularization = 1e-6 * np.eye(n_assets)  # Small regularization term
        cov_matrix_reg = cov_matrix + regularization
        
        # ranycs change: Verify positive definiteness and add more regularization if needed
        # This ensures the Cholesky decomposition will succeed
        min_eigenval = np.linalg.eigvals(cov_matrix_reg).min()
        if min_eigenval <= 0:
            # Add more regularization if still not positive definite
            additional_reg = abs(min_eigenval) + 1e-6
            cov_matrix_reg += additional_reg * np.eye(n_assets)
            print(f"Warning: Added additional regularization {additional_reg:.2e} for day {day}")
        
        covariances[day] = pd.DataFrame(cov_matrix_reg, index=covariance_df.loc[day].index, columns=covariance_df.loc[day].columns)
        cholesky_factorizations[day] = np.linalg.cholesky(covariances[day].values)
        # ranycs change ends here
    
    # Update indices to only include valid days
    indices = [i for i in indices if prices.index[i] in valid_days]
    print(f"Valid days for backtest: {len(valid_days)} out of {len(days)}")

    for t in indices:
        start_time = time.perf_counter()

        day = prices.index[t]

        if verbose:
            logger.info(f"Day {t} of {len(prices)-forward_smoothing}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1
        chol_t = cholesky_factorizations[day] 
        volas_t = np.sqrt(np.diag(covariance_t.values))

        # Get shorting fees for current day
        shorting_fees_t = get_shorting_fees(short_fee_data, prices.columns, day)
        
        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            chol_t,
            volas_t,
            spread_t,
            quantities,
            cash,
            risk_target,
            rf.iloc[t],
            shorting_fees_t,
        )

        # Runtime checks for NaNs/Infs in portfolio state before optimization
        latest_prices = prices.iloc[t] # At t
        portfolio_value = cash + quantities @ latest_prices
        if (
            np.isnan(quantities).any() or np.isinf(quantities).any()
            or np.isnan(cash) or np.isinf(cash)
            or np.isnan(portfolio_value) or np.isinf(portfolio_value)
        ):
            print(f"\n[ERROR] Invalid portfolio state at t={t}, day={day}")
            print(f"  Quantities has NaN: {np.isnan(quantities).any()}, Inf: {np.isinf(quantities).any()}")
            print(f"  Cash is NaN: {np.isnan(cash)}, Inf: {np.isinf(cash)}")
            print(f"  Portfolio value is NaN: {np.isnan(portfolio_value)}, Inf: {np.isinf(portfolio_value)}")
            print(f"  Quantities: {quantities}")
            print(f"  Cash: {cash}")
            print(f"  Latest prices: {latest_prices.values}")
            print(f"  Portfolio value: {portfolio_value}")
            raise ValueError("Portfolio state contains NaN or Inf before optimization.")
        # End runtime checks

        w, _, problem = strategy(inputs_t)

        latest_spread = spread.iloc[t]

        # Update cash with interest and fees
        prev_cash = cash
        cash += interest_and_fees(cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day)
        if np.isnan(cash) or np.isinf(cash):
            print(f"\n[ERROR] NaN/Inf in cash after interest_and_fees at t={t}, day={day}")
            print(f"  Previous cash: {prev_cash}")
            print(f"  Quantities: {quantities}")
            print(f"  Prices (t-1): {prices.iloc[t - 1].values}")
            print(f"  RF (t-1): {rf.iloc[t - 1]}")
            print(f"  New cash: {cash}")
            raise ValueError("NaN or Inf in cash after interest_and_fees.")

        # Update quantities with trade_quantities
        prev_quantities = quantities.copy()
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        if np.isnan(quantities).any() or np.isinf(quantities).any():
            print(f"\n[ERROR] NaN/Inf in quantities after trade at t={t}, day={day}")
            print(f"  Previous quantities: {prev_quantities}")
            print(f"  Trade quantities: {trade_quantities}")
            print(f"  New quantities: {quantities}")
            raise ValueError("NaN or Inf in quantities after trade.")

        # Update cash with order execution
        prev_cash2 = cash
        cash += execute_orders(latest_prices, trade_quantities, latest_spread)
        if np.isnan(cash) or np.isinf(cash):
            print(f"\n[ERROR] NaN/Inf in cash after execute_orders at t={t}, day={day}")
            print(f"  Previous cash: {prev_cash2}")
            print(f"  Trade quantities: {trade_quantities}")
            print(f"  Latest prices: {latest_prices.values}")
            print(f"  Latest spread: {latest_spread.values}")
            print(f"  New cash: {cash}")
            raise ValueError("NaN or Inf in cash after execute_orders.")

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        # Timings
        end_time = time.perf_counter()
        timings.append(Timing.get_timing(start_time, end_time, problem))

    post_trade_cash = pd.Series(post_trade_cash, index=prices.index[lookback:-forward_smoothing])
    post_trade_quantities = pd.DataFrame(
        post_trade_quantities,
        index=prices.index[lookback:-forward_smoothing],
        columns=prices.columns,
    )

    return BacktestResult(post_trade_cash, post_trade_quantities, risk_target, timings)


def create_orders(
    w: np.array, quantities: np.array, cash: float, latest_prices: pd.Series
) -> np.array:
    portfolio_value = cash + quantities @ latest_prices
    w_prev = (quantities * latest_prices) / portfolio_value

    z = w - w_prev
    trades = z * portfolio_value
    trade_quantities = trades / latest_prices
    return trade_quantities.values


def execute_orders(
    latest_prices: pd.Series,
    trade_quantities: np.array,
    latest_spread: pd.Series,
) -> float:
    sell_order_quantities = np.clip(trade_quantities, None, 0)
    buy_order_quantities = np.clip(trade_quantities, 0, None)

    sell_order_prices = latest_prices * (1 - latest_spread / 2)
    buy_order_prices = latest_prices * (1 + latest_spread / 2)

    sell_receipt = -sell_order_quantities @ sell_order_prices
    buy_payment = buy_order_quantities @ buy_order_prices

    return sell_receipt - buy_payment


def interest_and_fees(
    cash: float, rf: float, quantities: pd.Series, prices: pd.Series, day: pd.Timestamp
) -> float:
    """
    From t-1 to t we either earn interest on cash or pay interest on borrowed cash.
    We also pay a fee for shorting (stark simplification: using the same rate).

    cash: cash at t-1
    rf: risk free rate from t-1 to t
    quantities: quantities at t-1
    prices: prices at t-1
    day: day t
    Note on rf: the Effective Federal Funds Rate uses ACT/360.
    """
    days_t_to_t_minus_1 = (day - prices.name).days
    cash_interest = cash * (1 + rf) ** days_t_to_t_minus_1 - cash
    short_valuations = np.clip(quantities, None, 0) * prices
    short_value = short_valuations.sum()
    short_spread = 0.05 / 360
    shorting_fee = short_value * (1 + rf + short_spread) ** days_t_to_t_minus_1 - short_value
    return cash_interest + shorting_fee


@dataclass
class Timing:
    solver: float
    cvxpy: float
    other: float

    @property
    def total(self) -> float:
        return self.solver + self.cvxpy + self.other

    @classmethod
    def get_timing(cls, start_time: float, end_time: float, problem: cp.Problem | None) -> Timing:
        if problem:
            solver_time = problem.solver_stats.solve_time
            cvxpy_time = problem.compilation_time
            other_time = end_time - start_time - solver_time - cvxpy_time
            return cls(solver_time, cvxpy_time, other_time)
        else:
            return cls(0, 0, 0)


@dataclass
class BacktestResult:
    cash: pd.Series
    quantities: pd.DataFrame
    risk_target: float
    timings: list[Timing]
    dual_optimals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def valuations(self) -> pd.DataFrame:
        prices = load_data()[0].loc[self.history]
        return self.quantities * prices

    @property
    def portfolio_value(self) -> pd.Series:
        return self.cash + self.valuations.sum(axis=1)

    @property
    def portfolio_returns(self) -> pd.Series:
        return self.portfolio_value.pct_change().dropna()

    @property
    def periods_per_year(self) -> float:
        return len(self.history) / ((self.history[-1] - self.history[0]).days / 365.25)

    @property
    def history(self) -> pd.DatetimeIndex:
        return self.cash.index

    @property
    def cash_weight(self) -> pd.Series:
        return self.cash / self.portfolio_value

    @property
    def asset_weights(self) -> pd.DataFrame:
        return self.valuations.div(self.portfolio_value, axis=0)

    @property
    def daily_turnover(self) -> pd.Series:
        trades = self.quantities.diff()
        prices = load_data()[0].loc[self.history]
        valuation_trades = trades * prices
        relative_trades = valuation_trades.div(self.portfolio_value, axis=0)
        return relative_trades.abs().sum(axis=1) / 2

    @property
    def turnover(self) -> float:
        return self.daily_turnover.mean() * self.periods_per_year

    @property
    def mean_return(self) -> float:
        return self.portfolio_returns.mean() * self.periods_per_year

    @property
    def volatility(self) -> float:
        return self.portfolio_returns.std() * np.sqrt(self.periods_per_year)

    @property
    def max_drawdown(self) -> float:
        return self.portfolio_value.div(self.portfolio_value.cummax()).sub(1).min()

    @property
    def max_leverage(self) -> float:
        return self.asset_weights.abs().sum(axis=1).max()

    @property
    def sharpe(self) -> float:
        risk_free = load_data()[2].loc[self.history]
        excess_return = self.portfolio_returns - risk_free
        return excess_return.mean() / excess_return.std() * np.sqrt(self.periods_per_year)

    def active_return(self, benchmark: BacktestResult) -> float:
        return self.mean_return - benchmark.mean_return

    def active_risk(self, benchmark: BacktestResult) -> float:
        return self.portfolio_returns.sub(benchmark.portfolio_returns).std() * np.sqrt(
            self.periods_per_year
        )

    def information_ratio(self, benchmark: BacktestResult) -> float:
        return self.active_return(benchmark) / self.active_risk(benchmark)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> BacktestResult:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Example usage with equal weights
    n_assets = load_data()[0].shape[1]
    w_targets = np.ones(n_assets) / (n_assets + 1)
    c_target = 1 / (n_assets + 1)
    result = run_backtest(
        lambda _inputs: (w_targets, c_target, None), risk_target=0.0, verbose=True
    )
    logger.info(
        f"Mean return: {result.mean_return:.2%},\n"
        f"Volatility: {result.volatility:.2%},\n"
        f"Sharpe: {result.sharpe:.2f},\n"
        f"Turnover: {result.turnover:.2f},\n"
        f"Max leverage: {result.max_leverage:.2f}"
    )
