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
from utils import data_path, synthetic_returns


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(data_path() / "prices_full.csv", index_col=0, parse_dates=True)
    spread = pd.read_csv(data_path() / "spreads_full.csv", index_col=0, parse_dates=True)
    rf = pd.read_csv(data_path() / "rf.csv", index_col=0, parse_dates=True).iloc[:, 0]
    volume = pd.read_csv(data_path() / "volumes_shares_full.csv", index_col=0, parse_dates=True)
    if os.getenv("CI"):
        prices = prices.tail(1800)
        spread = spread.tail(1800)
        rf = rf.tail(1800)
        volume = volume.tail(1800)
    return prices, spread, rf, volume


@lru_cache(maxsize=1)
def load_shortfees() -> pd.DataFrame:
    """Load shortfees data with annualized rates converted to daily."""
    shortfees = pd.read_csv(data_path() / "shortfees_full.csv", index_col=0, parse_dates=True)
    if os.getenv("CI"):
        shortfees = shortfees.tail(1800)
    # Convert annualized percentage to daily rate: /100/360
    return shortfees / 100 / 360


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

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]


# In backtest.py, replace the entire run_backtest function with this

def run_backtest(
    strategy: Callable, 
    risk_target: float, 
    verbose: bool = False, 
    strategy_kwargs: dict | None = None,
    max_steps: int | None = None,
    start_time: str | pd.Timestamp | None = None,
    initial_weights: pd.Series | None = None,
    use_actual_shortfees: bool = True,
    ) -> BacktestResult:
    """
    Run a simplified backtest for a given strategy.
    This is the corrected version that handles start_time properly.
    """
    if strategy_kwargs is None: 
        strategy_kwargs = {}

    # STEP 1: Load all data (do not slice it yet)
    prices, spread, rf, _ = load_data()
    n_assets = prices.shape[1] 

    # STEP 2: Determine the simulation's true start index
    training_length = 1250 # Min days of data before simulation can start
    
    sim_start_day_idx = training_length # Default start
    if start_time:
        try:
            # Find the index for the user-provided start time
            user_start_idx = prices.index.get_loc(start_time)
            # The actual start must be after the warm-up period
            sim_start_day_idx = max(training_length, user_start_idx)
        except KeyError:
            logger.error(f"Start time {start_time} not in index. Aborting.")
            raise

    # STEP 3: Initialize portfolio state on the day BEFORE the simulation starts
    setup_day_idx = sim_start_day_idx - 1
    setup_day_timestamp = prices.index[setup_day_idx]
    
    cash = 1e6
    quantities = np.zeros(n_assets)

    if initial_weights is not None:
        logger.info(f"Initializing portfolio on {setup_day_timestamp.date()} before simulation starts.")
        initial_prices = prices.iloc[setup_day_idx]
        
        # Use total capital to calculate initial holdings
        initial_values = cash * initial_weights
        quantities = (initial_values / initial_prices).fillna(0)
        
        # Update cash to be what's left over
        cash = cash - (quantities @ initial_prices)

    # STEP 4: Initialize result lists and add the "Day 0" state
    post_trade_cash = [cash]
    post_trade_quantities = [quantities.copy()]
    timings = [Timing(0, 0, 0)] # Placeholder for Day 0
    daily_target_weights_history = [np.full(n_assets, np.nan)] # Placeholder for Day 0

    # STEP 5: Run the simulation loop starting from the correct day
    lookback = 500
    forward_smoothing = 5
    
    # Define the correct range of days for the simulation loop
    simulation_indices = range(sim_start_day_idx, len(prices) - forward_smoothing)

    # Pre-computation for the relevant days
    returns = prices.pct_change().dropna()
    means = synthetic_returns(prices, information_ratio=0.15, forward_smoothing=forward_smoothing).shift(-1).dropna()
    covariance_df = returns.ewm(halflife=125).cov()
    
    days_to_compute = [prices.index[t] for t in simulation_indices[:(max_steps or len(simulation_indices))]]
    covariances = {day: covariance_df.loc[day] for day in days_to_compute if day in covariance_df.index}
    cholesky_factorizations = {day: np.linalg.cholesky(cov.values) for day, cov in covariances.items()}

    for i, t in enumerate(simulation_indices):
        if max_steps is not None and i >= max_steps:
            break

        start_time_loop = time.perf_counter()
        day = prices.index[t]

        if day not in covariances:
            logger.warning(f"Covariance for day {day} not found. Using last known portfolio state.")
            post_trade_cash.append(cash)
            post_trade_quantities.append(quantities.copy())
            timings.append(Timing(0, 0, 0))
            daily_target_weights_history.append(np.full(n_assets, np.nan))
            continue
            
        if verbose:
            logger.info(f"Day {i+1}/{max_steps or len(simulation_indices)}, {day.date()}")

        prices_t = prices.iloc[t - lookback : t + 1]
        spread_t = spread.iloc[t - lookback : t + 1]
        
        inputs_t = OptimizationInput(
            prices=prices_t,
            mean=means.loc[day],
            chol=cholesky_factorizations[day],
            volas=np.sqrt(np.diag(covariances[day].values)),
            spread=spread_t,
            quantities=quantities,
            cash=cash,
            risk_target=risk_target,
            risk_free=rf.iloc[t],
        )

        w, _, problem = strategy(inputs_t, **strategy_kwargs)
        daily_target_weights_history.append(w)

        latest_prices = prices.iloc[t]
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day, use_actual_shortfees)
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(latest_prices, trade_quantities, latest_spread)

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())
        timings.append(Timing.get_timing(start_time_loop, time.perf_counter(), problem))

    # STEP 6: Construct the final BacktestResult object
    num_results = len(post_trade_cash)
    result_index = prices.index[setup_day_idx : setup_day_idx + num_results]

    post_trade_cash_series = pd.Series(post_trade_cash, index=result_index)
    post_trade_quantities_df = pd.DataFrame(
        post_trade_quantities, index=result_index, columns=prices.columns
    )
    daily_target_weights_df = pd.DataFrame(
        daily_target_weights_history, index=result_index, columns=prices.columns
    )

    original_target_weights = pd.Series(dtype=float)
    if 'target_weights' in strategy_kwargs:
        original_target_weights = pd.Series(strategy_kwargs['target_weights'], index=prices.columns)
    
    stored_initial_weights = pd.Series(dtype=float)
    if initial_weights is not None:
        stored_initial_weights = pd.Series(initial_weights, index=prices.columns)
    
    return BacktestResult(
        post_trade_cash_series, 
        post_trade_quantities_df, 
        risk_target, 
        timings, 
        daily_target_weights=daily_target_weights_df,
        original_target_weights=original_target_weights,
        initial_weights=stored_initial_weights,
        use_actual_shortfees=use_actual_shortfees
    )


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
    cash: float, rf: float, quantities: pd.Series, prices: pd.Series, day: pd.Timestamp,
    use_actual_shortfees: bool = True
) -> float:
    """
    From t-1 to t we either earn interest on cash or pay interest on borrowed cash.
    We also pay a fee for shorting.

    cash: cash at t-1
    rf: risk free rate from t-1 to t
    quantities: quantities at t-1
    prices: prices at t-1
    day: day t
    use_actual_shortfees: if True, use actual shortfees data; if False, use hardcoded 5bps
    Note on rf: the Effective Federal Funds Rate uses ACT/360.
    """
    days_t_to_t_minus_1 = (day - prices.name).days
    cash_interest = cash * (1 + rf) ** days_t_to_t_minus_1 - cash
    
    short_valuations = np.clip(quantities, None, 0) * prices
    short_value = short_valuations.sum()
    
    if use_actual_shortfees:
        # Use actual shortfees data (already converted to daily rates)
        shortfees = load_shortfees()
        if day in shortfees.index:
            # Weighted average short fee based on short positions
            short_weights = np.clip(quantities, None, 0) * prices
            short_weights = short_weights / short_weights.sum() if short_weights.sum() != 0 else short_weights
            daily_shortfees = shortfees.loc[day]
            short_spread = (short_weights * daily_shortfees).sum()
        else:
            # Fallback to hardcoded if date not available
            logger.warning(f"Shortfees data not available for date {day}, falling back to hardcoded 5bps annualized rate")
            short_spread = 0.05 / 360
    else:
        # Use hardcoded 5bps annualized
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

    # ADDED NEW FIELDS 
    daily_target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    original_target_weights: pd.Series = field(default_factory=pd.Series)
    initial_weights: pd.Series = field(default_factory=pd.Series)
    use_actual_shortfees: bool = True


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
