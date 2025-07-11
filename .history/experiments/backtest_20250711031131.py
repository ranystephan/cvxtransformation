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


def run_backtest(
    strategy: Callable, 
    risk_target: float, 
    verbose: bool = False, 
    strategy_kwargs: dict | None = None, # MODIFICATION 1: For passing extra params
    max_steps: int | None = None, # MODIFICATION 2: For fixed-period tests
    start_time: str | pd.Timestamp | None = None, # MODIFICATION 5: For Monte-Carlo analysis
    initial_weights: pd.Series | None = None, # MODIFICATION 6: For initial weights
    use_actual_shortfees: bool = True, # MODIFICATION 8: For using actual shortfees data
    ) -> BacktestResult:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-lookback to t to compute the optimal portfolio
    weights and then execute the trades at time t.
    """

    # STEP 1: Load all data
    if strategy_kwargs is None: 
        strategy_kwargs = {}

    prices, spread, rf, _ = load_data()
    n_assets = prices.shape[1] 

    # STEP 2: Determine the simulation start index
    training_length = 1250
    start_idx_from_time = 0 
    if start_time is not None: 
        try:
            start_idx_from_time = prices.index.get_loc(start_time)
        except KeyError:
            logger.warning(f"Start time {start_time} not found in prices index.")
            raise

    # The actual start index must be after the training/warm-up period
    start_idx = max(training_length, start_idx_from_time)

    # STEP 3: Initialize the portfolio (either cash or initial weights)
    cash = 1e6  # Initial cash
    quantities = np.zeros(n_assets)
    setup_day_timestamp = None



    if initial_weights is not None:
        # We are starting with a pre-defined portfolio
        logger.info("Initializing backtest with a starting portfolio.")
        setup_day_index = start_idx - 1
        setup_day_timestamp = prices.index[setup_day_index] # Get the timestamp for Day 0
        # We need the prices on the day before the simulation starts to value our initial holdings
        initial_prices = prices.iloc[start_idx -1]

        initial_values = cash * initial_weights
        quantities = (initial_values / initial_prices).fillna(0) # fillna(0) for robustness

        # The cash is what's left over. If initial_weights sum to 1, cash is 0.
        cash = cash - (quantities @ initial_prices)
        logger.info(f"Initial cash: {cash:.2f}, Initial quantities: {quantities}")
    else: 
        # Standard start: 100% cash
        quantities = np.zeros(n_assets)

    # STEP 4: Slide dataframes to the simulation window
    prices, spread, rf = (
        prices.iloc[training_length:],
        spread.iloc[training_length:],
        rf.iloc[training_length:],
    )

    # STEP 5: Run the simulation
    lookback = 500
    forward_smoothing = 5

    post_trade_cash = []
    post_trade_quantities = []
    timings = []
    result_timestamps = []

    if initial_weights is not None:
        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())
        timings.append(Timing(0, 0, 0)) # Placeholder for Day 0 timing 
        result_timestamps.append(setup_day_timestamp) # Add Day 0 timestamp

    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(prices, information_ratio=0.15, forward_smoothing=forward_smoothing)
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    indices = range(lookback, len(prices) - forward_smoothing)

    # Pre-computation
    days = [prices.index[t] for t in indices]
    covariances = {}
    cholesky_factorizations = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]
        cholesky_factorizations[day] = np.linalg.cholesky(covariances[day].values)

    daily_target_weights_history = []

    for i, t in enumerate(indices):
        # MODIFICATION 3: Check if we should stop early
        if max_steps is not None and i >= max_steps:
            break

        start_time_loop = time.perf_counter()
        day = prices.index[t]

        if i == 0 and initial_weights is None: # If starting from cash, Day 1 is the first point
             result_timestamps.append(day)
        elif initial_weights is not None: # Otherwise, we've already added Day 0
             result_timestamps.append(day)

        if day not in covariances: 
            logger.warning(f"Covariance for day {day} not found. Skipping.")
            continue

        if verbose:
            logger.info(f"Day {t} of {len(prices)-forward_smoothing}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1
        chol_t = cholesky_factorizations[day]
        volas_t = np.sqrt(np.diag(covariance_t.values))

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
        )

        # MODIFICATION 4: Pass the extra arguments to the strategy
        w, _, problem = strategy(inputs_t, **strategy_kwargs)

        # MODIFICATION 7?
        # Record the target weight vector for this day 
        daily_target_weights_history.append(w)

        latest_prices = prices.iloc[t]  # At t
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day, use_actual_shortfees)
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(latest_prices, trade_quantities, latest_spread)

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        # Timings
        end_time_loop = time.perf_counter()
        timings.append(Timing.get_timing(start_time_loop, end_time_loop, problem))


    # Instead of slicing the original index, we calculate the correct end point. 


    # The number of steps we actually ran 
    num_steps_ran = len(post_trade_cash)

    # The end of our slice is 'lookback' + the number of steps we ran
    end_index_slice = lookback + num_steps_ran

    # The index for our results corresponds to the days we actually simulated
    result_index = prices.index[lookback:end_index_slice]

    post_trade_cash = pd.Series(post_trade_cash, index=result_index)
    post_trade_quantities = pd.DataFrame(
        post_trade_quantities,
        index=result_index,
        columns=prices.columns,
    )

    # Convert the list of weights into a DataFrame
    daily_target_weights_df = pd.DataFrame(
        daily_target_weights_history, index=result_index, columns=prices.columns
    )

    # Extract original target weights if they were passed
    original_target_weights = pd.Series(dtype=float)
    if 'target_weights' in strategy_kwargs:
        original_target_weights = pd.Series(strategy_kwargs['target_weights'], index=prices.columns)
    
    # Store initial weights if they were provided
    stored_initial_weights = pd.Series(dtype=float)
    if initial_weights is not None:
        stored_initial_weights = pd.Series(initial_weights, index=prices.columns)
    
    return BacktestResult(
        post_trade_cash, 
        post_trade_quantities, 
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
