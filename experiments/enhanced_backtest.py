"""
Enhanced Backtest Runner with Detailed Data Capture

This module extends the basic backtest functionality to capture detailed tracking
data that can be used to create comprehensive dashboards and visualizations.
"""

import numpy as np
import pandas as pd
import time
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import cvxpy as cp

from backtest import (
    load_data, OptimizationInput, BacktestResult, Timing,
    create_orders, execute_orders, interest_and_fees
)
from dashboard import DetailedBacktestResult, BacktestDashboard


@dataclass
class EnhancedBacktestResult(BacktestResult):
    """
    Enhanced backtest result that extends the basic BacktestResult with
    detailed tracking data for dashboard creation.
    """
    # Additional detailed tracking data
    daily_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_spreads: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_optimization_info: List[Dict] = field(default_factory=list)
    strategy_name: str = "Unknown"
    strategy_params: Dict = field(default_factory=dict)
    
    def to_detailed_result(self) -> DetailedBacktestResult:
        """Convert to DetailedBacktestResult for dashboard creation."""
        return DetailedBacktestResult(
            cash=self.cash,
            quantities=self.quantities,
            portfolio_value=self.portfolio_value,
            portfolio_returns=self.portfolio_returns,
            risk_target=self.risk_target,
            daily_trades=self.daily_trades,
            daily_weights=self.daily_weights,
            daily_target_weights=self.daily_target_weights,
            daily_costs=self.daily_costs,
            strategy_name=self.strategy_name,
            strategy_params=self.strategy_params
        )


def run_enhanced_backtest(
    strategy: Callable, 
    risk_target: float, 
    strategy_name: str = "Unknown",
    strategy_params: Dict = None,
    verbose: bool = False
) -> EnhancedBacktestResult:
    """
    Run an enhanced backtest that captures detailed data for dashboard creation.
    
    Args:
        strategy: Strategy function that takes OptimizationInput and returns (w, c, problem)
        risk_target: Target risk level
        strategy_name: Name of the strategy for labeling
        strategy_params: Dictionary of strategy parameters
        verbose: Whether to print progress
        
    Returns:
        EnhancedBacktestResult with detailed tracking data
    """
    
    strategy_params = strategy_params or {}
    
    # Load data
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

    # Initialize portfolio
    quantities = np.zeros(n_assets)
    cash = 1e6

    # Tracking data containers
    post_trade_cash = []
    post_trade_quantities = []
    daily_trades_list = []
    daily_weights_list = []
    daily_prices_list = []
    daily_spreads_list = []
    daily_target_weights_list = []
    daily_costs_list = []
    daily_optimization_info = []
    timings = []
    
    # Prepare returns and covariance data (same as original backtest)
    returns = prices.pct_change().dropna()
    
    # Import synthetic_returns from utils
    from utils import synthetic_returns
    
    means = (
        synthetic_returns(prices, information_ratio=0.15, forward_smoothing=forward_smoothing)
        .shift(-1)
        .dropna()
    )
    
    covariance_df = returns.ewm(halflife=125).cov()
    indices = range(lookback, len(prices) - forward_smoothing)
    days = [prices.index[t] for t in indices]
    
    # Prepare covariance matrices (same as original backtest)
    covariances = {}
    cholesky_factorizations = {}
    valid_days = []
    
    for day in days:
        cov_matrix = covariance_df.loc[day].values
        
        if np.isnan(cov_matrix).any():
            if verbose:
                print(f"Warning: Skipping day {day} due to NaN covariance matrix")
            continue
            
        valid_days.append(day)
        
        # Add regularization
        n_assets = cov_matrix.shape[0]
        regularization = 1e-6 * np.eye(n_assets)
        cov_matrix_reg = cov_matrix + regularization
        
        # Ensure positive definiteness
        min_eigenval = np.linalg.eigvals(cov_matrix_reg).min()
        if min_eigenval <= 0:
            additional_reg = abs(min_eigenval) + 1e-6
            cov_matrix_reg += additional_reg * np.eye(n_assets)
            if verbose:
                print(f"Warning: Added additional regularization {additional_reg:.2e} for day {day}")
        
        covariances[day] = pd.DataFrame(cov_matrix_reg, index=covariance_df.loc[day].index, columns=covariance_df.loc[day].columns)
        cholesky_factorizations[day] = np.linalg.cholesky(covariances[day].values)
    
    # Update indices to only include valid days
    indices = [i for i in indices if prices.index[i] in valid_days]
    if verbose:
        print(f"Valid days for backtest: {len(valid_days)} out of {len(days)}")

    # Extract shorting fees function (same as original)
    def get_shorting_fees(short_fee_data, asset_names, date):
        """Extract shorting fees for specific assets on a specific date"""
        if date not in short_fee_data.index:
            return np.zeros(len(asset_names))
        
        fees = []
        for asset in asset_names:
            fee_col = f'fee_{asset}'
            assert fee_col in short_fee_data.columns, f"Missing shorting fee for {asset}"
            fee = short_fee_data.loc[date, fee_col]
            fees.append(fee / 100 / 252 if not pd.isna(fee) else 0.0)
        return np.array(fees)

    # Main backtest loop with enhanced tracking
    for t in indices:
        start_time = time.perf_counter()
        day = prices.index[t]

        if verbose:
            print(f"Day {t} of {len(prices)-forward_smoothing}, {day}")

        # Prepare input data
        prices_t = prices.iloc[t - lookback : t + 1]
        spread_t = spread.iloc[t - lookback : t + 1]
        mean_t = means.loc[day]
        covariance_t = covariances[day]
        chol_t = cholesky_factorizations[day]
        volas_t = np.sqrt(np.diag(covariance_t.values))
        shorting_fees_t = get_shorting_fees(short_fee_data, prices.columns, day)
        
        inputs_t = OptimizationInput(
            prices_t, mean_t, chol_t, volas_t, spread_t,
            quantities, cash, risk_target, rf.iloc[t], shorting_fees_t
        )

        # Get current prices and spreads for tracking
        latest_prices = prices.iloc[t]
        latest_spread = spread.iloc[t]

        # Calculate current portfolio value and weights
        portfolio_value = cash + quantities @ latest_prices
        current_weights = (quantities * latest_prices) / portfolio_value

        # Run strategy
        w, _, problem = strategy(inputs_t)

        # Store optimization information
        optimization_info = {
            'day': day,
            'portfolio_value': portfolio_value,
            'target_weights': w.copy() if hasattr(w, 'copy') else w,
            'current_weights': current_weights.copy(),
            'problem_status': problem.status if problem else None,
            'solver_time': problem.solver_stats.solve_time if problem else 0,
            'optimization_value': problem.value if problem else None
        }
        daily_optimization_info.append(optimization_info)

        # Update cash with interest and fees
        cash += interest_and_fees(cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day)

        # Calculate trades
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        trade_costs = execute_orders(latest_prices, trade_quantities, latest_spread)

        # Update portfolio
        quantities += trade_quantities
        cash += trade_costs

        # Store detailed tracking data
        daily_trades_list.append(trade_quantities.copy())
        daily_weights_list.append(current_weights.copy())
        daily_prices_list.append(latest_prices.copy())
        daily_spreads_list.append(latest_spread.copy())
        daily_target_weights_list.append(w.copy() if hasattr(w, 'copy') else w)
        
        # Calculate transaction costs
        total_trade_cost = -trade_costs  # Negative because trade_costs is cash flow
        daily_costs_list.append({
            'total_cost': total_trade_cost,
            'spread_cost': abs(trade_quantities) @ latest_spread * latest_prices / 2,
            'trade_volume': abs(trade_quantities) @ latest_prices
        })

        # Store results
        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        # Timing
        end_time = time.perf_counter()
        timings.append(Timing.get_timing(start_time, end_time, problem))

    # Create result DataFrames
    backtest_dates = prices.index[lookback:-forward_smoothing]
    
    post_trade_cash = pd.Series(post_trade_cash, index=backtest_dates)
    post_trade_quantities = pd.DataFrame(post_trade_quantities, index=backtest_dates, columns=prices.columns)
    
    # Detailed tracking DataFrames
    daily_trades_df = pd.DataFrame(daily_trades_list, index=backtest_dates, columns=prices.columns)
    daily_weights_df = pd.DataFrame(daily_weights_list, index=backtest_dates, columns=prices.columns)
    daily_prices_df = pd.DataFrame(daily_prices_list, index=backtest_dates, columns=prices.columns)
    daily_spreads_df = pd.DataFrame(daily_spreads_list, index=backtest_dates, columns=prices.columns)
    daily_target_weights_df = pd.DataFrame(daily_target_weights_list, index=backtest_dates, columns=prices.columns)
    daily_costs_df = pd.DataFrame(daily_costs_list, index=backtest_dates)

    # Create enhanced result
    result = EnhancedBacktestResult(
        cash=post_trade_cash,
        quantities=post_trade_quantities,
        risk_target=risk_target,
        timings=timings,
        daily_trades=daily_trades_df,
        daily_weights=daily_weights_df,
        daily_prices=daily_prices_df,
        daily_spreads=daily_spreads_df,
        daily_target_weights=daily_target_weights_df,
        daily_costs=daily_costs_df,
        daily_optimization_info=daily_optimization_info,
        strategy_name=strategy_name,
        strategy_params=strategy_params
    )

    return result


def create_dashboard_from_enhanced_backtest(
    result: EnhancedBacktestResult,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Create a comprehensive dashboard from an enhanced backtest result.
    
    Args:
        result: EnhancedBacktestResult from run_enhanced_backtest
        save_path: Optional path to save dashboard
        show: Whether to display plots
        
    Returns:
        Dictionary of created figures
    """
    detailed_result = result.to_detailed_result()
    dashboard = BacktestDashboard(detailed_result)
    return dashboard.create_full_dashboard(save_path=save_path, show=show)


# Example usage functions
def run_markowitz_with_dashboard(risk_target: float = 0.05, save_path: Optional[str] = None):
    """Example: Run Markowitz strategy with dashboard."""
    from markowitz import markowitz_strategy
    
    result = run_enhanced_backtest(
        strategy=markowitz_strategy,
        risk_target=risk_target,
        strategy_name="Markowitz",
        strategy_params={"risk_target": risk_target},
        verbose=True
    )
    
    figures = create_dashboard_from_enhanced_backtest(result, save_path=save_path)
    return result, figures


def run_transformation_with_dashboard(
    transformation_config,
    risk_target: float = 0.05,
    save_path: Optional[str] = None
):
    """Example: Run transformation strategy with dashboard."""
    from transformation import create_transformation_strategy
    
    strategy = create_transformation_strategy(transformation_config)
    
    result = run_enhanced_backtest(
        strategy=strategy,
        risk_target=risk_target,
        strategy_name=f"Transformation_{transformation_config.__class__.__name__}",
        strategy_params=transformation_config.__dict__,
        verbose=True
    )
    
    figures = create_dashboard_from_enhanced_backtest(result, save_path=save_path)
    return result, figures


if __name__ == "__main__":
    # Example usage
    print("Enhanced backtest module ready.")
    print("Example usage:")
    print("result, figures = run_markowitz_with_dashboard(risk_target=0.05, save_path='./dashboard')")
    print("result, figures = run_transformation_with_dashboard(config, save_path='./dashboard')") 