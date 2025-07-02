#!/usr/bin/env python3
"""
Script to display results from existing backtest checkpoints.
Usage: python show_results.py
"""

import sys
from pathlib import Path

# Add experiments to path
sys.path.append((Path(__file__).parent / "experiments").as_posix())

from experiments.backtest import BacktestResult
from experiments.utils import checkpoints_path, figures_path
from loguru import logger


def show_backtest_results(checkpoint_name: str = "scaling_parametrized_0.1.pickle") -> None:
    """Display comprehensive results from a backtest checkpoint."""
    
    checkpoint_path = checkpoints_path() / checkpoint_name
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        for file in checkpoints_path().glob("*.pickle"):
            logger.info(f"  - {file.name}")
        return
    
    # Load the backtest result
    result = BacktestResult.load(checkpoint_path)
    
    # Display comprehensive portfolio performance metrics
    logger.info("=" * 60)
    logger.info("PORTFOLIO PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Backtest Period: {result.history[0].strftime('%Y-%m-%d')} to {result.history[-1].strftime('%Y-%m-%d')}")
    logger.info(f"Number of Trading Days: {len(result.history)}")
    logger.info(f"Number of Assets: {result.quantities.shape[1]}")
    logger.info("")
    
    # Performance metrics
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  Annualized Return:     {result.mean_return:.2%}")
    logger.info(f"  Annualized Volatility: {result.volatility:.2%}")
    logger.info(f"  Sharpe Ratio:          {result.sharpe:.2f}")
    logger.info(f"  Maximum Drawdown:      {result.max_drawdown:.2%}")
    logger.info(f"  Maximum Leverage:      {result.max_leverage:.2f}")
    logger.info(f"  Annualized Turnover:   {result.turnover:.2f}")
    logger.info("")
    
    # Portfolio composition
    final_portfolio_value = result.portfolio_value.iloc[-1]
    initial_portfolio_value = result.portfolio_value.iloc[0]
    total_return = (final_portfolio_value / initial_portfolio_value - 1) * 100
    
    logger.info("PORTFOLIO SUMMARY:")
    logger.info(f"  Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
    logger.info(f"  Final Portfolio Value:   ${final_portfolio_value:,.2f}")
    logger.info(f"  Total Return:            {total_return:.2f}%")
    logger.info(f"  Final Cash Position:     ${result.cash.iloc[-1]:,.2f}")
    logger.info("")
    
    # Timing information
    total_time = sum(t.total for t in result.timings)
    cvxpy_time = sum(t.cvxpy for t in result.timings)
    solver_time = sum(t.solver for t in result.timings)
    other_time = sum(t.other for t in result.timings)
    
    logger.info("COMPUTATIONAL PERFORMANCE:")
    logger.info(f"  Total Computation Time: {total_time:.1f} seconds")
    logger.info(f"  Average Time per Day:   {total_time / len(result.timings):.3f} seconds")
    logger.info(f"  CVXPY Time:             {cvxpy_time/total_time:.1%}")
    logger.info(f"  Solver Time:            {solver_time/total_time:.1%}")
    logger.info(f"  Other Time:             {other_time/total_time:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    show_backtest_results() 