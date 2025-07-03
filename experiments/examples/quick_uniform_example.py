"""
Quick Uniform Transformation Example with Dashboard

This script shows how to run a uniform transformation strategy and generate plots.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from backtest import load_data
from transformation import (
    UniformTransformationPolicy,
    run_transformation_backtest
)
from dashboard import create_dashboard_from_backtest

def run_quick_uniform_example():
    """Run a simple uniform transformation example with plots."""
    print("Running Uniform Transformation Example...")
    
    # Load data to get number of assets
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Number of assets: {n_assets}")
    
    # Define a simple transformation scenario
    w_initial = np.zeros(n_assets)
    w_initial[0] = -0.02  # Short 2% in first asset
    
    w_target = np.zeros(n_assets)
    w_target[1] = 0.01   # Long 1% in second asset
    w_target[2] = 0.01   # Long 1% in third asset
    
    total_days = 10  # Transform over 10 days
    
    print(f"Initial: Short {abs(w_initial[0]):.1%} in asset 0")
    print(f"Target: Long {w_target[1]:.1%} in asset 1, {w_target[2]:.1%} in asset 2")
    print(f"Transformation period: {total_days} days")
    
    # Create transformation configuration and strategy
    policy = UniformTransformationPolicy()
    
    # Option 1: Use enhanced backtest for detailed data (recommended)
    try:
        from transformation import TransformationConfig, create_transformation_strategy
        from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest
        
        config = TransformationConfig(
            w_initial=w_initial,
            w_target=w_target,
            total_days=total_days,
            participation_limit=0.05
        )
        
        strategy = create_transformation_strategy(policy, config)
        
        print("Using enhanced backtest for detailed tracking...")
        result = run_enhanced_backtest(
            strategy=strategy,
            risk_target=0.05,  # 5% annual volatility target
            strategy_name="Uniform Transformation Example",
            verbose=True
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Sharpe Ratio: {result.sharpe:.2f}")
        print(f"Annual Return: {result.mean_return:.2%}")
        print(f"Annual Volatility: {result.volatility:.2%}")
        print(f"Annual Turnover: {result.turnover:.2f}")
        print(f"Max Leverage: {result.max_leverage:.2f}")
        
        # Check data availability
        print(f"\n=== DATA CAPTURED ===")
        print(f"Daily trades: {result.daily_trades.shape} ({(result.daily_trades != 0).sum().sum()} non-zero)")
        print(f"Daily weights: {result.daily_weights.shape} ({(result.daily_weights != 0).sum().sum()} non-zero)")
        
        # Create dashboard with enhanced data
        save_path = "./uniform_example_output"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nCreating enhanced dashboard plots...")
        figures = create_dashboard_from_enhanced_backtest(
            result=result,
            save_path=save_path,
            show=False  # Set to True if you want to display plots
        )
        
    except ImportError as e:
        print(f"Enhanced backtest not available ({e}), using basic backtest...")
        # Fallback to basic backtest
        result = run_transformation_backtest(
            policy=policy,
            w_initial=w_initial,
            w_target=w_target,
            total_days=total_days,
            risk_target=0.05,
            verbose=True
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Sharpe Ratio: {result.sharpe:.2f}")
        print(f"Annual Return: {result.mean_return:.2%}")
        print(f"Annual Volatility: {result.volatility:.2%}")
        print(f"Annual Turnover: {result.turnover:.2f}")
        print(f"Max Leverage: {result.max_leverage:.2f}")
        
        # Create dashboard with basic data (trades_analysis and weights_heatmap will be empty)
        save_path = "./uniform_example_output"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\nCreating basic dashboard plots...")
        print("Note: trades_analysis and weights_heatmap will be empty with basic backtest")
        figures = create_dashboard_from_backtest(
            backtest_result=result,
            strategy_name="Uniform Transformation Example",
            save_path=save_path,
            show=False
        )
    
    print(f"Dashboard created with {len(figures)} plots:")
    for name in figures.keys():
        print(f"  - {name}.png")
    
    print(f"\nPlots saved to: {save_path}/")
    print("You can view the PNG files to see the results!")
    
    return result, figures

if __name__ == "__main__":
    run_quick_uniform_example() 