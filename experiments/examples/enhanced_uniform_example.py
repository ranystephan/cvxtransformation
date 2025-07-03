"""
Enhanced Uniform Transformation Example with Full Dashboard

This script uses the enhanced backtest system to capture detailed trading data
for complete dashboard plots including trades analysis and weights heatmap.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from backtest import load_data
from transformation import (
    UniformTransformationPolicy,
    TransformationConfig,
    create_transformation_strategy
)
from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

def run_enhanced_uniform_example():
    """Run a uniform transformation example with enhanced tracking and full plots."""
    print("Running Enhanced Uniform Transformation Example...")
    
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
    
    total_days = 15  # Transform over 15 days
    risk_target = 0.05  # 5% annual volatility target
    
    print(f"Initial: Short {abs(w_initial[0]):.1%} in asset 0")
    print(f"Target: Long {w_target[1]:.1%} in asset 1, {w_target[2]:.1%} in asset 2")
    print(f"Transformation period: {total_days} days")
    print(f"Risk target: {risk_target:.1%} annual volatility")
    
    # Create transformation configuration
    policy = UniformTransformationPolicy()
    config = TransformationConfig(
        w_initial=w_initial,
        w_target=w_target,
        total_days=total_days,
        participation_limit=0.05
    )
    
    # Create transformation strategy
    strategy = create_transformation_strategy(policy, config)
    
    # Run enhanced backtest to capture detailed data
    print(f"\nRunning enhanced backtest...")
    result = run_enhanced_backtest(
        strategy=strategy,
        risk_target=risk_target,
        strategy_name="Enhanced Uniform Transformation",
        strategy_params={
            'policy': 'UniformTransformationPolicy',
            'total_days': total_days,
            'w_initial_nonzero': {i: w for i, w in enumerate(w_initial) if w != 0},
            'w_target_nonzero': {i: w for i, w in enumerate(w_target) if w != 0}
        },
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
    print(f"Daily trades shape: {result.daily_trades.shape}")
    print(f"Daily weights shape: {result.daily_weights.shape}")
    print(f"Daily prices shape: {result.daily_prices.shape}")
    print(f"Non-zero trades: {(result.daily_trades != 0).sum().sum()}")
    print(f"Non-zero weights: {(result.daily_weights != 0).sum().sum()}")
    
    # Create dashboard with enhanced data
    save_path = "./enhanced_uniform_output"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\nCreating enhanced dashboard plots...")
    figures = create_dashboard_from_enhanced_backtest(
        result=result,
        save_path=save_path,
        show=False  # Set to True if you want to display plots
    )
    
    print(f"Enhanced dashboard created with {len(figures)} plots:")
    for name in figures.keys():
        print(f"  - {name}.png")
    
    print(f"\nPlots saved to: {save_path}/")
    print("Now all plots including trades_analysis and weights_heatmap should have data!")
    
    return result, figures

if __name__ == "__main__":
    run_enhanced_uniform_example() 