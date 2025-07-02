"""
Compare Transformation Strategies

This script compares all three transformation strategies and generates plots for each.
"""

import numpy as np
import os
from backtest import load_data
from transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    run_transformation_backtest
)
from transformations.univariate_policy import UnivariateScalarTrackingPolicy
from dashboard import create_dashboard_from_backtest

def compare_transformation_strategies():
    """Compare all three transformation strategies."""
    print("=" * 70)
    print("COMPARING TRANSFORMATION STRATEGIES")
    print("=" * 70)
    
    # Load data
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Number of assets: {n_assets}")
    
    # Define transformation scenario
    w_initial = np.zeros(n_assets)
    w_initial[0] = -0.03  # Short 3% in first asset
    
    w_target = np.zeros(n_assets)
    w_target[1] = 0.015  # Long 1.5% in second asset
    w_target[2] = 0.015  # Long 1.5% in third asset
    
    total_days = 15  # Transform over 15 days
    risk_target = 0.06  # 6% annual volatility target
    
    print(f"Transformation: Short {abs(w_initial[0]):.1%} → Long {w_target[1]:.1%} + {w_target[2]:.1%}")
    print(f"Period: {total_days} days")
    print(f"Risk target: {risk_target:.1%} annual volatility")
    
    # Define the strategies to test
    strategies = {
        "Uniform": UniformTransformationPolicy(),
        "Dynamic_Uniform": DynamicUniformTransformationPolicy(),
        "Univariate_Tracking": UnivariateScalarTrackingPolicy(risk_aversion=1.0)
    }
    
    results = {}
    
    # Test each strategy
    for name, policy in strategies.items():
        print(f"\n{'-'*50}")
        print(f"Testing {name.replace('_', ' ')} Policy...")
        print(f"{'-'*50}")
        
        try:
            # Run backtest
            result = run_transformation_backtest(
                policy=policy,
                w_initial=w_initial,
                w_target=w_target,
                total_days=total_days,
                risk_target=risk_target,
                verbose=False  # Set to True for detailed output
            )
            
            results[name] = result
            
            # Print results
            print(f"✓ {name} completed successfully")
            print(f"  Sharpe Ratio: {result.sharpe:.3f}")
            print(f"  Annual Return: {result.mean_return:.2%}")
            print(f"  Annual Volatility: {result.volatility:.2%}")
            print(f"  Annual Turnover: {result.turnover:.2f}")
            print(f"  Max Leverage: {result.max_leverage:.2f}")
            
            # Create dashboard for this strategy
            save_path = f"./strategy_comparison/{name.lower()}"
            os.makedirs(save_path, exist_ok=True)
            
            figures = create_dashboard_from_backtest(
                backtest_result=result,
                strategy_name=name.replace('_', ' ') + " Transformation",
                save_path=save_path,
                show=False
            )
            
            print(f"  Dashboard saved to: {save_path}/")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = None
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Strategy':<20} | {'Sharpe':<6} | {'Return':<7} | {'Vol':<6} | {'Turnover':<8} | {'MaxLev':<6}")
    print("-" * 70)
    
    for name, result in results.items():
        if result is not None:
            print(f"{name.replace('_', ' '):<20} | {result.sharpe:6.3f} | {result.mean_return:6.2%} | "
                  f"{result.volatility:5.2%} | {result.turnover:8.2f} | {result.max_leverage:6.2f}")
        else:
            print(f"{name.replace('_', ' '):<20} | {'FAILED':<6} | {'FAILED':<7} | {'FAILED':<6} | {'FAILED':<8} | {'FAILED':<6}")
    
    print(f"\nPlots saved to ./strategy_comparison/ subdirectories")
    print("View the PNG files to compare strategy performance!")
    
    return results

if __name__ == "__main__":
    compare_transformation_strategies() 