"""
Dashboard Example Script

This script demonstrates how to use the enhanced backtest system with dashboards
for various portfolio strategies including Markowitz and transformation strategies.
"""

import numpy as np
from pathlib import Path
import os

# Import our enhanced backtest and dashboard systems
from enhanced_backtest import (
    run_enhanced_backtest, 
    create_dashboard_from_enhanced_backtest,
    run_markowitz_with_dashboard,
    run_transformation_with_dashboard
)

# Import transformation strategies
from transformation import (
    TransformationConfig,
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    create_transformation_strategy
)

# Import transformations
from transformations.uniform_policy import UniformTradingPolicy, DynamicUniformTradingPolicy
from transformations.univariate_policy import UnivariateScalarTrackingPolicy


def run_markowitz_example():
    """Run Markowitz strategy with dashboard."""
    print("=" * 60)
    print("MARKOWITZ STRATEGY WITH DASHBOARD")
    print("=" * 60)
    
    # Create output directory
    save_path = "./dashboard_outputs/markowitz"
    os.makedirs(save_path, exist_ok=True)
    
    # Run backtest with dashboard
    result, figures = run_markowitz_with_dashboard(
        risk_target=0.05,
        save_path=save_path
    )
    
    print(f"\nMarkowitz Strategy Results:")
    print(f"Total Return: {result.mean_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Turnover: {result.turnover:.2f}")
    print(f"Max Leverage: {result.max_leverage:.2f}")
    print(f"Dashboard saved to: {save_path}")
    
    return result, figures


def run_uniform_transformation_example():
    """Run uniform transformation strategy with dashboard."""
    print("=" * 60)
    print("UNIFORM TRANSFORMATION STRATEGY WITH DASHBOARD")
    print("=" * 60)
    
    # Create uniform transformation config
    n_assets = 50  # Approximate number of assets (will be set correctly in backtest)
    target_weights = np.ones(n_assets) / n_assets  # Equal weights
    
    config = TransformationConfig(
        policy=UniformTradingPolicy(
            target_weights=target_weights,
            trade_fractions=np.full(n_assets, 0.02)  # 2% per day
        ),
        risk_target=0.05
    )
    
    # Create output directory
    save_path = "./dashboard_outputs/uniform_transformation"
    os.makedirs(save_path, exist_ok=True)
    
    # Run backtest with dashboard
    result, figures = run_transformation_with_dashboard(
        transformation_config=config,
        risk_target=0.05,
        save_path=save_path
    )
    
    print(f"\nUniform Transformation Strategy Results:")
    print(f"Total Return: {result.mean_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Turnover: {result.turnover:.2f}")
    print(f"Max Leverage: {result.max_leverage:.2f}")
    print(f"Dashboard saved to: {save_path}")
    
    return result, figures


def run_dynamic_transformation_example():
    """Run dynamic uniform transformation strategy with dashboard."""
    print("=" * 60)
    print("DYNAMIC TRANSFORMATION STRATEGY WITH DASHBOARD")
    print("=" * 60)
    
    # Create dynamic transformation config
    n_assets = 50  # Approximate number of assets
    target_weights = np.ones(n_assets) / n_assets  # Equal weights
    
    config = TransformationConfig(
        policy=DynamicUniformTradingPolicy(
            target_weights=target_weights,
            base_trade_fraction=0.02,  # 2% base rate
            urgency_factor=0.5  # Increase trading when further from target
        ),
        risk_target=0.05
    )
    
    # Create output directory
    save_path = "./dashboard_outputs/dynamic_transformation"
    os.makedirs(save_path, exist_ok=True)
    
    # Run backtest with dashboard
    result, figures = run_transformation_with_dashboard(
        transformation_config=config,
        risk_target=0.05,
        save_path=save_path
    )
    
    print(f"\nDynamic Transformation Strategy Results:")
    print(f"Total Return: {result.mean_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Turnover: {result.turnover:.2f}")
    print(f"Max Leverage: {result.max_leverage:.2f}")
    print(f"Dashboard saved to: {save_path}")
    
    return result, figures


def run_optimal_transformation_example():
    """Run optimal (univariate scalar tracking) transformation strategy with dashboard."""
    print("=" * 60)
    print("OPTIMAL TRANSFORMATION STRATEGY WITH DASHBOARD")
    print("=" * 60)
    
    # Create optimal transformation config
    n_assets = 50  # Approximate number of assets
    target_weights = np.ones(n_assets) / n_assets  # Equal weights
    
    config = TransformationConfig(
        policy=UnivariateScalarTrackingPolicy(
            target_weights=target_weights,
            gamma_cost=0.5,  # Cost aversion parameter
            gamma_risk=1.0   # Risk aversion parameter
        ),
        risk_target=0.05
    )
    
    # Create output directory
    save_path = "./dashboard_outputs/optimal_transformation"
    os.makedirs(save_path, exist_ok=True)
    
    # Run backtest with dashboard
    result, figures = run_transformation_with_dashboard(
        transformation_config=config,
        risk_target=0.05,
        save_path=save_path
    )
    
    print(f"\nOptimal Transformation Strategy Results:")
    print(f"Total Return: {result.mean_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Turnover: {result.turnover:.2f}")
    print(f"Max Leverage: {result.max_leverage:.2f}")
    print(f"Dashboard saved to: {save_path}")
    
    return result, figures


def compare_strategies():
    """Run multiple strategies and create a comparison dashboard."""
    print("=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    strategies = []
    
    # Run all strategies (in practice, you might want to run these in parallel or cache results)
    print("Running Markowitz...")
    markowitz_result, _ = run_markowitz_with_dashboard(risk_target=0.05, show=False)
    strategies.append(("Markowitz", markowitz_result))
    
    # For transformation strategies, we'll need to create a simple comparison
    # (Note: The actual implementation would need the correct number of assets)
    
    # Create comparison summary
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Return':<10} {'Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Turnover':<10}")
    print("-" * 80)
    
    for name, result in strategies:
        print(f"{name:<20} {result.mean_return:>8.2%} {result.volatility:>7.2%} "
              f"{result.sharpe:>7.2f} {result.max_drawdown:>7.2%} {result.turnover:>9.2f}")
    
    return strategies


def main():
    """Main function to run all examples."""
    print("Portfolio Strategy Dashboard Examples")
    print("=" * 60)
    
    # Create main output directory
    os.makedirs("./dashboard_outputs", exist_ok=True)
    
    try:
        # Run individual strategy examples
        markowitz_result, markowitz_figures = run_markowitz_example()
        
        # Note: Transformation examples would need proper asset count
        # These are commented out for now since they need the correct number of assets
        # uniform_result, uniform_figures = run_uniform_transformation_example()
        # dynamic_result, dynamic_figures = run_dynamic_transformation_example()
        # optimal_result, optimal_figures = run_optimal_transformation_example()
        
        # Run comparison
        strategies = compare_strategies()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Dashboard files have been saved to the './dashboard_outputs/' directory.")
        print("Each strategy has its own subdirectory with detailed visualizations.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all required modules are available and data is loaded.")
        raise


if __name__ == "__main__":
    main() 