"""
Test Dashboard System

Simple test script to verify that the dashboard system works correctly 
with the existing backtest infrastructure.
"""

import numpy as np
import os
from pathlib import Path

# Import the basic backtest system
from backtest import run_backtest, load_data

# Import the dashboard system
from dashboard import create_dashboard_from_backtest


def test_basic_dashboard():
    """Test dashboard with a simple equal-weight strategy."""
    print("Testing dashboard with equal-weight strategy...")
    
    # Load data to get the correct number of assets
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Number of assets: {n_assets}")
    
    # Create a simple equal-weight strategy
    def equal_weight_strategy(inputs):
        """Simple equal-weight strategy."""
        n = inputs.n_assets
        w = np.ones(n) / (n + 1)  # Equal weights with some cash
        c = 1 / (n + 1)  # Cash weight
        return w, c, None
    
    # Run basic backtest
    print("Running backtest...")
    result = run_backtest(
        strategy=equal_weight_strategy,
        risk_target=0.05,
        verbose=False
    )
    
    print(f"Backtest completed. Portfolio value range: ${result.portfolio_value.min():.0f} - ${result.portfolio_value.max():.0f}")
    
    # Create dashboard output directory
    save_path = "./test_dashboard_output"
    os.makedirs(save_path, exist_ok=True)
    
    # Create dashboard
    print("Creating dashboard...")
    figures = create_dashboard_from_backtest(
        backtest_result=result,
        strategy_name="Equal Weight Test",
        save_path=save_path,
        show=False  # Don't show plots in test
    )
    
    print(f"Dashboard created with {len(figures)} figures:")
    for name, fig in figures.items():
        if fig is not None:
            print(f"  - {name}")
    
    print(f"Dashboard saved to: {save_path}")
    print("Test completed successfully!")
    
    return result, figures


def test_enhanced_backtest_compatibility():
    """Test that enhanced backtest produces compatible results."""
    try:
        from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest
        
        print("Testing enhanced backtest system...")
        
        # Create a simple strategy
        def simple_strategy(inputs):
            """Simple strategy for testing."""
            n = inputs.n_assets
            w = np.ones(n) / (n + 1)
            c = 1 / (n + 1)
            return w, c, None
        
        # Run enhanced backtest
        print("Running enhanced backtest...")
        enhanced_result = run_enhanced_backtest(
            strategy=simple_strategy,
            risk_target=0.05,
            strategy_name="Enhanced Test",
            verbose=False
        )
        
        print(f"Enhanced backtest completed.")
        print(f"Additional data captured:")
        print(f"  - Daily trades: {enhanced_result.daily_trades.shape}")
        print(f"  - Daily weights: {enhanced_result.daily_weights.shape}")
        print(f"  - Daily prices: {enhanced_result.daily_prices.shape}")
        
        # Create dashboard
        save_path = "./test_enhanced_dashboard_output"
        os.makedirs(save_path, exist_ok=True)
        
        figures = create_dashboard_from_enhanced_backtest(
            result=enhanced_result,
            save_path=save_path,
            show=False
        )
        
        print(f"Enhanced dashboard created with {len(figures)} figures")
        print(f"Enhanced dashboard saved to: {save_path}")
        
        return enhanced_result, figures
        
    except ImportError as e:
        print(f"Enhanced backtest not available: {e}")
        return None, None


def main():
    """Run all tests."""
    print("Dashboard System Tests")
    print("=" * 50)
    
    try:
        # Test basic dashboard
        basic_result, basic_figures = test_basic_dashboard()
        
        print("\n" + "=" * 50)
        
        # Test enhanced backtest (if available)
        enhanced_result, enhanced_figures = test_enhanced_backtest_compatibility()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Basic dashboard: {len(basic_figures) if basic_figures else 0} figures")
        if enhanced_figures:
            print(f"Enhanced dashboard: {len(enhanced_figures)} figures")
        
        print(f"\nDashboard outputs saved to:")
        print(f"  - ./test_dashboard_output/")
        if enhanced_result:
            print(f"  - ./test_enhanced_dashboard_output/")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 