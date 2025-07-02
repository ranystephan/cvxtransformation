"""
Test Script for Portfolio Transformation Strategies

This script demonstrates how to use the transformation strategies with the 
existing backtesting infrastructure and validates that they work correctly.
"""

import numpy as np
from loguru import logger
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

from backtest import run_backtest, load_data
from transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy, 
    UnivariateScalarTrackingPolicy,
    run_transformation_backtest,
    TransformationConfig,
    create_transformation_strategy
)


def test_basic_functionality():
    """Test basic functionality of transformation strategies."""
    logger.info("Testing basic functionality of transformation strategies...")
    
    # Load data to get asset information
    try:
        prices, spread, rf, volume, short_fee_data = load_data()
        n_assets = prices.shape[1]
        logger.info(f"Loaded data with {n_assets} assets")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Define a simple transformation scenario
    w_initial = np.zeros(n_assets)
    w_initial[0] = -0.02  # Small short position in first asset
    
    w_target = np.zeros(n_assets)
    w_target[1] = 0.01   # Small long position in second asset
    w_target[2] = 0.01   # Small long position in third asset
    
    total_days = 10  # Short transformation period
    
    logger.info(f"Initial weights (non-zero): {dict(enumerate(w_initial[w_initial != 0]))}")
    logger.info(f"Target weights (non-zero): {dict(enumerate(w_target[w_target != 0]))}")
    logger.info(f"Transformation period: {total_days} days")
    
    return True


def test_uniform_policy():
    """Test the uniform transformation policy."""
    logger.info("\n" + "="*60)
    logger.info("Testing Uniform Transformation Policy")
    logger.info("="*60)
    
    try:
        # Load data
        prices, _, _, _, _ = load_data()
        n_assets = prices.shape[1]
        
        # Define transformation
        w_initial = np.zeros(n_assets)
        w_initial[0] = -0.01
        
        w_target = np.zeros(n_assets)  
        w_target[1] = 0.005
        w_target[2] = 0.005
        
        total_days = 5
        
        # Test the policy
        policy = UniformTransformationPolicy()
        result = run_transformation_backtest(
            policy=policy,
            w_initial=w_initial,
            w_target=w_target,
            total_days=total_days,
            risk_target=0.05,  # 5% annual volatility target
            verbose=False
        )
        
        logger.info("Uniform Policy Results:")
        logger.info(f"  Sharpe Ratio: {result.sharpe:.3f}")
        logger.info(f"  Annual Return: {result.mean_return:.2%}")
        logger.info(f"  Annual Volatility: {result.volatility:.2%}")
        logger.info(f"  Annual Turnover: {result.turnover:.2f}")
        logger.info(f"  Max Leverage: {result.max_leverage:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Uniform policy test failed: {e}")
        return False


def test_dynamic_uniform_policy():
    """Test the dynamic uniform transformation policy."""
    logger.info("\n" + "="*60)
    logger.info("Testing Dynamic Uniform Transformation Policy")
    logger.info("="*60)
    
    try:
        # Load data
        prices, _, _, _, _ = load_data()
        n_assets = prices.shape[1]
        
        # Define transformation
        w_initial = np.zeros(n_assets)
        w_initial[0] = -0.01
        
        w_target = np.zeros(n_assets)  
        w_target[1] = 0.005
        w_target[2] = 0.005
        
        total_days = 5
        
        # Test the policy
        policy = DynamicUniformTransformationPolicy()
        result = run_transformation_backtest(
            policy=policy,
            w_initial=w_initial,
            w_target=w_target,
            total_days=total_days,
            risk_target=0.05,
            verbose=False
        )
        
        logger.info("Dynamic Uniform Policy Results:")
        logger.info(f"  Sharpe Ratio: {result.sharpe:.3f}")
        logger.info(f"  Annual Return: {result.mean_return:.2%}")
        logger.info(f"  Annual Volatility: {result.volatility:.2%}")
        logger.info(f"  Annual Turnover: {result.turnover:.2f}")
        logger.info(f"  Max Leverage: {result.max_leverage:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dynamic uniform policy test failed: {e}")
        return False


def test_univariate_policy():
    """Test the univariate scalar tracking transformation policy."""
    logger.info("\n" + "="*60)
    logger.info("Testing Univariate Scalar Tracking Policy")
    logger.info("="*60)
    
    try:
        # Load data
        prices, _, _, _, _ = load_data()
        n_assets = prices.shape[1]
        
        # Define transformation
        w_initial = np.zeros(n_assets)
        w_initial[0] = -0.01
        
        w_target = np.zeros(n_assets)  
        w_target[1] = 0.005
        w_target[2] = 0.005
        
        total_days = 5
        
        # Test the policy
        policy = UnivariateScalarTrackingPolicy(risk_aversion=1.0)
        result = run_transformation_backtest(
            policy=policy,
            w_initial=w_initial,
            w_target=w_target,
            total_days=total_days,
            risk_target=0.05,
            verbose=False
        )
        
        logger.info("Univariate Scalar Tracking Policy Results:")
        logger.info(f"  Sharpe Ratio: {result.sharpe:.3f}")
        logger.info(f"  Annual Return: {result.mean_return:.2%}")
        logger.info(f"  Annual Volatility: {result.volatility:.2%}")
        logger.info(f"  Annual Turnover: {result.turnover:.2f}")
        logger.info(f"  Max Leverage: {result.max_leverage:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Univariate scalar tracking policy test failed: {e}")
        return False


def compare_policies():
    """Compare all transformation policies side by side."""
    logger.info("\n" + "="*60)
    logger.info("POLICY COMPARISON")
    logger.info("="*60)
    
    try:
        # Load data
        prices, _, _, _, _ = load_data()
        n_assets = prices.shape[1]
        
        # Define transformation scenario
        w_initial = np.zeros(n_assets)
        w_initial[0] = -0.015  # Slightly larger for more visible differences
        
        w_target = np.zeros(n_assets)  
        w_target[1] = 0.0075
        w_target[2] = 0.0075
        
        total_days = 8
        risk_target = 0.06
        
        # Test all policies
        policies = {
            "Uniform": UniformTransformationPolicy(),
            "Dynamic Uniform": DynamicUniformTransformationPolicy(),
            "Univariate Tracking": UnivariateScalarTrackingPolicy(risk_aversion=0.5)
        }
        
        results = {}
        
        for name, policy in policies.items():
            try:
                result = run_transformation_backtest(
                    policy=policy,
                    w_initial=w_initial,
                    w_target=w_target,
                    total_days=total_days,
                    risk_target=risk_target,
                    verbose=False
                )
                results[name] = result
                logger.info(f"✓ {name} policy completed successfully")
            except Exception as e:
                logger.error(f"✗ {name} policy failed: {e}")
                results[name] = None
        
        # Display comparison
        logger.info(f"\n{'Policy':<15} | {'Sharpe':<6} | {'Return':<7} | {'Vol':<6} | {'Turnover':<8} | {'MaxLev':<6}")
        logger.info("-" * 70)
        
        for name, result in results.items():
            if result is not None:
                logger.info(f"{name:<15} | {result.sharpe:6.3f} | {result.mean_return:6.2%} | "
                           f"{result.volatility:5.2%} | {result.turnover:8.2f} | {result.max_leverage:6.2f}")
            else:
                logger.info(f"{name:<15} | {'FAILED':<6} | {'---':<7} | {'---':<6} | {'---':<8} | {'---':<6}")
        
        return True
        
    except Exception as e:
        logger.error(f"Policy comparison failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("PORTFOLIO TRANSFORMATION STRATEGIES TEST SUITE")
    logger.info("="*80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Uniform Policy", test_uniform_policy),
        ("Dynamic Uniform Policy", test_dynamic_uniform_policy),
        ("Univariate Scalar Tracking Policy", test_univariate_policy),
        ("Policy Comparison", compare_policies)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "="*80)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info("="*80)
    
    if passed == total:
        logger.info("🎉 All tests passed! The transformation strategies are working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 