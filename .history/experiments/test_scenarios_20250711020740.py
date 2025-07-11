# experiments/test_scenarios.py

import numpy as np
import pandas as pd
from loguru import logger
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from backtest import run_backtest, load_data
from transformation_strategies import dynamic_uniform_strategy, front_loaded_strategy
from portfolio_construction import get_minimum_variance_portfolio
from utils import checkpoints_path, figures_path, estimate_transition_period
from analyze_results import create_comprehensive_report


def create_random_portfolio(asset_names, n_stocks=20, max_weight=0.25, seed=42):
    """
    Create a random portfolio with n_stocks randomly selected assets.
    
    Args:
        asset_names: List of asset names in the universe
        n_stocks: Number of stocks to include in portfolio
        max_weight: Maximum weight any single stock can have
        seed: Random seed for reproducibility
    
    Returns:
        pd.Series: Portfolio weights (sums to 1.0)
    """
    rng = np.random.default_rng(seed)
    
    # Randomly select n_stocks from the universe
    selected_assets = rng.choice(asset_names, size=n_stocks, replace=False)
    
    # Generate random weights that sum to 1.0
    weights = rng.uniform(0.01, max_weight, size=n_stocks)
    weights = weights / weights.sum()  # Normalize to sum to 1.0
    
    # Create a Series with zeros for all assets, then fill selected ones
    all_weights = pd.Series(0.0, index=asset_names)
    all_weights[selected_assets] = weights
    
    return all_weights


def create_liquidation_target(asset_names):
    """
    Create a target portfolio for liquidation (all zeros).
    
    Args:
        asset_names: List of asset names in the universe
    
    Returns:
        pd.Series: Target weights (all zeros)
    """
    return pd.Series(0.0, index=asset_names)


def create_equal_weight_target(asset_names, target_weight=0.6):
    """
    Create an equal-weight target portfolio.
    
    Args:
        asset_names: List of asset names in the universe
        target_weight: Total weight to allocate to stocks (rest goes to cash)
    
    Returns:
        pd.Series: Target weights (equal weight for all assets)
    """
    return pd.Series(target_weight / len(asset_names), index=asset_names)


def create_minimum_variance_portfolios(asset_names, construction_date="2022-01-01"):
    """
    Create minimum variance portfolios for different risk targets.
    
    Args:
        asset_names: List of asset names in the universe
        construction_date: Date to use for historical data
    
    Returns:
        tuple: (low_risk_weights, high_risk_weights)
    """
    prices, _, _, _ = load_data()
    
    # Split universe into two halves
    group_A_assets = asset_names[:len(asset_names)//2]
    group_B_assets = asset_names[len(asset_names)//2:]
    
    # Get historical data up to construction date
    historical_prices_A = prices.loc[:construction_date][group_A_assets]
    historical_prices_B = prices.loc[:construction_date][group_B_assets]
    
    # Portfolio A: Low risk (5% annualized vol target)
    low_risk_weights = get_minimum_variance_portfolio(historical_prices_A, risk_target=0.05)
    # Portfolio B: Higher risk (15% annualized vol target)
    high_risk_weights = get_minimum_variance_portfolio(historical_prices_B, risk_target=0.15)
    
    # Combine the weights into a single series, filling missing with 0
    low_risk_weights = low_risk_weights.reindex(asset_names, fill_value=0)
    high_risk_weights = high_risk_weights.reindex(asset_names, fill_value=0)
    
    return low_risk_weights, high_risk_weights


def calculate_implementation_shortfall(result, initial_portfolio_value):
    """Calculates the normalized cost of the liquidation."""
    
    # The actual cash we have at the end of the liquidation
    final_cash = result.portfolio_value.iloc[-1]
    
    # Shortfall is the difference. A positive value means it cost us money.
    shortfall_dollars = initial_portfolio_value - final_cash
    
    # Return the shortfall as a fraction of the initial value
    return shortfall_dollars / initial_portfolio_value


def test_liquidation_scenario():
    """
    Test liquidation scenario: 20 random stocks to all cash.
    """
    # Load data to get asset names
    prices, _, _, _ = load_data()
    asset_names = prices.columns
    
    # Create initial portfolio: 20 random stocks with random weights
    initial_weights = create_random_portfolio(asset_names, n_stocks=20, max_weight=0.25)
    
    # Create target: all cash (all zeros)
    target_weights = create_liquidation_target(asset_names)
    
    # Run backtest
    logger.info("Testing liquidation scenario: 20 random stocks → all cash")
    
    day_counter = 0
    def strategy_wrapper(inputs, **kwargs):
        nonlocal day_counter
        kwargs['days_remaining'] = 30 - day_counter
        result = dynamic_uniform_strategy(inputs, **kwargs)
        day_counter += 1
        return result
    
    result = run_backtest(
        strategy=strategy_wrapper,
        risk_target=0.0,
        initial_weights=initial_weights,
        start_time="2022-01-03",
        max_steps=30,
        strategy_kwargs={'target_weights': target_weights},
        verbose=True
    )
    
    # Save and analyze results
    name = "Liquidation_20_Stocks_Dynamic Uniform"
    result_filename = f"liquidation_result_{name.replace(' ', '_')}.pickle"
    result.save(checkpoints_path() / result_filename)
    
    # Create dashboard report
    create_comprehensive_report(result, name)
    
    logger.info(f"Liquidation test complete! Results saved as '{name}'")
    return result


def test_transition_scenario():
    """
    Test transition scenario: equal weight to minimum variance portfolio.
    """
    # Load data
    prices, _, _, _ = load_data()
    asset_names = prices.columns
    
    # Create initial portfolio: equal weight
    initial_weights = create_equal_weight_target(asset_names, target_weight=1.0)
    
    # Create target: minimum variance portfolio
    target_weights = get_minimum_variance_portfolio(prices.loc[:'2022-01-01'], risk_target=0.10)
    
    # Run backtest
    logger.info("Testing transition scenario: equal weight → minimum variance")
    
    day_counter = 0
    def strategy_wrapper(inputs, **kwargs):
        nonlocal day_counter
        kwargs['days_remaining'] = 30 - day_counter
        result = dynamic_uniform_strategy(inputs, **kwargs)
        day_counter += 1
        return result
    
    result = run_backtest(
        strategy=strategy_wrapper,
        risk_target=0.0,
        initial_weights=initial_weights,
        start_time="2022-01-03",
        max_steps=30,
        strategy_kwargs={'target_weights': target_weights},
        verbose=True
    )
    
    # Save and analyze results
    name = "Transition_Equal_to_MinVar_Dynamic Uniform"
    result_filename = f"transition_result_{name.replace(' ', '_')}.pickle"
    result.save(checkpoints_path() / result_filename)
    
    # Create dashboard report
    create_comprehensive_report(result, name)
    
    logger.info(f"Transition test complete! Results saved as '{name}'")
    return result


def test_advanced_transition_scenario():
    """
    Test advanced transition scenario: minimum variance portfolios with different risk targets.
    """
    # Load data
    prices, _, _, _ = load_data()
    asset_names = prices.columns
    
    # Create minimum variance portfolios
    initial_weights, target_weights = create_minimum_variance_portfolios(asset_names)
    
    # Run backtest
    logger.info("Testing advanced transition: low risk → high risk minimum variance")
    
    day_counter = 0
    def strategy_wrapper(inputs, **kwargs):
        nonlocal day_counter
        kwargs['days_remaining'] = 30 - day_counter
        result = dynamic_uniform_strategy(inputs, **kwargs)
        day_counter += 1
        return result
    
    result = run_backtest(
        strategy=strategy_wrapper,
        risk_target=0.0,
        initial_weights=initial_weights,
        start_time="2022-01-03",
        max_steps=30,
        strategy_kwargs={'target_weights': target_weights},
        verbose=True
    )
    
    # Save and analyze results
    name = "Advanced_Transition_MinVar_Dynamic Uniform"
    result_filename = f"advanced_transition_result_{name.replace(' ', '_')}.pickle"
    result.save(checkpoints_path() / result_filename)
    
    # Create dashboard report
    create_comprehensive_report(result, name)
    
    logger.info(f"Advanced transition test complete! Results saved as '{name}'")
    return result


def test_liquidation_monte_carlo(n_simulations=100):
    """
    Test liquidation scenario with Monte Carlo simulation.
    """
    logger.info("--- Setting up Liquidation Monte Carlo Experiment ---")
    
    # Load data
    prices, _, _, _ = load_data()
    asset_names = prices.columns
    liquidation_period_days = 30
    initial_portfolio_value = 1_000_000
    
    # Create equal-weight initial portfolio
    initial_weights = pd.Series(np.ones(len(asset_names)) / len(asset_names), index=asset_names)
    target_weights = pd.Series(0.0, index=asset_names)
    
    # Select random start dates
    possible_start_dates = prices.index[1250:-liquidation_period_days]
    rng = np.random.default_rng(seed=42)
    start_dates = rng.choice(possible_start_dates, n_simulations, replace=False)
    logger.info(f"Testing on {n_simulations} random start dates...")
    
    # Run Monte Carlo simulation
    all_shortfalls = []
    
    for i, start_date in enumerate(start_dates):
        logger.debug(f"Running simulation {i+1}/{n_simulations} starting on {pd.Timestamp(start_date).date()}...")
        
        day_counter = 0
        def strategy_wrapper(inputs, **kwargs):
            nonlocal day_counter
            kwargs['days_remaining'] = liquidation_period_days - day_counter
            result = dynamic_uniform_strategy(inputs, **kwargs)
            day_counter += 1
            return result
        
        result = run_backtest(
            strategy=strategy_wrapper,
            risk_target=0.0,
            start_time=start_date,
            max_steps=liquidation_period_days,
            initial_weights=initial_weights,
            strategy_kwargs={'target_weights': target_weights},
            verbose=False,
        )
        
        shortfall = calculate_implementation_shortfall(result, initial_portfolio_value)
        all_shortfalls.append(shortfall)
    
    # Analyze results (in BPS)
    shortfall_bps = pd.Series(all_shortfalls) * 10000
    
    logger.info("\n--- Liquidation Cost Analysis ---")
    logger.info(f"Strategy: Dynamic Uniform")
    logger.info(f"Mean Implementation Shortfall: {shortfall_bps.mean():.2f} BPS")
    logger.info(f"Median Implementation Shortfall: {shortfall_bps.median():.2f} BPS")
    logger.info(f"Std Dev of Shortfall: {shortfall_bps.std():.2f} BPS")
    logger.info(f"Worst Case (95th percentile): {shortfall_bps.quantile(0.95):.2f} BPS")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.hist(shortfall_bps, bins=20, edgecolor='black')
    plt.title('Distribution of Implementation Shortfall (Costs in Basis Points)')
    plt.xlabel('Implementation Shortfall (BPS)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    
    figures_path().mkdir(exist_ok=True)
    plt.savefig(figures_path() / "liquidation_shortfall_distribution.pdf")
    plt.show()
    
    return shortfall_bps


def run_all_scenarios():
    """
    Run all test scenarios.
    """
    logger.info("=== Running All Test Scenarios ===")
    
    # Test 1: Liquidation
    logger.info("\n1. Testing Liquidation Scenario")
    test_liquidation_scenario()
    
    # Test 2: Transition
    logger.info("\n2. Testing Transition Scenario")
    test_transition_scenario()
    
    # Test 3: Advanced Transition
    logger.info("\n3. Testing Advanced Transition Scenario")
    test_advanced_transition_scenario()
    
    # Test 4: Monte Carlo Liquidation
    logger.info("\n4. Testing Monte Carlo Liquidation")
    test_liquidation_monte_carlo(n_simulations=50)  # Reduced for speed
    
    logger.info("\n=== All Scenarios Complete ===")


if __name__ == "__main__":
    # You can run individual scenarios or all of them
    # test_liquidation_scenario()
    # test_transition_scenario()
    # test_advanced_transition_scenario()
    # test_liquidation_monte_carlo()
    
    # Or run all scenarios
    run_all_scenarios() 