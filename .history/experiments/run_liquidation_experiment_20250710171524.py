# experiments/run_liquidation_experiment.py

import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

from backtest import run_backtest
from transformation_strategies import dynamic_uniform_strategy
from utils import data_path, load_data

def calculate_implementation_shortfall(result, initial_portfolio_weights, start_price_series):
    """Calculates the total cost of the liquidation."""
    
    # Paper value of the portfolio at the start of the liquidation
    # We assume the total initial portfolio value is the starting cash of the backtest (1e6)
    initial_portfolio_value = 1_000_000
    initial_paper_value = (initial_portfolio_weights * initial_portfolio_value) @ start_price_series

    # The actual cash we have at the end of the liquidation
    final_cash = result.cash.iloc[-1]
    
    # Shortfall is the difference. A positive value means it cost us money.
    # We normalize by the initial value to get a percentage cost.
    shortfall = initial_paper_value - final_cash
    return shortfall / initial_portfolio_value

def main():
    """
    Runs a Monte Carlo simulation for a portfolio liquidation strategy.
    """
    logger.info("--- Setting up Liquidation Experiment ---")
    
    # --- 1. Define Experiment Parameters ---
    n_assets = load_data()[0].shape[1]
    liquidation_period_days = 30  # Liquidate over 30 trading days

    # The task: Liquidate a random initial portfolio
    # For reproducibility, we'll generate one random portfolio and use it for all tests
    rng = np.random.default_rng(seed=42)
    random_weights = rng.random(n_assets)
    initial_portfolio_weights = random_weights / random_weights.sum()
    
    # The target is always all zeros (full liquidation)
    target_weights = np.zeros(n_assets)
    
    # --- 2. Select Random Start Dates ---
    prices, _, _, _ = load_data()
    # We need to leave enough room for the liquidation period at the end of the dataset
    possible_start_dates = prices.index[1250:-liquidation_period_days]
    
    n_simulations = 100
    start_dates = rng.choice(possible_start_dates, n_simulations, replace=False)
    logger.info(f"Testing on {n_simulations} random start dates...")

    # --- 3. Run the Monte Carlo Simulation ---
    all_shortfalls = []
    
    for i, start_date in enumerate(start_dates):
        logger.info(f"Running simulation {i+1}/{n_simulations} starting on {start_date.date()}...")

        # Create the strategy wrapper for this run
        day_counter = 0
        def strategy_wrapper(inputs: dict, **kwargs):
            nonlocal day_counter
            kwargs['days_remaining'] = liquidation_period_days - day_counter
            result_tuple = dynamic_uniform_strategy(inputs, **kwargs)
            day_counter += 1
            return result_tuple
        
        day_counter = 0

        # Run the backtest for this specific period
        result = run_backtest(
            strategy=strategy_wrapper,
            risk_target=0.0, # Unused
            start_time=start_date, # Use our new parameter
            max_steps=liquidation_period_days, # Use our new parameter
            strategy_kwargs={'target_weights': target_weights},
            verbose=False, # Turn off daily logging for cleaner output
        )
        
        # We need the prices on the start date to value the initial portfolio
        start_prices = prices.loc[start_date]
        shortfall = calculate_implementation_shortfall(result, initial_portfolio_weights, start_prices)
        all_shortfalls.append(shortfall)

    # --- 4. Analyze and Report Results ---
    shortfall_series = pd.Series(all_shortfalls)
    
    logger.info("\n--- Liquidation Cost Analysis ---")
    logger.info(f"Strategy: Dynamic Uniform")
    logger.info(f"Mean Implementation Shortfall: {shortfall_series.mean():.4%} BPS")
    logger.info(f"Median Implementation Shortfall: {shortfall_series.median():.4%} BPS")
    logger.info(f"Std Dev of Shortfall: {shortfall_series.std():.4%} BPS")
    logger.info(f"Worst Case (95th percentile): {shortfall_series.quantile(0.95):.4%} BPS")
    
    # Create a histogram of the costs
    plt.figure(figsize=(10, 6))
    plt.hist(shortfall_series * 10000, bins=20, edgecolor='black')
    plt.title('Distribution of Implementation Shortfall (Costs in Basis Points)')
    plt.xlabel('Implementation Shortfall (BPS)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    
    # Save the figure
    # from utils import figures_path
    # figures_path().mkdir(exist_ok=True)
    # plt.savefig(figures_path() / "liquidation_shortfall_distribution.pdf")
    
    plt.show()

if __name__ == "__main__":
    main()