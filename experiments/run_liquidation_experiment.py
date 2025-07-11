# experiments/run_liquidation_experiment.py (Complete, Corrected Code)

import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

from backtest import run_backtest, load_data
from transformation_strategies import dynamic_uniform_strategy
from utils import figures_path

def calculate_implementation_shortfall(result, initial_portfolio_value):
    """Calculates the normalized cost of the liquidation."""
    
    # The actual cash we have at the end of the liquidation
    final_cash = result.portfolio_value.iloc[-1]
    
    # Shortfall is the difference. A positive value means it cost us money.
    shortfall_dollars = initial_portfolio_value - final_cash
    
    # Return the shortfall as a fraction of the initial value
    return shortfall_dollars / initial_portfolio_value

def main():
    """Runs a Monte Carlo simulation for a portfolio liquidation strategy."""
    logger.info("--- Setting up Liquidation Experiment ---")
    
    # --- 1. Define Experiment Parameters ---
    prices, _, _, _ = load_data()
    n_assets = prices.shape[1]
    liquidation_period_days = 30
    initial_portfolio_value = 1_000_000 # Matches the backtester's starting value

    # Create an equal-weight initial portfolio as a pandas Series
    initial_weights_values = np.ones(n_assets) / n_assets
    initial_weights = pd.Series(initial_weights_values, index=prices.columns)
    
    target_weights = np.zeros(n_assets)
    
    # --- 2. Select Random Start Dates ---
    possible_start_dates = prices.index[1250:-liquidation_period_days]
    n_simulations = 100
    rng = np.random.default_rng(seed=42)
    start_dates = rng.choice(possible_start_dates, n_simulations, replace=False)
    logger.info(f"Testing on {n_simulations} random start dates...")

    # --- 3. Run the Monte Carlo Simulation ---
    all_shortfalls = []
    
    for i, start_date in enumerate(start_dates):
        logger.debug(f"Running simulation {i+1}/{n_simulations} starting on {pd.Timestamp(start_date).date()}...")

        day_counter = 0
        def strategy_wrapper(inputs: dict, **kwargs):
            nonlocal day_counter
            kwargs['days_remaining'] = liquidation_period_days - day_counter
            result_tuple = dynamic_uniform_strategy(inputs, **kwargs)
            day_counter += 1
            return result_tuple
        
        day_counter = 0

        result = run_backtest(
            strategy=strategy_wrapper,
            risk_target=0.0,
            start_time=start_date,
            max_steps=liquidation_period_days,
            initial_weights=initial_weights, # <-- USE THE NEW PARAMETER
            strategy_kwargs={'target_weights': target_weights},
            verbose=False,
        )
        
        shortfall = calculate_implementation_shortfall(result, initial_portfolio_value)
        all_shortfalls.append(shortfall)

    # --- 4. Analyze and Report Results (in BPS) ---
    # 1 BPS = 0.0001. Multiply the decimal shortfall by 10,000.
    shortfall_bps = pd.Series(all_shortfalls) * 10000
    
    logger.info("\n--- Liquidation Cost Analysis ---")
    logger.info(f"Strategy: Dynamic Uniform")
    logger.info(f"Mean Implementation Shortfall: {shortfall_bps.mean():.2f} BPS")
    logger.info(f"Median Implementation Shortfall: {shortfall_bps.median():.2f} BPS")
    logger.info(f"Std Dev of Shortfall: {shortfall_bps.std():.2f} BPS")
    logger.info(f"Worst Case (95th percentile): {shortfall_bps.quantile(0.95):.2f} BPS")
    
    plt.figure(figsize=(10, 6))
    plt.hist(shortfall_bps, bins=20, edgecolor='black')
    plt.title('Distribution of Implementation Shortfall (Costs in Basis Points)')
    plt.xlabel('Implementation Shortfall (BPS)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', alpha=0.5)
    
    figures_path().mkdir(exist_ok=True)
    plt.savefig(figures_path() / "liquidation_shortfall_distribution.pdf")
    plt.show()

if __name__ == "__main__":
    main()