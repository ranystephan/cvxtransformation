# experiments/run_advanced_transitions.py

import numpy as np
import pandas as pd
from loguru import logger

from backtest import run_backtest, load_data
from transformation_strategies import dynamic_uniform_strategy
from portfolio_construction import get_minimum_variance_portfolio
from utils import estimate_transition_period, checkpoints_path
from loguru import logger

def main():
    logger.info("--- Setting up Advanced Portfolio Transition Experiment ---")

    # --- 1. Load and Prepare Data ---
    prices, spread, rf, volumes = load_data() # Using the screened, liquid universe
    
    # --- 2. Define Asset Groups ---
    # Example: Split universe into two halves
    all_assets = prices.columns
    group_A_assets = all_assets[:len(all_assets)//2]
    group_B_assets = all_assets[len(all_assets)//2:]

    # --- 3. Construct the Initial and Target Portfolios ---
    # For this example, we need historical data to calculate covariance
    # Let's use data up to a certain point to construct the portfolios
    construction_date = "2022-01-01"
    historical_prices_A = prices.loc[:construction_date][group_A_assets]
    historical_prices_B = prices.loc[:construction_date][group_B_assets]

    # Portfolio A: Low risk (5% annualized vol target)
    initial_weights = get_minimum_variance_portfolio(historical_prices_A, risk_target=0.05)
    
    # Portfolio B: Higher risk (15% annualized vol target)
    target_weights = get_minimum_variance_portfolio(historical_prices_B, risk_target=0.15)
    
    # Combine the weights into a single series for the backtester, filling missing with 0
    initial_weights = initial_weights.reindex(all_assets, fill_value=0)
    target_weights = target_weights.reindex(all_assets, fill_value=0)

    # --- 4. Estimate Transition Period ---
    # Use data up to the transition start date for the estimation
    transition_period_days = 30

    transition_start_date = "2022-01-03" # The first trading day after construction
    # --- 5. Run the Backtest Simulation ---
    logger.info(f"Starting transition from Portfolio A to B over {transition_period_days} days.")
    
    day_counter = 0
    def strategy_wrapper(inputs, **kwargs):
        nonlocal day_counter
        kwargs['days_remaining'] = transition_period_days - day_counter
        result = dynamic_uniform_strategy(inputs, **kwargs)
        day_counter += 1
        return result

    result = run_backtest(
        strategy=strategy_wrapper,
        risk_target=0.0, # Unused
        initial_weights=initial_weights,
        start_time=transition_start_date,
        max_steps=transition_period_days,
        strategy_kwargs={'target_weights': target_weights},
        verbose=True
    )

    # Define results dictionary and name variable
    results = {}
    name = "Dynamic Uniform"

    results[name] = result

    # SAVE THE RESULT
    result_filename = f"transition_result_{name.replace(' ', '_')}.pickle"
    result.save(checkpoints_path() / result_filename)

    logger.info("--- Advanced Transition Complete ---")
    logger.info(f"Final portfolio value: {result.portfolio_value.iloc[-1]:,.2f}")
    # Add other analysis here...

if __name__ == "__main__":
    main()