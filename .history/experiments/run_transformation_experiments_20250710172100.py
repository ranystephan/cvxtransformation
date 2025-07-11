# experiments/run_transformation_experiments.py

import numpy as np
import pandas as pd
from loguru import logger

# Import the modified backtest engine and result object
from backtest import run_backtest, BacktestResult

# Import all your new strategies
from transformation_strategies import (
    dynamic_uniform_strategy,
    front_loaded_strategy
)
# Import for saving/loading results
from utils import checkpoints_path, load_data

# MODIFICATION 1: Add target_weights as an argument
def generate_comparison_table(results: dict, target_weights: np.ndarray):
    """Creates a table comparing the performance of each strategy."""
    
    # Focus on metrics relevant to transformation execution
    df = pd.DataFrame(
        index=results.keys(),
        columns=["Turnover", "Drawdown", "Final Tracking Error"],
    )

    for name, result in results.items():
        # Target is now passed in directly
        final_weights = result.asset_weights.iloc[-1]
        error = np.linalg.norm(final_weights - target_weights)
        
        df.loc[name, "Turnover"] = f"{result.turnover:.2f}"
        df.loc[name, "Drawdown"] = f"{result.max_drawdown:.2%}"
        df.loc[name, "Final Tracking Error"] = f"{error:.4f}"

    logger.info("--- Transformation Strategy Comparison ---")
    print(df)
    
def main():
    """Runs all transformation experiments and generates a comparison table."""
    
    # --- 1. Define Experiment Parameters ---
    n_assets = load_data()[0].shape[1]
    transformation_period_days = 60
    
    # The task: Transform from 100% cash to a 60/40 style portfolio
    # MODIFICATION 2: Remove the 'global' keyword
    target_weights = (np.ones(n_assets) / n_assets) * 0.60
    
    # --- 2. Define Strategies to Test ---
    strategies_to_test = {
        "Dynamic Uniform": dynamic_uniform_strategy,
        "Front-Loaded 20%": front_loaded_strategy,
    }

    results = {}
    
    # --- 3. Run Backtest for Each Strategy ---
    for name, strategy_func in strategies_to_test.items():
        logger.info(f"--- Running backtest for: {name} ---")

        # The wrapper function is perfect as is.
        day_counter = 0
        def strategy_wrapper(inputs: dict, **kwargs):
            nonlocal day_counter
            kwargs['days_remaining'] = transformation_period_days - day_counter
            result_tuple = strategy_func(inputs, **kwargs)
            day_counter += 1
            return result_tuple
        
        day_counter = 0 # This reset is critical and you have it right.

        # Pass static parameters for the strategy via strategy_kwargs
        static_kwargs = {'target_weights': target_weights}
        
        # This makes it easy to add other static params if needed, e.g.:
        # if "Front-Loaded" in name:
        #     static_kwargs['trade_fraction'] = 0.20

        result = run_backtest(
            strategy=strategy_wrapper,
            risk_target=0.0,
            max_steps=transformation_period_days,
            strategy_kwargs=static_kwargs,
            verbose=True,
        )
        
        results[name] = result

    # --- 4. Analyze and Compare Results ---
    # MODIFICATION 3: Pass target_weights to the function
    generate_comparison_table(results, target_weights)

if __name__ == "__main__":
    main()