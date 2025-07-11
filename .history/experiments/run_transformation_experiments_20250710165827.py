# experiments/run_transformation_experiments.py (Complete Code)

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

def generate_comparison_table(results: dict):
    """Creates a table comparing the performance of each strategy."""
    
    # Focus on metrics relevant to transformation execution
    df = pd.DataFrame(
        index=results.keys(),
        columns=["Turnover", "Drawdown", "Final Tracking Error"],
    )

    for name, result in results.items():
        # Target is defined outside but needed here for error calculation
        final_weights = result.asset_weights.iloc[-1]
        error = np.linalg.norm(final_weights - target_weights)
        
        df.loc[name, "Turnover"] = result.turnover
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
    # (60% equal weight stocks, 40% cash)
    global target_weights 
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

        # The backtester needs to pass `days_remaining` to the strategy.
        # We can create a simple wrapper (a "closure") to handle this.
        day_counter = 0
        def strategy_wrapper(inputs: OptimizationInput, **kwargs):
            nonlocal day_counter
            # This is where we calculate the dynamic argument
            kwargs['days_remaining'] = transformation_period_days - day_counter
            
            # Call the actual strategy with the complete set of arguments
            result_tuple = strategy_func(inputs, **kwargs)
            
            day_counter += 1
            return result_tuple
        
        # Reset counter for each run
        day_counter = 0

        # Call the modified backtester
        result = run_backtest(
            strategy=strategy_wrapper,
            risk_target=0.0, # risk_target is unused, pass a dummy value
            max_steps=transformation_period_days, # IMPORTANT: Stop after the period
            strategy_kwargs={'target_weights': target_weights}, # Pass static params
            verbose=True,
        )
        
        results[name] = result
        # Optional: Save each result
        # result.save(checkpoints_path() / f"transform_{name}.pickle")

    # --- 4. Analyze and Compare Results ---
    generate_comparison_table(results)

if __name__ == "__main__":
    main()