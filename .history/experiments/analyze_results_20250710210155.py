# experiments/analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from loguru import logger

from backtest import BacktestResult
from utils import checkpoints_path, figures_path

def generate_full_report(result: BacktestResult, result_name: str):
    """
    Takes a BacktestResult object and generates a full suite of
    plots and summary statistics.
    """
    logger.info(f"--- Generating Full Report for: {result_name} ---")
    
    # Create a directory to save the figures
    report_path = figures_path() / result_name
    report_path.mkdir(exist_ok=True)

    # --- 1. Key Performance Indicators (KPIs) Table ---
    # We can expand the table from before
    kpi_data = {
        "Final Portfolio Value": f"${result.portfolio_value.iloc[-1]:,.2f}",
        "Mean Return (Annualized)": f"{result.mean_return:.2%}",
        "Volatility (Annualized)": f"{result.volatility:.2%}",
        "Sharpe Ratio": f"{result.sharpe:.2f}",
        "Max Drawdown": f"{result.max_drawdown:.2%}",
        "Turnover (Annualized)": f"{result.turnover:.2f}",
        "Average Leverage": f"{result.asset_weights.abs().sum(axis=1).mean():.2f}",
    }
    kpis = pd.Series(kpi_data, name=result_name)
    logger.info("\n" + kpis.to_string())

    # --- 2. Plot: Portfolio Value (Equity Curve) ---
    plt.figure(figsize=(12, 6))
    result.portfolio_value.plot(grid=True)
    plt.title(f"Portfolio Value Over Time: {result_name}")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.savefig(report_path / "equity_curve.pdf")
    plt.close()
    logger.info(f"Saved equity curve plot.")

    # --- 3. Plot: Asset Weights Over Time ---
    # This is crucial for visualizing the transition
    plt.figure(figsize=(12, 8))
    
    # Combine asset weights with cash weight for a complete picture
    weights_with_cash = result.asset_weights.copy()
    weights_with_cash["_CASH"] = result.cash_weight
    
    # For clarity, if there are too many assets, we can group small ones
    if len(result.asset_weights.columns) > 15:
        # Get top 14 assets by final weight, group rest into "Other"
        top_assets = result.asset_weights.iloc[-1].abs().sort_values(ascending=False).head(14).index
        other_weights = weights_with_cash.drop(columns=top_assets).sum(axis=1)
        weights_to_plot = weights_with_cash[top_assets].copy()
        weights_to_plot["_OTHER_ASSETS"] = other_weights
        weights_to_plot["_CASH"] = weights_with_cash["_CASH"] # Re-add cash
    else:
        weights_to_plot = weights_with_cash.copy()

    # Handle negative cash weights for stacked area plot
    # Split cash into positive and negative components
    cash_positive = weights_to_plot["_CASH"].clip(lower=0)
    cash_negative = weights_to_plot["_CASH"].clip(upper=0)
    
    # Remove original cash column and add split components
    weights_to_plot = weights_to_plot.drop(columns=["_CASH"])
    weights_to_plot["_CASH_POSITIVE"] = cash_positive
    weights_to_plot["_CASH_NEGATIVE"] = cash_negative
    
    # Create the stacked area plot
    weights_to_plot.plot.area(stacked=True, linewidth=0)
    
    plt.title(f"Portfolio Composition Over Time: {result_name}")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust layout to make room for legend
    plt.savefig(report_path / "weights_over_time.pdf")
    plt.close()
    logger.info(f"Saved weights transition plot.")
    
    # --- 4. Plot: Daily Turnover ---
    plt.figure(figsize=(12, 6))
    result.daily_turnover.plot(grid=True, style='-')
    plt.title(f"Daily Turnover: {result_name}")
    plt.ylabel("Turnover Rate")
    plt.xlabel("Date")
    plt.savefig(report_path / "daily_turnover.pdf")
    plt.close()
    logger.info(f"Saved daily turnover plot.")

    # --- 5. Plot: Distribution of Daily Returns ---
    plt.figure(figsize=(10, 6))
    result.portfolio_returns.hist(bins=50)
    plt.title(f"Distribution of Daily Returns: {result_name}")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.savefig(report_path / "returns_distribution.pdf")
    plt.close()
    logger.info(f"Saved returns distribution plot.")
    
    logger.info(f"--- Report generation complete. Files saved in: {report_path} ---")

# ... existing code ...

if __name__ == "__main__":
    # --- How to use this analysis script ---

    # 1. Define the name of the result file you saved earlier
    # This should match the filename from your runner script
    result_filename = "transition_result_Dynamic_Uniform.pickle"
    result_name = "Dynamic_Uniform_Strategy" # A nice name for titles and labels

    # 2. Load the BacktestResult object
    try:
        result_path = checkpoints_path() / result_filename
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Successfully loaded result from: {result_path}")
    except FileNotFoundError:
        logger.error(f"Could not find result file: {result_path}")
        logger.error("Please run the experiment script first to generate the result file.")
        exit()

    # 3. Generate the full report
    generate_full_report(result, result_name)