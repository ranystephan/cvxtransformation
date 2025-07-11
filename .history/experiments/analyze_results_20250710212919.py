# experiments/analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from backtest import BacktestResult
from utils import checkpoints_path, figures_path

def calculate_trade_vectors(result: BacktestResult) -> pd.DataFrame:
    """Calculate trade vectors (daily changes in quantities)"""
    trades = result.quantities.diff()
    prices = result.valuations.div(result.quantities, axis=0).fillna(0)
    trade_values = trades * prices
    return trade_values

def calculate_weight_changes(result: BacktestResult) -> pd.DataFrame:
    """Calculate daily changes in portfolio weights"""
    return result.asset_weights.diff()

def generate_comprehensive_dashboard(result: BacktestResult, result_name: str, 
                                   initial_weights: pd.Series = None, 
                                   target_weights: pd.Series = None):
    """
    Generate a comprehensive dashboard with enhanced statistics and visualizations.
    """
    logger.info(f"--- Generating Comprehensive Dashboard for: {result_name} ---")
    
    # Create a directory to save the figures
    report_path = figures_path() / result_name
    report_path.mkdir(exist_ok=True)

    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # --- 1. Enhanced Performance Metrics ---
    logger.info("Calculating enhanced performance metrics...")
    
    # Basic metrics
    final_value = result.portfolio_value.iloc[-1]
    total_return = (final_value / result.portfolio_value.iloc[0]) - 1
    annualized_return = result.mean_return
    annualized_vol = result.volatility
    sharpe_ratio = result.sharpe
    max_dd = result.max_drawdown
    annualized_turnover = result.turnover
    avg_leverage = result.asset_weights.abs().sum(axis=1).mean()
    max_leverage = result.max_leverage
    
    # Additional metrics
    returns = result.portfolio_returns
    positive_days = (returns > 0).sum()
    negative_days = (returns < 0).sum()
    win_rate = positive_days / len(returns)
    
    # Risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Create comprehensive metrics table
    metrics_data = {
        "Portfolio Value": f"${final_value:,.2f}",
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_vol:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "VaR (95%)": f"{var_95:.2%}",
        "CVaR (95%)": f"{cvar_95:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Annualized Turnover": f"{annualized_turnover:.2f}",
        "Average Leverage": f"{avg_leverage:.2f}",
        "Max Leverage": f"{max_leverage:.2f}",
        "Trading Days": f"{len(result.history)}",
        "Positive Days": f"{positive_days}",
        "Negative Days": f"{negative_days}"
    }
    
    metrics_df = pd.Series(metrics_data, name=result_name)
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*60)
    logger.info("\n" + metrics_df.to_string())

    # --- 2. Portfolio Weights Analysis ---
    logger.info("\nAnalyzing portfolio weights...")
    
    # Get initial, target, and final weights
    initial_weights_actual = result.asset_weights.iloc[0]
    final_weights_actual = result.asset_weights.iloc[-1]
    
    weights_comparison = pd.DataFrame({
        'Initial': initial_weights_actual,
        'Final': final_weights_actual
    })
    
    if initial_weights is not None:
        weights_comparison['Target_Initial'] = initial_weights
    if target_weights is not None:
        weights_comparison['Target_Final'] = target_weights
    
    # Calculate weight changes
    weight_changes = final_weights_actual - initial_weights_actual
    weights_comparison['Change'] = weight_changes
    
    # Sort by absolute change
    weights_comparison = weights_comparison.sort_values('Change', key=abs, ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info("PORTFOLIO WEIGHTS ANALYSIS")
    logger.info("="*60)
    logger.info(f"\nTop 10 weight changes:")
    logger.info(weights_comparison.head(10).to_string())

    # --- 3. Trade Analysis ---
    logger.info("\nAnalyzing trade vectors...")
    
    trade_vectors = calculate_trade_vectors(result)
    weight_changes_daily = calculate_weight_changes(result)
    
    # Trade statistics
    total_trades = trade_vectors.abs().sum().sum()
    avg_daily_trades = trade_vectors.abs().sum(axis=1).mean()
    max_daily_trades = trade_vectors.abs().sum(axis=1).max()
    
    # Most traded assets
    asset_trade_volumes = trade_vectors.abs().sum().sort_values(ascending=False)
    
    logger.info(f"\nTrade Statistics:")
    logger.info(f"Total Trade Volume: ${total_trades:,.2f}")
    logger.info(f"Average Daily Trade Volume: ${avg_daily_trades:,.2f}")
    logger.info(f"Max Daily Trade Volume: ${max_daily_trades:,.2f}")
    logger.info(f"\nTop 10 Most Traded Assets:")
    logger.info(asset_trade_volumes.head(10).to_string())

    # --- 4. Create Comprehensive Dashboard Plots ---
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 4.1 Portfolio Value (Equity Curve)
    plt.subplot(4, 3, 1)
    result.portfolio_value.plot(grid=True, linewidth=2, color='blue')
    plt.title(f"Portfolio Value Over Time\n{result_name}", fontsize=12, fontweight='bold')
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    # 4.2 Portfolio Returns Distribution
    plt.subplot(4, 3, 2)
    returns.hist(bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2%}')
    plt.title("Distribution of Daily Returns", fontsize=12, fontweight='bold')
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend()
    
    # 4.3 Cumulative Returns
    plt.subplot(4, 3, 3)
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns.plot(linewidth=2, color='purple')
    plt.title("Cumulative Returns", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    # 4.4 Portfolio Weights Over Time (Fixed version)
    plt.subplot(4, 3, 4)
    weights_with_cash = result.asset_weights.copy()
    weights_with_cash["_CASH"] = result.cash_weight
    
    # Handle negative cash weights for stacked area plot
    cash_positive = weights_with_cash["_CASH"].clip(lower=0)
    cash_negative = weights_with_cash["_CASH"].clip(upper=0)
    
    weights_to_plot = weights_with_cash.drop(columns=["_CASH"]).copy()
    weights_to_plot["_CASH_POSITIVE"] = cash_positive
    weights_to_plot["_CASH_NEGATIVE"] = cash_negative
    
    # Select top assets for clarity
    if len(weights_to_plot.columns) > 10:
        top_assets = weights_to_plot.iloc[-1].abs().sort_values(ascending=False).head(9).index
        other_weights = weights_to_plot.drop(columns=top_assets).sum(axis=1)
        weights_to_plot = weights_to_plot[top_assets].copy()
        weights_to_plot["_OTHER"] = other_weights
    
    weights_to_plot.plot.area(stacked=True, linewidth=0, alpha=0.8)
    plt.title("Portfolio Composition Over Time", fontsize=12, fontweight='bold')
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4.5 Daily Turnover
    plt.subplot(4, 3, 5)
    result.daily_turnover.plot(linewidth=1, color='orange')
    plt.title("Daily Turnover", fontsize=12, fontweight='bold')
    plt.ylabel("Turnover Rate")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    # 4.6 Leverage Over Time
    plt.subplot(4, 3, 6)
    leverage = result.asset_weights.abs().sum(axis=1)
    leverage.plot(linewidth=2, color='red')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Leverage')
    plt.title("Portfolio Leverage Over Time", fontsize=12, fontweight='bold')
    plt.ylabel("Leverage")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4.7 Weight Changes (Initial vs Final)
    plt.subplot(4, 3, 7)
    top_changes = weights_comparison.head(15)
    plt.barh(range(len(top_changes)), top_changes['Change'], 
             color=['green' if x > 0 else 'red' for x in top_changes['Change']])
    plt.yticks(range(len(top_changes)), top_changes.index)
    plt.title("Top 15 Weight Changes\n(Initial to Final)", fontsize=12, fontweight='bold')
    plt.xlabel("Weight Change")
    plt.grid(True, alpha=0.3)
    
    # 4.8 Trade Volume by Asset
    plt.subplot(4, 3, 8)
    top_traded = asset_trade_volumes.head(15)
    plt.barh(range(len(top_traded)), top_traded.values, color='skyblue')
    plt.yticks(range(len(top_traded)), top_traded.index)
    plt.title("Total Trade Volume by Asset", fontsize=12, fontweight='bold')
    plt.xlabel("Trade Volume ($)")
    plt.grid(True, alpha=0.3)
    
    # 4.9 Daily Trade Volume
    plt.subplot(4, 3, 9)
    daily_trade_volume = trade_vectors.abs().sum(axis=1)
    daily_trade_volume.plot(linewidth=1, color='brown')
    plt.title("Daily Trade Volume", fontsize=12, fontweight='bold')
    plt.ylabel("Trade Volume ($)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    # 4.10 Drawdown Analysis
    plt.subplot(4, 3, 10)
    running_max = result.portfolio_value.cummax()
    drawdown = (result.portfolio_value - running_max) / running_max
    drawdown.plot(linewidth=2, color='red', fillstyle='full')
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    plt.title("Portfolio Drawdown", fontsize=12, fontweight='bold')
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    # 4.11 Rolling Sharpe Ratio
    plt.subplot(4, 3, 11)
    window = min(60, len(returns) // 4)  # 60-day window or 1/4 of data
    if window > 10:
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe.plot(linewidth=2, color='green')
        plt.title(f"Rolling Sharpe Ratio\n({window}-day window)", fontsize=12, fontweight='bold')
        plt.ylabel("Sharpe Ratio")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Insufficient data\nfor rolling Sharpe", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Rolling Sharpe Ratio", fontsize=12, fontweight='bold')
    
    # 4.12 Cash Position Over Time
    plt.subplot(4, 3, 12)
    result.cash_weight.plot(linewidth=2, color='gray')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title("Cash Position Over Time", fontsize=12, fontweight='bold')
    plt.ylabel("Cash Weight")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_path / "comprehensive_dashboard.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved comprehensive dashboard plot.")

    # --- 5. Save Detailed Data Tables ---
    logger.info("Saving detailed data tables...")
    
    # Save weights comparison
    weights_comparison.to_csv(report_path / "weights_comparison.csv")
    
    # Save trade vectors
    trade_vectors.to_csv(report_path / "trade_vectors.csv")
    
    # Save daily metrics
    daily_metrics = pd.DataFrame({
        'Portfolio_Value': result.portfolio_value,
        'Cash_Weight': result.cash_weight,
        'Leverage': result.asset_weights.abs().sum(axis=1),
        'Daily_Turnover': result.daily_turnover,
        'Daily_Return': result.portfolio_returns
    })
    daily_metrics.to_csv(report_path / "daily_metrics.csv")
    
    # Save asset trade volumes
    asset_trade_volumes.to_csv(report_path / "asset_trade_volumes.csv")
    
    # --- 6. Generate Summary Report ---
    summary_report = f"""
COMPREHENSIVE PORTFOLIO TRANSITION ANALYSIS
==========================================
Strategy: {result_name}
Period: {result.history[0].strftime('%Y-%m-%d')} to {result.history[-1].strftime('%Y-%m-%d')}
Trading Days: {len(result.history)}

PERFORMANCE SUMMARY:
- Final Portfolio Value: ${final_value:,.2f}
- Total Return: {total_return:.2%}
- Annualized Return: {annualized_return:.2%}
- Annualized Volatility: {annualized_vol:.2%}
- Sharpe Ratio: {sharpe_ratio:.3f}
- Maximum Drawdown: {max_dd:.2%}
- Win Rate: {win_rate:.2%}

TRADING ACTIVITY:
- Annualized Turnover: {annualized_turnover:.2f}
- Total Trade Volume: ${total_trades:,.2f}
- Average Daily Trade Volume: ${avg_daily_trades:,.2f}
- Maximum Daily Trade Volume: ${max_daily_trades:,.2f}

RISK METRICS:
- VaR (95%): {var_95:.2%}
- CVaR (95%): {cvar_95:.2%}
- Average Leverage: {avg_leverage:.2f}
- Maximum Leverage: {max_leverage:.2f}

Files saved in: {report_path}
"""
    
    with open(report_path / "summary_report.txt", 'w') as f:
        f.write(summary_report)
    
    logger.info(summary_report)
    logger.info(f"--- Comprehensive Dashboard Complete ---")
    logger.info(f"All files saved in: {report_path}")

if __name__ == "__main__":
    # --- How to use this analysis script ---

    # 1. Define the name of the result file you saved earlier
    result_filename = "transition_result_Dynamic_Uniform.pickle"
    result_name = "Dynamic_Uniform_Strategy"

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

    # 3. Try to load initial and target weights if available
    initial_weights = None
    target_weights = None
    
    # You can manually specify these if you have them from your experiment
    # For example, from run_advanced_transitions.py:
    # initial_weights = pd.Series(...)  # Your initial weights
    # target_weights = pd.Series(...)   # Your target weights

    # 4. Generate the comprehensive dashboard
    generate_comprehensive_dashboard(result, result_name, initial_weights, target_weights)