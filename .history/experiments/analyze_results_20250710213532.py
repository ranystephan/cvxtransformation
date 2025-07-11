# experiments/analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from loguru import logger
from pathlib import Path

from backtest import BacktestResult
from utils import checkpoints_path, figures_path

def generate_comprehensive_report(result: BacktestResult, result_name: str, 
                                target_weights: np.ndarray | None = None,
                                initial_weights: np.ndarray | None = None):
    """
    Generates a comprehensive analysis report including:
    - Initial, target, and final weights comparison
    - Daily trade vectors
    - Complete performance dashboard
    - All relevant statistics
    """
    logger.info(f"--- Generating Comprehensive Report for: {result_name} ---")
    
    # Create directory for the report
    report_path = figures_path() / result_name
    report_path.mkdir(exist_ok=True)

    # --- 1. COMPREHENSIVE STATISTICS DASHBOARD ---
    generate_statistics_dashboard(result, result_name, report_path, target_weights, initial_weights)
    
    # --- 2. WEIGHTS ANALYSIS ---
    generate_weights_analysis(result, result_name, report_path, target_weights, initial_weights)
    
    # --- 3. TRADE ANALYSIS ---
    generate_trade_analysis(result, result_name, report_path)
    
    # --- 4. PERFORMANCE PLOTS ---
    generate_performance_plots(result, result_name, report_path)
    
    # --- 5. RISK ANALYSIS ---
    generate_risk_analysis(result, result_name, report_path)
    
    logger.info(f"--- Comprehensive report complete. Files saved in: {report_path} ---")

def generate_statistics_dashboard(result: BacktestResult, result_name: str, 
                                report_path: Path, target_weights: np.ndarray | None = None,
                                initial_weights: np.ndarray | None = None):
    """Generate comprehensive statistics dashboard"""
    
    # Calculate additional metrics
    final_weights = result.asset_weights.iloc[-1].values if len(result.asset_weights) > 0 else np.zeros(result.quantities.shape[1])
    
    # Tracking error if target weights provided
    tracking_error = None
    if target_weights is not None:
        tracking_error = np.linalg.norm(final_weights - target_weights)
    
    # Initial weights tracking error
    initial_tracking_error = None
    if initial_weights is not None:
        initial_tracking_error = np.linalg.norm(final_weights - initial_weights)
    
    # Enhanced statistics
    stats_data = {
        "Portfolio Metrics": {
            "Final Portfolio Value": f"${result.portfolio_value.iloc[-1]:,.2f}",
            "Total Return": f"{(result.portfolio_value.iloc[-1] / result.portfolio_value.iloc[0] - 1):.2%}",
            "Mean Return (Annualized)": f"{result.mean_return:.2%}",
            "Volatility (Annualized)": f"{result.volatility:.2%}",
            "Sharpe Ratio": f"{result.sharpe:.2f}",
            "Max Drawdown": f"{result.max_drawdown:.2%}",
            "Calmar Ratio": f"{(result.mean_return / abs(result.max_drawdown)):.2f}" if result.max_drawdown != 0 else "N/A",
        },
        "Trading Metrics": {
            "Turnover (Annualized)": f"{result.turnover:.2f}",
            "Average Daily Turnover": f"{result.daily_turnover.mean():.4f}",
            "Max Daily Turnover": f"{result.daily_turnover.max():.4f}",
            "Total Number of Trades": f"{len(result.history)}",
        },
        "Risk Metrics": {
            "Average Leverage": f"{result.asset_weights.abs().sum(axis=1).mean():.2f}",
            "Max Leverage": f"{result.max_leverage:.2f}",
            "Average Cash Weight": f"{result.cash_weight.mean():.2%}",
            "Min Cash Weight": f"{result.cash_weight.min():.2%}",
            "Max Cash Weight": f"{result.cash_weight.max():.2%}",
        }
    }
    
    # Add tracking errors if available
    if tracking_error is not None:
        stats_data["Transformation Metrics"] = {
            "Final Tracking Error": f"{tracking_error:.4f}",
            "Tracking Error (vs Initial)": f"{initial_tracking_error:.4f}" if initial_tracking_error is not None else "N/A",
        }
    
    # Create comprehensive statistics table
    all_stats = {}
    for category, metrics in stats_data.items():
        for metric, value in metrics.items():
            all_stats[f"{category} - {metric}"] = value
    
    stats_df = pd.Series(all_stats, name=result_name)
    
    # Save statistics to file
    stats_file = report_path / "comprehensive_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"COMPREHENSIVE STATISTICS DASHBOARD\n")
        f.write(f"Strategy: {result_name}\n")
        f.write(f"Period: {result.history[0].strftime('%Y-%m-%d')} to {result.history[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Total Days: {len(result.history)}\n\n")
        
        for category, metrics in stats_data.items():
            f.write(f"{category}:\n")
            f.write("-" * len(category) + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")
    
    logger.info(f"Saved comprehensive statistics to: {stats_file}")
    logger.info("\n" + stats_df.to_string())

def generate_weights_analysis(result: BacktestResult, result_name: str, 
                            report_path: Path, target_weights: np.ndarray | None = None,
                            initial_weights: np.ndarray | None = None):
    """Generate comprehensive weights analysis"""
    
    # Get asset names
    asset_names = result.asset_weights.columns.tolist() if len(result.asset_weights.columns) > 0 else [f"Asset_{i}" for i in range(result.quantities.shape[1])]
    
    # Create weights comparison DataFrame
    weights_comparison = pd.DataFrame(index=pd.Index(asset_names))
    
    # Initial weights
    if initial_weights is not None:
        weights_comparison['Initial'] = initial_weights
    else:
        weights_comparison['Initial'] = result.asset_weights.iloc[0].values if len(result.asset_weights) > 0 else np.zeros(len(asset_names))
    
    # Target weights
    if target_weights is not None:
        weights_comparison['Target'] = target_weights
    else:
        weights_comparison['Target'] = np.nan
    
    # Final weights
    weights_comparison['Final'] = result.asset_weights.iloc[-1].values if len(result.asset_weights) > 0 else np.zeros(len(asset_names))
    
    # Calculate differences
    if target_weights is not None:
        weights_comparison['Final_vs_Target'] = weights_comparison['Final'] - weights_comparison['Target']
    weights_comparison['Final_vs_Initial'] = weights_comparison['Final'] - weights_comparison['Initial']
    
    # Save weights comparison
    weights_file = report_path / "weights_comparison.csv"
    weights_comparison.to_csv(weights_file)
    logger.info(f"Saved weights comparison to: {weights_file}")
    
    # Create weights comparison plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Weights comparison bar chart
    plt.subplot(2, 2, 1)
    x = np.arange(len(asset_names))
    width = 0.25
    
    if initial_weights is not None:
        plt.bar(x - width, weights_comparison['Initial'], width, label='Initial', alpha=0.7)
    if target_weights is not None:
        plt.bar(x, weights_comparison['Target'], width, label='Target', alpha=0.7)
    plt.bar(x + width, weights_comparison['Final'], width, label='Final', alpha=0.7)
    
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Weights Comparison: Initial vs Target vs Final')
    plt.legend()
    plt.xticks(x, asset_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Weights evolution over time
    plt.subplot(2, 2, 2)
    if len(result.asset_weights) > 0:
        # Show top 10 assets by final weight
        top_assets = result.asset_weights.iloc[-1].abs().sort_values(ascending=False).head(10).index
        result.asset_weights[top_assets].plot(ax=plt.gca(), linewidth=1)
        plt.title('Top 10 Assets: Weight Evolution')
        plt.ylabel('Weight')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    # Subplot 3: Cash weight over time
    plt.subplot(2, 2, 3)
    result.cash_weight.plot(ax=plt.gca(), linewidth=2, color='red')
    plt.title('Cash Weight Over Time')
    plt.ylabel('Cash Weight')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Leverage over time
    plt.subplot(2, 2, 4)
    leverage = result.asset_weights.abs().sum(axis=1)
    leverage.plot(ax=plt.gca(), linewidth=2, color='purple')
    plt.title('Portfolio Leverage Over Time')
    plt.ylabel('Leverage')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_path / "weights_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved weights analysis plots.")

def generate_trade_analysis(result: BacktestResult, result_name: str, report_path: Path):
    """Generate comprehensive trade analysis"""
    
    # Calculate daily trade vectors
    daily_trades = result.quantities.diff()
    prices = result.valuations.div(result.quantities, axis=0).fillna(0)
    trade_values = daily_trades * prices
    
    # Save trade vectors to CSV
    trade_file = report_path / "daily_trade_vectors.csv"
    trade_values.to_csv(trade_file)
    logger.info(f"Saved daily trade vectors to: {trade_file}")
    
    # Create trade analysis plots
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Daily turnover
    plt.subplot(3, 2, 1)
    result.daily_turnover.plot(ax=plt.gca(), linewidth=1, color='blue')
    plt.title('Daily Turnover')
    plt.ylabel('Turnover Rate')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative turnover
    plt.subplot(3, 2, 2)
    cumulative_turnover = result.daily_turnover.cumsum()
    cumulative_turnover.plot(ax=plt.gca(), linewidth=2, color='green')
    plt.title('Cumulative Turnover')
    plt.ylabel('Cumulative Turnover')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Trade distribution
    plt.subplot(3, 2, 3)
    result.daily_turnover.hist(bins=50, ax=plt.gca(), alpha=0.7, color='orange')
    plt.title('Distribution of Daily Turnover')
    plt.xlabel('Daily Turnover')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Largest trades (top 10 by absolute value)
    plt.subplot(3, 2, 4)
    if len(trade_values) > 0:
        max_trades = trade_values.abs().max().sort_values(ascending=False).head(10)
        max_trades.plot(kind='bar', ax=plt.gca(), color='red', alpha=0.7)
        plt.title('Largest Single-Day Trades by Asset')
        plt.ylabel('Trade Value ($)')
        plt.xlabel('Assets')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    
    # Subplot 5: Trade heatmap (if not too many assets)
    plt.subplot(3, 2, 5)
    if len(trade_values.columns) <= 20:
        # Show heatmap of trade values over time
        trade_heatmap = trade_values.T  # Transpose for better visualization
        sns.heatmap(trade_heatmap, cmap='RdBu_r', center=0, ax=plt.gca(), 
                   cbar_kws={'label': 'Trade Value ($)'})
        plt.title('Trade Heatmap Over Time')
        plt.xlabel('Date')
        plt.ylabel('Assets')
    else:
        # Show aggregate trade statistics
        trade_stats = trade_values.describe()
        trade_stats.plot(kind='bar', ax=plt.gca())
        plt.title('Trade Statistics by Asset')
        plt.ylabel('Trade Value ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    
    # Subplot 6: Trade timing analysis
    plt.subplot(3, 2, 6)
    # Show when largest trades occurred
    largest_trade_days = result.daily_turnover.nlargest(10)
    largest_trade_days.plot(kind='bar', ax=plt.gca(), color='purple', alpha=0.7)
    plt.title('Days with Highest Turnover')
    plt.ylabel('Turnover Rate')
    plt.xlabel('Date')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_path / "trade_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved trade analysis plots.")

def generate_performance_plots(result: BacktestResult, result_name: str, report_path: Path):
    """Generate comprehensive performance plots"""
    
    plt.figure(figsize=(15, 12))
    
    # Subplot 1: Portfolio value (equity curve)
    plt.subplot(3, 2, 1)
    result.portfolio_value.plot(ax=plt.gca(), linewidth=2, color='blue')
    plt.title(f'Portfolio Value Over Time: {result_name}')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Portfolio returns
    plt.subplot(3, 2, 2)
    result.portfolio_returns.plot(ax=plt.gca(), linewidth=1, color='green', alpha=0.7)
    plt.title('Daily Portfolio Returns')
    plt.ylabel('Daily Return')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Returns distribution
    plt.subplot(3, 2, 3)
    result.portfolio_returns.hist(bins=50, ax=plt.gca(), alpha=0.7, color='orange')
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Drawdown
    plt.subplot(3, 2, 4)
    cumulative_returns = (1 + result.portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown.plot(ax=plt.gca(), linewidth=2, color='red')
    plt.title('Portfolio Drawdown')
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Rolling Sharpe ratio (30-day window)
    plt.subplot(3, 2, 5)
    if len(result.portfolio_returns) > 30:
        rolling_sharpe = result.portfolio_returns.rolling(30).mean() / result.portfolio_returns.rolling(30).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=plt.gca(), linewidth=2, color='purple')
        plt.title('Rolling Sharpe Ratio (30-day)')
        plt.ylabel('Sharpe Ratio')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
    
    # Subplot 6: Rolling volatility (30-day window)
    plt.subplot(3, 2, 6)
    if len(result.portfolio_returns) > 30:
        rolling_vol = result.portfolio_returns.rolling(30).std() * np.sqrt(252)
        rolling_vol.plot(ax=plt.gca(), linewidth=2, color='brown')
        plt.title('Rolling Volatility (30-day)')
        plt.ylabel('Annualized Volatility')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_path / "performance_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved performance analysis plots.")

def generate_risk_analysis(result: BacktestResult, result_name: str, report_path: Path):
    """Generate comprehensive risk analysis"""
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Portfolio composition over time (stacked area)
    plt.subplot(2, 3, 1)
    if len(result.asset_weights) > 0:
        # Handle negative cash weights for stacked area plot
        weights_with_cash = result.asset_weights.copy()
        weights_with_cash["_CASH"] = result.cash_weight
        
        # For clarity, if there are too many assets, group small ones
        if len(result.asset_weights.columns) > 15:
            top_assets = result.asset_weights.iloc[-1].abs().sort_values(ascending=False).head(14).index
            other_weights = weights_with_cash.drop(columns=top_assets).sum(axis=1)
            weights_to_plot = weights_with_cash[top_assets].copy()
            weights_to_plot["_OTHER_ASSETS"] = other_weights
            weights_to_plot["_CASH"] = weights_with_cash["_CASH"]
        else:
            weights_to_plot = weights_with_cash.copy()

        # Split cash into positive and negative components for stacked plot
        cash_positive = np.maximum(weights_to_plot["_CASH"], 0.0)
        cash_negative = np.minimum(weights_to_plot["_CASH"], 0.0)
        
        weights_to_plot = weights_to_plot.drop(columns=["_CASH"])
        weights_to_plot["_CASH_POSITIVE"] = cash_positive
        weights_to_plot["_CASH_NEGATIVE"] = cash_negative
        
        weights_to_plot.plot.area(stacked=True, linewidth=0, ax=plt.gca())
        
        plt.title('Portfolio Composition Over Time')
        plt.ylabel('Portfolio Weight')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 2: Risk decomposition (if we have covariance data)
    plt.subplot(2, 3, 2)
    # This would require additional data - placeholder for now
    plt.text(0.5, 0.5, 'Risk Decomposition\n(Requires covariance data)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Risk Decomposition')
    
    # Subplot 3: Correlation matrix of asset weights
    plt.subplot(2, 3, 3)
    if len(result.asset_weights) > 0 and len(result.asset_weights.columns) <= 20:
        corr_matrix = result.asset_weights.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=plt.gca(),
                   cbar_kws={'label': 'Correlation'})
        plt.title('Asset Weight Correlations')
    else:
        plt.text(0.5, 0.5, 'Correlation Matrix\n(Too many assets to display)', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Asset Weight Correlations')
    
    # Subplot 4: VaR analysis
    plt.subplot(2, 3, 4)
    returns = result.portfolio_returns
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    
    returns.hist(bins=50, ax=plt.gca(), alpha=0.7, color='lightblue')
    plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.4f}')
    plt.axvline(var_99, color='darkred', linestyle='--', label=f'99% VaR: {var_99:.4f}')
    plt.title('Value at Risk Analysis')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Rolling beta (if benchmark available)
    plt.subplot(2, 3, 5)
    plt.text(0.5, 0.5, 'Rolling Beta\n(Requires benchmark data)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Rolling Beta vs Benchmark')
    
    # Subplot 6: Information ratio over time
    plt.subplot(2, 3, 6)
    if len(result.portfolio_returns) > 30:
        # Calculate rolling information ratio (assuming risk-free rate is 0 for simplicity)
        rolling_ir = result.portfolio_returns.rolling(30).mean() / result.portfolio_returns.rolling(30).std()
        rolling_ir.plot(ax=plt.gca(), linewidth=2, color='green')
        plt.title('Rolling Information Ratio (30-day)')
        plt.ylabel('Information Ratio')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_path / "risk_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved risk analysis plots.")

if __name__ == "__main__":
    # Example usage
    result_filename = "transition_result_Dynamic_Uniform.pickle"
    result_name = "Dynamic_Uniform_Strategy"

    try:
        result_path = checkpoints_path() / result_filename
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Successfully loaded result from: {result_path}")
        
        # For transformation experiments, you might have target weights
        # This would typically come from your experiment setup
        target_weights = None  # Set this if available
        initial_weights = None  # Set this if available
        
        generate_comprehensive_report(result, result_name, target_weights, initial_weights)
        
    except FileNotFoundError:
        logger.error(f"Could not find result file: {result_path}")
        logger.error("Please run the experiment script first to generate the result file.")
        exit()