# experiments/analyze_results.py (Comprehensive Multi-Page Report)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
import pickle
from loguru import logger
from datetime import datetime

from backtest import BacktestResult, load_data
from utils import checkpoints_path, figures_path

# Constants - no more hardcoded values
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW_DAYS = 30
MAX_ASSETS_FOR_PLOTS = 20
MAX_DAYS_FOR_HEATMAP = 100
MAX_ASSETS_FOR_HEATMAP = 20
MAX_TOP_CHANGES = 15
MAX_TOP_WEIGHTS_TABLE = 20
DEFAULT_LOOKBACK_DAYS = 60
MAX_PARTICIPATION_RATE = 0.10

def get_report_config(
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    max_assets_for_plots: int = MAX_ASSETS_FOR_PLOTS,
    max_days_for_heatmap: int = MAX_DAYS_FOR_HEATMAP,
    max_assets_for_heatmap: int = MAX_ASSETS_FOR_HEATMAP,
    max_top_changes: int = MAX_TOP_CHANGES,
    max_top_weights_table: int = MAX_TOP_WEIGHTS_TABLE
) -> dict:
    """
    Get configuration for the report generation.
    
    Returns:
        dict: Configuration dictionary with all report parameters
    """
    return {
        'trading_days_per_year': trading_days_per_year,
        'rolling_window_days': rolling_window_days,
        'max_assets_for_plots': max_assets_for_plots,
        'max_days_for_heatmap': max_days_for_heatmap,
        'max_assets_for_heatmap': max_assets_for_heatmap,
        'max_top_changes': max_top_changes,
        'max_top_weights_table': max_top_weights_table
    }

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_portfolio_statistics(portfolio_value, asset_weights, daily_turnover, config: dict | None = None):
    """Calculate comprehensive portfolio statistics."""
    if config is None:
        config = get_report_config()
    
    returns = portfolio_value.pct_change().dropna()
    
    # Basic statistics - derive from actual data
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    periods_per_year = config['trading_days_per_year']
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = np.where(volatility > 0, annualized_return / volatility, np.nan)
    
    # Drawdown analysis
    rolling_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Turnover statistics
    avg_daily_turnover = daily_turnover.mean()
    annualized_turnover = daily_turnover.sum() * periods_per_year / len(daily_turnover)
    
    # Leverage statistics
    leverage = asset_weights.abs().sum(axis=1)
    avg_leverage = leverage.mean()
    max_leverage = leverage.max()
    
    # Risk metrics
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else np.nan
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Calmar ratio
    calmar_ratio = np.where(abs(max_drawdown) > 0, annualized_return / abs(max_drawdown), np.nan)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_daily_turnover': avg_daily_turnover,
        'annualized_turnover': annualized_turnover,
        'avg_leverage': avg_leverage,
        'max_leverage': max_leverage,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'calmar_ratio': calmar_ratio,
        'returns': returns,
        'drawdown': drawdown,
        'leverage': leverage
    }

def analyze_nan_data(prices, spreads, shortfees, volumes):
    """Analyze NaN patterns in the data."""
    nan_analysis = {}
    
    # Count NaNs in each dataset
    nan_analysis['prices_nan_count'] = prices.isna().sum().sum()
    nan_analysis['spreads_nan_count'] = spreads.isna().sum().sum()
    nan_analysis['shortfees_nan_count'] = shortfees.isna().sum().sum()
    nan_analysis['volumes_nan_count'] = volumes.isna().sum().sum()
    
    # Percentage of NaNs
    nan_analysis['prices_nan_pct'] = (nan_analysis['prices_nan_count'] / (prices.shape[0] * prices.shape[1])) * 100
    nan_analysis['spreads_nan_pct'] = (nan_analysis['spreads_nan_count'] / (spreads.shape[0] * spreads.shape[1])) * 100
    nan_analysis['shortfees_nan_pct'] = (nan_analysis['shortfees_nan_count'] / (shortfees.shape[0] * shortfees.shape[1])) * 100
    nan_analysis['volumes_nan_pct'] = (nan_analysis['volumes_nan_count'] / (volumes.shape[0] * volumes.shape[1])) * 100
    
    # Assets with most NaNs
    nan_analysis['prices_nan_by_asset'] = prices.isna().sum().sort_values(ascending=False).head(10)
    nan_analysis['spreads_nan_by_asset'] = spreads.isna().sum().sort_values(ascending=False).head(10)
    nan_analysis['shortfees_nan_by_asset'] = shortfees.isna().sum().sort_values(ascending=False).head(10)
    nan_analysis['volumes_nan_by_asset'] = volumes.isna().sum().sort_values(ascending=False).head(10)
    
    # Time periods with most NaNs
    nan_analysis['prices_nan_by_date'] = prices.isna().sum(axis=1).sort_values(ascending=False).head(10)
    nan_analysis['spreads_nan_by_date'] = spreads.isna().sum(axis=1).sort_values(ascending=False).head(10)
    
    return nan_analysis

def analyze_cost_data(spreads, shortfees, volumes, result):
    """Analyze trading costs and their impact using vectorized operations."""
    cost_analysis = {}
    
    # Spread analysis
    cost_analysis['avg_spread'] = spreads.mean().mean()
    cost_analysis['max_spread'] = spreads.max().max()
    cost_analysis['spread_std'] = spreads.std().mean()
    cost_analysis['high_spread_assets'] = spreads.mean().sort_values(ascending=False).head(10)
    
    # Short fee analysis
    cost_analysis['avg_shortfee'] = shortfees.mean().mean()
    cost_analysis['max_shortfee'] = shortfees.max().max()
    cost_analysis['high_shortfee_assets'] = shortfees.mean().sort_values(ascending=False).head(10)
    
    # Volume analysis
    cost_analysis['avg_volume'] = volumes.mean().mean()
    cost_analysis['low_volume_assets'] = volumes.mean().sort_values().head(10)
    
    # Vectorized cost impact analysis
    if hasattr(result, 'daily_turnover') and hasattr(result, 'asset_weights'):
        # Align data by date
        aligned_spreads = spreads.reindex(result.asset_weights.index, method='ffill')
        
        # Calculate weight changes (vectorized)
        weight_diff = result.asset_weights.diff().abs()
        
        # Estimate trading costs: half spread on trades (vectorized)
        daily_costs = (weight_diff * aligned_spreads / 2).sum(axis=1)
        
        # Remove first day (no previous weights to diff against)
        daily_costs = daily_costs.dropna()
        
        if len(daily_costs) > 0:
            cost_analysis['total_estimated_cost'] = daily_costs.sum()
            cost_analysis['avg_daily_cost'] = daily_costs.mean()
            # Use actual initial portfolio value instead of hardcoded 1e6
            initial_portfolio_value = result.portfolio_value.iloc[0]
            cost_analysis['cost_as_pct_of_portfolio'] = (cost_analysis['total_estimated_cost'] / initial_portfolio_value) * 100
            cost_analysis['daily_costs'] = daily_costs
    
    return cost_analysis


def analyze_shortfees_comparison(result):
    """Analyze comparison between actual shortfees and hardcoded rates."""
    from backtest import load_shortfees
    
    comparison = {}
    
    # Load actual shortfees (daily rates)
    actual_shortfees_daily = load_shortfees()
    
    # Convert to annualized percentages for comparison
    actual_shortfees_annual = actual_shortfees_daily * 100 * 360
    hardcoded_annual = 5.0  # 5 basis points annualized
    
    # Align with backtest period
    backtest_dates = result.history  # Use history instead of asset_weights.index
    aligned_actual = actual_shortfees_annual.reindex(backtest_dates, method='ffill')
    
    # Calculate portfolio-weighted shortfees over time
    portfolio_shortfees = []
    for date in backtest_dates:
        if date in aligned_actual.index:
            # Get short positions (negative quantities)
            short_quantities = result.quantities.loc[date]
            short_quantities = short_quantities[short_quantities < 0].abs()
            
            if len(short_quantities) > 0:
                # Get shortfees for this date
                daily_shortfees = aligned_actual.loc[date]
                
                # Calculate weighted average based on quantities
                weighted_shortfee = (short_quantities * daily_shortfees).sum() / short_quantities.sum()
                portfolio_shortfees.append(weighted_shortfee)
            else:
                portfolio_shortfees.append(0)
        else:
            portfolio_shortfees.append(hardcoded_annual)
    
    comparison['portfolio_shortfees'] = pd.Series(portfolio_shortfees, index=backtest_dates)
    comparison['hardcoded_rate'] = hardcoded_annual
    comparison['actual_avg'] = actual_shortfees_annual.mean().mean()
    comparison['actual_std'] = actual_shortfees_annual.std().mean()
    comparison['actual_min'] = actual_shortfees_annual.min().min()
    comparison['actual_max'] = actual_shortfees_annual.max().max()
    
    # Calculate cost difference
    if hasattr(result, 'use_actual_shortfees') and result.use_actual_shortfees:
        comparison['cost_savings'] = (hardcoded_annual - comparison['portfolio_shortfees'].mean()) / hardcoded_annual * 100
    else:
        comparison['cost_savings'] = 0
    
    return comparison

def create_comprehensive_report(
    result: BacktestResult, 
    result_name: str,
    config: dict | None = None
):
    """
    Creates a comprehensive multi-page PDF report with detailed analysis.
    
    Returns:
        dict: Dictionary containing paths to generated files
    """
    logger.info(f"--- Generating Comprehensive Report for: {result_name} ---")
    
    report_path = figures_path() / result_name
    report_path.mkdir(exist_ok=True)
    
    # Add logging to the report directory
    log_file = report_path / "report.log"
    logger.add(log_file, rotation="1 MB")
    
    # Use provided config or default
    if config is None:
        config = get_report_config()
    
    # Extract weights from the BacktestResult
    if not hasattr(result, 'original_target_weights') or len(result.original_target_weights) == 0:
        raise ValueError("BacktestResult must contain original_target_weights. Please run backtest with target_weights in strategy_kwargs.")
    
    target_weights = result.original_target_weights
    
    # Handle initial weights - use stored if available, otherwise derive from first day
    if hasattr(result, 'initial_weights') and len(result.initial_weights) > 0:
        initial_weights = result.initial_weights
        logger.info("Using stored initial weights from BacktestResult")
    else:
        # Backward compatibility: derive initial weights from first day's asset weights
        logger.warning("No initial_weights found in BacktestResult. Deriving from first day's asset weights.")
        initial_weights = result.asset_weights.iloc[0]
        logger.info("Derived initial weights from first day's asset weights")
    
    logger.info(f"Using initial weights: {len(initial_weights[initial_weights > 0])} assets")
    logger.info(f"Using target weights: {len(target_weights[target_weights > 0])} assets")
    
    # Load all data for analysis
    prices, spreads, rf, volumes = load_data()
    from utils import data_path
    from backtest import load_shortfees
    
    # Use shortfees based on what the backtest actually used
    if hasattr(result, 'use_actual_shortfees') and result.use_actual_shortfees:
        shortfees = load_shortfees() * 100 * 360  # Convert back to annualized percentages for analysis
        logger.info("Using actual shortfees data for analysis (as used in backtest)")
    else:
        # Load raw data for comparison
        shortfees = pd.read_csv(data_path() / "shortfees_full.csv", index_col=0, parse_dates=True)
        logger.info("Using raw shortfees data for analysis (backtest used hardcoded rates)")
    
    # Calculate portfolio statistics
    stats = calculate_portfolio_statistics(result.portfolio_value, result.asset_weights, result.daily_turnover, config)
    
    # Analyze NaN patterns
    nan_analysis = analyze_nan_data(prices, spreads, shortfees, volumes)
    
    # Analyze costs
    cost_analysis = analyze_cost_data(spreads, shortfees, volumes, result)
    
    # Analyze shortfees comparison
    shortfees_comparison = analyze_shortfees_comparison(result)
    
    # Create weights comparison
    final_weights = result.asset_weights.iloc[-1]
    weights_comparison = pd.DataFrame({
        'Initial_Weight': initial_weights,
        'Target_Weight': target_weights,
        'Final_Weight': final_weights,
        'Weight_Change': final_weights - initial_weights,
        'Target_Error': final_weights - target_weights
    })
    # Sort by absolute weight change - compatible with pandas < 2.0
    weights_comparison = weights_comparison.sort_values('Weight_Change', key=lambda s: s.abs(), ascending=False)
    
    # Create PDF with multiple pages
    pdf_path = report_path / "comprehensive_report.pdf"
    
    with PdfPages(pdf_path) as pdf:
        
        # ===== PAGE 1: EXECUTIVE SUMMARY =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Portfolio Value
        ax1 = fig.add_subplot(gs[0, :])
        result.portfolio_value.plot(ax=ax1, linewidth=2, color='blue')
        ax1.set_title("Portfolio Value Over Time", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)
        
        # Key Statistics Table
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        stats_text = (
            f"EXECUTIVE SUMMARY - {result_name}\n"
            f"{'='*60}\n\n"
            f"PERFORMANCE METRICS:\n"
            f"Total Return:           {stats['total_return']:.2%}\n"
            f"Annualized Return:      {stats['annualized_return']:.2%}\n"
            f"Volatility:             {stats['volatility']:.2%}\n"
            f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}\n"
            f"Maximum Drawdown:       {stats['max_drawdown']:.2%}\n\n"
            f"TRADING METRICS:\n"
            f"Annualized Turnover:    {stats['annualized_turnover']:.2f}\n"
            f"Average Daily Turnover: {stats['avg_daily_turnover']:.2%}\n"
            f"Average Leverage:       {stats['avg_leverage']:.2f}\n"
            f"Maximum Leverage:       {stats['max_leverage']:.2f}\n\n"
            f"PORTFOLIO COMPOSITION:\n"
            f"Initial Stocks:         {len([w for w in initial_weights if w > 0])}\n"
            f"Target Stocks:          {len([w for w in target_weights if w > 0])}\n"
            f"Final Stocks:           {len([w for w in final_weights if w > 0])}\n"
        )
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        # Implementation Shortfall - derive from actual data
        ax3 = fig.add_subplot(gs[2, 0])
        initial_notional = result.portfolio_value.iloc[0]
        final_value = result.portfolio_value.iloc[-1]
        shortfall = (initial_notional - final_value) / initial_notional * 10000
        ax3.bar(['Implementation\nShortfall'], [shortfall], color='red' if shortfall > 0 else 'green')
        ax3.set_ylabel('Basis Points')
        ax3.set_title(f'Implementation Shortfall: {shortfall:.2f} BPS')
        ax3.grid(True, alpha=0.3)
        
        # Tracking Error
        ax4 = fig.add_subplot(gs[2, 1])
        tracking_error = np.linalg.norm(final_weights - target_weights)
        ax4.bar(['Tracking\nError'], [tracking_error], color='orange')
        ax4.set_ylabel('Error Magnitude')
        ax4.set_title(f'Final Tracking Error: {tracking_error:.4f}')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle("EXECUTIVE SUMMARY", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 2: PORTFOLIO BEHAVIOR =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Drawdown Analysis
        ax1 = fig.add_subplot(gs[0, :])
        stats['drawdown'].plot(ax=ax1, kind='area', alpha=0.7, color='red')
        ax1.set_title("Portfolio Drawdown Analysis", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Drawdown")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Daily Turnover
        ax2 = fig.add_subplot(gs[1, 0])
        result.daily_turnover.plot(ax=ax2, alpha=0.7, color='purple')
        ax2.set_title("Daily Turnover")
        ax2.set_ylabel("Turnover Rate")
        ax2.grid(True, alpha=0.3)
        
        # Leverage Over Time
        ax3 = fig.add_subplot(gs[1, 1])
        stats['leverage'].plot(ax=ax3, alpha=0.7, color='green')
        ax3.set_title("Portfolio Leverage")
        ax3.set_ylabel("Leverage")
        ax3.grid(True, alpha=0.3)
        
        # Weight Trajectories (All Assets)
        ax4 = fig.add_subplot(gs[2, :])
        # Plot all assets with non-zero weights at any point
        active_assets = result.asset_weights.columns[result.asset_weights.abs().sum() > 0]
        
        # Use daily_target_weights if available, otherwise use asset_weights
        if hasattr(result, 'daily_target_weights') and len(result.daily_target_weights) > 0:
            weight_data = result.daily_target_weights
        else:
            weight_data = result.asset_weights
            logger.warning("daily_target_weights not available, using asset_weights for trajectories")
        
        # Limit to first max_assets_for_plots for readability
        for asset in active_assets[:config['max_assets_for_plots']]:
            if asset in weight_data.columns:
                ax4.plot(weight_data.index, weight_data[asset], 
                        alpha=0.6, linewidth=1)
        ax4.set_title(f"Weight Trajectories (Top {config['max_assets_for_plots']} Active Assets)", fontsize=14)
        ax4.set_ylabel("Weight")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        plt.suptitle("PORTFOLIO BEHAVIOR ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 3: DATA QUALITY ANALYSIS =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # NaN Summary
        ax1 = fig.add_subplot(gs[0, :])
        datasets = ['Prices', 'Spreads', 'Short Fees', 'Volumes']
        nan_counts = [nan_analysis['prices_nan_count'], nan_analysis['spreads_nan_count'], 
                     nan_analysis['shortfees_nan_count'], nan_analysis['volumes_nan_count']]
        nan_pcts = [nan_analysis['prices_nan_pct'], nan_analysis['spreads_nan_pct'],
                   nan_analysis['shortfees_nan_pct'], nan_analysis['volumes_nan_pct']]
        
        bars = ax1.bar(datasets, nan_pcts, color=['blue', 'orange', 'red', 'green'])
        ax1.set_title("NaN Percentage by Dataset", fontsize=16, fontweight='bold')
        ax1.set_ylabel("NaN Percentage (%)")
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars, nan_pcts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{pct:.2f}%', ha='center', va='bottom')
        
        # Assets with Most NaNs
        ax2 = fig.add_subplot(gs[1, 0])
        top_nan_assets = nan_analysis['prices_nan_by_asset'].head(10)
        ax2.barh(range(len(top_nan_assets)), top_nan_assets.values, color='red', alpha=0.7)
        ax2.set_yticks(range(len(top_nan_assets)))
        ax2.set_yticklabels(top_nan_assets.index)
        ax2.set_title("Assets with Most NaN Prices")
        ax2.set_xlabel("Number of NaN Days")
        
        # Dates with Most NaNs
        ax3 = fig.add_subplot(gs[1, 1])
        top_nan_dates = nan_analysis['prices_nan_by_date'].head(10)
        ax3.bar(range(len(top_nan_dates)), top_nan_dates.values, color='orange', alpha=0.7)
        ax3.set_title("Dates with Most NaN Prices")
        ax3.set_ylabel("Number of NaN Assets")
        ax3.tick_params(axis='x', rotation=45)
        
        # Data Completeness Heatmap
        ax4 = fig.add_subplot(gs[2, :])
        # Sample of assets for heatmap
        sample_assets = prices.columns[:config['max_assets_for_heatmap']]  # First max_assets_for_heatmap assets
        sample_prices = prices[sample_assets].iloc[-config['max_days_for_heatmap']:]  # Last max_days_for_heatmap days
        completeness = ~sample_prices.isna()
        
        im = ax4.imshow(completeness.T, cmap='RdYlGn', aspect='auto')
        ax4.set_title(f"Data Completeness Heatmap (Last {config['max_days_for_heatmap']} Days, First {config['max_assets_for_heatmap']} Assets)")
        ax4.set_xlabel("Days")
        ax4.set_ylabel("Assets")
        ax4.set_yticks(range(len(sample_assets)))
        ax4.set_yticklabels(sample_assets)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Data Available')
        
        plt.suptitle("DATA QUALITY ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 4: EXPLORATORY DATA ANALYSIS =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Returns Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        stats['returns'].hist(ax=ax1, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title("Portfolio Returns Distribution")
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        
        # Q-Q Plot
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats as scipy_stats
        scipy_stats.probplot(stats['returns'], dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)")
        ax2.grid(True, alpha=0.3)
        
        # Rolling Statistics
        ax3 = fig.add_subplot(gs[1, :])
        rolling_vol = stats['returns'].rolling(window=config['rolling_window_days']).std(ddof=0) * np.sqrt(config['trading_days_per_year'])
        rolling_vol.plot(ax=ax3, label=f'{config["rolling_window_days"]}-Day Rolling Volatility', color='red')
        ax3.axhline(y=stats['volatility'], color='blue', linestyle='--', 
                   label=f'Overall Volatility: {stats["volatility"]:.2%}')
        ax3.set_title(f"Rolling Volatility ({config['rolling_window_days']}-Day Window)")
        ax3.set_ylabel("Annualized Volatility")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Asset Performance Distribution
        ax4 = fig.add_subplot(gs[2, :])
        # Calculate individual asset returns during the period
        start_date = result.asset_weights.index[0]
        end_date = result.asset_weights.index[-1]
        if start_date in prices.index and end_date in prices.index:
            start_idx = prices.index.get_loc(start_date)
            end_idx = prices.index.get_loc(end_date)
            asset_returns = (prices.iloc[end_idx] / prices.iloc[start_idx] - 1).dropna()
            
            ax4.hist(asset_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(x=stats['total_return'], color='red', linestyle='--', 
                       label=f'Portfolio Return: {stats["total_return"]:.2%}')
            ax4.set_title("Individual Asset Returns Distribution")
            ax4.set_xlabel("Total Return")
            ax4.set_ylabel("Number of Assets")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle("EXPLORATORY DATA ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 5: COST ANALYSIS =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Spread Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        spread_stats = [cost_analysis['avg_spread'], cost_analysis['max_spread'], cost_analysis['spread_std']]
        spread_labels = ['Average\nSpread', 'Maximum\nSpread', 'Spread\nStd Dev']
        bars = ax1.bar(spread_labels, spread_stats, color=['blue', 'red', 'orange'])
        ax1.set_title("Spread Statistics")
        ax1.set_ylabel("Spread (BPS)")
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, spread_stats):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # High Spread Assets
        ax2 = fig.add_subplot(gs[0, 1])
        high_spreads = cost_analysis['high_spread_assets'].head(10)
        ax2.barh(range(len(high_spreads)), high_spreads.values, color='red', alpha=0.7)
        ax2.set_yticks(range(len(high_spreads)))
        ax2.set_yticklabels(high_spreads.index)
        ax2.set_title("Assets with Highest Average Spreads")
        ax2.set_xlabel("Average Spread (BPS)")
        
        # Short Fee Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        shortfee_stats = [cost_analysis['avg_shortfee'], cost_analysis['max_shortfee']]
        shortfee_labels = ['Average\nShort Fee', 'Maximum\nShort Fee']
        bars = ax3.bar(shortfee_labels, shortfee_stats, color=['purple', 'darkred'])
        ax3.set_title("Short Fee Statistics")
        ax3.set_ylabel("Short Fee (%)")
        ax3.grid(True, alpha=0.3)
        
        # High Short Fee Assets
        ax4 = fig.add_subplot(gs[1, 1])
        high_shortfees = cost_analysis['high_shortfee_assets'].head(10)
        ax4.barh(range(len(high_shortfees)), high_shortfees.values, color='purple', alpha=0.7)
        ax4.set_yticks(range(len(high_shortfees)))
        ax4.set_yticklabels(high_shortfees.index)
        ax4.set_title("Assets with Highest Short Fees")
        ax4.set_xlabel("Average Short Fee (%)")
        
        # Cost Impact Summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        cost_text = (
            f"TRADING COST ANALYSIS\n"
            f"{'='*40}\n\n"
            f"SPREAD COSTS:\n"
            f"Average Spread:         {cost_analysis['avg_spread']:.3f} BPS\n"
            f"Maximum Spread:         {cost_analysis['max_spread']:.3f} BPS\n"
            f"Spread Standard Dev:    {cost_analysis['spread_std']:.3f} BPS\n\n"
            f"SHORT FEE COSTS:\n"
            f"Average Short Fee:      {cost_analysis['avg_shortfee']:.3f}%\n"
            f"Maximum Short Fee:      {cost_analysis['max_shortfee']:.3f}%\n\n"
            f"VOLUME ANALYSIS:\n"
            f"Average Volume:         {cost_analysis['avg_volume']:,.0f}\n\n"
        )
        
        if 'total_estimated_cost' in cost_analysis:
            cost_text += (
                f"ESTIMATED TRADING COSTS:\n"
                f"Total Estimated Cost:  ${cost_analysis['total_estimated_cost']:,.2f}\n"
                f"Average Daily Cost:    ${cost_analysis['avg_daily_cost']:,.2f}\n"
                f"Cost as % of Portfolio: {cost_analysis['cost_as_pct_of_portfolio']:.2f}%\n"
            )
        
        ax5.text(0.05, 0.95, cost_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        plt.suptitle("TRADING COST ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 6: SHORTFEES COMPARISON =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Portfolio Shortfees Over Time
        ax1 = fig.add_subplot(gs[0, :])
        shortfees_comparison['portfolio_shortfees'].plot(ax=ax1, alpha=0.7, color='blue', linewidth=1)
        ax1.axhline(y=shortfees_comparison['hardcoded_rate'], color='red', linestyle='--', 
                   label=f"Hardcoded Rate: {shortfees_comparison['hardcoded_rate']:.2f}%")
        ax1.set_title("Portfolio-Weighted Shortfees Over Time", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Short Fee (%)")
        ax1.set_xlabel("Date")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Shortfees Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        portfolio_shortfees = shortfees_comparison['portfolio_shortfees']
        portfolio_shortfees[portfolio_shortfees > 0].hist(ax=ax2, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=shortfees_comparison['hardcoded_rate'], color='red', linestyle='--', 
                   label=f"Hardcoded: {shortfees_comparison['hardcoded_rate']:.2f}%")
        ax2.set_title("Portfolio Shortfees Distribution")
        ax2.set_xlabel("Short Fee (%)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Comparison Statistics
        ax3 = fig.add_subplot(gs[1, 1])
        comparison_stats = [
            shortfees_comparison['portfolio_shortfees'].mean(),
            shortfees_comparison['hardcoded_rate'],
            shortfees_comparison['actual_avg'],
            shortfees_comparison['actual_min'],
            shortfees_comparison['actual_max']
        ]
        comparison_labels = [
            'Portfolio\nAverage',
            'Hardcoded\nRate',
            'Market\nAverage',
            'Market\nMinimum',
            'Market\nMaximum'
        ]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        bars = ax3.bar(comparison_labels, comparison_stats, color=colors, alpha=0.7)
        ax3.set_title("Shortfees Comparison")
        ax3.set_ylabel("Short Fee (%)")
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, comparison_stats):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # Summary Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        shortfees_text = (
            f"SHORTFEES COMPARISON ANALYSIS\n"
            f"{'='*50}\n\n"
            f"BACKTEST CONFIGURATION:\n"
            f"Used Actual Shortfees:           {'Yes' if hasattr(result, 'use_actual_shortfees') and result.use_actual_shortfees else 'No'}\n"
            f"Hardcoded Rate:                  {shortfees_comparison['hardcoded_rate']:.2f}% (5 bps)\n\n"
            f"PORTFOLIO ANALYSIS:\n"
            f"Average Portfolio Shortfee:      {shortfees_comparison['portfolio_shortfees'].mean():.2f}%\n"
            f"Portfolio Shortfee Std Dev:      {shortfees_comparison['portfolio_shortfees'].std():.2f}%\n"
            f"Days with Short Positions:       {(shortfees_comparison['portfolio_shortfees'] > 0).sum()} ({(shortfees_comparison['portfolio_shortfees'] > 0).mean():.1%})\n\n"
            f"MARKET DATA:\n"
            f"Market Average Shortfee:         {shortfees_comparison['actual_avg']:.2f}%\n"
            f"Market Shortfee Std Dev:         {shortfees_comparison['actual_std']:.2f}%\n"
            f"Market Range:                    {shortfees_comparison['actual_min']:.2f}% - {shortfees_comparison['actual_max']:.2f}%\n\n"
        )
        
        if hasattr(result, 'use_actual_shortfees') and result.use_actual_shortfees:
            shortfees_text += (
                f"COST IMPACT:\n"
                f"Cost Savings vs Hardcoded:     {shortfees_comparison['cost_savings']:.1f}%\n"
                f"Average Daily Savings:        {shortfees_comparison['cost_savings']/100 * shortfees_comparison['hardcoded_rate']/360:.4f}% of portfolio\n"
            )
        else:
            shortfees_text += (
                f"COST IMPACT:\n"
                f"Using hardcoded rates - no actual shortfees data used in backtest\n"
            )
        
        ax4.text(0.02, 0.98, shortfees_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        plt.suptitle("SHORTFEES COMPARISON ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 7: WEIGHTS COMPARISON =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Weight Changes Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        weight_changes = weights_comparison['Weight_Change']
        ax1.hist(weight_changes, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title("Distribution of Weight Changes")
        ax1.set_xlabel("Weight Change")
        ax1.set_ylabel("Number of Assets")
        ax1.grid(True, alpha=0.3)
        
        # Target Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        target_errors = weights_comparison['Target_Error']
        ax2.hist(target_errors, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title("Distribution of Target Errors")
        ax2.set_xlabel("Target Error")
        ax2.set_ylabel("Number of Assets")
        ax2.grid(True, alpha=0.3)
        
        # Top Weight Changes
        ax3 = fig.add_subplot(gs[1, :])
        top_changes = weights_comparison.head(config['max_top_changes'])
        colors = ['green' if x > 0 else 'red' for x in top_changes['Weight_Change']]
        bars = ax3.barh(range(len(top_changes)), top_changes['Weight_Change'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(top_changes)))
        ax3.set_yticklabels(top_changes.index)
        ax3.set_title(f"Top {config['max_top_changes']} Assets by Weight Change")
        ax3.set_xlabel("Weight Change")
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Weights Comparison Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Prepare table data
        table_data = weights_comparison.head(config['max_top_weights_table']).copy()
        table_data = table_data.round(4)
        
        # Create table
        table = ax4.table(
            cellText=table_data.values.tolist(),
            rowLabels=table_data.index.tolist(),
            colLabels=['Initial', 'Target', 'Final', 'Change', 'Target Error'],
            cellLoc='center',
            loc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color header row
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax4.set_title(f"Weights Comparison (Top {config['max_top_weights_table']} Assets)", fontsize=14, pad=20)
        
        plt.suptitle("PORTFOLIO WEIGHTS ANALYSIS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 8: DETAILED STATISTICS =====
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        
        # Comprehensive Statistics Table
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')
        
        detailed_stats = (
            f"DETAILED PERFORMANCE STATISTICS\n"
            f"{'='*50}\n\n"
            f"RETURN METRICS:\n"
            f"Total Return:                    {stats['total_return']:.4f} ({stats['total_return']:.2%})\n"
            f"Annualized Return:               {stats['annualized_return']:.4f} ({stats['annualized_return']:.2%})\n"
            f"Volatility (Annualized):         {stats['volatility']:.4f} ({stats['volatility']:.2%})\n"
            f"Sharpe Ratio:                    {stats['sharpe_ratio']:.4f}\n"
            f"Maximum Drawdown:                {stats['max_drawdown']:.4f} ({stats['max_drawdown']:.2%})\n"
            f"Calmar Ratio:                    {(stats['annualized_return'] / abs(stats['max_drawdown'])):.4f}\n\n"
            f"RISK METRICS:\n"
            f"VaR (95%):                       {np.percentile(stats['returns'], 5):.4f} ({np.percentile(stats['returns'], 5):.2%})\n"
            f"CVaR (95%):                      {stats['returns'][stats['returns'] <= np.percentile(stats['returns'], 5)].mean():.4f}\n"
            f"Skewness:                        {stats['returns'].skew():.4f}\n"
            f"Kurtosis:                        {stats['returns'].kurtosis():.4f}\n\n"
            f"TRADING METRICS:\n"
            f"Annualized Turnover:             {stats['annualized_turnover']:.4f}\n"
            f"Average Daily Turnover:          {stats['avg_daily_turnover']:.4f} ({stats['avg_daily_turnover']:.2%})\n"
            f"Turnover Volatility:             {result.daily_turnover.std():.4f}\n"
            f"Maximum Daily Turnover:          {result.daily_turnover.max():.4f} ({result.daily_turnover.max():.2%})\n\n"
            f"LEVERAGE METRICS:\n"
            f"Average Leverage:                {stats['avg_leverage']:.4f}\n"
            f"Maximum Leverage:                {stats['max_leverage']:.4f}\n"
            f"Leverage Volatility:             {stats['leverage'].std():.4f}\n"
            f"Days with Leverage > 1:          {(stats['leverage'] > 1).sum()} ({(stats['leverage'] > 1).mean():.2%})\n\n"
            f"IMPLEMENTATION METRICS:\n"
            f"Implementation Shortfall:        {shortfall:.2f} BPS\n"
            f"Final Tracking Error:            {tracking_error:.6f}\n"
            f"Average Daily Tracking Error:    {np.mean([np.linalg.norm(result.asset_weights.iloc[i] - target_weights) for i in range(len(result.asset_weights))]):.6f}\n"
        )
        
        ax1.text(0.02, 0.98, detailed_stats, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        # Portfolio Composition Summary
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        initial_stocks = [asset for asset, weight in initial_weights.items() if weight > 0]
        target_stocks = [asset for asset, weight in target_weights.items() if weight > 0]
        final_stocks = [asset for asset, weight in final_weights.items() if weight > 0]
        
        composition_text = (
            f"PORTFOLIO COMPOSITION ANALYSIS\n"
            f"{'='*40}\n\n"
            f"INITIAL PORTFOLIO:\n"
            f"Number of Stocks:                {len(initial_stocks)}\n"
            f"Total Weight:                    {initial_weights.sum():.4f}\n"
            f"Largest Position:                {initial_weights.max():.4f} ({initial_weights.idxmax()})\n"
            f"Smallest Position:               {initial_weights[initial_weights > 0].min():.4f}\n"
            f"Average Position Size:           {initial_weights[initial_weights > 0].mean():.4f}\n\n"
            f"TARGET PORTFOLIO:\n"
            f"Number of Stocks:                {len(target_stocks)}\n"
            f"Total Weight:                    {target_weights.sum():.4f}\n"
            f"Largest Position:                {target_weights.max():.4f} ({target_weights.idxmax()})\n"
            f"Smallest Position:               {target_weights[target_weights > 0].min():.4f}\n"
            f"Average Position Size:           {target_weights[target_weights > 0].mean():.4f}\n\n"
            f"FINAL PORTFOLIO:\n"
            f"Number of Stocks:                {len(final_stocks)}\n"
            f"Total Weight:                    {final_weights.sum():.4f}\n"
            f"Largest Position:                {final_weights.max():.4f} ({final_weights.idxmax()})\n"
            f"Smallest Position:               {final_weights[final_weights > 0].min():.4f}\n"
            f"Average Position Size:           {final_weights[final_weights > 0].mean():.4f}\n\n"
            f"TRANSITION ANALYSIS:\n"
            f"Assets Added:                    {len(set(final_stocks) - set(initial_stocks))}\n"
            f"Assets Removed:                  {len(set(initial_stocks) - set(final_stocks))}\n"
            f"Assets Maintained:               {len(set(initial_stocks) & set(final_stocks))}\n"
            f"Total Weight Change:             {final_weights.sum() - initial_weights.sum():.4f}\n"
        )
        
        ax2.text(0.02, 0.98, composition_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        plt.suptitle("DETAILED STATISTICS", fontsize=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Save additional files
    weights_csv_path = report_path / "weights_comparison.csv"
    weights_comparison.to_csv(weights_csv_path)
    
    # Save portfolio lists
    for portfolio_name, weights in [('initial', initial_weights), ('target', target_weights), ('final', final_weights)]:
        portfolio_file_path = report_path / f"{portfolio_name}_portfolio_stocks.txt"
        stocks = [asset for asset, weight in weights.items() if weight > 0]
        with open(portfolio_file_path, 'w') as f:
            f.write(f"{portfolio_name.title()} Portfolio Stocks ({len(stocks)} stocks, total weight: {weights.sum():.4f})\n")
            f.write("=" * 60 + "\n")
            for i, stock in enumerate(stocks, 1):
                f.write(f"{i:3d}. {stock:<15} {weights[stock]:.4f}\n")
    
    logger.info(f"Comprehensive report saved to {pdf_path}")
    logger.info(f"Additional files saved to {report_path}")
    
    # Return dictionary with all generated file paths
    return {
        "pdf": pdf_path,
        "weights_csv": weights_csv_path,
        "initial_portfolio": report_path / "initial_portfolio_stocks.txt",
        "target_portfolio": report_path / "target_portfolio_stocks.txt", 
        "final_portfolio": report_path / "final_portfolio_stocks.txt",
        "log": log_file,
        "report_directory": report_path
    }

# Keep the old function for backward compatibility
def create_dashboard_report(
    result: BacktestResult, 
    initial_weights: pd.Series,
    target_weights: pd.Series,
    result_name: str,
    config: dict | None = None
):
    """
    Legacy function - now calls the comprehensive report.
    """
    logger.warning("create_dashboard_report is deprecated. Use create_comprehensive_report directly.")
    return create_comprehensive_report(result, result_name, config)


if __name__ == "__main__":
    # Define which result to analyze
    result_name = "Dynamic_Uniform"
    result_filename = f"transition_result_{result_name}.pickle"
    
    # Load the BacktestResult object
    try:
        result_path = checkpoints_path() / result_filename
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Successfully loaded result from: {result_path}")
    except FileNotFoundError:
        logger.error(f"Could not find result file: {result_path}")
        exit()

    # Create comprehensive report using weights stored in BacktestResult
    create_comprehensive_report(result, result_name)