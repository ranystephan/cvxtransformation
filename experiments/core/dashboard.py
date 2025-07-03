"""
Portfolio Backtesting Dashboard

This module provides comprehensive visualization and analysis tools for portfolio
backtesting results, including detailed tracking of portfolio evolution, trades,
and performance metrics over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class DetailedBacktestResult:
    """Extended backtest result with detailed tracking data for dashboard creation."""
    # Basic results
    cash: pd.Series
    quantities: pd.DataFrame
    portfolio_value: pd.Series
    portfolio_returns: pd.Series
    risk_target: float
    
    # Detailed tracking data
    daily_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_name: str = "Unknown"
    strategy_params: Dict = field(default_factory=dict)


class BacktestDashboard:
    """Comprehensive dashboard for portfolio backtesting visualization."""
    
    def __init__(self, result: DetailedBacktestResult):
        self.result = result
        self.figures = {}
    
    def create_full_dashboard(self, save_path: Optional[str] = None, show: bool = True) -> Dict:
        """Create a comprehensive dashboard with all visualizations."""
        print(f"Creating dashboard for {self.result.strategy_name} strategy...")
        
        # Create main dashboard
        self.figures['main'] = self._create_main_dashboard()
        
        # Create detailed analysis plots
        self.figures['portfolio_evolution'] = self._create_portfolio_evolution_plot()
        self.figures['trades_analysis'] = self._create_trades_analysis()
        self.figures['weights_heatmap'] = self._create_weights_heatmap()
        
        # Transformation-specific plots if applicable
        if not self.result.daily_target_weights.empty:
            self.figures['transformation_tracking'] = self._create_transformation_tracking()
        
        # Save figures if requested
        if save_path:
            self._save_dashboard(save_path)
        
        if show:
            plt.show()
        
        return self.figures
    
    def _create_main_dashboard(self):
        """Create the main dashboard with key metrics overview."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Portfolio value evolution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.result.portfolio_value.index, self.result.portfolio_value.values, 
                linewidth=2, color='navy', label='Portfolio Value')
        ax1.set_title('Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Returns distribution
        ax2 = fig.add_subplot(gs[0, 2])
        returns = self.result.portfolio_returns.dropna()
        ax2.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Returns')
        ax2.legend()
        
        # Key metrics
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        
        # Calculate key metrics
        total_return = (self.result.portfolio_value.iloc[-1] / self.result.portfolio_value.iloc[0] - 1) * 100
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = self._calculate_max_drawdown()
        volatility = returns.std() * np.sqrt(252) * 100
        
        metrics_text = f"""
Key Metrics:

Total Return: {total_return:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
Volatility: {volatility:.2f}%
Max Drawdown: {max_dd:.2f}%

Strategy: {self.result.strategy_name}
Risk Target: {self.result.risk_target:.3f}
Period: {len(self.result.portfolio_value)} days
        """
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Cash vs invested
        ax4 = fig.add_subplot(gs[1, :2])
        invested_value = self.result.portfolio_value - self.result.cash
        ax4.plot(self.result.cash.index, self.result.cash.values, 
                label='Cash', color='green', alpha=0.7)
        ax4.plot(invested_value.index, invested_value.values, 
                label='Invested', color='orange', alpha=0.7)
        ax4.fill_between(self.result.cash.index, 0, self.result.cash.values, 
                        alpha=0.3, color='green')
        ax4.fill_between(invested_value.index, self.result.cash.values, 
                        self.result.portfolio_value.values, alpha=0.3, color='orange')
        ax4.set_title('Cash vs Invested Capital')
        ax4.set_ylabel('Value ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax5 = fig.add_subplot(gs[1, 2:])
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', alpha=0.8)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_title('60-Day Rolling Sharpe Ratio')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.grid(True, alpha=0.3)
        
        # Daily trades summary (if available)
        if not self.result.daily_trades.empty:
            ax6 = fig.add_subplot(gs[2, :2])
            daily_trade_value = self.result.daily_trades.abs().sum(axis=1)
            ax6.plot(daily_trade_value.index, daily_trade_value.values, 
                    color='red', alpha=0.7, label='Daily Trade Volume')
            ax6.set_title('Daily Trading Activity')
            ax6.set_ylabel('Trade Volume')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Weight evolution (top 5 assets)
        if not self.result.daily_weights.empty:
            ax7 = fig.add_subplot(gs[2, 2:])
            top_assets = self.result.daily_weights.iloc[-1].abs().nlargest(5).index
            for asset in top_assets:
                ax7.plot(self.result.daily_weights.index, 
                        self.result.daily_weights[asset], 
                        label=f'{asset}', alpha=0.8)
            ax7.set_title('Top 5 Asset Weights Evolution')
            ax7.set_ylabel('Weight')
            ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'Portfolio Dashboard - {self.result.strategy_name}', 
                    fontsize=16, fontweight='bold')
        return fig
    
    def _create_portfolio_evolution_plot(self):
        """Create detailed portfolio evolution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value with drawdown
        ax1 = axes[0, 0]
        cumulative_max = self.result.portfolio_value.cummax()
        drawdown = (self.result.portfolio_value / cumulative_max - 1) * 100
        
        ax1_twin = ax1.twinx()
        ax1.plot(self.result.portfolio_value.index, self.result.portfolio_value.values, 
                color='blue', label='Portfolio Value')
        ax1_twin.fill_between(drawdown.index, drawdown.values, 0, 
                             color='red', alpha=0.3, label='Drawdown %')
        ax1.set_title('Portfolio Value and Drawdown')
        ax1.set_ylabel('Portfolio Value ($)', color='blue')
        ax1_twin.set_ylabel('Drawdown (%)', color='red')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Returns vs benchmark
        ax2 = axes[0, 1]
        returns = self.result.portfolio_returns.dropna()
        cumulative_returns = (1 + returns).cumprod()
        ax2.plot(cumulative_returns.index, (cumulative_returns - 1) * 100, 
                color='green', label='Strategy Returns')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Volatility evolution
        ax3 = axes[1, 0]
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        ax3.plot(rolling_vol.index, rolling_vol.values, color='orange', alpha=0.8)
        ax3.axhline(y=self.result.risk_target * np.sqrt(252) * 100, 
                   color='red', linestyle='--', label='Risk Target')
        ax3.set_title('30-Day Rolling Volatility')
        ax3.set_ylabel('Volatility (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cash allocation over time
        ax4 = axes[1, 1]
        cash_pct = (self.result.cash / self.result.portfolio_value) * 100
        ax4.plot(cash_pct.index, cash_pct.values, color='purple', alpha=0.8)
        ax4.set_title('Cash Allocation Over Time')
        ax4.set_ylabel('Cash Allocation (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_trades_analysis(self):
        """Create detailed trades analysis."""
        if self.result.daily_trades.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trade data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Trades Analysis - No Data')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Daily trade volume
        ax1 = axes[0, 0]
        daily_volume = self.result.daily_trades.abs().sum(axis=1)
        ax1.plot(daily_volume.index, daily_volume.values, color='red', alpha=0.7)
        ax1.set_title('Daily Trading Volume')
        ax1.set_ylabel('Trade Volume')
        ax1.grid(True, alpha=0.3)
        
        # Trade distribution
        ax2 = axes[0, 1]
        all_trades = self.result.daily_trades.values.flatten()
        all_trades = all_trades[all_trades != 0]  # Remove zero trades
        if len(all_trades) > 0:
            ax2.hist(all_trades, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Trade Size Distribution')
            ax2.set_xlabel('Trade Size')
            ax2.set_ylabel('Frequency')
        
        # Turnover over time
        ax3 = axes[1, 0]
        turnover = daily_volume / self.result.portfolio_value * 100
        ax3.plot(turnover.index, turnover.values, color='orange', alpha=0.8)
        ax3.set_title('Daily Turnover Rate')
        ax3.set_ylabel('Turnover (%)')
        ax3.grid(True, alpha=0.3)
        
        # Most active assets
        ax4 = axes[1, 1]
        asset_activity = self.result.daily_trades.abs().sum().sort_values(ascending=False).head(10)
        asset_activity.plot(kind='bar', ax=ax4, color='skyblue')
        ax4.set_title('Most Actively Traded Assets')
        ax4.set_ylabel('Total Trade Volume')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _create_weights_heatmap(self):
        """Create portfolio weights heatmap over time."""
        if self.result.daily_weights.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No weights data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Portfolio Weights - No Data')
            return fig
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Select top assets by maximum absolute weight
        max_weights = self.result.daily_weights.abs().max()
        top_assets = max_weights.nlargest(20).index
        
        # Weights heatmap
        weights_subset = self.result.daily_weights[top_assets].T
        
        im = axes[0].imshow(weights_subset.values, aspect='auto', cmap='RdBu_r', 
                           vmin=-weights_subset.abs().max().max(), 
                           vmax=weights_subset.abs().max().max())
        
        axes[0].set_title('Portfolio Weights Heatmap (Top 20 Assets)')
        axes[0].set_ylabel('Assets')
        axes[0].set_yticks(range(len(top_assets)))
        axes[0].set_yticklabels(top_assets)
        
        # Set x-axis ticks for dates
        n_ticks = min(10, len(weights_subset.columns))
        tick_indices = np.linspace(0, len(weights_subset.columns)-1, n_ticks, dtype=int)
        axes[0].set_xticks(tick_indices)
        axes[0].set_xticklabels([weights_subset.columns[i].strftime('%Y-%m-%d') 
                                for i in tick_indices], rotation=45)
        
        plt.colorbar(im, ax=axes[0], label='Weight')
        
        # Portfolio concentration over time
        concentration = (self.result.daily_weights ** 2).sum(axis=1)
        axes[1].plot(concentration.index, concentration.values, color='purple', alpha=0.8)
        axes[1].set_title('Portfolio Concentration (Sum of Squared Weights)')
        axes[1].set_ylabel('Concentration')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_transformation_tracking(self):
        """Create transformation-specific tracking plots."""
        if self.result.daily_target_weights.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Select a few key assets to track
        initial_weights = self.result.daily_target_weights.iloc[0]
        final_weights = self.result.daily_target_weights.iloc[-1]
        weight_changes = (final_weights - initial_weights).abs()
        key_assets = weight_changes.nlargest(4).index
        
        # Plot target vs actual paths for key assets
        for i, asset in enumerate(key_assets):
            ax = axes[i//2, i%2]
            
            if asset in self.result.daily_weights.columns:
                ax.plot(self.result.daily_target_weights.index, 
                       self.result.daily_target_weights[asset], 
                       label='Target Path', linestyle='--', color='red', linewidth=2)
                ax.plot(self.result.daily_weights.index, 
                       self.result.daily_weights[asset], 
                       label='Actual Path', color='blue', alpha=0.8)
                
                ax.set_title(f'{asset} - Target vs Actual Path')
                ax.set_ylabel('Weight')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{asset}\nNo actual data', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Transformation Tracking: Target vs Actual Paths', fontsize=14)
        plt.tight_layout()
        return fig
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        cumulative_max = self.result.portfolio_value.cummax()
        drawdown = (self.result.portfolio_value / cumulative_max - 1) * 100
        return drawdown.min()
    
    def _save_dashboard(self, save_path: str):
        """Save all figures to files."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for name, fig in self.figures.items():
            if fig is not None:
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
        
        print(f"Dashboard saved to {save_path}")


def create_dashboard_from_backtest(backtest_result, strategy_name: str = "Unknown", 
                                  additional_data: Dict = None, 
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> Dict:
    """
    Convenience function to create a dashboard from a basic backtest result.
    
    Args:
        backtest_result: Basic BacktestResult object
        strategy_name: Name of the strategy for labeling
        additional_data: Additional tracking data if available
        save_path: Optional path to save dashboard
        show: Whether to display plots
        
    Returns:
        Dictionary of created figures
    """
    # Convert to detailed result
    additional_data = additional_data or {}
    
    detailed_result = DetailedBacktestResult(
        cash=backtest_result.cash,
        quantities=backtest_result.quantities,
        portfolio_value=backtest_result.portfolio_value,
        portfolio_returns=backtest_result.portfolio_returns,
        risk_target=backtest_result.risk_target,
        strategy_name=strategy_name,
        strategy_params=additional_data.get('strategy_params', {}),
        daily_trades=additional_data.get('daily_trades', pd.DataFrame()),
        daily_weights=additional_data.get('daily_weights', pd.DataFrame()),
        daily_target_weights=additional_data.get('daily_target_weights', pd.DataFrame()),
        daily_costs=additional_data.get('daily_costs', pd.DataFrame())
    )
    
    # Create dashboard
    dashboard = BacktestDashboard(detailed_result)
    return dashboard.create_full_dashboard(save_path=save_path, show=show)


if __name__ == "__main__":
    print("Dashboard module ready. Use create_dashboard_from_backtest() to create visualizations.")
    print("Example:")
    print("figures = create_dashboard_from_backtest(result, 'My Strategy', save_path='./dashboard')") 