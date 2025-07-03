# Portfolio Backtesting Dashboard System

This dashboard system provides comprehensive visualization and analysis tools for portfolio backtesting results. It captures detailed tracking data and creates rich visualizations to help understand portfolio behavior, trading patterns, and performance metrics.

## Features

### Dashboard Visualizations

1. **Main Dashboard** (`main.png`)
   - Portfolio value evolution over time
   - Returns distribution histogram
   - Key performance metrics summary
   - Cash vs invested capital breakdown
   - Rolling Sharpe ratio
   - Daily trading activity
   - Top assets weight evolution

2. **Portfolio Evolution** (`portfolio_evolution.png`)
   - Portfolio value with drawdown visualization
   - Cumulative returns chart
   - Rolling volatility vs risk target
   - Cash allocation percentage over time

3. **Trades Analysis** (`trades_analysis.png`)
   - Daily trading volume
   - Trade size distribution
   - Turnover rate over time
   - Most actively traded assets

4. **Weights Heatmap** (`weights_heatmap.png`)
   - Portfolio weights heatmap for top assets over time
   - Portfolio concentration (sum of squared weights)

5. **Transformation Tracking** (`transformation_tracking.png`) - *For transformation strategies*
   - Target vs actual weight paths for key assets
   - Shows how well the strategy follows the intended transformation path

## Quick Start

### Option 1: Basic Dashboard (Works with existing backtest results)

```python
from dashboard import create_dashboard_from_backtest
from backtest import run_backtest

# Run your existing backtest
def my_strategy(inputs):
    # Your strategy logic here
    return w, c, problem

result = run_backtest(my_strategy, risk_target=0.05)

# Create dashboard
figures = create_dashboard_from_backtest(
    backtest_result=result,
    strategy_name="My Strategy",
    save_path="./my_dashboard",
    show=True  # Set to False to not display plots
)
```

### Option 2: Enhanced Dashboard (More detailed tracking)

```python
from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

# Run enhanced backtest with detailed tracking
result = run_enhanced_backtest(
    strategy=my_strategy,
    risk_target=0.05,
    strategy_name="My Strategy",
    strategy_params={"param1": "value1"},
    verbose=True
)

# Create comprehensive dashboard
figures = create_dashboard_from_enhanced_backtest(
    result=result,
    save_path="./my_enhanced_dashboard",
    show=True
)
```

### Option 3: Convenience Functions

```python
from enhanced_backtest import run_markowitz_with_dashboard, run_transformation_with_dashboard

# Run Markowitz strategy with dashboard
result, figures = run_markowitz_with_dashboard(
    risk_target=0.05,
    save_path="./markowitz_dashboard"
)

# Run transformation strategy with dashboard
from transformation import TransformationConfig
# ... create your transformation config ...
result, figures = run_transformation_with_dashboard(
    transformation_config=config,
    risk_target=0.05,
    save_path="./transformation_dashboard"
)
```

## Example Usage

### Testing the System

```bash
# Run the test script to verify everything works
python test_dashboard.py
```

### Running Examples

```bash
# Run the example script (demonstrates multiple strategies)
python dashboard_example.py
```

## Dashboard Output Structure

When you run a dashboard, it creates:

```
your_dashboard_folder/
├── main.png                    # Main overview dashboard
├── portfolio_evolution.png     # Detailed portfolio analysis
├── trades_analysis.png         # Trading activity analysis
├── weights_heatmap.png         # Portfolio weights visualization
└── transformation_tracking.png # Target vs actual paths (if applicable)
```

## What the Dashboard Shows

### For Any Strategy
- **Portfolio Performance**: Total return, Sharpe ratio, volatility, max drawdown
- **Portfolio Evolution**: Value over time, drawdowns, cumulative returns
- **Risk Management**: Rolling volatility vs targets, cash allocation
- **Trading Activity**: Daily volumes, turnover, most active assets
- **Portfolio Composition**: Weight evolution, concentration metrics

### For Transformation Strategies (Additional)
- **Path Tracking**: Target vs actual weight paths for key assets
- **Transformation Quality**: How well the strategy follows intended paths
- **Adaptation Behavior**: How the strategy responds to market conditions

## Interpreting the Visualizations

### Main Dashboard
- **Portfolio Value**: Shows overall performance and any major drawdowns
- **Returns Distribution**: Indicates return consistency and tail risk
- **Key Metrics Box**: Quick summary of performance statistics
- **Cash vs Invested**: Shows how much capital is deployed vs held in cash
- **Rolling Sharpe**: Indicates consistency of risk-adjusted returns
- **Trading Activity**: Higher spikes indicate more active rebalancing
- **Top Assets**: Shows which assets have the largest allocations

### Portfolio Evolution
- **Value + Drawdown**: Red areas show periods of losses from peak
- **Cumulative Returns**: Total strategy performance over time
- **Rolling Volatility**: Shows if strategy maintains target risk level
- **Cash Allocation**: Higher percentages indicate more conservative positioning

### Trades Analysis
- **Daily Volume**: Indicates trading intensity
- **Trade Distribution**: Shows typical trade sizes (centered around zero is normal)
- **Turnover Rate**: Higher values indicate more frequent rebalancing
- **Most Active Assets**: Identifies which assets are traded most frequently

### Weights Heatmap
- **Heatmap**: Red = negative weights (short), Blue = positive (long), White = no position
- **Concentration**: Lower values indicate more diversified portfolios

### Transformation Tracking (Transformation Strategies Only)
- **Target vs Actual**: Shows how closely the strategy follows intended paths
- **Divergence**: Large gaps indicate implementation challenges or market constraints
- **Adaptation**: How quickly the strategy adjusts to reach targets

## Customization

You can extend the dashboard by:

1. **Adding Custom Metrics**: Modify the `DetailedBacktestResult` class
2. **Creating New Visualizations**: Add methods to the `BacktestDashboard` class
3. **Strategy-Specific Plots**: Add conditional plots based on strategy type
4. **Interactive Dashboards**: Integrate with Plotly for interactive visualizations

## Requirements

- `matplotlib` for plotting
- `seaborn` for enhanced visualizations
- `pandas` and `numpy` for data handling
- Existing backtest infrastructure

## Notes

- The basic dashboard works with any existing `BacktestResult`
- The enhanced dashboard provides more detailed tracking at the cost of slightly longer runtimes
- All visualizations are saved as high-resolution PNG files
- The system automatically handles missing data and edge cases
- For transformation strategies, the dashboard shows both target paths and actual execution

## Troubleshooting

### Common Issues

1. **Missing Plots**: Make sure `matplotlib` backend is properly configured
2. **Memory Issues**: For very long backtests, consider reducing the tracking frequency
3. **Import Errors**: Ensure all required modules are in the Python path
4. **Data Alignment**: The system handles most alignment issues automatically

### Performance Tips

- Use `show=False` when generating many dashboards
- Save dashboards to different directories to avoid overwrites
- For batch processing, consider running dashboards in parallel 