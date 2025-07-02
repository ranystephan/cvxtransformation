# Transformation Experiments Guide

This guide describes three comprehensive experiments designed to test portfolio transformation strategies across different scenarios and market conditions.

## Overview

Based on the questions about transformations between asset groups, portfolio setup/liquidation, and volume-based execution, I've created three experiment files:

1. **Disjoint Asset Groups Experiment** (`disjoint_groups_experiment.py`)
2. **Portfolio Lifecycle Experiment** (`portfolio_lifecycle_experiment.py`) 
3. **Volume-Adaptive Experiment** (`volume_adaptive_experiment.py`)

## Experiment Descriptions

### 1. Disjoint Asset Groups Experiment

**Purpose**: Tests transformations between disjoint groups of assets with different risk characteristics.

**What it does**:
- Creates asset groups based on volatility, Sharpe ratio, and market correlation
- Tests transformations between groups (e.g., low-vol to high-vol assets)
- Compares different transformation policies across varying time periods
- Measures implementation quality and trading costs

**Key Scenarios**:
- Low volatility → High volatility assets (5% → 15% risk)
- High Sharpe → Low correlation assets (equal risk)

**Run it**:
```bash
cd experiments
python disjoint_groups_experiment.py
```

**Outputs**:
- Results saved to `./disjoint_groups_output/`
- CSV files with detailed metrics
- Dashboards for best-performing transformations

### 2. Portfolio Lifecycle Experiment

**Purpose**: Studies critical portfolio lifecycle events - setup, liquidation, and rebalancing.

**What it does**:
- Tests cash-to-portfolio transformations (setup)
- Tests portfolio-to-cash transformations (liquidation)
- Tests portfolio rebalancing scenarios
- Adapts execution based on transformation complexity
- Considers different market conditions (calm, normal, volatile)

**Key Scenarios**:
- Cash → Equal-weight portfolio
- Cash → Concentrated portfolio  
- Portfolio → Cash (liquidation)
- Concentrated → Diversified rebalancing

**Run it**:
```bash
cd experiments
python portfolio_lifecycle_experiment.py
```

**Outputs**:
- Results saved to `./portfolio_lifecycle_output/`
- Policy effectiveness summaries
- Implementation shortfall analysis

### 3. Volume-Adaptive Experiment

**Purpose**: Demonstrates volume-aware transformation execution that adapts to asset liquidity.

**What it does**:
- Classifies assets by volume/liquidity characteristics
- Adapts transformation periods based on volume profiles
- Tests different execution approaches
- Measures implementation quality vs. volume exposure

**Key Scenarios**:
- High-volume focused transformations
- Low-volume challenged transformations
- Mixed-volume portfolio transitions

**Run it**:
```bash
cd experiments
python volume_adaptive_experiment.py
```

**Outputs**:
- Results saved to `./volume_adaptive_output/`
- Volume-based effectiveness analysis
- Adaptive period recommendations

## Common Features Across All Experiments

### Transformation Policies Tested
- **Uniform**: Fixed daily transformation amounts
- **Dynamic Uniform**: Adjusts based on market conditions
- **Univariate Tracking**: Risk-aware with market impact consideration

### Metrics Analyzed
- **Sharpe Ratio**: Risk-adjusted returns
- **Turnover**: Trading activity intensity  
- **Implementation Shortfall**: Distance from target portfolio
- **Maximum Drawdown**: Risk management effectiveness
- **Leverage**: Portfolio leverage usage

### Output Structure
Each experiment creates:
- CSV files with detailed results
- Summary statistics by policy/scenario
- Visual dashboards for best results
- Implementation quality metrics

## Running All Experiments

To run all three experiments in sequence:

```bash
cd experiments

# Run disjoint groups experiment
echo "Running Disjoint Groups Experiment..."
python disjoint_groups_experiment.py

# Run portfolio lifecycle experiment  
echo "Running Portfolio Lifecycle Experiment..."
python portfolio_lifecycle_experiment.py

# Run volume-adaptive experiment
echo "Running Volume-Adaptive Experiment..."
python volume_adaptive_experiment.py

echo "All experiments completed!"
```

## Understanding the Results

### Key Questions Answered

1. **Which transformation policy works best?**
   - Look at Sharpe ratios and implementation quality across scenarios
   - Consider turnover costs vs. performance gains

2. **How do transformation periods affect outcomes?**
   - Compare results across different period lengths
   - Notice volume-dependent optimal periods

3. **What are the trade-offs?**
   - Fast execution vs. market impact
   - Risk targeting vs. implementation costs
   - Policy complexity vs. robustness

### Interpretation Tips

- **High Sharpe + Low Turnover**: Efficient transformation
- **Low Implementation Shortfall**: Good target tracking
- **Scenario-dependent results**: No one-size-fits-all policy
- **Volume considerations**: Liquidity matters for execution quality

## Customization

### Modifying Scenarios
Each experiment can be customized by editing the scenario definitions:

```python
scenarios = [
    {
        'name': 'Custom_Scenario',
        'description': 'Your custom description',
        'target_risk': 0.08,
        # ... other parameters
    }
]
```

### Adding New Policies
Create new transformation policies by extending the base classes:

```python
from transformation import TransformationPolicy

class CustomPolicy(TransformationPolicy):
    def __init__(self, custom_param):
        self.custom_param = custom_param
    
    def get_next_weights(self, current_weights, target_weights, day, **kwargs):
        # Your custom logic here
        return next_weights
```

### Adjusting Risk Targets
Modify risk targets based on your requirements:
- Conservative: 4-6% annual volatility
- Moderate: 8-10% annual volatility  
- Aggressive: 12-15% annual volatility

## Dependencies

Make sure you have these modules available:
- `backtest.py` - Data loading and basic backtest framework
- `transformation.py` - Core transformation policies and configuration
- `enhanced_backtest.py` - Enhanced backtesting with dashboards
- `transformations/` - Additional policy implementations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required modules are in the Python path
2. **Data Loading**: Check that data files exist in `data_ranycs/`
3. **Memory Issues**: Reduce the number of assets or scenarios for large datasets
4. **Slow Execution**: Start with shorter transformation periods for testing

### Performance Tips

- Run experiments on subsets first to validate
- Use parallel processing for multiple scenarios
- Cache intermediate results for repeated runs
- Monitor memory usage with large asset universes

## Next Steps

After running these experiments, consider:

1. **Analyzing sensitivity** to different market conditions
2. **Testing custom transformation policies** based on your specific needs
3. **Integrating actual volume data** for more realistic volume-adaptive strategies
4. **Extending to multi-asset class** transformations
5. **Adding transaction cost models** for more realistic implementation analysis 