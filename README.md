# CVX Portfolio Transformation Framework

This repository extends the [cvxgrp/markowitz-reference](https://github.com/cvxgrp/markowitz-reference) implementation with advanced portfolio transformation strategies, comprehensive backtesting infrastructure, and visualization tools. It accompanies the paper [Markowitz Portfolio Construction at Seventy](https://web.stanford.edu/~boyd/papers/markowitz.html) and adds significant new functionality for portfolio transition optimization.

## Key Features

### 🎯 Portfolio Transformation Strategies
- **Uniform Transformation**: Linear interpolation between initial and target weights
- **Dynamic Uniform Transformation**: Adaptive rebalancing that accounts for market drift
- **Univariate Scalar Tracking**: Optimization-based approach balancing tracking error with transaction costs
- **Custom Policy Framework**: Extensible architecture for implementing new transformation strategies

### 📊 Comprehensive Backtesting Infrastructure
- **Enhanced Backtesting**: Detailed tracking of portfolio evolution, trades, and performance metrics
- **Interactive Dashboards**: Rich visualizations including portfolio evolution, trading analysis, and weight heatmaps
- **Parameter Tuning**: Systematic optimization of strategy parameters with extensive results analysis
- **Yearly Rebalancing**: Long-term strategy evaluation with periodic re-optimization

### 🔬 Experimental Framework
- **Scaling Studies**: Performance analysis across different portfolio sizes and market conditions
- **Comparative Analysis**: Head-to-head comparison of different transformation approaches
- **Transaction Cost Analysis**: Detailed modeling of bid-ask spreads, market impact, and execution costs
- **Risk Management**: Volatility targeting and drawdown control mechanisms

## Quick Start

### Basic Setup

```bash
# Clone the repository
git clone <repository-url>
cd cvxtransformation

# Install dependencies
pip install -r requirements.txt

# Run all experiments
make experiments
```

### Simple Portfolio Transformation Example

```python
from experiments.transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    UnivariateScalarTrackingPolicy,
    run_transformation_backtest
)
import numpy as np

# Load data
from experiments.backtest import load_data
prices, _, _, _, _ = load_data()
n_assets = prices.shape[1]

# Define transformation: rebalance from 60/40 to 40/60
w_initial = np.array([0.6, 0.4] + [0.0] * (n_assets - 2))
w_target = np.array([0.4, 0.6] + [0.0] * (n_assets - 2))

# Test different transformation policies
policies = {
    'Uniform': UniformTransformationPolicy(),
    'Dynamic': DynamicUniformTransformationPolicy(),
    'Univariate': UnivariateScalarTrackingPolicy(risk_aversion=1.0)
}

for name, policy in policies.items():
    result = run_transformation_backtest(
        policy=policy,
        w_initial=w_initial,
        w_target=w_target,
        total_days=20,
        risk_target=0.10,
        verbose=True
    )
    print(f"{name}: Sharpe={result.sharpe:.2f}, Turnover={result.turnover:.2f}")
```

### Dashboard Generation

```python
from experiments.enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

# Run enhanced backtest with detailed tracking
result = run_enhanced_backtest(
    strategy=my_strategy,
    risk_target=0.05,
    strategy_name="My Strategy",
    verbose=True
)

# Create comprehensive dashboard
figures = create_dashboard_from_enhanced_backtest(
    result=result,
    save_path="./my_dashboard",
    show=True
)
```

## Portfolio Transformation Strategies

### 1. Uniform Transformation Policy
**Best for**: Predictable execution with minimal market impact
- Executes trades at a constant rate
- Simple linear interpolation between initial and target weights
- Static "open-loop" approach

```python
policy = UniformTransformationPolicy()
```

### 2. Dynamic Uniform Transformation Policy
**Best for**: Adaptive execution that accounts for market drift
- Recalculates required trades each day based on current portfolio state
- Closed-loop approach that ensures reaching the target
- Adapts to market movements and unexpected deviations

```python
policy = DynamicUniformTransformationPolicy()
```

### 3. Univariate Scalar Tracking Policy
**Best for**: Optimal execution with transaction cost considerations
- Uses convex optimization (CVXPY) to balance tracking error with costs
- Asset-by-asset optimization considering bid-ask spreads and market impact
- Most sophisticated approach for cost-sensitive transformations

```python
policy = UnivariateScalarTrackingPolicy(
    risk_aversion=1.0,      # Higher = more risk averse
    impact_coeff=0.01,      # Market impact coefficient
    max_position=0.1,       # Position limits
    min_position=-0.1
)
```

## Dashboard System

The dashboard system provides comprehensive visualization and analysis tools:

### Visualizations Generated
- **Main Dashboard**: Portfolio performance, returns distribution, key metrics
- **Portfolio Evolution**: Value over time, drawdowns, rolling volatility
- **Trades Analysis**: Trading volume, turnover, most active assets
- **Weights Heatmap**: Portfolio composition over time
- **Transformation Tracking**: Target vs actual paths for transformation strategies

### Example Usage
```python
from experiments.dashboard import create_dashboard_from_backtest
from experiments.backtest import run_backtest

result = run_backtest(my_strategy, risk_target=0.05)
figures = create_dashboard_from_backtest(
    backtest_result=result,
    strategy_name="My Strategy",
    save_path="./dashboard_output"
)
```

## Experimental Framework

### Parameter Tuning
```bash
# Run systematic parameter optimization
python experiments/tuning_example.py

# Analyze tuning results
python experiments/tuning_utils.py
```

### Scaling Studies
```bash
# Test performance on different portfolio sizes
python experiments/scaling_small.py  # Small portfolios
python experiments/scaling_large.py  # Large portfolios
```

### Yearly Rebalancing
```bash
# Long-term strategy evaluation
python experiments/yearly_metrics.py
python experiments/tuning_yearly_retune.py
```

## Data and Reproducibility

### Data Structure
- **Prices**: Daily asset prices (tickers obfuscated per data provider terms)
- **Risk-free Rate**: Daily risk-free rates for Sharpe ratio calculations
- **Spreads**: Bid-ask spreads for transaction cost modeling
- **Volumes**: Share volumes for market impact calculations

### Reproducibility
- **Python Version**: 3.10.13
- **Main Packages**: Specified in `requirements.txt`
- **Frozen Environment**: Complete dependency tree in `requirements_frozen.txt`
- **Solver**: MOSEK (license required) - apply for [Trial License](https://www.mosek.com/try/)

## File Structure

```
cvxtransformation/
├── experiments/
│   ├── transformations/          # Transformation strategy implementations
│   │   ├── uniform_policy.py     # Uniform transformation
│   │   ├── univariate_policy.py  # Univariate scalar tracking
│   │   └── README.md             # Detailed transformation docs
│   ├── transformation.py         # Main transformation interface
│   ├── backtest.py              # Core backtesting infrastructure
│   ├── enhanced_backtest.py     # Enhanced tracking and analysis
│   ├── dashboard.py             # Visualization dashboard
│   ├── tuning_utils.py          # Parameter optimization tools
│   ├── scaling_*.py             # Scaling studies
│   └── yearly_*.py              # Long-term analysis
├── data/                        # Market data
└── requirements.txt             # Dependencies
```

## Advanced Usage

### Custom Transformation Policies
```python
from experiments.transformations.uniform_policy import TransformationPolicy

class MyCustomPolicy(TransformationPolicy):
    def get_target_weights(self, inputs, config, current_day):
        # Your custom transformation logic here
        return target_weights

# Use with existing infrastructure
policy = MyCustomPolicy()
result = run_transformation_backtest(policy, w_initial, w_target, total_days)
```

### Integration with Existing Strategies
```python
# Combine transformation with Markowitz optimization
from experiments.tuning_utils import full_markowitz

def transformation_markowitz_strategy(inputs):
    # Get transformation target for this day
    w_transform_target = policy.get_target_weights(inputs, config, current_day)
    
    # Use Markowitz optimization around transformation target
    # ... custom logic combining both approaches ...
    
    return full_markowitz(inputs, hyperparams, targets)
```

## Performance Considerations

- **Uniform Policies**: Very fast, minimal computation overhead
- **Dynamic Uniform**: Fast daily recalculation
- **Univariate Scalar Tracking**: Slower due to optimization, but most sophisticated
- **Large Portfolios**: Consider pre-computing paths or parallelizing asset optimization

## Testing

```bash
# Run transformation strategy tests
python experiments/test_transformations.py

# Run dashboard tests
python experiments/test_dashboard.py

# Run full experimental suite
make experiments
```

## Citation

If you use this work in your research, please cite both the original paper and this extension:

### Original Paper
```bibtex
@article{boyd2024markowitz,
      title={Markowitz Portfolio Construction at Seventy},
      author={S. Boyd and K. Johansson and R. Kahn and P. Schiele and T. Schmelzer},
      journal={Journal of Portfolio Management},
      volume={50},
      number={8},
      pages={117--160},
      year={2024}
}
```

### arXiv Version
```bibtex
@misc{boyd2024markowitz,
      title={Markowitz Portfolio Construction at Seventy},
      author={Stephen Boyd and Kasper Johansson and Ronald Kahn and Philipp Schiele and Thomas Schmelzer},
      year={2024},
      doi = {10.48550/arXiv.2401.05080},
      url = {https://arxiv.org/abs/2401.05080}
}
```

## License

This project extends the original [cvxgrp/markowitz-reference](https://github.com/cvxgrp/markowitz-reference) implementation. Please see the LICENSE file for details.

## Contributing

We welcome contributions! Please see the individual module README files for detailed documentation on specific components:

- [`experiments/transformations/README.md`](experiments/transformations/README.md) - Detailed transformation strategies documentation
- [`experiments/DASHBOARD_README.md`](experiments/DASHBOARD_README.md) - Dashboard system documentation

## Requirements

- Python 3.10+
- CVXPY for optimization
- MOSEK solver (license required)
- Standard scientific computing stack (NumPy, Pandas, Matplotlib, Seaborn)
- See `requirements.txt` for complete dependency list
