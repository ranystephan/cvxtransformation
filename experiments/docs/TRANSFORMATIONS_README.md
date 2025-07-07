# Portfolio Transformation Strategies

This module implements various portfolio transformation strategies that work with the existing Markowitz backtesting infrastructure. These strategies handle the gradual transition from an initial portfolio to a target portfolio over a specified time period, considering transaction costs and market dynamics.

## Overview

Portfolio transformation is a critical problem in quantitative finance - how do you optimally transition from one portfolio to another while minimizing costs and risks? This module provides several approaches:

1. **Uniform Transformation** - Execute a constant fraction of the total transformation each day
2. **Dynamic Uniform Transformation** - Adjust the transformation path based on current portfolio state
3. **Univariate Scalar Tracking** - Use optimization to balance tracking error with transaction costs on an asset-by-asset basis

## Architecture

The transformation strategies are designed to integrate seamlessly with the existing backtesting infrastructure:

```
backtest.py (existing) 
    ↓
transformation.py (main interface)
    ↓
transformations/ (strategy implementations)
    ├── uniform_policy.py
    ├── univariate_policy.py
    └── __init__.py
```

## Quick Start

### Basic Usage

```python
from experiments.transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    UnivariateScalarTrackingPolicy,
    run_transformation_backtest
)
import numpy as np

# Load data to get number of assets
from experiments.backtest import load_data
prices, _, _, _, _ = load_data()
n_assets = prices.shape[1]

# Define transformation: short GOOG, long AAPL+MSFT
w_initial = np.zeros(n_assets)
w_initial[0] = -0.05  # Short 5% in first asset

w_target = np.zeros(n_assets)
w_target[1] = 0.025   # Long 2.5% in second asset  
w_target[2] = 0.025   # Long 2.5% in third asset

# Run transformation over 20 days
total_days = 20

# Test uniform policy
uniform_policy = UniformTransformationPolicy()
result = run_transformation_backtest(
    policy=uniform_policy,
    w_initial=w_initial,
    w_target=w_target,
    total_days=total_days,
    risk_target=0.10,  # 10% annual volatility target
    verbose=True
)

print(f"Sharpe Ratio: {result.sharpe:.2f}")
print(f"Annual Return: {result.mean_return:.2%}")
print(f"Annual Volatility: {result.volatility:.2%}")
```

### Integration with Existing Backtesting

```python
from experiments.backtest import run_backtest
from experiments.transformation import create_transformation_strategy, TransformationConfig

# Create a transformation strategy that works with run_backtest()
policy = DynamicUniformTransformationPolicy()
config = TransformationConfig(w_initial, w_target, total_days)
strategy = create_transformation_strategy(policy, config)

# Use with existing backtesting infrastructure
result = run_backtest(
    strategy=strategy,
    risk_target=0.10 / np.sqrt(252),  # Daily risk target
    verbose=True
)
```

## Transformation Strategies

### 1. Uniform Transformation Policy

**Purpose**: Execute trades at a constant rate, ignoring market drift.

**Characteristics**:
- Static, "open-loop" policy
- Simple linear interpolation between initial and target weights
- Predictable execution schedule
- May deviate from target if market moves significantly

**Formula**: `w(t) = w_initial + (t/T) * (w_target - w_initial)`

**Use When**: You want predictable, systematic execution and market drift is minimal.

```python
policy = UniformTransformationPolicy()
```

### 2. Dynamic Uniform Transformation Policy

**Purpose**: Adjust transformation path based on current portfolio state.

**Characteristics**:
- Dynamic, "closed-loop" policy  
- Recalculates required trades each day
- Accounts for market drift and deviations
- More adaptive but potentially higher turnover

**Formula**: `w(t+1) = w_current + (w_target - w_current) / (T - t)`

**Use When**: Market drift is significant or you want to ensure reaching the target.

```python
policy = DynamicUniformTransformationPolicy()
```

### 3. Univariate Scalar Tracking Policy

**Purpose**: Optimize trades considering transaction costs and risk on an asset-by-asset basis.

**Characteristics**:
- Uses convex optimization (CVXPY)
- Balances tracking error with transaction costs
- Asset-by-asset optimization (univariate)
- Considers bid-ask spreads and market impact
- Most sophisticated but computationally intensive

**Objective**: For each asset i, minimize:
```
risk_aversion * σ²ᵢ * (wᵢ - wᵢᵗᵃʳᵍᵉᵗ)² + spread_cost * |zᵢ| + impact_cost * |zᵢ|^1.5
```

**Use When**: Transaction costs are significant and you want optimized execution.

```python
policy = UnivariateScalarTrackingPolicy(
    risk_aversion=1.0,      # Higher = more risk averse
    impact_coeff=0.01,      # Market impact coefficient
    max_position=0.1,       # Position limits
    min_position=-0.1
)
```

## Configuration Options

### TransformationConfig

```python
@dataclass
class TransformationConfig:
    w_initial: np.ndarray           # Initial portfolio weights
    w_target: np.ndarray            # Target portfolio weights  
    total_days: int                 # Transformation period
    participation_limit: float = 0.05  # Max % of daily volume
```

### Policy Parameters

**UnivariateScalarTrackingPolicy**:
- `risk_aversion`: Controls trade-off between tracking error and costs
- `impact_coeff`: Market impact cost parameter  
- `max_position` / `min_position`: Position size limits

## Testing and Validation

Run the test suite to validate functionality:

```bash
cd experiments/
python test_transformations.py
```

The test suite includes:
- Basic functionality tests
- Individual policy tests  
- Policy comparison
- Integration with existing infrastructure

## Examples and Use Cases

### Example 1: Simple Rebalancing

```python
# Rebalance from 60/40 to 40/60 over 2 weeks
w_initial = np.array([0.6, 0.4] + [0.0] * (n_assets - 2))
w_target = np.array([0.4, 0.6] + [0.0] * (n_assets - 2))

policy = DynamicUniformTransformationPolicy()
result = run_transformation_backtest(
    policy=policy,
    w_initial=w_initial,
    w_target=w_target,
    total_days=14
)
```

### Example 2: Unwinding Short Positions

```python
# Unwind short position gradually to minimize market impact
w_initial = np.zeros(n_assets)
w_initial[0] = -0.20  # Large short position

w_target = np.zeros(n_assets)  # Close to cash

policy = UnivariateScalarTrackingPolicy(risk_aversion=2.0)  # Conservative
result = run_transformation_backtest(
    policy=policy,
    w_initial=w_initial,
    w_target=w_target,
    total_days=30  # Longer period for large position
)
```

### Example 3: Factor Rotation

```python
# Rotate from value to growth factors
w_initial = np.zeros(n_assets)
w_initial[:5] = 0.02  # Value stocks

w_target = np.zeros(n_assets)  
w_target[5:10] = 0.02  # Growth stocks

# Compare different approaches
policies = {
    'Uniform': UniformTransformationPolicy(),
    'Dynamic': DynamicUniformTransformationPolicy(),
    'Univariate': UnivariateScalarTrackingPolicy()
}

for name, policy in policies.items():
    result = run_transformation_backtest(policy, w_initial, w_target, 15)
    print(f"{name}: Sharpe={result.sharpe:.2f}, Turnover={result.turnover:.2f}")
```

## Extensions and Customization

### Creating Custom Policies

```python
from experiments.transformations.uniform_policy import TransformationPolicy

class MyCustomPolicy(TransformationPolicy):
    def get_target_weights(self, inputs, config, current_day):
        # Your custom logic here
        # Access: inputs.prices, inputs.volas, inputs.spread, etc.
        # Return: target weights for this day
        return target_weights
```

### Integration with Existing Strategies

```python
# Combine with Markowitz optimization
from experiments.tuning_utils import full_markowitz, HyperParameters, Targets

def transformation_markowitz_strategy(inputs):
    # First, determine transformation target for this day
    policy = DynamicUniformTransformationPolicy()
    config = TransformationConfig(w_initial, w_target, total_days)
    w_transform_target = policy.get_target_weights(inputs, config, current_day)
    
    # Then, use Markowitz optimization around the transformation target
    hyperparams = HyperParameters(1, 1, 0.005, 0.0005, 0.05)
    targets = Targets(0.1, 1.6, 0.1/np.sqrt(252))
    
    # Modify targets to bias toward transformation target
    # ... custom logic ...
    
    return full_markowitz(inputs, hyperparams, targets)
```

## Performance Considerations

- **Uniform policies**: Very fast, minimal computation
- **Dynamic uniform**: Fast, simple recalculation each day  
- **Univariate scalar tracking**: Slower due to optimization, but more sophisticated

For large portfolios or high-frequency rebalancing, consider:
- Using uniform policies for speed
- Pre-computing univariate paths offline
- Parallelizing asset-by-asset optimization

## References

The transformation strategies implement concepts from:
- Portfolio transformation literature
- Optimal execution algorithms
- Transaction cost analysis
- Market microstructure research

For more details on the theoretical foundations, see the accompanying research papers. 