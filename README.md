# Portfolio Transformation Framework

This repository focuses on **portfolio transformation strategies** - techniques for efficiently transitioning from one portfolio composition to another over time.

The backtesting framework is adapted from [cvxgrp/markowitz-reference](https://github.com/cvxgrp/markowitz-reference).

## Overview

Portfolio transformation addresses the practical challenge of changing portfolio allocations while minimizing transaction costs and market impact. This framework implements and compares different transformation strategies.

## Data

The main dataset is in `data_ranycs/` and contains:

- **Prices** (`prices_final.csv`): 235 assets over 2,502 trading days (July 2015 - June 2025)
- **Bid-Ask Spreads** (`spread_final.csv`): Transaction cost data for all assets
- **Volume** (`volume_final.csv`): Trading volume data
- **Risk-Free Rate** (`rf.csv`): Daily risk-free rates
- **Short Fees** (`short_fee_data_cleaned.csv`): Cost of shorting data

All data has been cleaned and aligned across the same time period and asset universe.

## Usage

### Basic Example
```python
from experiments.transformation import run_transformation_backtest
from experiments.backtest import load_data
import numpy as np

# Load data
prices, _, _, _, _ = load_data()
n_assets = prices.shape[1]

# Define a simple rebalancing: concentrated → diversified
w_initial = np.zeros(n_assets)
w_initial[:5] = 0.02  # 10% in top 5 assets

w_target = np.zeros(n_assets) 
w_target[:50] = 0.002  # 10% spread across top 50 assets

# Run transformation
result = run_transformation_backtest(
    policy='uniform',
    w_initial=w_initial,
    w_target=w_target,
    total_days=20
)
```

### Running Experiments
```bash
# Test setup
python test_experiments.py

# Individual experiments
cd experiments
python volume_adaptive_experiment.py
python disjoint_groups_experiment.py
python portfolio_lifecycle_experiment.py

# All experiments
python run_all_experiments.py
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
- CVXPY for optimization (uses default solvers)

## Experiments

The `experiments/` directory contains three main studies:

1. **Disjoint Asset Groups**: Transformations between different risk characteristics
2. **Portfolio Lifecycle**: Setup, liquidation, and rebalancing scenarios  
3. **Volume-Adaptive**: Execution strategies based on asset liquidity

See `experiments/EXPERIMENT_GUIDE.md` for detailed instructions.
