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
