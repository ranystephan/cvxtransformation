import pandas as pd
import numpy as np
from pathlib import Path

# Load the spread data
data_path = Path("data_ranycs")
spread = pd.read_csv(data_path / "spread_final.csv", index_col=0, parse_dates=True)

print("=== Spread Data Analysis ===")
print(f"Spread shape: {spread.shape}")
print(f"Date range: {spread.index[0]} to {spread.index[-1]}")

# Check for NaNs in spread
print(f"\n=== NaN Analysis ===")
nan_counts = spread.isna().sum()
print(f"NaN counts per asset in spread:")
print(nan_counts.describe())
print(f"\nAssets with NaNs: {nan_counts[nan_counts > 0].count()} out of {len(nan_counts)}")

# Check for NaNs by date
nan_by_date = spread.isna().sum(axis=1)
print(f"\nNaN counts per date in spread:")
print(nan_by_date.describe())
print(f"\nDates with NaNs: {nan_by_date[nan_by_date > 0].count()} out of {len(nan_by_date)}")

# Show first few dates with NaNs
if nan_by_date[nan_by_date > 0].count() > 0:
    print(f"\nFirst 10 dates with NaNs:")
    print(nan_by_date[nan_by_date > 0].head(10))

# Check for infinite values
inf_counts = np.isinf(spread).sum()
print(f"\nInf counts per asset in spread:")
print(inf_counts.describe())
print(f"\nAssets with Infs: {inf_counts[inf_counts > 0].count()} out of {len(inf_counts)}")

# Check specific problematic assets
if nan_counts.max() > 0:
    print(f"\nAssets with most NaNs:")
    print(nan_counts.nlargest(10))

# Check if there are any completely NaN rows
completely_nan_rows = spread.isna().all(axis=1)
print(f"\nCompletely NaN rows: {completely_nan_rows.sum()}")

# Check the specific date that caused the error (2023-02-16)
error_date = pd.Timestamp('2023-02-16')
if error_date in spread.index:
    print(f"\n=== Analysis of Error Date ({error_date}) ===")
    error_spread = spread.loc[error_date]
    error_nan_count = error_spread.isna().sum()
    print(f"NaNs on {error_date}: {error_nan_count}")
    
    if error_nan_count > 0:
        print(f"Assets with NaNs on {error_date}:")
        nan_assets = error_spread[error_spread.isna()].index.tolist()
        print(nan_assets[:10])  # Show first 10
        if len(nan_assets) > 10:
            print(f"... and {len(nan_assets) - 10} more")

# Check the backtest date range specifically
print(f"\n=== Backtest Date Range Analysis ===")
# The backtest starts at training_length=1250, so let's check from that point
training_length = 1250
backtest_spread = spread.iloc[training_length:]
print(f"Backtest spread shape: {backtest_spread.shape}")
print(f"Backtest date range: {backtest_spread.index[0]} to {backtest_spread.index[-1]}")

backtest_nan_by_date = backtest_spread.isna().sum(axis=1)
print(f"NaN counts per date in backtest period:")
print(backtest_nan_by_date.describe())
print(f"Dates with NaNs in backtest: {backtest_nan_by_date[backtest_nan_by_date > 0].count()} out of {len(backtest_nan_by_date)}")

if backtest_nan_by_date[backtest_nan_by_date > 0].count() > 0:
    print(f"\nFirst 10 backtest dates with NaNs:")
    print(backtest_nan_by_date[backtest_nan_by_date > 0].head(10))

# Check for negative or zero spreads (which might be problematic)
print(f"\n=== Spread Value Analysis ===")
print(f"Negative spreads: {(spread < 0).sum().sum()}")
print(f"Zero spreads: {(spread == 0).sum().sum()}")
print(f"Very small spreads (< 0.001): {(spread < 0.001).sum().sum()}")

# Show spread statistics
print(f"\nSpread statistics:")
print(spread.describe())

print(f"\n=== Summary ===")
print(f"Total NaNs in spread: {spread.isna().sum().sum()}")
print(f"Total Infs in spread: {np.isinf(spread).sum().sum()}")
print(f"Percentage of NaNs in spread: {spread.isna().sum().sum() / spread.size * 100:.2f}%") 