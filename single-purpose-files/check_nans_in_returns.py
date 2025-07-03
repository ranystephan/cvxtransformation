import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
data_path = Path("data_ranycs")
prices = pd.read_csv(data_path / "prices_final.csv", index_col=0, parse_dates=True)
spread = pd.read_csv(data_path / "spread_final.csv", index_col=0, parse_dates=True)
volume = pd.read_csv(data_path / "volume_final.csv", index_col=0, parse_dates=True)
rf_data = pd.read_csv(data_path / "rf.csv", index_col=0, parse_dates=True)
rf = rf_data['rf']

print("=== Data Loading Info ===")
print(f"Prices shape: {prices.shape}")
print(f"Spread shape: {spread.shape}")
print(f"Volume shape: {volume.shape}")
print(f"RF shape: {rf.shape}")

# Drop first date as done in backtest
prices = prices.iloc[1:]
spread = spread.iloc[1:]
volume = volume.iloc[1:]
rf = rf.iloc[1:]

print(f"\nAfter dropping first date:")
print(f"Prices shape: {prices.shape}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# Calculate returns
returns = prices.pct_change()
print(f"\n=== Returns Analysis ===")
print(f"Returns shape: {returns.shape}")

# Check for NaNs in returns
nan_counts = returns.isna().sum()
print(f"\nNaN counts per asset in returns:")
print(nan_counts.describe())
print(f"\nAssets with NaNs: {nan_counts[nan_counts > 0].count()} out of {len(nan_counts)}")

# Check for NaNs by date
nan_by_date = returns.isna().sum(axis=1)
print(f"\nNaN counts per date in returns:")
print(nan_by_date.describe())
print(f"\nDates with NaNs: {nan_by_date[nan_by_date > 0].count()} out of {len(nan_by_date)}")

# Show first few dates with NaNs
if nan_by_date[nan_by_date > 0].count() > 0:
    print(f"\nFirst 10 dates with NaNs:")
    print(nan_by_date[nan_by_date > 0].head(10))

# Check for infinite values
inf_counts = np.isinf(returns).sum()
print(f"\nInf counts per asset in returns:")
print(inf_counts.describe())
print(f"\nAssets with Infs: {inf_counts[inf_counts > 0].count()} out of {len(inf_counts)}")

# Check specific problematic assets
if nan_counts.max() > 0:
    print(f"\nAssets with most NaNs:")
    print(nan_counts.nlargest(10))

# Check if there are any completely NaN rows
completely_nan_rows = returns.isna().all(axis=1)
print(f"\nCompletely NaN rows: {completely_nan_rows.sum()}")

# Check synthetic returns
print(f"\n=== Synthetic Returns Analysis ===")
from experiments.utils import synthetic_returns

synth_returns = synthetic_returns(prices, information_ratio=0.15, forward_smoothing=5)
synth_nan_counts = synth_returns.isna().sum()
print(f"NaN counts per asset in synthetic returns:")
print(synth_nan_counts.describe())
print(f"Assets with NaNs in synthetic returns: {synth_nan_counts[synth_nan_counts > 0].count()}")

# Check means calculation
means = synth_returns.shift(-1).dropna()
print(f"\nMeans shape: {means.shape}")
means_nan_counts = means.isna().sum()
print(f"NaN counts per asset in means:")
print(means_nan_counts.describe())

# Check covariance calculation
returns_clean = returns.dropna()
print(f"\n=== Covariance Analysis ===")
print(f"Returns after dropna shape: {returns_clean.shape}")
covariance_df = returns_clean.ewm(halflife=125).cov()
print(f"Covariance shape: {covariance_df.shape}")

# Check if covariance has any NaNs
if hasattr(covariance_df, 'isna'):
    cov_nan_counts = covariance_df.isna().sum()
    print(f"NaN counts in covariance: {cov_nan_counts.sum()}")
else:
    print("Covariance is a MultiIndex DataFrame, checking for NaNs in values...")
    # For MultiIndex DataFrames, we need to check differently
    sample_cov = covariance_df.loc[covariance_df.index.get_level_values(0)[0]]
    print(f"Sample covariance shape: {sample_cov.shape}")
    print(f"Sample covariance has NaNs: {sample_cov.isna().sum().sum()}")

print(f"\n=== Summary ===")
print(f"Total NaNs in returns: {returns.isna().sum().sum()}")
print(f"Total Infs in returns: {np.isinf(returns).sum().sum()}")
print(f"Percentage of NaNs in returns: {returns.isna().sum().sum() / returns.size * 100:.2f}%") 