import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
data_path = Path("data_ranycs")
prices = pd.read_csv(data_path / "prices_final.csv", index_col=0, parse_dates=True)

# Drop first date as done in backtest
prices = prices.iloc[1:]

# Calculate returns
returns = prices.pct_change().dropna()
print(f"Returns shape after dropna: {returns.shape}")

# Calculate covariance
covariance_df = returns.ewm(halflife=125).cov()
print(f"Covariance shape: {covariance_df.shape}")

# Check for NaNs in covariance
print(f"\n=== Covariance NaN Analysis ===")

# Get unique dates from the MultiIndex
dates = covariance_df.index.get_level_values(0).unique()
print(f"Number of unique dates in covariance: {len(dates)}")

# Check a few sample covariance matrices
sample_dates = dates[:5]
for date in sample_dates:
    cov_matrix = covariance_df.loc[date]
    nan_count = cov_matrix.isna().sum().sum()
    total_elements = cov_matrix.size
    print(f"Date {date}: {nan_count}/{total_elements} NaNs ({nan_count/total_elements*100:.1f}%)")

# Check if there are any dates with completely NaN covariance matrices
completely_nan_dates = []
for date in dates:
    cov_matrix = covariance_df.loc[date]
    if cov_matrix.isna().all().all():
        completely_nan_dates.append(date)

print(f"\nDates with completely NaN covariance matrices: {len(completely_nan_dates)}")
if completely_nan_dates:
    print(f"First few: {completely_nan_dates[:5]}")

# Check the backtest date range
lookback = 500
forward_smoothing = 5
indices = range(lookback, len(prices) - forward_smoothing)
backtest_days = [prices.index[t] for t in indices]

print(f"\n=== Backtest Date Range Analysis ===")
print(f"Backtest days: {len(backtest_days)}")
print(f"Date range: {backtest_days[0]} to {backtest_days[-1]}")

# Check which backtest days have NaNs in covariance
backtest_days_with_nans = []
for day in backtest_days:
    if day in dates:  # Make sure the day exists in covariance
        cov_matrix = covariance_df.loc[day]
        if cov_matrix.isna().any().any():
            backtest_days_with_nans.append(day)

print(f"Backtest days with NaNs in covariance: {len(backtest_days_with_nans)}")
if backtest_days_with_nans:
    print(f"First few: {backtest_days_with_nans[:5]}")

# Check the specific issue: look at the first few backtest days
print(f"\n=== First Few Backtest Days ===")
for i, day in enumerate(backtest_days[:10]):
    if day in dates:
        cov_matrix = covariance_df.loc[day]
        nan_count = cov_matrix.isna().sum().sum()
        total_elements = cov_matrix.size
        print(f"Day {i+1} ({day}): {nan_count}/{total_elements} NaNs ({nan_count/total_elements*100:.1f}%)")
        
        # Check if the matrix is positive definite
        try:
            # Replace NaNs with zeros for testing
            cov_matrix_clean = cov_matrix.fillna(0)
            eigenvals = np.linalg.eigvals(cov_matrix_clean.values)
            min_eigenval = eigenvals.min()
            print(f"  Min eigenvalue: {min_eigenval:.2e}")
            if min_eigenval <= 0:
                print(f"  WARNING: Matrix not positive definite!")
        except Exception as e:
            print(f"  ERROR checking eigenvalues: {e}")
    else:
        print(f"Day {i+1} ({day}): Not found in covariance DataFrame")

# Check if the issue is with the ewm calculation
print(f"\n=== EWM Analysis ===")
# Try a simpler covariance calculation
simple_cov = returns.cov()
print(f"Simple covariance shape: {simple_cov.shape}")
print(f"Simple covariance NaNs: {simple_cov.isna().sum().sum()}")

# Check if the issue is with the halflife parameter
print(f"\n=== Testing Different Halflife Values ===")
for halflife in [10, 50, 125, 250]:
    test_cov = returns.ewm(halflife=halflife).cov()
    nan_count = test_cov.isna().sum().sum()
    print(f"Halflife {halflife}: {nan_count} NaNs")

# Check if the issue is with the returns data itself
print(f"\n=== Returns Data Quality ===")
print(f"Returns has any NaNs: {returns.isna().any().any()}")
print(f"Returns has any Infs: {np.isinf(returns).any().any()}")
print(f"Returns has any negative values: {(returns < 0).any().any()}")

# Check for zero or negative prices that could cause issues
print(f"\n=== Price Data Quality ===")
print(f"Prices has any NaNs: {prices.isna().any().any()}")
print(f"Prices has any zeros: {(prices == 0).any().any()}")
print(f"Prices has any negative values: {(prices < 0).any().any()}")

# Check for assets with constant prices (which would cause division by zero in pct_change)
constant_price_assets = []
for col in prices.columns:
    if prices[col].std() == 0:
        constant_price_assets.append(col)

print(f"Assets with constant prices: {len(constant_price_assets)}")
if constant_price_assets:
    print(f"First few: {constant_price_assets[:5]}") 