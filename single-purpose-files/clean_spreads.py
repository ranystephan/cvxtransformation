import pandas as pd
import numpy as np
from pathlib import Path

# Load the spread data
data_path = Path("data_ranycs")
spread = pd.read_csv(data_path / "spread_final.csv", index_col=0, parse_dates=True)

print("=== Original Spread Data Analysis ===")
print(f"Spread shape: {spread.shape}")
print(f"Negative spreads: {(spread < 0).sum().sum()}")
print(f"Zero spreads: {(spread == 0).sum().sum()}")
print(f"NaNs: {spread.isna().sum().sum()}")

# Step 1: Take absolute values of all spreads
spread_clean = spread.abs()
print(f"\n=== After Taking Absolute Values ===")
print(f"Negative spreads: {(spread_clean < 0).sum().sum()}")
print(f"Zero spreads: {(spread_clean == 0).sum().sum()}")

# Step 2: Analyze NaN patterns to decide on removal vs filling
print(f"\n=== NaN Analysis for Decision Making ===")

# Count NaNs per asset
nan_counts = spread_clean.isna().sum()
print(f"Assets with NaNs: {nan_counts[nan_counts > 0].count()} out of {len(nan_counts)}")

# Show assets with most NaNs
print(f"\nAssets with most NaNs:")
print(nan_counts.nlargest(15))

# Check which assets have NaNs in the backtest period
training_length = 1250
backtest_spread = spread_clean.iloc[training_length:]
backtest_nan_counts = backtest_spread.isna().sum()
backtest_assets_with_nans = backtest_nan_counts[backtest_nan_counts > 0]

print(f"\nAssets with NaNs in backtest period: {len(backtest_assets_with_nans)}")
if len(backtest_assets_with_nans) > 0:
    print("Assets with NaNs in backtest:")
    for asset, count in backtest_assets_with_nans.items():
        print(f"  {asset}: {count} NaNs")

# Analyze the impact of removing assets
print(f"\n=== Impact Analysis ===")

# Option 1: Remove assets with any NaNs in backtest period
assets_to_remove_backtest = backtest_assets_with_nans.index.tolist()
print(f"Option 1: Remove {len(assets_to_remove_backtest)} assets with NaNs in backtest")
print(f"Remaining assets: {len(spread_clean.columns) - len(assets_to_remove_backtest)}")

# Option 2: Remove assets with more than X NaNs total
thresholds = [1, 5, 10, 50, 100]
for threshold in thresholds:
    assets_to_remove_threshold = nan_counts[nan_counts > threshold].index.tolist()
    print(f"Option 2a: Remove assets with >{threshold} NaNs: {len(assets_to_remove_threshold)} assets")
    print(f"  Remaining assets: {len(spread_clean.columns) - len(assets_to_remove_threshold)}")

# Option 3: Fill NaNs with different strategies
print(f"\n=== Fill Strategies Analysis ===")

# Strategy 1: Fill with median per asset
spread_filled_median = spread_clean.copy()
for col in spread_filled_median.columns:
    median_val = spread_filled_median[col].median()
    spread_filled_median[col].fillna(median_val, inplace=True)

# Strategy 2: Fill with 5% (typical high spread)
spread_filled_5pct = spread_clean.copy()
spread_filled_5pct.fillna(0.05, inplace=True)

# Strategy 3: Fill with max spread per asset
spread_filled_max = spread_clean.copy()
for col in spread_filled_max.columns:
    max_val = spread_filled_max[col].max()
    spread_filled_max[col].fillna(max_val, inplace=True)

# Compare the strategies
print(f"Original NaNs: {spread_clean.isna().sum().sum()}")
print(f"After median fill: {spread_filled_median.isna().sum().sum()}")
print(f"After 5% fill: {spread_filled_5pct.isna().sum().sum()}")
print(f"After max fill: {spread_filled_max.isna().sum().sum()}")

# Check spread statistics after each strategy
print(f"\n=== Spread Statistics Comparison ===")
print("Original (with NaNs):")
print(spread_clean.describe().loc[['mean', 'std', 'min', 'max']])

print("\nAfter median fill:")
print(spread_filled_median.describe().loc[['mean', 'std', 'min', 'max']])

print("\nAfter 5% fill:")
print(spread_filled_5pct.describe().loc[['mean', 'std', 'min', 'max']])

print("\nAfter max fill:")
print(spread_filled_max.describe().loc[['mean', 'std', 'min', 'max']])

# Recommendation
print(f"\n=== Recommendation ===")
print("Based on the analysis:")
print("1. Only 5 assets have NaNs in the backtest period")
print("2. Removing these 5 assets would preserve 464/469 assets (98.9%)")
print("3. Filling with median values would preserve all assets but may introduce bias")
print("4. Filling with 5% is conservative but may overestimate spreads")

print(f"\nRecommended approach:")
print("- Take absolute values of all spreads (done)")
print("- Remove the 5 assets with NaNs in backtest period")
print("- This preserves 98.9% of assets while ensuring no NaNs in backtest")

# Create the cleaned spread file
print(f"\n=== Creating Cleaned Spread File ===")
spread_cleaned = spread_clean.drop(columns=assets_to_remove_backtest)
print(f"Cleaned spread shape: {spread_cleaned.shape}")
print(f"NaNs in cleaned spread: {spread_cleaned.isna().sum().sum()}")

# Save the cleaned spread file
output_path = data_path / "spread_cleaned.csv"
spread_cleaned.to_csv(output_path)
print(f"Saved cleaned spread to: {output_path}")

# Also create a version with median fill for comparison
spread_median_filled = spread_clean.copy()
for col in spread_median_filled.columns:
    median_val = spread_median_filled[col].median()
    spread_median_filled[col].fillna(median_val, inplace=True)

output_path_filled = data_path / "spread_median_filled.csv"
spread_median_filled.to_csv(output_path_filled)
print(f"Saved median-filled spread to: {output_path_filled}") 