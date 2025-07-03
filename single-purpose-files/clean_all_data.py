import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
data_path = Path("data_ranycs")
prices = pd.read_csv(data_path / "prices_final.csv", index_col=0, parse_dates=True)
spread = pd.read_csv(data_path / "spread_cleaned.csv", index_col=0, parse_dates=True)
volume = pd.read_csv(data_path / "volume_final.csv", index_col=0, parse_dates=True)

print("=== Data Consistency Check ===")
print(f"Prices shape: {prices.shape}")
print(f"Spread cleaned shape: {spread.shape}")
print(f"Volume shape: {volume.shape}")

# Get the assets that are in the cleaned spread file
spread_assets = set(spread.columns)
prices_assets = set(prices.columns)
volume_assets = set(volume.columns)

print(f"\nAssets in spread_cleaned: {len(spread_assets)}")
print(f"Assets in prices_final: {len(prices_assets)}")
print(f"Assets in volume_final: {len(volume_assets)}")

# Find assets that need to be removed from prices and volume
assets_to_remove = prices_assets - spread_assets
print(f"\nAssets to remove from prices/volume: {len(assets_to_remove)}")
if len(assets_to_remove) > 0:
    print("Assets to remove:", sorted(assets_to_remove))

# Clean prices and volume to match spread_cleaned
prices_cleaned = prices.drop(columns=assets_to_remove)
volume_cleaned = volume.drop(columns=assets_to_remove)

print(f"\nAfter cleaning:")
print(f"Prices cleaned shape: {prices_cleaned.shape}")
print(f"Volume cleaned shape: {volume_cleaned.shape}")

# Verify consistency
prices_assets_cleaned = set(prices_cleaned.columns)
volume_assets_cleaned = set(volume_cleaned.columns)

print(f"\nConsistency check:")
print(f"Prices assets == Spread assets: {prices_assets_cleaned == spread_assets}")
print(f"Volume assets == Spread assets: {volume_assets_cleaned == spread_assets}")

# Save the cleaned files
prices_output_path = data_path / "prices_cleaned.csv"
volume_output_path = data_path / "volume_cleaned.csv"

prices_cleaned.to_csv(prices_output_path)
volume_cleaned.to_csv(volume_output_path)

print(f"\nSaved cleaned files:")
print(f"  {prices_output_path}")
print(f"  {volume_output_path}")

# Final verification
print(f"\n=== Final Data Summary ===")
print(f"All files now have {len(spread_assets)} assets")
print(f"Date range: {prices_cleaned.index[0]} to {prices_cleaned.index[-1]}")
print(f"Total dates: {len(prices_cleaned)}")

# Check for any remaining NaNs
print(f"\n=== NaN Check ===")
print(f"NaNs in prices_cleaned: {prices_cleaned.isna().sum().sum()}")
print(f"NaNs in spread_cleaned: {spread.isna().sum().sum()}")
print(f"NaNs in volume_cleaned: {volume_cleaned.isna().sum().sum()}") 