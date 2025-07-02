import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('experiments')
from backtest import load_data, run_backtest
from scaling_small import parameter_scaling_markowitz

# Load data
prices, spread, rf, volume = load_data()
training_length = 1250
prices, spread, rf = (
    prices.iloc[training_length:],
    spread.iloc[training_length:],
    rf.iloc[training_length:],
)

n_assets = prices.shape[1]
lookback = 500
forward_smoothing = 5

# Calculate returns and means
returns = prices.pct_change().dropna()
means = (
    pd.read_csv(Path("data_ranycs") / "prices_final.csv", index_col=0, parse_dates=True)
    .iloc[1:]  # Drop first date
    .pipe(lambda df: df.pct_change().dropna())
    .pipe(lambda df: df.rolling(forward_smoothing).mean().shift(-(forward_smoothing - 1)))
    .dropna()
    .iloc[training_length-1:]  # Align with training data
)

# Calculate covariance
covariance_df = returns.ewm(halflife=125).cov()

# Get indices and days
indices = range(lookback, len(prices) - forward_smoothing)
days = [prices.index[t] for t in indices]

# Filter out days with NaN covariance matrices
valid_days = []
for day in days:
    if day in covariance_df.index.get_level_values(0):
        cov_matrix = covariance_df.loc[day].values
        if not np.isnan(cov_matrix).any():
            valid_days.append(day)

indices = [i for i in indices if prices.index[i] in valid_days]

print(f"Testing w_prev calculation for first few days...")

# Test the first few days
for i, t in enumerate(indices[:5]):
    day = prices.index[t]
    print(f"\n=== Day {i+1} ({day}) ===")
    
    # Get the data for this day
    prices_t = prices.iloc[t - lookback : t + 1]
    spread_t = spread.iloc[t - lookback : t + 1]
    
    # Get mean and covariance
    mean_t = means.loc[day]
    cov_matrix = covariance_df.loc[day].values
    n_assets = cov_matrix.shape[0]
    regularization = 1e-6 * np.eye(n_assets)
    cov_matrix_reg = cov_matrix + regularization
    min_eigenval = np.linalg.eigvals(cov_matrix_reg).min()
    if min_eigenval <= 0:
        additional_reg = abs(min_eigenval) + 1e-6
        cov_matrix_reg += additional_reg * np.eye(n_assets)
    
    covariance_t = pd.DataFrame(cov_matrix_reg, index=covariance_df.loc[day].index, columns=covariance_df.loc[day].columns)
    chol_t = np.linalg.cholesky(covariance_t.values)
    volas_t = np.sqrt(np.diag(covariance_t.values))
    
    # Initialize portfolio state
    quantities = np.zeros(n_assets)
    cash = 1e6
    
    # Create OptimizationInput
    from experiments.backtest import OptimizationInput
    inputs_t = OptimizationInput(
        prices_t,
        mean_t,
        chol_t,
        volas_t,
        spread_t,
        quantities,
        cash,
        0.1 / np.sqrt(252),  # risk_target
        rf.iloc[t],
    )
    
    # Debug the w_prev calculation
    latest_prices = inputs_t.prices.iloc[-1]
    portfolio_value = inputs_t.cash + inputs_t.quantities @ latest_prices
    
    print(f"Latest prices shape: {latest_prices.shape}")
    print(f"Quantities shape: {inputs_t.quantities.shape}")
    print(f"Cash: {inputs_t.cash}")
    print(f"Portfolio value: {portfolio_value}")
    
    # Check for NaNs/Infs in individual components
    print(f"Latest prices has NaNs: {np.isnan(latest_prices).any()}")
    print(f"Latest prices has Infs: {np.isinf(latest_prices).any()}")
    print(f"Quantities has NaNs: {np.isnan(inputs_t.quantities).any()}")
    print(f"Quantities has Infs: {np.isinf(inputs_t.quantities).any()}")
    print(f"Portfolio value is NaN: {np.isnan(portfolio_value)}")
    print(f"Portfolio value is Inf: {np.isinf(portfolio_value)}")
    
    # Calculate w_prev step by step
    step1 = inputs_t.quantities * latest_prices
    print(f"Step 1 (quantities * prices) has NaNs: {np.isnan(step1).any()}")
    print(f"Step 1 has Infs: {np.isinf(step1).any()}")
    
    step2 = step1 / portfolio_value
    print(f"Step 2 (step1 / portfolio_value) has NaNs: {np.isnan(step2).any()}")
    print(f"Step 2 has Infs: {np.isinf(step2).any()}")
    
    w_prev = step2.values
    print(f"Final w_prev has NaNs: {np.isnan(w_prev).any()}")
    print(f"Final w_prev has Infs: {np.isinf(w_prev).any()}")
    print(f"w_prev shape: {w_prev.shape}")
    print(f"w_prev dtype: {w_prev.dtype}")
    
    # Check if any values are complex
    if np.iscomplexobj(w_prev):
        print(f"WARNING: w_prev contains complex values!")
        print(f"Complex values: {np.iscomplex(w_prev).sum()}")
    
    # Try to call the strategy function
    try:
        w, _, problem = parameter_scaling_markowitz(inputs_t)
        print(f"Strategy succeeded! w shape: {w.shape}")
    except Exception as e:
        print(f"Strategy failed with error: {e}")
        print(f"Error type: {type(e)}")
        
        # Check if it's the CVXPY parameter error
        if "Parameter value must be real" in str(e):
            print("This is the CVXPY parameter error we're looking for!")
            
            # Let's check all the parameters being set
            from experiments.scaling_small import get_parametrized_problem
            problem, param_dict, w, c = get_parametrized_problem(inputs_t.n_assets, inputs_t.risk_target)
            
            print("\nChecking all parameter values:")
            for param_name, param in param_dict.items():
                if hasattr(param, 'value') and param.value is not None:
                    value = param.value
                    if hasattr(value, 'shape'):
                        print(f"{param_name}: shape {value.shape}, has NaNs: {np.isnan(value).any()}, has Infs: {np.isinf(value).any()}")
                    else:
                        print(f"{param_name}: scalar {value}, is NaN: {np.isnan(value)}, is Inf: {np.isinf(value)}")
    
    print("-" * 50) 