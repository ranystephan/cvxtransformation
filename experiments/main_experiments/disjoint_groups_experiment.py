"""
Disjoint Asset Groups Experiment

This experiment tests portfolio transformations between disjoint groups of assets
with different risk characteristics. It explores how different transformation 
policies perform when moving between distinct asset universes.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Sequence
import sys
import os
import signal
import time
import logging
import json

# Optional progress bar; falls back to no-op if tqdm unavailable
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.backtest import load_data
from core.transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    TransformationConfig,
    create_transformation_strategy,
    UnivariateScalarTrackingPolicy
)
from core.enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

def calculate_asset_risk_metrics(prices: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """Calculate risk metrics for each asset to inform group selection."""
    returns = prices.pct_change().dropna()
    
    # Calculate rolling statistics
    rolling_vol = returns.rolling(lookback_days).std() * np.sqrt(252)
    rolling_corr_to_market = returns.corrwith(returns.mean(axis=1), axis=0)
    
    # Calculate average metrics
    metrics = pd.DataFrame({
        'volatility': rolling_vol.mean(),
        'market_correlation': rolling_corr_to_market,
        'sharpe_proxy': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(returns)
    })
    
    return metrics.fillna(0)

def calculate_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """Calculate maximum drawdown for each asset."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def create_asset_groups(
    asset_metrics: pd.DataFrame,
    n_assets: int,
    min_group_size: int = 5,
) -> Dict[str, List[str]]:
    """Create (roughly) disjoint asset groups based on risk characteristics.

    If after deduplication a group is smaller than *min_group_size*, extra assets
    are pulled (without replacement) from the leftover universe to meet the
    size requirement.  A warning is logged whenever this happens.
    """

    sorted_by_vol = asset_metrics.sort_values("volatility")

    group_size = min(25, n_assets // 10)  # heuristic size

    raw_groups: Dict[str, List[str]] = {
        "low_vol": sorted_by_vol.head(group_size).index.tolist(),
        "high_vol": sorted_by_vol.tail(group_size).index.tolist(),
        "high_sharpe": asset_metrics.nlargest(group_size, "sharpe_proxy").index.tolist(),
        "low_correlation": asset_metrics.nsmallest(group_size, "market_correlation").index.tolist(),
    }

    # Build disjoint groups
    used_assets: set[str] = set()
    clean_groups: Dict[str, List[str]] = {}

    for name, asset_list in raw_groups.items():
        group_assets = [a for a in asset_list if a not in used_assets]

        # Ensure minimum viable size
        if len(group_assets) < min_group_size:
            # Grab additional assets (still avoiding overlap)
            additional_candidates = [a for a in asset_metrics.index if a not in used_assets and a not in group_assets]
            needed = min_group_size - len(group_assets)
            group_assets.extend(additional_candidates[:needed])

            if len(group_assets) < min_group_size:
                logging.warning(
                    f"Group '{name}' only has {len(group_assets)} assets (<{min_group_size}). Experiment results may be unstable."
                )

        if group_assets:
            clean_groups[name] = group_assets
            used_assets.update(group_assets)

    return clean_groups

def create_minimum_variance_portfolio(
    asset_names: Sequence[str],
    prices: pd.DataFrame,
    target_risk: float = 0.10,
    method: str = "equal",
) -> np.ndarray:
    """Construct a simple proxy for a minimum-variance portfolio.

    Parameters
    ----------
    asset_names   : List/Sequence of tickers to include.
    prices        : Full price DataFrame.
    target_risk   : Total weight assigned to this group (acts as risk proxy).
    method        : "equal" (default) ⇒ equal-weight inside the group.
                    "inv_var" ⇒ inverse-variance weighting using the last 252-day
                    sample variance of each asset.
    """

    n_assets = prices.shape[1]
    w = np.zeros(n_assets)

    if not asset_names:
        return w

    asset_names = [a for a in asset_names if a in prices.columns]
    if not asset_names:
        return w

    if method == "inv_var":
        returns = prices[asset_names].pct_change().dropna()
        variances = returns.var()
        # Guard against zero variance
        inv_var = 1.0 / np.where(variances == 0, 1e-8, variances)
        raw_weights = inv_var / inv_var.sum()
    else:  # equal weight fallback
        raw_weights = np.ones(len(asset_names)) / len(asset_names)

    scaled_weights = raw_weights * target_risk

    for asset_name, weight in zip(asset_names, scaled_weights):
        idx = prices.columns.get_loc(asset_name)
        w[idx] = weight

    return w

def run_disjoint_groups_experiment():
    """Run the main disjoint asset groups experiment."""
    print("="*80)
    print("DISJOINT ASSET GROUPS TRANSFORMATION EXPERIMENT")
    print("="*80)
    
    # Load data
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Total assets available: {n_assets}")
    
    # Calculate asset risk metrics
    print("Calculating asset risk metrics...")
    asset_metrics = calculate_asset_risk_metrics(prices)
    
    # Create disjoint asset groups
    print("Creating disjoint asset groups...")
    asset_groups = create_asset_groups(asset_metrics, n_assets)
    
    print(f"Created {len(asset_groups)} asset groups:")
    for group_name, assets in asset_groups.items():
        avg_vol = asset_metrics.loc[assets, 'volatility'].mean()
        avg_sharpe = asset_metrics.loc[assets, 'sharpe_proxy'].mean()
        print(f"  {group_name}: {len(assets)} assets, avg vol={avg_vol:.2%}, avg sharpe={avg_sharpe:.2f}")
    
    # Define experiment scenarios
    scenarios = [
        {
            'name': 'Low_Vol_to_High_Vol',
            'initial_group': 'low_vol',
            'target_group': 'high_vol',
            'initial_risk': 0.05,
            'target_risk': 0.15,
            'description': 'Transform from low volatility to high volatility assets'
        },
        {
            'name': 'High_Sharpe_to_Low_Correlation',
            'initial_group': 'high_sharpe',
            'target_group': 'low_correlation',
            'initial_risk': 0.10,
            'target_risk': 0.10,
            'description': 'Transform from high Sharpe assets to low correlation assets'
        }
    ]
    
    # Test different transformation periods
    transformation_periods = [5, 10, 20, 30]
    
    # Test all transformation policies
    policies = {
        'Uniform': UniformTransformationPolicy(),
        'Dynamic_Uniform': DynamicUniformTransformationPolicy(),
        'Univariate_Tracking': UnivariateScalarTrackingPolicy(risk_aversion=1.0)
    }
    
    results = []
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        print(f"\n{'-'*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*60}")
        
        # Create initial and target portfolios
        if scenario['initial_group'] not in asset_groups or scenario['target_group'] not in asset_groups:
            print(f"Skipping scenario - missing asset groups")
            continue
            
        initial_assets = asset_groups[scenario['initial_group']]
        target_assets = asset_groups[scenario['target_group']]
        
        w_initial = create_minimum_variance_portfolio(initial_assets, prices, scenario['initial_risk'])
        w_target = create_minimum_variance_portfolio(target_assets, prices, scenario['target_risk'])
        
        print(f"Initial portfolio: {len(initial_assets)} assets, total weight={w_initial.sum():.2%}")
        print(f"Target portfolio: {len(target_assets)} assets, total weight={w_target.sum():.2%}")
        
        for period in transformation_periods:
            for policy_name, policy in policies.items():
                print(f"\nTesting {policy_name} with {period}-day transformation...")
                
                try:
                    # Create transformation configuration
                    config = TransformationConfig(
                        w_initial=w_initial,
                        w_target=w_target,
                        total_days=period,
                        participation_limit=0.05
                    )
                    
                    # Create strategy
                    strategy = create_transformation_strategy(policy, config)
                    
                    # Set up timeout for backtest
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Backtest timed out after 30 seconds")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 second timeout (shorter since we limited backtest duration)
                    
                    try:
                        # Run enhanced backtest with limited duration for transformation
                        result = run_enhanced_backtest(
                            strategy=strategy,
                            risk_target=0.08,  # 8% annual volatility target
                            strategy_name=f"{scenario['name']}_{policy_name}_{period}d",
                            strategy_params={
                                'scenario': scenario['name'],
                                'policy': policy_name,
                                'transformation_days': period,
                                'initial_group': scenario['initial_group'],
                                'target_group': scenario['target_group']
                            },
                            verbose=False,
                            max_days=period * 2  # Run for 2x transformation period to see results
                        )
                        signal.alarm(0)  # Cancel timeout
                    except TimeoutError:
                        signal.alarm(0)  # Cancel timeout
                        print(f"  ERROR: Backtest timed out after 30 seconds")
                        continue
                    
                    # Validate result; skip if too short (e.g., all days skipped)
                    if not is_valid_result(result):
                        logging.warning("Result for %s/%s/%sd is invalid (too few days). Skipping.", scenario['name'], policy_name, period)
                        continue

                    # Compute average L1 tracking error if data available
                    tracking_error = np.nan
                    if not result.daily_weights.empty and not result.daily_target_weights.empty:
                        diff = result.daily_weights.sub(result.daily_target_weights, fill_value=0)
                        tracking_error = diff.abs().sum(axis=1).mean()

                    # Build weight snapshots
                    init_w_dict = {prices.columns[i]: w for i, w in enumerate(w_initial) if abs(w) > 0}
                    target_w_dict = {prices.columns[i]: w for i, w in enumerate(w_target) if abs(w) > 0}
                    final_w_dict = (
                        result.daily_weights.iloc[-1].round(6).loc[lambda s: s != 0].to_dict()
                        if not result.daily_weights.empty else {}
                    )

                    # Policy hyper-parameters (public attrs only)
                    policy_params = {k: v for k, v in vars(policy).items() if not k.startswith("_")}

                    # Store results
                    results.append({
                        'scenario': scenario['name'],
                        'policy': policy_name,
                        'period': period,
                        'sharpe': result.sharpe,
                        'return': result.mean_return,
                        'volatility': result.volatility,
                        'turnover': result.turnover,
                        'max_leverage': result.max_leverage,
                        'max_drawdown': result.max_drawdown,
                        'tracking_error': tracking_error,
                        'initial_weights': _to_compact_json(init_w_dict),
                        'target_weights': _to_compact_json(target_w_dict),
                        'final_weights': _to_compact_json(final_w_dict),
                        'policy_params': json.dumps(policy_params),
                        'risk_target': 0.08,
                        'result': result
                    })
                    
                    print(f"  Sharpe: {result.sharpe:.2f}, Return: {result.mean_return:.2%}, "
                          f"Vol: {result.volatility:.2%}, Turnover: {result.turnover:.2f}")
                    
                except (Exception, TimeoutError) as e:
                    print(f"  ERROR: {e}")
                    continue
    
    # Create comparison analysis
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Group by scenario and policy for analysis
        for scenario_name in results_df['scenario'].unique():
            scenario_results = results_df[results_df['scenario'] == scenario_name]
            
            print(f"\n{scenario_name}:")
            print(f"{'Policy':<20} {'Period':<8} {'Sharpe':<8} {'Return':<8} {'Vol':<8} {'Turnover':<10}")
            print("-" * 70)
            
            for _, row in scenario_results.iterrows():
                print(f"{row['policy']:<20} {row['period']:<8} {row['sharpe']:<8.2f} "
                      f"{row['return']:<8.2%} {row['volatility']:<8.2%} {row['turnover']:<10.2f}")
        
        # Save detailed results
        save_path = "./disjoint_groups_output"
        os.makedirs(save_path, exist_ok=True)
        
        results_df.drop('result', axis=1).to_csv(f"{save_path}/experiment_results.csv", index=False)
        
        # Create dashboards for best performing combinations
        print(f"\nCreating dashboards for top results...")
        
        # Find best result for each scenario
        for scenario_name in results_df['scenario'].unique():
            scenario_results = results_df[results_df['scenario'] == scenario_name]
            best_result = scenario_results.loc[scenario_results['sharpe'].idxmax()]
            
            print(f"Creating dashboard for {scenario_name} (best: {best_result['policy']} {best_result['period']}d)")
            
            scenario_save_path = f"{save_path}/{scenario_name}"
            os.makedirs(scenario_save_path, exist_ok=True)
            
            try:
                figures = create_dashboard_from_enhanced_backtest(
                    result=best_result['result'],
                    save_path=scenario_save_path,
                    show=False
                )
                print(f"  Dashboard saved with {len(figures)} plots")
            except Exception as e:
                print(f"  Dashboard creation failed: {e}")
        
        print(f"\nAll results saved to: {save_path}/")
        return results_df
    else:
        print("No results to analyze - all experiments failed!")
        return None

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def is_valid_result(result, min_days: int = 5) -> bool:
    """Basic sanity-check to ensure backtest produced data."""
    try:
        return len(result.portfolio_value) >= min_days
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Helper for serialising weight dictionaries
# ---------------------------------------------------------------------------

def _to_compact_json(mapping: Dict[str, float]) -> str:
    """Serialize dict to a compact JSON string for CSV storage."""
    # Keep only non-zero weights for brevity
    return json.dumps({k: round(v, 6) for k, v in mapping.items() if abs(v) > 1e-8})

if __name__ == "__main__":
    run_disjoint_groups_experiment() 