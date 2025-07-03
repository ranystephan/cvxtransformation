"""
Disjoint Asset Groups Experiment

This experiment tests portfolio transformations between disjoint groups of assets
with different risk characteristics. It explores how different transformation 
policies perform when moving between distinct asset universes.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import sys
import os
import signal
import time
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

def create_asset_groups(asset_metrics: pd.DataFrame, n_assets: int) -> Dict[str, List[int]]:
    """Create disjoint asset groups based on risk characteristics."""
    
    # Sort assets by volatility
    sorted_by_vol = asset_metrics.sort_values('volatility')
    
    # Create groups with different risk profiles
    group_size = min(25, n_assets // 10)  # Reasonable group size
    
    groups = {
        'low_vol': sorted_by_vol.head(group_size).index.tolist(),
        'high_vol': sorted_by_vol.tail(group_size).index.tolist(),
        'high_sharpe': asset_metrics.nlargest(group_size, 'sharpe_proxy').index.tolist(),
        'low_correlation': asset_metrics.nsmallest(group_size, 'market_correlation').index.tolist()
    }
    
    # Ensure disjoint groups by removing overlaps
    used_assets = set()
    clean_groups = {}
    
    for group_name, assets in groups.items():
        clean_assets = [a for a in assets if a not in used_assets][:group_size//2]  # Smaller to avoid overlaps
        if len(clean_assets) >= 3:  # Minimum viable group size
            clean_groups[group_name] = clean_assets
            used_assets.update(clean_assets)
    
    return clean_groups

def create_minimum_variance_portfolio(asset_names: List[str], prices: pd.DataFrame, target_risk: float = 0.10) -> np.ndarray:
    """Create a minimum variance portfolio for given assets with target risk."""
    n_assets = prices.shape[1]
    w = np.zeros(n_assets)
    
    if len(asset_names) == 0:
        return w
    
    # Equal weight within the group as a simple proxy for minimum variance
    weight_per_asset = target_risk / len(asset_names)
    
    for asset_name in asset_names:
        if asset_name in prices.columns:
            asset_idx = prices.columns.get_loc(asset_name)
            w[asset_idx] = weight_per_asset
    
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
    
    for scenario in scenarios:
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

if __name__ == "__main__":
    run_disjoint_groups_experiment() 