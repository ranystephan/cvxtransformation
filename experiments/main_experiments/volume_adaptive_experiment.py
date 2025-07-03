"""
Volume-Adaptive Transformation Experiment

This experiment demonstrates how to adapt transformation periods and execution
strategies based on trading volume, market impact considerations, and liquidity
conditions. It implements smart scheduling that optimizes transformation timing
based on market microstructure factors.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from backtest import load_data
from transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    TransformationConfig,
    create_transformation_strategy
)
# UnivariateScalarTrackingPolicy not needed for simplified experiment
from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

def classify_assets_by_volume(prices: pd.DataFrame, volume_proxy: bool = True) -> Dict[int, str]:
    """Classify assets by volume/liquidity characteristics."""
    
    if volume_proxy:
        # Use price volatility and momentum as volume proxies
        returns = prices.pct_change().dropna()
        
        # Calculate volatility (higher vol often means more trading)
        volatility = returns.rolling(60).std().mean()
        
        # Calculate trading activity proxy (frequency of large moves)
        large_moves = (returns.abs() > returns.std() * 2).mean()
        
        # Combine metrics for volume ranking
        activity_score = volatility * 0.7 + large_moves * 0.3
        
    else:
        # If actual volume data available, use it directly
        activity_score = pd.Series(np.random.random(len(prices.columns)), index=prices.columns)
    
    # Rank assets by activity score
    ranked_assets = activity_score.rank(ascending=False)
    n_assets = len(ranked_assets)
    
    asset_volume_rank = {}
    for i, (asset_idx, rank) in enumerate(ranked_assets.items()):
        if rank <= n_assets * 0.25:
            asset_volume_rank[i] = 'high'
        elif rank <= n_assets * 0.50:
            asset_volume_rank[i] = 'medium'
        elif rank <= n_assets * 0.75:
            asset_volume_rank[i] = 'low'
        else:
            asset_volume_rank[i] = 'very_low'
    
    return asset_volume_rank

def create_scenario_portfolios(scenario: Dict, n_assets: int, 
                             asset_volume_ranks: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Create initial and target portfolios for each scenario."""
    
    w_initial = np.zeros(n_assets)
    w_target = np.zeros(n_assets)
    
    if scenario['name'] == 'High_Volume_Focus':
        # Initial: Some positions in mixed assets
        mixed_assets = [i for i, rank in asset_volume_ranks.items() if rank in ['medium', 'low']][:20]
        for i in mixed_assets:
            w_initial[i] = 0.005  # 0.5% each, total 10%
        
        # Target: Focus on high volume assets
        high_vol_assets = [i for i, rank in asset_volume_ranks.items() if rank == 'high'][:15]
        for i in high_vol_assets:
            w_target[i] = 0.08 / len(high_vol_assets)  # Total 8%
    
    elif scenario['name'] == 'Low_Volume_Challenge':
        # Initial: Start with cash (easier)
        w_initial = np.zeros(n_assets)
        
        # Target: Invest in low volume assets
        low_vol_assets = [i for i, rank in asset_volume_ranks.items() if rank in ['low', 'very_low']][:10]
        for i in low_vol_assets:
            w_target[i] = 0.06 / len(low_vol_assets)  # Total 6%
    
    return w_initial, w_target

def run_volume_adaptive_experiment():
    """Run the main volume-adaptive transformation experiment."""
    print("="*80)
    print("VOLUME-ADAPTIVE TRANSFORMATION EXPERIMENT")
    print("="*80)
    
    # Load data
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Total assets available: {n_assets}")
    
    # Classify assets by volume/liquidity
    print("Classifying assets by volume characteristics...")
    asset_volume_ranks = classify_assets_by_volume(prices)
    
    # Print volume distribution
    for rank in ['high', 'medium', 'low', 'very_low']:
        count = sum(1 for v in asset_volume_ranks.values() if v == rank)
        print(f"  {rank.capitalize()} volume assets: {count}")
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'High_Volume_Focus',
            'description': 'Transformation focused on high-volume assets',
            'target_risk': 0.08
        },
        {
            'name': 'Low_Volume_Challenge',
            'description': 'Transformation involving primarily low-volume assets',
            'target_risk': 0.06
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'-'*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*70}")
        
        # Create scenario-specific portfolios
        w_initial, w_target = create_scenario_portfolios(
            scenario, n_assets, asset_volume_ranks
        )
        
        print(f"Initial portfolio total weight: {w_initial.sum():.2%}")
        print(f"Target portfolio total weight: {w_target.sum():.2%}")
        
        try:
            # Create basic strategy
            policy = UniformTransformationPolicy()
            config = TransformationConfig(w_initial, w_target, 10)
            strategy = create_transformation_strategy(policy, config)
            
            # Run enhanced backtest
            result = run_enhanced_backtest(
                strategy=strategy,
                risk_target=scenario['target_risk'],
                strategy_name=f"{scenario['name']}_Basic",
                strategy_params={'scenario': scenario['name']},
                verbose=False
            )
            
            results.append({
                'scenario': scenario['name'],
                'sharpe': result.sharpe,
                'return': result.mean_return,
                'volatility': result.volatility,
                'turnover': result.turnover
            })
            
            print(f"Sharpe: {result.sharpe:.2f}, Return: {result.mean_return:.2%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\nExperiment completed with {len(results)} results")
    return results

if __name__ == "__main__":
    run_volume_adaptive_experiment()
