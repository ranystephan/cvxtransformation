"""
Portfolio Lifecycle Experiment

This experiment studies portfolio setup and liquidation scenarios:
1. Setup: Transitioning from cash to a diversified portfolio
2. Liquidation: Transitioning from a portfolio back to cash
3. Rebalancing: Moving between different portfolio configurations

Tests how different transformation policies handle these critical lifecycle events.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from backtest import load_data
from transformation import (
    UniformTransformationPolicy,
    DynamicUniformTransformationPolicy,
    TransformationConfig,
    create_transformation_strategy
)
from transformations.univariate_policy import UnivariateScalarTrackingPolicy
from enhanced_backtest import run_enhanced_backtest, create_dashboard_from_enhanced_backtest

def create_diversified_portfolio(n_assets: int, target_risk: float = 0.10, 
                                concentration: str = 'equal') -> np.ndarray:
    """Create a diversified portfolio with different concentration levels."""
    w = np.zeros(n_assets)
    
    if concentration == 'equal':
        # Equal weight across top N assets
        n_selected = min(50, n_assets)  # Diversify across top 50 assets
        weight_per_asset = target_risk / n_selected
        w[:n_selected] = weight_per_asset
        
    elif concentration == 'concentrated':
        # Concentrated in top 10 assets
        n_selected = min(10, n_assets)
        weight_per_asset = target_risk / n_selected
        w[:n_selected] = weight_per_asset
        
    elif concentration == 'factor':
        # Factor-based allocation (simplified)
        # Growth stocks (higher indices), Value stocks (lower indices)
        growth_assets = min(20, n_assets // 3)
        value_assets = min(15, n_assets // 4)
        
        # Allocate 60% to growth, 40% to value
        growth_weight = 0.6 * target_risk / growth_assets
        value_weight = 0.4 * target_risk / value_assets
        
        w[:value_assets] = value_weight
        w[-growth_assets:] = growth_weight
        
    elif concentration == 'market_cap':
        # Market cap weighted (proxy using index position)
        n_selected = min(100, n_assets)
        weights = 1.0 / np.arange(1, n_selected + 1)  # 1/rank weighting
        weights = weights / weights.sum() * target_risk
        w[:n_selected] = weights
    
    return w

def estimate_transformation_complexity(w_initial: np.ndarray, w_target: np.ndarray) -> Dict[str, float]:
    """Estimate the complexity of a transformation for planning purposes."""
    
    # Calculate trade requirements
    trades = w_target - w_initial
    total_trade_size = np.abs(trades).sum()
    n_assets_traded = np.sum(trades != 0)
    
    # Concentration measures
    initial_concentration = np.sum(w_initial ** 2)  # Herfindahl index
    target_concentration = np.sum(w_target ** 2)
    
    # Directional changes
    long_positions = np.sum(trades > 0)
    short_positions = np.sum(trades < 0)
    
    return {
        'total_trade_size': total_trade_size,
        'n_assets_traded': n_assets_traded,
        'initial_concentration': initial_concentration,
        'target_concentration': target_concentration,
        'concentration_change': target_concentration - initial_concentration,
        'long_positions': long_positions,
        'short_positions': short_positions,
        'trade_complexity': total_trade_size * n_assets_traded / 10000  # Normalized complexity score
    }

def suggest_transformation_period(complexity: Dict[str, float], market_conditions: str = 'normal') -> int:
    """Suggest optimal transformation period based on complexity and market conditions."""
    
    base_days = max(5, int(complexity['trade_complexity'] * 10))
    
    # Adjust for market conditions
    if market_conditions == 'volatile':
        base_days = int(base_days * 1.5)  # Slower in volatile markets
    elif market_conditions == 'calm':
        base_days = int(base_days * 0.8)  # Faster in calm markets
    
    # Bound the result
    return min(max(base_days, 3), 60)  # Between 3 and 60 days

def run_portfolio_lifecycle_experiment():
    """Run the main portfolio lifecycle experiment."""
    print("="*80)
    print("PORTFOLIO LIFECYCLE TRANSFORMATION EXPERIMENT")
    print("="*80)
    
    # Load data
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    print(f"Total assets available: {n_assets}")
    
    # Define lifecycle scenarios
    scenarios = [
        {
            'name': 'Cash_to_Equal_Weight',
            'description': 'Setup: From cash to equal-weight diversified portfolio',
            'w_initial': np.zeros(n_assets),
            'w_target': create_diversified_portfolio(n_assets, 0.10, 'equal'),
            'scenario_type': 'setup'
        },
        {
            'name': 'Cash_to_Concentrated',
            'description': 'Setup: From cash to concentrated portfolio',
            'w_initial': np.zeros(n_assets),
            'w_target': create_diversified_portfolio(n_assets, 0.12, 'concentrated'),
            'scenario_type': 'setup'
        },
        {
            'name': 'Cash_to_Factor',
            'description': 'Setup: From cash to factor-based portfolio',
            'w_initial': np.zeros(n_assets),
            'w_target': create_diversified_portfolio(n_assets, 0.08, 'factor'),
            'scenario_type': 'setup'
        },
        {
            'name': 'Equal_Weight_to_Cash',
            'description': 'Liquidation: From equal-weight portfolio to cash',
            'w_initial': create_diversified_portfolio(n_assets, 0.10, 'equal'),
            'w_target': np.zeros(n_assets),
            'scenario_type': 'liquidation'
        },
        {
            'name': 'Concentrated_to_Cash',
            'description': 'Liquidation: From concentrated portfolio to cash',
            'w_initial': create_diversified_portfolio(n_assets, 0.12, 'concentrated'),
            'w_target': np.zeros(n_assets),
            'scenario_type': 'liquidation'
        },
        {
            'name': 'Concentrated_to_Diversified',
            'description': 'Rebalancing: From concentrated to diversified portfolio',
            'w_initial': create_diversified_portfolio(n_assets, 0.12, 'concentrated'),
            'w_target': create_diversified_portfolio(n_assets, 0.10, 'equal'),
            'scenario_type': 'rebalancing'
        },
        {
            'name': 'Factor_Rotation',
            'description': 'Rebalancing: From factor portfolio to market-cap weighted',
            'w_initial': create_diversified_portfolio(n_assets, 0.08, 'factor'),
            'w_target': create_diversified_portfolio(n_assets, 0.08, 'market_cap'),
            'scenario_type': 'rebalancing'
        }
    ]
    
    # Test transformation policies
    policies = {
        'Uniform': UniformTransformationPolicy(),
        'Dynamic_Uniform': DynamicUniformTransformationPolicy(),
        'Univariate_Tracking_Conservative': UnivariateScalarTrackingPolicy(risk_aversion=2.0),
        'Univariate_Tracking_Aggressive': UnivariateScalarTrackingPolicy(risk_aversion=0.5)
    }
    
    # Market condition scenarios for adaptive timing
    market_conditions = ['calm', 'normal', 'volatile']
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'-'*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Type: {scenario['scenario_type'].upper()}")
        print(f"Description: {scenario['description']}")
        print(f"{'-'*70}")
        
        w_initial = scenario['w_initial']
        w_target = scenario['w_target']
        
        # Analyze transformation complexity
        complexity = estimate_transformation_complexity(w_initial, w_target)
        
        print(f"Transformation Analysis:")
        print(f"  Total trade size: {complexity['total_trade_size']:.2%}")
        print(f"  Assets to trade: {complexity['n_assets_traded']}")
        print(f"  Complexity score: {complexity['trade_complexity']:.2f}")
        print(f"  Initial concentration: {complexity['initial_concentration']:.4f}")
        print(f"  Target concentration: {complexity['target_concentration']:.4f}")
        
        # Test different market conditions and corresponding transformation periods
        for market_condition in market_conditions:
            suggested_period = suggest_transformation_period(complexity, market_condition)
            print(f"\n  Market condition: {market_condition} → Suggested period: {suggested_period} days")
            
            for policy_name, policy in policies.items():
                print(f"    Testing {policy_name}...")
                
                try:
                    # Create transformation configuration
                    config = TransformationConfig(
                        w_initial=w_initial,
                        w_target=w_target,
                        total_days=suggested_period,
                        participation_limit=0.10 if market_condition == 'volatile' else 0.05
                    )
                    
                    # Create strategy
                    strategy = create_transformation_strategy(policy, config)
                    
                    # Determine risk target based on scenario type
                    if scenario['scenario_type'] == 'setup':
                        risk_target = 0.06  # Conservative for portfolio setup
                    elif scenario['scenario_type'] == 'liquidation':
                        risk_target = 0.04  # Very conservative for liquidation
                    else:  # rebalancing
                        risk_target = 0.08  # Normal for rebalancing
                    
                    # Run enhanced backtest
                    result = run_enhanced_backtest(
                        strategy=strategy,
                        risk_target=risk_target,
                        strategy_name=f"{scenario['name']}_{policy_name}_{market_condition}",
                        strategy_params={
                            'scenario': scenario['name'],
                            'scenario_type': scenario['scenario_type'],
                            'policy': policy_name,
                            'market_condition': market_condition,
                            'transformation_days': suggested_period,
                            'complexity_score': complexity['trade_complexity']
                        },
                        verbose=False
                    )
                    
                    # Calculate scenario-specific metrics
                    portfolio_turnover = result.turnover
                    implementation_shortfall = calculate_implementation_shortfall(result, w_initial, w_target)
                    
                    # Store results
                    results.append({
                        'scenario': scenario['name'],
                        'scenario_type': scenario['scenario_type'],
                        'policy': policy_name,
                        'market_condition': market_condition,
                        'period': suggested_period,
                        'complexity_score': complexity['trade_complexity'],
                        'sharpe': result.sharpe,
                        'return': result.mean_return,
                        'volatility': result.volatility,
                        'turnover': result.turnover,
                        'max_leverage': result.max_leverage,
                        'max_drawdown': result.max_drawdown,
                        'implementation_shortfall': implementation_shortfall,
                        'result': result
                    })
                    
                    print(f"      Sharpe: {result.sharpe:.2f}, Turnover: {result.turnover:.2f}, "
                          f"Shortfall: {implementation_shortfall:.4f}")
                    
                except Exception as e:
                    print(f"      ERROR: {e}")
                    continue
    
    # Analysis and results summary
    print(f"\n{'='*80}")
    print("PORTFOLIO LIFECYCLE EXPERIMENT RESULTS")
    print(f"{'='*80}")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Analyze by scenario type
        scenario_types = results_df['scenario_type'].unique()
        
        for scenario_type in scenario_types:
            type_results = results_df[results_df['scenario_type'] == scenario_type]
            
            print(f"\n{scenario_type.upper()} SCENARIOS:")
            print(f"{'Scenario':<25} {'Policy':<25} {'Market':<10} {'Sharpe':<8} {'Turnover':<10} {'Shortfall':<10}")
            print("-" * 100)
            
            for _, row in type_results.iterrows():
                print(f"{row['scenario']:<25} {row['policy']:<25} {row['market_condition']:<10} "
                      f"{row['sharpe']:<8.2f} {row['turnover']:<10.2f} {row['implementation_shortfall']:<10.4f}")
        
        # Find best policies for each scenario type
        print(f"\nBEST POLICIES BY SCENARIO TYPE:")
        for scenario_type in scenario_types:
            type_results = results_df[results_df['scenario_type'] == scenario_type]
            
            # Use composite score (Sharpe - penalty for high turnover and shortfall)
            type_results = type_results.copy()
            type_results['composite_score'] = (type_results['sharpe'] - 
                                             0.1 * type_results['turnover'] - 
                                             10 * type_results['implementation_shortfall'])
            
            best_result = type_results.loc[type_results['composite_score'].idxmax()]
            print(f"  {scenario_type}: {best_result['policy']} "
                  f"(Sharpe: {best_result['sharpe']:.2f}, Score: {best_result['composite_score']:.2f})")
        
        # Save results
        save_path = "./portfolio_lifecycle_output"
        os.makedirs(save_path, exist_ok=True)
        
        results_df.drop('result', axis=1).to_csv(f"{save_path}/lifecycle_experiment_results.csv", index=False)
        
        # Create summary by policy effectiveness
        policy_summary = results_df.groupby(['policy', 'scenario_type']).agg({
            'sharpe': 'mean',
            'turnover': 'mean',
            'implementation_shortfall': 'mean',
            'volatility': 'mean'
        }).round(3)
        
        policy_summary.to_csv(f"{save_path}/policy_effectiveness_summary.csv")
        
        # Create dashboards for representative scenarios
        print(f"\nCreating dashboards for representative scenarios...")
        
        representative_scenarios = ['Cash_to_Equal_Weight', 'Equal_Weight_to_Cash', 'Concentrated_to_Diversified']
        
        for scenario_name in representative_scenarios:
            if scenario_name in results_df['scenario'].values:
                scenario_results = results_df[results_df['scenario'] == scenario_name]
                best_result = scenario_results.loc[scenario_results['sharpe'].idxmax()]
                
                print(f"Creating dashboard for {scenario_name}...")
                
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
        print(f"Key insights:")
        print(f"  - Analyzed {len(results)} transformation scenarios")
        print(f"  - Tested {len(scenario_types)} scenario types: {', '.join(scenario_types)}")
        print(f"  - Best overall policy varies by scenario type")
        
        return results_df
    else:
        print("No results to analyze - all experiments failed!")
        return None

def calculate_implementation_shortfall(result, w_initial: np.ndarray, w_target: np.ndarray) -> float:
    """Calculate implementation shortfall as a measure of transformation quality."""
    
    # Get final weights
    final_day = result.daily_weights.iloc[-1] if not result.daily_weights.empty else np.zeros_like(w_target)
    
    # Calculate shortfall as distance from target
    shortfall = np.sum(np.abs(final_day.values - w_target))
    
    # Normalize by target trade size
    target_trade_size = np.sum(np.abs(w_target - w_initial))
    
    return shortfall / max(target_trade_size, 1e-6)

if __name__ == "__main__":
    run_portfolio_lifecycle_experiment() 