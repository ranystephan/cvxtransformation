"""
Univariate Scalar Tracking Policy

This module implements the univariate scalar tracking transformation policy from 
the transformation literature. It optimizes each asset independently, considering 
costs and risk through asset-by-asset optimization.
"""

import numpy as np
import cvxpy as cp
import sys
import os
from typing import Optional

# Add parent directory to path to find backtest module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backtest import OptimizationInput
from .uniform_policy import TransformationPolicy


class UnivariateScalarTrackingPolicy(TransformationPolicy):
    """
    Univariate scalar tracking transformation policy.
    
    This implements the univariate scalar tracking baseline policy from the
    transformation literature. The policy is myopic (single-period) and optimizes 
    each asset's trade independently, ignoring all cross-asset correlations. 
    It's a step up from uniform trading as it considers costs and risk for each asset.
    
    For each asset i, it solves:
    min_{z_i} 0.5 * σ_i^2 * (w_i + z_i - w_i^target)^2 + κ_spread * |z_i| + κ_impact * |z_i|^1.5
    
    Where:
    - z_i is the trade in asset i
    - σ_i^2 is the variance forecast for asset i  
    - w_i^target is the target weight for asset i
    - κ_spread and κ_impact are transaction cost parameters
    """
    
    def __init__(self, 
                 risk_aversion: float = 1.0,
                 impact_coeff: float = 0.01,
                 max_position: float = 0.1,
                 min_position: float = -0.1):
        """
        Initialize the optimal transformation policy.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            impact_coeff: Market impact coefficient
            max_position: Maximum allowed position size
            min_position: Minimum allowed position size (for shorts)
        """
        self.risk_aversion = risk_aversion
        self.impact_coeff = impact_coeff
        self.max_position = max_position
        self.min_position = min_position
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config,
        current_day: int
    ) -> np.ndarray:
        """
        Calculate optimal weights using univariate optimization for each asset.
        
        This solves a separate optimization problem for each asset, balancing
        tracking error against transaction costs including spreads and market impact.
        """
        # Get current portfolio state
        latest_prices = inputs.prices.iloc[-1]
        portfolio_value = inputs.cash + inputs.quantities @ latest_prices
        
        if portfolio_value > 0:
            w_current = (inputs.quantities * latest_prices / portfolio_value).values
        else:
            w_current = np.zeros(inputs.n_assets)
        
        # Get market data for cost estimation
        spreads = inputs.spread.iloc[-1].values  # Most recent spreads
        volas = inputs.volas  # Volatility forecasts from EWMA
        
        # Estimate daily volumes (simplified - ideally you'd have real volume data)
        # Using a placeholder that scales with portfolio value
        avg_daily_volume = np.ones(inputs.n_assets) * portfolio_value * 10
        
        # Initialize optimal weights
        w_optimal = np.zeros(inputs.n_assets)
        
        # Solve optimization for each asset independently
        for i in range(inputs.n_assets):
            w_optimal[i] = self._optimize_single_asset(
                current_weight=w_current[i],
                target_weight=config.w_target[i],
                volatility=volas[i],
                spread=spreads[i],
                daily_volume=avg_daily_volume[i],
                portfolio_value=portfolio_value,
                participation_limit=config.participation_limit
            )
        
        return w_optimal
    
    def _optimize_single_asset(
        self,
        current_weight: float,
        target_weight: float,
        volatility: float,
        spread: float,
        daily_volume: float,
        portfolio_value: float,
        participation_limit: float
    ) -> float:
        """
        Optimize the position for a single asset.
        
        This solves the univariate optimization problem for one asset,
        balancing tracking error with transaction costs.
        """
        # Decision variable: new weight for this asset
        w_i = cp.Variable()
        
        # Trade size
        z_i = w_i - current_weight
        
        # Objective components
        # 1. Tracking error (quadratic in distance from target)
        tracking_error = cp.square(w_i - target_weight)
        
        # 2. Spread costs (linear in trade size)
        spread_cost = spread * cp.abs(z_i)
        
        # 3. Market impact costs (power 1.5 in trade size)
        impact_cost = self.impact_coeff * cp.power(cp.abs(z_i), 1.5)
        
        # Combined objective
        objective = cp.Minimize(
            self.risk_aversion * volatility**2 * tracking_error +
            spread_cost + 
            impact_cost
        )
        
        # Constraints
        constraints = [
            # Position limits
            w_i >= self.min_position,
            w_i <= self.max_position,
            
            # Participation constraint (limit trade size relative to daily volume)
            cp.abs(z_i) * portfolio_value <= participation_limit * daily_volume
        ]
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return float(w_i.value)
            else:
                # If optimization fails, return current weight (no trade)
                return current_weight
                
        except Exception as e:
            # Fallback to current weight if solver fails
            print(f"Optimization failed for asset: {e}")
            return current_weight


# Keep the old alias for backward compatibility if needed
class OptimalTransformationPolicy(UnivariateScalarTrackingPolicy):
    """
    Alias for UnivariateScalarTrackingPolicy for backward compatibility.
    
    This maintains compatibility if anyone was using the old name.
    """
    pass


# Utility function for creating strategies
def create_univariate_strategy_function(
    w_initial: np.ndarray, 
    w_target: np.ndarray, 
    total_days: int,
    risk_aversion: float = 1.0,
    impact_coeff: float = 0.01,
    max_position: float = 0.1,
    min_position: float = -0.1
):
    """
    Create a univariate scalar tracking transformation strategy function.
    
    Args:
        w_initial: Initial portfolio weights
        w_target: Target portfolio weights
        total_days: Total transformation period in days
        risk_aversion: Risk aversion parameter
        impact_coeff: Market impact coefficient
        max_position: Maximum position size
        min_position: Minimum position size
        
    Returns:
        Strategy function compatible with run_backtest()
    """
    import transformation
    
    policy = UnivariateScalarTrackingPolicy(
        risk_aversion=risk_aversion,
        impact_coeff=impact_coeff,
        max_position=max_position,
        min_position=min_position
    )
    config = transformation.TransformationConfig(w_initial, w_target, total_days)
    return transformation.create_transformation_strategy(policy, config)


# Backward compatibility alias
def create_optimal_strategy_function(*args, **kwargs):
    """Backward compatibility alias for create_univariate_strategy_function."""
    return create_univariate_strategy_function(*args, **kwargs) 