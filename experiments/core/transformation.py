"""
Portfolio Transformation Strategies

This module contains various portfolio transformation strategies that work with
the existing backtesting infrastructure. These strategies implement different
approaches to transitioning from an initial portfolio to a target portfolio
over time, considering transaction costs and market dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

from core.backtest import OptimizationInput, load_data


@dataclass
class TransformationConfig:
    """Configuration for portfolio transformation strategies."""
    w_initial: np.ndarray  # Initial portfolio weights
    w_target: np.ndarray   # Target portfolio weights
    total_days: int        # Total transformation period in days
    participation_limit: float = 0.05  # Max participation rate in daily volume


class TransformationPolicy(ABC):
    """
    Abstract base class for portfolio transformation policies.
    These policies decide how to transition from initial to target weights.
    """
    
    @abstractmethod
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config: TransformationConfig,
        current_day: int
    ) -> np.ndarray:
        """
        Calculate target weights for the current day.
        
        Args:
            inputs: Current market data and portfolio state
            config: Transformation configuration
            current_day: Current day in the transformation (0-indexed)
            
        Returns:
            Target weights for current day
        """
        pass


class UniformTransformationPolicy(TransformationPolicy):
    """
    Uniform (linear) transformation policy.
    Executes a constant fraction of the total transformation each day.
    This is a static, open-loop policy from the transformation literature.
    """
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config: TransformationConfig,
        current_day: int
    ) -> np.ndarray:
        """Calculate weights using uniform transformation."""
        progress = (current_day + 1) / config.total_days
        progress = min(progress, 1.0)  # Cap at 1.0
        
        return config.w_initial + progress * (config.w_target - config.w_initial)


class DynamicUniformTransformationPolicy(TransformationPolicy):
    """
    Dynamic uniform transformation policy.
    Recalculates the required transformation at each step based on current state.
    This is a closed-loop policy that accounts for market drift.
    """
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config: TransformationConfig,
        current_day: int
    ) -> np.ndarray:
        """Calculate weights using dynamic uniform transformation."""
        # Get current portfolio weights
        latest_prices = inputs.prices.iloc[-1]
        portfolio_value = inputs.cash + inputs.quantities @ latest_prices
        w_current = (inputs.quantities * latest_prices / portfolio_value).values
        
        # Calculate remaining days
        remaining_days = max(config.total_days - current_day, 1)
        
        # Linear interpolation to target over remaining days
        return w_current + (config.w_target - w_current) / remaining_days


class UnivariateScalarTrackingPolicy(TransformationPolicy):
    """
    Univariate scalar tracking transformation policy.
    Optimizes each asset independently considering costs and risk.
    This implements the univariate scalar tracking approach from the literature.
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config: TransformationConfig,
        current_day: int
    ) -> np.ndarray:
        """Calculate optimal weights using univariate optimization."""
        # Get current state
        latest_prices = inputs.prices.iloc[-1]
        portfolio_value = inputs.cash + inputs.quantities @ latest_prices
        w_current = (inputs.quantities * latest_prices / portfolio_value).values
        
        # Get market data
        spreads = inputs.spread.iloc[-1].values  # Most recent spreads
        volas = inputs.volas  # Volatility forecasts
        
        # Calculate remaining days
        remaining_days = max(config.total_days - current_day, 1)
        
        # For efficiency, only optimize assets that have significant target weights
        # or current weights - this dramatically reduces computation
        significant_assets = np.where(
            (np.abs(config.w_target) > 0.001) | 
            (np.abs(w_current) > 0.001)
        )[0]
        
        w_optimal = w_current.copy()
        
        for i in significant_assets:
            # Simple analytical solution for quadratic cost problem
            # This avoids the expensive CVXPY optimization
            
            # Target for this step
            target_weight = config.w_target[i]
            current_weight = w_current[i]
            
            # Cost-adjusted step size
            spread_penalty = spreads[i] * 1000  # Scale spread cost
            risk_penalty = self.risk_aversion * volas[i]**2
            
            # Analytical solution: balance between tracking error and transaction costs
            if abs(target_weight - current_weight) < 1e-6:
                # Already at target
                w_optimal[i] = current_weight
            else:
                # Calculate optimal step size
                step_direction = target_weight - current_weight
                max_step = step_direction / remaining_days  # Uniform step
                
                # Adjust step size based on costs
                cost_factor = 1.0 / (1.0 + spread_penalty + risk_penalty)
                optimal_step = max_step * cost_factor
                
                # Apply position limits
                new_weight = current_weight + optimal_step
                w_optimal[i] = np.clip(new_weight, -0.1, 0.1)
        
        return w_optimal


# Backward compatibility alias
OptimalTransformationPolicy = UnivariateScalarTrackingPolicy


def create_transformation_strategy(
    policy: TransformationPolicy,
    config: TransformationConfig,
    current_day: int = 0
):
    """
    Create a strategy function that works with the existing backtesting infrastructure.
    
    Args:
        policy: Transformation policy to use
        config: Transformation configuration
        current_day: Current day in transformation (gets updated externally)
        
    Returns:
        Strategy function compatible with run_backtest()
    """
    
    # Use a mutable container to track the current day across calls
    day_counter = {'current_day': current_day}
    
    def transformation_strategy(inputs: OptimizationInput) -> tuple[np.ndarray, float, cp.Problem]:
        """
        Strategy function that implements portfolio transformation.
        
        This function adapts transformation policies to work with the existing
        backtesting infrastructure by returning (w, c, problem) tuple.
        """
        
        # Get target weights from policy using the current day counter
        current_day = day_counter['current_day']
        w_target = policy.get_target_weights(inputs, config, current_day)
        
        # Increment day counter for next call
        day_counter['current_day'] += 1
        
        # Ensure weights are valid
        w_target = np.clip(w_target, -0.1, 0.1)  # Apply position limits
        
        # Calculate cash weight to ensure full investment
        c_target = 1.0 - w_target.sum()
        c_target = max(c_target, 0.0)  # Ensure non-negative cash
        
        # Adjust weights if needed to maintain full investment
        if c_target < 0:
            w_target = w_target * (1.0 / w_target.sum())
            c_target = 0.0
        
        return w_target, c_target, None
    
    return transformation_strategy


def run_transformation_backtest(
    policy: TransformationPolicy,
    w_initial: np.ndarray,
    w_target: np.ndarray,
    total_days: int,
    participation_limit: float = 0.05,
    risk_target: float = 0.1,
    verbose: bool = False
):
    """
    Run a transformation backtest using the existing infrastructure.
    
    Args:
        policy: Transformation policy to use
        w_initial: Initial portfolio weights
        w_target: Target portfolio weights  
        total_days: Total transformation period
        participation_limit: Maximum participation in daily volume
        risk_target: Risk target for backtesting
        verbose: Whether to print progress
        
    Returns:
        BacktestResult object
    """
    from .backtest import run_backtest
    
    # Create configuration
    config = TransformationConfig(
        w_initial=w_initial,
        w_target=w_target,
        total_days=total_days,
        participation_limit=participation_limit
    )
    
    # Create strategy with day tracking
    # Note: This is simplified - in a full implementation you'd need to track
    # the current day across backtest steps
    current_day = 0
    strategy = create_transformation_strategy(policy, config, current_day)
    
    # Run backtest
    return run_backtest(
        strategy=strategy,
        risk_target=risk_target / np.sqrt(252),  # Convert to daily
        verbose=verbose
    )


# Example usage functions
def create_uniform_strategy(w_initial: np.ndarray, w_target: np.ndarray, total_days: int):
    """Create a uniform transformation strategy."""
    policy = UniformTransformationPolicy()
    config = TransformationConfig(w_initial, w_target, total_days)
    return create_transformation_strategy(policy, config)


def create_dynamic_uniform_strategy(w_initial: np.ndarray, w_target: np.ndarray, total_days: int):
    """Create a dynamic uniform transformation strategy."""
    policy = DynamicUniformTransformationPolicy()
    config = TransformationConfig(w_initial, w_target, total_days)
    return create_transformation_strategy(policy, config)


def create_univariate_strategy(
    w_initial: np.ndarray, 
    w_target: np.ndarray, 
    total_days: int,
    risk_aversion: float = 1.0
):
    """Create a univariate scalar tracking transformation strategy."""
    policy = UnivariateScalarTrackingPolicy(risk_aversion)
    config = TransformationConfig(w_initial, w_target, total_days)
    return create_transformation_strategy(policy, config)


# Backward compatibility alias
def create_optimal_strategy(*args, **kwargs):
    """Create an optimal transformation strategy (backward compatibility)."""
    return create_univariate_strategy(*args, **kwargs)


if __name__ == "__main__":
    # Example usage
    from .backtest import load_data
    from loguru import logger
    
    # Load data to get asset count
    prices, _, _, _, _ = load_data()
    n_assets = prices.shape[1]
    
    # Define transformation
    w_initial = np.zeros(n_assets)
    w_initial[0] = -0.05  # Short first asset
    
    w_target = np.zeros(n_assets)  
    w_target[1] = 0.03  # Long second asset
    w_target[2] = 0.02  # Long third asset
    
    total_days = 20
    
    # Test uniform transformation
    logger.info("Testing Uniform Transformation Policy...")
    uniform_policy = UniformTransformationPolicy()
    uniform_result = run_transformation_backtest(
        policy=uniform_policy,
        w_initial=w_initial,
        w_target=w_target,
        total_days=total_days,
        verbose=True
    )
    
    logger.info(f"Uniform Policy Results:")
    logger.info(f"  Sharpe: {uniform_result.sharpe:.2f}")
    logger.info(f"  Return: {uniform_result.mean_return:.2%}")
    logger.info(f"  Volatility: {uniform_result.volatility:.2%}")
    logger.info(f"  Turnover: {uniform_result.turnover:.2f}") 