"""
Uniform Transformation Policies

This module implements uniform transformation policies that execute trades
at a constant rate over the transformation period.
"""

import numpy as np
import sys
import os
from abc import ABC, abstractmethod

# Add parent directory to path to find backtest module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backtest import OptimizationInput


class TransformationPolicy(ABC):
    """
    Abstract base class for portfolio transformation policies.
    These policies decide how to transition from initial to target weights.
    """
    
    @abstractmethod
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config,  # TransformationConfig
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
    Uniform (linear) transformation policy from the transformation literature.
    
    This implements the uniform trading baseline policy. It calculates the total 
    required trade at the beginning and executes a constant fraction of it each day,
    ignoring any market drift. This is a static, "open-loop" policy.
    
    The policy progresses linearly from w_initial to w_target over total_days:
    w(t) = w_initial + (t / T) * (w_target - w_initial)
    
    Where t is the current day and T is the total transformation period.
    """
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config,
        current_day: int
    ) -> np.ndarray:
        """
        Calculate weights using uniform (linear) transformation.
        
        This implements a simple linear interpolation between initial and target
        weights based on the elapsed time.
        """
        # Calculate progress as fraction of total period
        progress = (current_day + 1) / config.total_days
        progress = min(progress, 1.0)  # Cap at 1.0 to avoid overshooting
        
        # Linear interpolation between initial and target weights
        return config.w_initial + progress * (config.w_target - config.w_initial)


class DynamicUniformTransformationPolicy(TransformationPolicy):
    """
    Dynamic uniform transformation policy.
    
    This implements a dynamic version of the uniform trading policy. It's a 
    "closed-loop" policy that recalculates the required trade at each step 
    based on the current portfolio state, spreading the remaining gap to target
    over the remaining time. This accounts for market drift and deviations
    from the planned trajectory.
    
    At each step t, it calculates:
    w(t+1) = w_current + (w_target - w_current) / (T - t)
    
    Where w_current is the actual current weights (potentially drifted due to
    market movements), and (T - t) is the remaining time.
    """
    
    def get_target_weights(
        self, 
        inputs: OptimizationInput, 
        config,
        current_day: int
    ) -> np.ndarray:
        """
        Calculate weights using dynamic uniform transformation.
        
        This recalculates the transformation path at each step based on the
        current portfolio state, accounting for any drift from the original plan.
        """
        # Get current portfolio weights from actual holdings
        latest_prices = inputs.prices.iloc[-1]
        portfolio_value = inputs.cash + inputs.quantities @ latest_prices
        
        if portfolio_value > 0:
            w_current = (inputs.quantities * latest_prices / portfolio_value).values
        else:
            # Fallback if portfolio value is zero or negative
            w_current = np.zeros(inputs.n_assets)
        
        # Calculate remaining days in transformation
        remaining_days = max(config.total_days - current_day, 1)
        
        # Calculate the trade needed to close the gap over remaining time
        # This is the key difference from uniform policy - we use current weights
        # rather than the planned trajectory
        gap_to_target = config.w_target - w_current
        daily_adjustment = gap_to_target / remaining_days
        
        # Target weights for next period
        return w_current + daily_adjustment


# Utility functions for creating strategies
def create_uniform_strategy_function(w_initial: np.ndarray, w_target: np.ndarray, total_days: int):
    """
    Create a uniform transformation strategy function.
    
    Args:
        w_initial: Initial portfolio weights
        w_target: Target portfolio weights
        total_days: Total transformation period in days
        
    Returns:
        Strategy function compatible with run_backtest()
    """
    import transformation
    
    policy = UniformTransformationPolicy()
    config = transformation.TransformationConfig(w_initial, w_target, total_days)
    return transformation.create_transformation_strategy(policy, config)


def create_dynamic_uniform_strategy_function(w_initial: np.ndarray, w_target: np.ndarray, total_days: int):
    """
    Create a dynamic uniform transformation strategy function.
    
    Args:
        w_initial: Initial portfolio weights
        w_target: Target portfolio weights
        total_days: Total transformation period in days
        
    Returns:
        Strategy function compatible with run_backtest()
    """
    import transformation
    
    policy = DynamicUniformTransformationPolicy()
    config = transformation.TransformationConfig(w_initial, w_target, total_days)
    return transformation.create_transformation_strategy(policy, config) 