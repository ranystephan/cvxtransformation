# experiments/save_experiment_metadata.py

import pickle
import pandas as pd
from loguru import logger
from utils import checkpoints_path

def save_experiment_metadata(experiment_name: str, metadata: dict):
    """
    Save experiment metadata including target weights, initial weights, and other parameters
    for comprehensive analysis.
    
    Args:
        experiment_name: Name of the experiment (e.g., "Dynamic_Uniform")
        metadata: Dictionary containing experiment metadata
    """
    
    metadata_path = checkpoints_path() / f"experiment_metadata_{experiment_name}.pickle"
    
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Saved experiment metadata to: {metadata_path}")
    logger.info(f"Metadata keys: {list(metadata.keys())}")

def load_experiment_metadata(experiment_name: str) -> dict | None:
    """
    Load experiment metadata for analysis.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary containing experiment metadata or None if not found
    """
    
    metadata_path = checkpoints_path() / f"experiment_metadata_{experiment_name}.pickle"
    
    try:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        logger.info(f"Successfully loaded experiment metadata from: {metadata_path}")
        return metadata
        
    except FileNotFoundError:
        logger.warning(f"No metadata found for experiment: {experiment_name}")
        return None

# Example usage in your experiment script:
"""
# In run_advanced_transitions.py, add this after creating target_weights:

from save_experiment_metadata import save_experiment_metadata

# Save experiment metadata
experiment_metadata = {
    'initial_weights': initial_weights,
    'target_weights': target_weights,
    'transition_period_days': transition_period_days,
    'transition_start_date': transition_start_date,
    'construction_date': construction_date,
    'group_A_assets': list(group_A_assets),
    'group_B_assets': list(group_B_assets),
    'risk_target_initial': 0.05,
    'risk_target_final': 0.15,
    'strategy_name': 'dynamic_uniform_strategy'
}

save_experiment_metadata("Dynamic_Uniform", experiment_metadata)
""" 