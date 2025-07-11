# experiments/extract_weights.py

import pickle
import pandas as pd
from loguru import logger
from utils import checkpoints_path

def extract_weights_from_experiment():
    """
    Extract initial and target weights from the experiment and save them
    for use in the comprehensive analysis.
    """
    
    # Load the result file
    result_filename = "transition_result_Dynamic_Uniform.pickle"
    result_path = checkpoints_path() / result_filename
    
    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Successfully loaded result from: {result_path}")
    except FileNotFoundError:
        logger.error(f"Could not find result file: {result_path}")
        return None, None
    
    # Extract initial and final weights from the result
    initial_weights = result.asset_weights.iloc[0]
    final_weights = result.asset_weights.iloc[-1]
    
    # Note: The target weights are not stored in the BacktestResult
    # You would need to extract them from your experiment script
    # For now, we'll use the final weights as a proxy for target weights
    # In practice, you should modify your experiment to save the target weights
    
    logger.info("Extracted weights:")
    logger.info(f"Initial weights shape: {initial_weights.shape}")
    logger.info(f"Final weights shape: {final_weights.shape}")
    
    # Save the weights for use in analysis
    weights_data = {
        'initial_weights': initial_weights,
        'final_weights': final_weights,
        # 'target_weights': target_weights  # Add this when available
    }
    
    weights_path = checkpoints_path() / "extracted_weights.pickle"
    with open(weights_path, "wb") as f:
        pickle.dump(weights_data, f)
    
    logger.info(f"Saved weights to: {weights_path}")
    
    return initial_weights, final_weights

def load_extracted_weights():
    """
    Load the extracted weights for use in analysis.
    """
    weights_path = checkpoints_path() / "extracted_weights.pickle"
    
    try:
        with open(weights_path, "rb") as f:
            weights_data = pickle.load(f)
        
        initial_weights = weights_data['initial_weights']
        final_weights = weights_data['final_weights']
        target_weights = weights_data.get('target_weights', None)
        
        logger.info("Successfully loaded extracted weights")
        return initial_weights, final_weights, target_weights
        
    except FileNotFoundError:
        logger.warning("No extracted weights found. Run extract_weights_from_experiment() first.")
        return None, None, None

if __name__ == "__main__":
    extract_weights_from_experiment() 