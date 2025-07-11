# experiments/example_analysis_usage.py
# Example usage of the enhanced analyze_results.py

import numpy as np
import pickle
from loguru import logger

from analyze_results import generate_comprehensive_report
from utils import checkpoints_path

def analyze_transformation_experiment():
    """
    Example of how to analyze a transformation experiment with target and initial weights.
    """
    
    # Example result filename (adjust to match your actual file)
    result_filename = "transition_result_Dynamic_Uniform.pickle"
    result_name = "Dynamic_Uniform_Strategy"
    
    try:
        # Load the backtest result
        result_path = checkpoints_path() / result_filename
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Successfully loaded result from: {result_path}")
        
        # For transformation experiments, define your target and initial weights
        # These should match what you used in your experiment
        n_assets = result.quantities.shape[1]
        
        # Example: transforming from 100% cash to 60/40 portfolio
        initial_weights = np.zeros(n_assets)  # 100% cash initially
        target_weights = np.ones(n_assets) / n_assets * 0.60  # 60% equally weighted
        
        # Generate comprehensive analysis
        generate_comprehensive_report(
            result=result,
            result_name=result_name,
            target_weights=target_weights,
            initial_weights=initial_weights
        )
        
        logger.info("Analysis complete! Check the figures directory for results.")
        
    except FileNotFoundError:
        logger.error(f"Could not find result file: {result_path}")
        logger.error("Please run the experiment script first to generate the result file.")
        return False
    
    return True

def analyze_multiple_strategies():
    """
    Example of how to analyze multiple strategies and compare them.
    """
    
    # List of result files to analyze
    result_files = [
        ("transition_result_Dynamic_Uniform.pickle", "Dynamic_Uniform_Strategy"),
        ("transition_result_Front_Loaded.pickle", "Front_Loaded_Strategy"),
    ]
    
    # Common target weights for comparison
    n_assets = 100  # Adjust based on your data
    target_weights = np.ones(n_assets) / n_assets * 0.60
    initial_weights = np.zeros(n_assets)
    
    for result_filename, result_name in result_files:
        logger.info(f"Analyzing {result_name}...")
        
        try:
            result_path = checkpoints_path() / result_filename
            with open(result_path, "rb") as f:
                result = pickle.load(f)
            
            generate_comprehensive_report(
                result=result,
                result_name=result_name,
                target_weights=target_weights,
                initial_weights=initial_weights
            )
            
        except FileNotFoundError:
            logger.warning(f"Could not find {result_filename}, skipping...")
            continue
    
    logger.info("Multi-strategy analysis complete!")

if __name__ == "__main__":
    # Run single experiment analysis
    analyze_transformation_experiment()
    
    # Uncomment to analyze multiple strategies
    # analyze_multiple_strategies() 