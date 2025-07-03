#!/usr/bin/env python3
"""
Run All Transformation Experiments

This script runs all three transformation experiments in sequence:
1. Disjoint Asset Groups Experiment
2. Portfolio Lifecycle Experiment  
3. Volume-Adaptive Experiment

Usage:
    python run_all_experiments.py
"""

import sys
import time
import traceback
from pathlib import Path

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent / "experiments" / "main_experiments"))

def run_experiment(experiment_name: str, experiment_module: str):
    """Run a single experiment with error handling."""
    print(f"\n{'='*80}")
    print(f"STARTING: {experiment_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run the experiment
        module = __import__(experiment_module)
        
        if hasattr(module, 'run_disjoint_groups_experiment'):
            results = module.run_disjoint_groups_experiment()
        elif hasattr(module, 'run_portfolio_lifecycle_experiment'):
            results = module.run_portfolio_lifecycle_experiment()
        elif hasattr(module, 'run_volume_adaptive_experiment'):
            results = module.run_volume_adaptive_experiment()
        else:
            print(f"ERROR: No main experiment function found in {experiment_module}")
            return False
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {experiment_name} completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        if results is not None:
            if hasattr(results, '__len__'):
                print(f"📊 Generated {len(results)} results")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error in {experiment_name}: {e}")
        print(f"   Make sure all required modules are available")
        return False
        
    except Exception as e:
        print(f"❌ Error in {experiment_name}: {e}")
        print(f"   Traceback:")
        traceback.print_exc()
        return False

def main():
    """Run all experiments."""
    print("🚀 Starting All Transformation Experiments")
    print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Define experiments to run
    experiments = [
        ("Disjoint Asset Groups Experiment", "disjoint_groups_experiment"),
        ("Portfolio Lifecycle Experiment", "portfolio_lifecycle_experiment"),
        ("Volume-Adaptive Experiment", "volume_adaptive_experiment")
    ]
    
    results = []
    
    for experiment_name, experiment_module in experiments:
        success = run_experiment(experiment_name, experiment_module)
        results.append((experiment_name, success))
        
        if not success:
            print(f"\n⚠️  {experiment_name} failed, but continuing with remaining experiments...")
    
    # Summary
    overall_end = time.time()
    total_duration = overall_end - overall_start
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Total experiments: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {total - successful}")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds")
    
    print(f"\nDetailed results:")
    for experiment_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {experiment_name}: {status}")
    
    # Final status
    if successful == total:
        print(f"\n🎉 All experiments completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - successful} experiment(s) failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 