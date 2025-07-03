#!/usr/bin/env python3
"""
Quick Test for Transformation Experiments

This script tests that all the experiment files can be imported and their 
basic functionality works before running the full experiments.
"""

import sys
from pathlib import Path

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent / "experiments" / "main_experiments"))
sys.path.append(str(Path(__file__).parent / "experiments" / "core"))

def test_imports():
    """Test that all experiment modules can be imported."""
    print("Testing imports...")
    
    try:
        import disjoint_groups_experiment
        print("✅ disjoint_groups_experiment imported successfully")
    except Exception as e:
        print(f"❌ disjoint_groups_experiment import failed: {e}")
        return False
    
    try:
        import portfolio_lifecycle_experiment
        print("✅ portfolio_lifecycle_experiment imported successfully")
    except Exception as e:
        print(f"❌ portfolio_lifecycle_experiment import failed: {e}")
        return False
    
    try:
        import volume_adaptive_experiment
        print("✅ volume_adaptive_experiment imported successfully")
    except Exception as e:
        print(f"❌ volume_adaptive_experiment import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that data loading works."""
    print("\nTesting data loading...")
    
    try:
        from backtest import load_data
        prices, _, _, _, _ = load_data()
        print(f"✅ Data loaded successfully: {prices.shape[1]} assets, {prices.shape[0]} days")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_transformation_imports():
    """Test that transformation policy imports work."""
    print("\nTesting transformation policy imports...")
    
    try:
        from transformation import (
            UniformTransformationPolicy,
            DynamicUniformTransformationPolicy,
            TransformationConfig,
            create_transformation_strategy
        )
        print("✅ Core transformation imports working")
    except Exception as e:
        print(f"❌ Core transformation imports failed: {e}")
        return False
    
    try:
        from transformation import UnivariateScalarTrackingPolicy
        print("✅ Univariate policy import working")
    except Exception as e:
        print(f"❌ Univariate policy import failed: {e}")
        return False
    
    return True

def test_enhanced_backtest():
    """Test that enhanced backtest can be imported."""
    print("\nTesting enhanced backtest import...")
    
    try:
        from enhanced_backtest import run_enhanced_backtest
        print("✅ Enhanced backtest import working")
        return True
    except Exception as e:
        print(f"❌ Enhanced backtest import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Transformation Experiments Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_transformation_imports,
        test_enhanced_backtest
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Experiments are ready to run.")
        print("\nYou can now run:")
        print("  python run_all_experiments.py")
        print("  or individual experiments in the experiments/ directory")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues before running experiments.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 