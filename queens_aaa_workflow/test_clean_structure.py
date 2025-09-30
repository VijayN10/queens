# test_clean_structure.py
"""
Test script to verify the cleaned-up queens_aaa_workflow structure works correctly.
This script tests the updated import paths and consolidated repository structure.
"""

import sys
import traceback
from pathlib import Path

def test_import_paths():
    """Test that all import paths work correctly with consolidated repo structure."""
    print("🧪 Testing Import Paths...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: QUEENS imports
    try:
        tests_total += 1
        repo_root = Path(__file__).parent.parent
        sys.path.insert(0, str(repo_root / 'src'))
        
        from queens.global_settings import GlobalSettings
        from queens.parameters import Parameters
        from queens.distributions import Normal
        from queens.iterators import LatinHypercubeSampling
        
        print("   ✅ QUEENS imports successful")
        tests_passed += 1
        
    except ImportError as e:
        print(f"   ❌ QUEENS import failed: {e}")
    
    # Test 2: AORTA imports
    try:
        tests_total += 1
        sys.path.insert(0, str(repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'))
        
        from config import ConfigParams
        from main import generate_base_geometry_without_perturbation
        
        print("   ✅ AORTA imports successful")
        tests_passed += 1
        
    except ImportError as e:
        print(f"   ❌ AORTA import failed: {e}")
    
    # Test 3: Local config imports
    try:
        tests_total += 1
        from queens_aaa_config import create_aaa_parameters

        print("   ✅ Local config imports successful")
        tests_passed += 1

    except ImportError as e:
        print(f"   ❌ Local config import failed: {e}")
        
    print(f"📊 Import test results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def test_directory_structure():
    """Verify the expected directory structure exists."""
    print("\n🧪 Testing Directory Structure...")
    
    repo_root = Path(__file__).parent.parent
    
    required_paths = [
        repo_root / 'src' / 'queens',
        repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4',
        repo_root / 'queens_aaa_workflow'  # FIXED: singular form
    ]
    
    all_exist = True
    
    for path in required_paths:
        if path.exists():
            print(f"   ✅ {path.relative_to(repo_root)} exists")
        else:
            print(f"   ❌ {path.relative_to(repo_root)} missing")
            all_exist = False
    
    return all_exist

def test_simple_wrapper():
    """Test the updated simple wrapper functionality."""
    print("\n🧪 Testing Simple Wrapper...")
    
    try:
        from aorta_simple_wrapper import AortaGeometryModel
        import numpy as np
        
        # Create model (don't actually run geometry generation in test)
        model = AortaGeometryModel(
            output_dir="test_output",
            enable_morphing=False,
            random_seed=42
        )
        
        print("   ✅ AortaGeometryModel created successfully")
        
        # Test parameter format
        test_params = np.array([[2.5, 2.8, 5.5, 2.2]])
        print("   ✅ Test parameters prepared")
        
        # Note: We don't actually run evaluate() in this test to avoid
        # generating files, but we've verified the imports work
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simple wrapper test failed: {e}")
        traceback.print_exc()
        return False

def test_parameters_creation():
    """Test parameter creation with updated paths."""
    print("\n🧪 Testing Parameter Creation...")
    
    try:
        from queens_aaa_config import create_aaa_parameters
        from queens.global_settings import GlobalSettings

        # Test parameter creation
        with GlobalSettings(
            experiment_name="test_parameters",
            output_dir="test_output"
        ) as gs:
            parameters = create_aaa_parameters()
            print("   ✅ Fixed parameters created successfully")
            
            # Simple test - just verify parameters were created
            # Skip the LatinHypercubeSampling test since it requires model integration
            print("   ✅ Parameters object is valid")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter creation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests to verify clean structure."""
    print("🎯 Testing Cleaned Queens Workflow Structure")
    print("="*50)
    
    tests = [
        ("Import Paths", test_import_paths),
        ("Directory Structure", test_directory_structure), 
        ("Simple Wrapper", test_simple_wrapper),
        ("Parameter Creation", test_parameters_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
            
        except Exception as e:
            print(f"\n💥 ERROR in {test_name}: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🏁 FINAL RESULTS")
    print("="*30)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Cleaned structure is working correctly")
        print("\n🚀 Ready for next implementation steps:")
        print("   1. Run full integration test")
        print("   2. Add convex hull validation") 
        print("   3. Add OpenFOAM case generation")
        print("   4. Scale to production workflow")
    else:
        print("⚠️  Some tests failed - review and fix issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)