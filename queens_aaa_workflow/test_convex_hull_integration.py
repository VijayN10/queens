#!/usr/bin/env python3
"""
Test convex hull integration with QUEENS workflow
"""

import sys
import numpy as np
from pathlib import Path

# Get repo root
repo_root = Path(__file__).parent.parent

# Ensure we can import QUEENS
sys.path.insert(0, '/home/a11evina/queens/src')

# Import our integration modules
from aorta_simple_wrapper import AortaGeometryModel

def test_convex_hull_validation():
    """Test convex hull validation without generating full geometries."""
    print("🧪 Testing Convex Hull Validation Integration")
    print("=" * 50)

    # Define convex hull metadata path (from Method 4)
    convex_hull_path = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4' / 'data' / 'processed' / 'convex_hull_metadata.json'

    print(f"\n📍 Convex hull metadata path: {convex_hull_path}")
    print(f"   Exists: {convex_hull_path.exists()}")

    if not convex_hull_path.exists():
        print("❌ Convex hull metadata file not found!")
        return False

    # Create model with convex hull validation
    model = AortaGeometryModel(
        output_dir='./test_output',
        create_openfoam_cases=False,  # Disable OpenFOAM for quick test
        enable_morphing=False,
        random_seed=42,
        # Enable convex hull validation
        convex_hull_metadata_path=str(convex_hull_path),
        gender='F',  # Female
        age_group='70-79',  # 70-79 age group
        outlier_method='manual'  # Use manual outlier detection
    )

    print("\n✅ Model created with convex hull validation")

    # Test parameter validation with known good and bad samples
    print("\n" + "=" * 50)
    print("Testing parameter validation:")
    print("=" * 50)

    # Test samples (these should be representative of the F, 70-79 cohort)
    test_samples = [
        np.array([22.0, 24.0, 45.0, 18.0]),  # Should be valid (typical values)
        np.array([15.0, 18.0, 30.0, 12.0]),  # Might be invalid (small values)
        np.array([30.0, 35.0, 65.0, 25.0]),  # Might be invalid (large values)
        np.array([23.5, 26.0, 50.0, 20.0]),  # Should be valid (moderate values)
    ]

    for i, sample in enumerate(test_samples):
        print(f"\n🔍 Sample {i+1}: {sample}")
        validation = model.validate_parameters(sample)

        if validation['all_valid']:
            print(f"   ✅ VALID - All parameter pairs inside convex hull")
        else:
            print(f"   ⚠️  INVALID - Some parameters outside convex hull:")
            for comparison, result in validation.items():
                if comparison != 'all_valid' and isinstance(result, dict):
                    status = "✅" if result.get('valid') else "❌"
                    print(f"      {status} {comparison}")

    print("\n" + "=" * 50)
    print("✅ Convex hull validation test completed successfully!")
    print("=" * 50)

    return True

if __name__ == "__main__":
    success = test_convex_hull_validation()

    if success:
        print("\n✅ Test PASSED!")
        print("\n📋 Next steps:")
        print("   1. Run full integration with: python run_full_integration.py")
        print("   2. Check validation results in: geometries/validation_summary.json")
        print("   3. Inspect individual case validations in: geometries/case_*/validation_results.json")
    else:
        print("\n❌ Test FAILED!")

    sys.exit(0 if success else 1)