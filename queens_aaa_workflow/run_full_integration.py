# run_full_integration.py
"""
UPDATED: Full QUEENS-AORTA integration with consolidated repo structure.
This script demonstrates the complete workflow with updated import paths.
"""

import sys
import numpy as np
from pathlib import Path

# NEW consolidated repo paths
repo_root = Path(__file__).parent.parent  # Go up to queens/ root
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4'))

# QUEENS imports
from queens.global_settings import GlobalSettings
from queens.distributions import Normal
from queens.parameters import Parameters
from queens.iterators import LatinHypercubeSampling

# AORTA imports (now with relative paths)
from config import ConfigParams
from main import generate_base_geometry_without_perturbation, apply_perturbation
from src.vesselGen.save_stl_from_patches import save_stl_from_patches

# Local imports
from queens_aaa_config import create_aaa_parameters, create_fixed_aaa_parameters

class AortaGeometryModel:
    """UPDATED: Simple AORTA wrapper with consolidated repo paths."""
    
    def __init__(self, output_dir="queens_output", enable_morphing=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create geometry output directory
        self.geometry_dir = Path("geometries")
        self.geometry_dir.mkdir(exist_ok=True)
        
        self.enable_morphing = enable_morphing
        self.case_counter = 0
        
        print(f"✅ AortaGeometryModel initialized")
        print(f"   📁 Output directory: {self.output_dir.absolute()}")
        print(f"   📁 Geometry directory: {self.geometry_dir.absolute()}")
        print(f"   🔄 Morphing enabled: {enable_morphing}")

    def evaluate(self, samples):
        """Generate AORTA geometries for given parameter samples."""
        print(f"\n🚀 Starting AORTA geometry generation for {len(samples)} samples...")
        
        results = []
        
        for i, sample in enumerate(samples):
            case_id = f"case_{i:03d}"
            print(f"\n📐 Generating geometry {i+1}/{len(samples)} - {case_id}")
            
            try:
                # Create config with current parameters
                config = ConfigParams()
                
                # Update geometric parameters
                config.neck_diameter_1 = float(sample[0])
                config.neck_diameter_2 = float(sample[1]) 
                config.max_diameter = float(sample[2])
                config.distal_diameter = float(sample[3])
                
                print(f"   Parameters: neck_d1={sample[0]:.3f}, neck_d2={sample[1]:.3f}, "
                      f"max_d={sample[2]:.3f}, distal_d={sample[3]:.3f}")
                
                # Generate base geometry
                patches = generate_base_geometry_without_perturbation(config)
                print(f"   ✅ Base geometry generated: {len(patches)} patches")
                
                # Apply morphing if enabled
                if self.enable_morphing:
                    patches = apply_perturbation(patches, config)
                    print(f"   🔄 Morphing applied")
                
                # Save geometry as STL
                stl_file = self.geometry_dir / f"{case_id}.stl"
                save_stl_from_patches(patches, str(stl_file))
                print(f"   💾 STL saved: {stl_file}")
                
                # Store result
                results.append({
                    'case_id': case_id,
                    'geometry_file': str(stl_file),
                    'parameters': sample.tolist(),
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error generating geometry: {str(e)}")
                results.append({
                    'case_id': case_id,
                    'geometry_file': None,
                    'parameters': sample.tolist(),
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n✅ Geometry generation complete!")
        print(f"   📊 Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")
        
        return np.array(results)

def main():
    """Main QUEENS-AORTA integration workflow."""
    print("🎯 Starting QUEENS-AORTA Integration Test")
    print("="*50)
    
    # Test with GlobalSettings
    with GlobalSettings(
        experiment_name="queens_aorta_integration", 
        output_dir="queens_output"
    ) as gs:
        print("✅ GlobalSettings context active")
        
        # Try to load fitted parameters, fall back to fixed if needed
        try:
            parameters = create_aaa_parameters(gender='F', age_group='70-79')
            print("✅ Using fitted parameter distributions")
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️  Could not load fitted distributions: {e}")
            print("🔄 Using fixed parameter distributions instead")
            parameters = create_fixed_aaa_parameters()
        
        # Create sampling iterator
        n_samples = 5  # Start with 5 samples for testing
        iterator = LatinHypercubeSampling(
            parameters=parameters,
            num_samples=n_samples,
            seed=42  # For reproducibility
        )
        print(f"✅ LatinHypercubeSampling configured: {n_samples} samples")
        
        # Create AORTA model
        model = AortaGeometryModel(
            output_dir="queens_output",
            enable_morphing=False  # Start without morphing
        )
        
        # Generate samples
        print(f"\n📊 Generating {n_samples} parameter samples...")
        samples = iterator.get_samples()
        print(f"✅ Samples generated: shape {samples.shape}")
        print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Run AORTA evaluation
        print(f"\n🚀 Running AORTA evaluation...")
        results = model.evaluate(samples)
        
        # Display results
        print(f"\n📋 INTEGRATION TEST RESULTS:")
        print("="*50)
        successful_cases = [r for r in results if r['success']]
        failed_cases = [r for r in results if not r['success']]
        
        print(f"✅ Successful cases: {len(successful_cases)}/{len(results)}")
        for case in successful_cases:
            print(f"   📁 {case['case_id']}: {Path(case['geometry_file']).name}")
        
        if failed_cases:
            print(f"❌ Failed cases: {len(failed_cases)}")
            for case in failed_cases:
                print(f"   ⚠️  {case['case_id']}: {case.get('error', 'Unknown error')}")
        
        print(f"\n🎉 QUEENS-AORTA integration test complete!")
        
        return len(successful_cases) == len(results)  # True if all successful

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎊 ALL TESTS PASSED! Integration is working.")
        print("🔄 Next steps:")
        print("   1. Add convex hull validation")
        print("   2. Add OpenFOAM case generation")
        print("   3. Scale to larger sample sets")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("🔧 Debug and retry integration.")
    
    sys.exit(0 if success else 1)