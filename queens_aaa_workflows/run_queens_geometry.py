# run_queens_geometry.py
import numpy as np
from queens.global_settings import GlobalSettings
from queens.parameters import Parameters  
from queens.iterators import LatinHypercubeSampling
from queens_aaa_config import create_aaa_parameters
from aorta_queens_model import AortaGeometryModel

def main():
    """Run geometry generation through QUEENS."""
    
    print("="*60)
    print("AORTA + QUEENS: Geometry Generation")
    print("="*60)
    
    # 1. Set up QUEENS global settings
    global_settings = GlobalSettings(
        experiment_name="aaa_geometry_generation",
        output_dir="./queens_output"
    )
    
    with global_settings:
        # 2. Define parameters from fitted distributions
        print("\n📊 Setting up parameters...")
        parameters = create_aaa_parameters(gender='M', age_group='60-69')
        
        # 3. Create geometry model
        print("\n🏗️ Initializing AORTA geometry model...")
        model = AortaGeometryModel(output_dir='./geometries')
        
        # 4. Set up Latin Hypercube Sampling
        print("\n🎲 Setting up Latin Hypercube Sampling...")
        lhs = LatinHypercubeSampling(
            parameters=parameters,
            seed=42
        )
        
        # 5. Generate samples
        n_samples = 10  # Start with 10 for testing
        print(f"\n📐 Generating {n_samples} geometry samples...")
        samples = lhs.draw(n_samples)
        
        # 6. Generate geometries
        print("\n🔄 Generating geometries...")
        results = model.evaluate(samples)
        
        # 7. Summary
        print("\n" + "="*60)
        print("✅ GEOMETRY GENERATION COMPLETE")
        print("="*60)
        print(f"Generated {n_samples} geometries")
        print(f"Output directory: ./geometries")
        print(f"Results shape: {results.shape}")
        
        return results

if __name__ == "__main__":
    main()