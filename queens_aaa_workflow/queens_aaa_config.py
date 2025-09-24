# queens_aaa_config.py
import sys
import json
from pathlib import Path

# NEW consolidated repo paths
repo_root = Path(__file__).parent.parent  # Go up to queens/ root
sys.path.insert(0, str(repo_root / 'src'))

from queens.distributions import Normal, Uniform
from queens.parameters import Parameters
from queens.global_settings import GlobalSettings

# Point to AORTA's data (now inside queens repo)
AORTA_DATA = repo_root / 'Aorta' / 'ofCaseGen' / 'Method_4' / 'data'

def load_fitted_distributions(gender='F', age_group='70-79'):
    """Load fitted distributions from AORTA data."""
    fitted_file = AORTA_DATA / 'processed' / 'fitted_distributions.json'
    
    if not fitted_file.exists():
        raise FileNotFoundError(f"Fitted distributions file not found: {fitted_file}")
    
    with open(fitted_file, 'r') as f:
        fitted_data = json.load(f)
    
    # Extract for specific demographic
    key = f"{gender}_{age_group}"
    if key not in fitted_data:
        raise ValueError(f"No data for {gender} age {age_group}")
    
    return fitted_data[key]

def create_aaa_parameters(gender='F', age_group='70-79'):
    """Create QUEENS parameter object from fitted distributions."""
    
    # Load your fitted distributions
    fitted = load_fitted_distributions(gender, age_group)
    
    # Create QUEENS parameters with distributions
    parameters = Parameters(
        neck_diameter_1=Normal(
            mean=fitted['neck_d1']['mean'],
            covariance=fitted['neck_d1']['std']**2  # FIXED: use 'covariance' (variance = std²)
        ),
        neck_diameter_2=Normal(
            mean=fitted['neck_d2']['mean'], 
            covariance=fitted['neck_d2']['std']**2  # FIXED: use 'covariance' (variance = std²)
        ),
        max_diameter=Normal(
            mean=fitted['max_d']['mean'],
            covariance=fitted['max_d']['std']**2    # FIXED: use 'covariance' (variance = std²)
        ),
        distal_diameter=Normal(
            mean=fitted['distal_d']['mean'],
            covariance=fitted['distal_d']['std']**2  # FIXED: use 'covariance' (variance = std²)
        )
    )
    
    return parameters

def create_fixed_aaa_parameters():
    """Create fixed parameters for testing (if no fitted data available)."""
    parameters = Parameters(
        neck_diameter_1=Normal(mean=2.5, covariance=0.2**2),    # FIXED: covariance = variance = std²
        neck_diameter_2=Normal(mean=2.8, covariance=0.2**2),    # FIXED: covariance = variance = std²  
        max_diameter=Normal(mean=5.5, covariance=0.5**2),       # FIXED: covariance = variance = std²
        distal_diameter=Normal(mean=2.2, covariance=0.2**2)     # FIXED: covariance = variance = std²
    )
    
    return parameters