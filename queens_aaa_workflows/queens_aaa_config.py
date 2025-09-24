# queens_aaa_config.py
from queens.distributions import Normal, Uniform
from queens.parameters import Parameters
from queens.global_settings import GlobalSettings
import json

# Point to AORTA's data
AORTA_DATA = Path('/home/a11evina/Aorta/ofCaseGen/Method_4/data')

def load_fitted_distributions(gender='M', age_group='60-69'):
    fitted_file = AORTA_DATA / 'processed/fitted_distributions.json'
    
    with open(fitted_file, 'r') as f:
        fitted_data = json.load(f)
    
    # Extract for specific demographic
    key = f"{gender}_{age_group}"
    if key not in fitted_data:
        raise ValueError(f"No data for {gender} age {age_group}")
    
    return fitted_data[key]

def create_aaa_parameters(gender='M', age_group='60-69'):
    """Create QUEENS parameter object from fitted distributions."""
    
    # Load your fitted distributions
    fitted = load_fitted_distributions(gender, age_group)
    
    # Create QUEENS parameters with distributions
    parameters = Parameters(
        neck_diameter_1=Normal(
            mean=fitted['neck_d1']['mean'],
            std=fitted['neck_d1']['std']
        ),
        neck_diameter_2=Normal(
            mean=fitted['neck_d2']['mean'], 
            std=fitted['neck_d2']['std']
        ),
        max_diameter=Normal(
            mean=fitted['max_d']['mean'],
            std=fitted['max_d']['std']
        ),
        distal_diameter=Normal(
            mean=fitted['distal_d']['mean'],
            std=fitted['distal_d']['std']
        )
    )
    
    return parameters