# Aorta OpenFOAM Case Generator

This project provides a framework for generating patient-specific computational models of Abdominal Aortic Aneurysms (AAA) for computational fluid dynamics (CFD) simulations using OpenFOAM.

## Project Overview

The framework allows for the generation of customized AAA geometries based on demographic information (gender and age) and can produce multiple statistical variations and morphed variations of the base geometry. These geometries are then automatically set up as OpenFOAM cases ready for CFD simulation.

### Key Features

- Generate AAA geometries based on statistical distributions derived from patient data
- Customize parameters based on gender and age group
- Create multiple variations of geometries with different statistical properties
- Apply morphing to generate additional geometric variations
- Automatic setup of OpenFOAM cases with appropriate boundary conditions
- Visualization tools for generated geometries
- Patient data boundary analysis for physiologically realistic virtual populations
- Statistical validation of morphed geometries

## Patient Data Integration and Virtual Population Generation

The framework leverages real patient data to create physiologically realistic in silico models:

1. **Statistical Parameter Fitting**: The system analyzes real AAA measurements from patient data, fitting statistical distributions to key geometric parameters (neck diameters, maximum aneurysm diameter, distal diameter) for different demographic groups.

2. **Virtual Population Generation**: Using these fitted distributions, the framework can generate an unlimited number of virtual patients with statistically realistic AAA geometries. These virtual populations maintain the same statistical properties as the original patient cohort.

3. **Morphological Variation**: Beyond statistical sampling, the framework applies morphing techniques to introduce additional geometric diversity while maintaining anatomical plausibility.

4. **Convex Hull Boundary Analysis**: The system creates a multi-dimensional convex hull representing the bounds of physiologically realistic AAA geometries based on real patient data. This allows for:
   - Identification of "interior cases" that fall completely within the convex hull boundaries
   - Validation of morphed geometries against physiological constraints
   - Selection of representative geometries for in silico trials

5. **In Silico Trials**: The generated virtual patient cohorts can be used for in silico clinical trials, enabling computational testing of medical devices or treatment strategies across a diverse patient population.

## Directory Structure

- `config.py`: Main configuration parameters for the framework
- `main.py`: Entry point for the framework
- `data/`: Input and output data
  - `input/`: Input data including patient data and base geometries
  - `output/`: Generated cases and results
  - `processed/`: Processed statistical data and distributions
- `src/`: Source code for the framework
  - `vesselGen/`: Vessel geometry generation modules
  - `vesselMorph/`: Morphing functionality for vessel geometries
  - `vesselStats/`: Statistical analysis and parameter sampling
  - `ofCaseGen/`: OpenFOAM case generation
  - `visualization/`: Visualization tools

## How to Create Geometries

### 1. Understanding Configuration Parameters

The main configuration is handled in `config.py`. Key parameters include:

#### Demographics

```python
demographics = Demographics(
    gender='M',  # M for Male, F for Female
    age_group='60-69',  # Available age groups: '50-59', '60-69', '70-79', '80-89'
    stat_variant=5,  # Number of statistical variations to generate
    random_seed=42  # Seed for reproducibility
)
```

#### Morphing Settings

```python
morphing_settings = MorphingSettings(
    enable_morphing=True,  # Whether to generate morphed variations
    num_variations=10,  # Number of morphed variations per statistical variation
    sphere_radius=5.0,  # Parameters for spherical morphing
    num_control_points=50,
    influence_radius=10.0
)
```

#### Vessel Settings

```python
vessel_settings = VesselSettings(
    num_points=100,  # Number of points along vessel centerline
    num_circumference_vertices=100,  # Number of vertices around circumference
    perturbation_range=0.15,  # Range for surface perturbation
    num_cycles=4  # Number of cardiac cycles
)
```

### 2. Setting the Initial Spline and Anatomical Points

There are two ways to define the initial spline points for the AAA geometry:

#### Option 1: Using the Vessel Designer GUI

The framework includes an interactive geometry designer that allows visual placement of anatomical points:

1. Run the geometry designer:
   ```bash
   python -m src.vesselGen.geometry_designer
   ```

2. Use the interface to:
   - Add anatomical points by clicking the "Add Point" button and then clicking on the plot
   - Drag points to adjust their positions
   - Edit point names and diameters using the text boxes
   - Export the points to use in your configuration

3. The designer will generate Python code that you can copy into your configuration file.

#### Option 2: Direct Configuration in Code

You can directly define anatomical points in `config.py` or when creating a `ConfigParams` instance:

```python
from src.vesselGen.vessel_spline import AnatomicalPoint
from config import ConfigParams

# Define anatomical points
points = [
    AnatomicalPoint(0, 0, "Inlet", 22.0),
    AnatomicalPoint(0, 30, "Neck 1", 22.0),
    AnatomicalPoint(10, 47, "Neck 2", 22.0),
    AnatomicalPoint(20, 75.95, "Maximum Aneurysm", 50.0),
    AnatomicalPoint(10, 124.9, "Distal", 18.0)
]

# Create config with these points
config = ConfigParams()
config.anatomical_points = points
```

Key anatomical points include:
- **Inlet**: Proximal end of the AAA
- **Neck 1**: Proximal neck
- **Neck 2**: Distal neck
- **Maximum Aneurysm**: Point of maximum diameter
- **Distal**: Distal end before bifurcation

Each point requires x, y coordinates, a name, and diameter in millimeters.

### 3. Interior Case Analysis and Validation

The framework implements a sophisticated boundary analysis to ensure generated geometries are physiologically plausible:

1. **Convex Hull Construction**:
   - For each pair of geometric parameters (e.g., neck diameter vs. max diameter), the system constructs a convex hull based on patient data
   - These convex hulls define the boundaries of realistic parameter combinations

2. **Interior Case Identification**:
   - Morphed geometries are measured at key anatomical locations
   - A case is considered an "interior case" if all measurements fall within the convex hulls for all parameter pairs
   - Interior cases represent geometries that are statistically similar to real patient data

3. **Validation Process**:
   - Each morphed geometry undergoes validation against these boundaries
   - The system can be configured to only save valid interior cases
   - Validation results and statistics are stored with each case

To generate and analyze interior cases:

```python
# In your configuration
config.demographics.custom_suffix = 'prob_distribution'  # Identifies this batch

# After running the framework, analyze interior cases
python -m data.data_bound_with_morphed_data_manual \
    --data_path data/input/aaa_data.xlsx \
    --output_dir data/processed/bound_plots \
    --ofcases_dir data/output/ofCases \
    --gender F \
    --age_group 70-79 \
    --hull_data_path data/processed/convex_hull_metadata.json \
    --custom_suffix prob_distribution
```

### 4. Modifying the Configuration

To create geometries for different demographics:

1. Open `config.py`
2. Modify the `Demographics` class parameters:
   - `gender`: Set to 'M' for male or 'F' for female
   - `age_group`: Set to '50-59', '60-69', '70-79', or '80-89'

To adjust the morphing behavior:
1. Modify the `MorphingSettings` class parameters
2. Set `enable_morphing` to True/False to enable/disable morphing
3. Adjust `num_variations` to control how many morphed variations to create

To adjust the vessel mesh quality:
1. Modify the `VesselSettings` class parameters
2. Increase `num_points` for smoother centerlines
3. Increase `num_circumference_vertices` for finer mesh resolution around the circumference

### 5. Running the Framework

To execute the framework:

```bash
python main.py
```

This will:
1. Generate base AAA geometry using parameters sampled from statistical distributions
2. Create statistical variations based on demographic information
3. Generate morphed variations of each statistical variation if morphing is enabled
4. Create OpenFOAM cases for all generated geometries
5. Validate each morphed geometry against physiological bounds

### 6. Available Options for Customization

#### Gender Options
- `'M'`: Male
- `'F'`: Female

#### Age Group Options
- `'50-59'`: Ages 50 to 59
- `'60-69'`: Ages 60 to 69
- `'70-79'`: Ages 70 to 79
- `'80-89'`: Ages 80 to 89

#### Geometric Parameters (automatically determined from distributions)
- `neck_diameter_1`: Proximal neck diameter
- `neck_diameter_2`: Distal neck diameter
- `max_diameter`: Maximum aneurysm diameter
- `distal_diameter`: Diameter at bifurcation

## Understanding Generated Cases

Each generated case is organized in a directory with a name following this pattern:
`AAA_{gender}_{age_group}_stat_{stat_variant}_morph_{morph_variant}`

For example: `AAA_M_60-69_stat_1_morph_3` represents:
- Male patient
- Age group 60-69
- Statistical variation 1
- Morphed variation 3

Each case directory contains:
- OpenFOAM case structure (0, constant, system folders)
- Geometry files in STL format
- Parameter files documenting the geometric parameters used
- Visualization images of the geometry
- Validation results for morphed cases

## In Silico Trial Setup

The framework is designed to facilitate in silico clinical trials:

1. **Generate Virtual Population**: Create a large cohort of virtual patients with statistical and morphological diversity:
   ```bash
   python main.py
   ```

2. **Identify Interior Cases**: Use boundary analysis to select physiologically realistic cases:
   ```bash
   python -m data.data_bound_with_morphed_data_manual
   ```

3. **Export Selected Cases**: Package the interior cases for CFD simulation:
   ```bash
   python -m data.zip_it
   ```

4. **Run Simulations**: The generated OpenFOAM cases can be run locally or on computing clusters for your in silico trial.

5. **Analyze Results**: Compare simulation results across your virtual population to draw statistically meaningful conclusions.

## Advanced Usage

For advanced users who want to modify the underlying behavior:

- Modify the morphing algorithm in `src/vesselMorph/spherical_morphing.py`
- Adjust the statistical distributions in `data/processed/fitted_distributions.json`
- Customize OpenFOAM case settings in `src/ofCaseGen/`
- Add new geometric parameters in `src/vesselStats/parameter_sampler.py`

## Prerequisites

- Python 3.8 or higher
- NumPy, SciPy
- OpenFOAM (for running the generated cases)
- VTK (for visualization)

## Contact

For questions or support, please contact vijay.nanadurdikar@postgrad.manchester.ac.uk