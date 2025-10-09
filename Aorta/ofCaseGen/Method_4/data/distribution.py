import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple

class AgeGroups:
    """Define age group categories"""
    GROUPS = {
        '<50': {'min': 0, 'max': 49},
        '50-59': {'min': 50, 'max': 59},
        '60-69': {'min': 60, 'max': 69},
        '70-79': {'min': 70, 'max': 79},
        '80+': {'min': 80, 'max': float('inf')}
    }
    
    @classmethod
    def get_group(cls, age: int) -> str:
        """Get age group for a given age"""
        for group, limits in cls.GROUPS.items():
            if limits['min'] <= age <= limits['max']:
                return group
        return 'unknown'

class MorphologyAnalyzer:
    def __init__(self, base_dir: Path, fitted_dist_file: str, data_path: str):
        self.base_dir = base_dir
        self.fitted_dist_file = fitted_dist_file
        self.df = pd.read_excel(data_path, header=None).iloc[10:268]
        self.setup_parameters()
        self.setup_style()
        
    def setup_parameters(self):
        self.param_indices = {
            'neck_diameter_1': 9,
            'neck_diameter_2': 10,
            'max_diameter': 13,
            'distal_diameter': 14
        }
        
        self.param_mapping = {
            'neck_diameter_1': ['Inlet', 'Neck 1'],
            'neck_diameter_2': ['Neck 2'],
            'max_diameter': ['Maximum Aneurysm'],
            'distal_diameter': ['Distal']
        }
        
    def setup_style(self):
        sns.set_theme(style="white")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14
        })

    def get_morphed_cases(self) -> Dict[int, List[Path]]:
        morphed_cases = {}
        pattern = r'AAA_([MF])_(\d+-\d+)_stat_(\d+)_prob_distribution_morph_\d+'
        
        print(f"Searching in directory: {self.base_dir}")
        for case_dir in self.base_dir.glob("*prob_distribution_morph*"):
            print(f"Found case: {case_dir.name}")
            match = re.match(pattern, case_dir.name)
            if match:
                stat_var = int(match.group(3))
                if stat_var not in morphed_cases:
                    morphed_cases[stat_var] = []
                morphed_cases[stat_var].append(case_dir)
        
        return morphed_cases

    def collect_measurements(self, cases: List[Path], validation_name: str) -> List[float]:
        measurements = []
        for case_dir in cases:
            validation = self.load_validation_results(case_dir)
            if validation and validation_name in validation['measurements']:
                measurements.append(validation['measurements'][validation_name])
        return measurements

    def analyze_distributions(self, all_together: bool = False):
        morphed_cases = self.get_morphed_cases()
        
        if all_together:
            # Combine all morphed cases but still pass as a dictionary with key "combined"
            all_cases = [case for cases in morphed_cases.values() for case in cases]
            self.create_distribution_plots({"combined": all_cases}, "combined")
        else:
            # Keep existing behavior for separate analysis
            for stat_var, cases in morphed_cases.items():
                if cases:
                    self.create_distribution_plots({stat_var: cases}, f"stat{stat_var}")

    def create_distribution_plots(self, cases_dict: Dict[int, List[Path]], suffix: str):
        for stat_group, cases in cases_dict.items():
            if not cases:
                continue
                
            gender, age_group = self.parse_case_demographics(cases[0].name)
            fitted_dists = self.load_fitted_distribution(gender, age_group)
            
            original_data = self.get_original_data(gender, age_group)
            
            for param_name, validation_names in self.param_mapping.items():
                if param_name not in fitted_dists:
                    continue
                
                param_data = original_data[param_name]
                
                for validation_name in validation_names:
                    morphed_measurements = self.collect_measurements(cases, validation_name)
                    
                    if not morphed_measurements:
                        continue
                    
                    dist_info = fitted_dists[param_name]
                    morphed_dist = self.fit_morphed_distribution(morphed_measurements, dist_info['range'])
                    
                    if not morphed_dist:
                        continue
                    
                    self.plot_distributions(
                        param_data, 
                        morphed_measurements,
                        dist_info,
                        morphed_dist,
                        validation_name,
                        gender,
                        age_group,
                        stat_group,
                        param_name,
                        suffix
                    )

    def plot_distributions(self, param_data, morphed_measurements, dist_info, morphed_dist,
                        validation_name, gender, age_group, stat_var, param_name, suffix):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate common bins based on combined range of both datasets
        min_val = min(min(param_data), min(morphed_measurements))
        max_val = max(max(param_data), max(morphed_measurements))
        n_bins = int(np.sqrt(len(param_data) + len(morphed_measurements)))  # Sturges' formula
        bins = np.linspace(min_val, max_val, n_bins)
        
        # Plot histograms using common bins
        n, _, _ = ax.hist(param_data, bins=bins, density=True, alpha=0.4,
                        color='lightblue', label='Patient Data')
        n, _, _ = ax.hist(morphed_measurements, bins=bins, density=True, 
                        alpha=0.4, color='lightgreen', label='Morphed Data')
        
        
        x_range = np.linspace(dist_info['range'][0], dist_info['range'][1], 100)
        
        orig_dist = self.get_distribution_function(dist_info['distribution'], 
                                                 dist_info['parameters'])
        ax.plot(x_range, orig_dist.pdf(x_range), 'b-', lw=2, 
                label='Patient Data Distribution')
        
        morph_dist = self.get_distribution_function(morphed_dist['distribution'], 
                                                  morphed_dist['parameters'])
        ax.plot(x_range, morph_dist.pdf(x_range), 'g-', lw=2,
                label='Morphed Data Distribution')
        
        # Add statistics
        stats_text = (
            f"Patient Data:\n"
            f"Distribution: {dist_info['distribution']}\n"
            f"Mean: {dist_info['mean']:.1f} mm\n"
            f"Std: {dist_info['std']:.1f} mm\n"
            f"Sample Size: {len(param_data)}\n"
            f"\nMorphed Data:\n"
            f"Distribution: {morphed_dist['distribution']}\n"
            f"Mean: {morphed_dist['mean']:.1f} mm\n"
            f"Std: {morphed_dist['std']:.1f} mm\n"
            f"Sample Size: {len(morphed_measurements)}"
        )
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top', horizontalalignment='right',
                fontsize=9)
        
        ax.set_xlabel('Diameter (mm)')
        ax.set_ylabel('Probability Density')
        
        title = f'{validation_name} Distribution Comparison\n{gender}, Age {age_group}'
        if suffix != "combined":
            title += f', Statistical Variant {stat_var}'
        ax.set_title(title)
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                 fontsize=8, framealpha=0.8)
        
        plt.tight_layout()
        save_dir = Path('data/processed/distribution_plots')
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / f'{param_name}_{validation_name.lower()}_{suffix}_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Helper methods that were previously defined
    @staticmethod
    def load_validation_results(case_dir: Path) -> Optional[Dict]:
        validation_file = case_dir / 'parameters' / 'validation_results.json'
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                return json.load(f)
        return None

    @staticmethod
    def parse_case_demographics(case_name: str) -> Tuple[str, str]:
        match = re.match(r'AAA_([MF])_(\d+-\d+)_.*', case_name)
        if not match:
            raise ValueError(f"Invalid case directory name format: {case_name}")
        return match.group(1), match.group(2)

    def load_fitted_distribution(self, gender: str, age_group: str) -> Dict:
        with open(self.fitted_dist_file, 'r') as f:
            data = json.load(f)
        try:
            return data['demographics'][gender][age_group]
        except KeyError as e:
            print(f"Error: Could not find distribution for gender={gender}, age_group={age_group}")
            raise

    @staticmethod
    def get_distribution_function(dist_name: str, params: list):
        if dist_name == 'norm':
            return stats.norm(*params)
        elif dist_name == 'lognorm':
            return stats.lognorm(*params)
        elif dist_name == 'gamma':
            return stats.gamma(*params)
        elif dist_name == 'weibull_min':
            return stats.weibull_min(*params)
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")

    @staticmethod
    def fit_morphed_distribution(measurements: list, param_range: list) -> Optional[Dict]:
        if len(measurements) < 3:
            return None
            
        data = np.array(measurements)
        distributions = [
            ('norm', stats.norm),
            ('lognorm', stats.lognorm),
            ('gamma', stats.gamma),
            ('weibull_min', stats.weibull_min)
        ]
        
        best_fit = {'sse': float('inf')}
        
        for dist_name, dist in distributions:
            try:
                params = dist.fit(data)
                fitted_dist = dist(*params)
                
                hist, bin_edges = np.histogram(data, bins='auto', density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf = fitted_dist.pdf(bin_centers)
                
                sse = np.sum((hist - pdf) ** 2)
                
                if sse < best_fit['sse']:
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': [float(p) for p in params],
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'range': param_range,
                    }
            except:
                continue
                
        return best_fit

    def get_original_data(self, gender: str, age_group: str) -> Dict[str, np.ndarray]:
        param_data = {}
        
        gender_mask = slice(None) if gender == 'All' else (self.df.iloc[:, 8] == gender)
        
        if age_group == 'All':
            age_mask = slice(None)
        else:
            age_limits = AgeGroups.GROUPS[age_group]
            ages = pd.to_numeric(self.df.iloc[:, 7], errors='coerce')
            age_mask = (ages >= age_limits['min']) & (ages <= age_limits['max'])
        
        for param_name, idx in self.param_indices.items():
            if gender == 'All' and age_group == 'All':
                data = self.df.iloc[:, idx]
            elif gender == 'All':
                data = self.df[age_mask].iloc[:, idx]
            elif age_group == 'All':
                data = self.df[gender_mask].iloc[:, idx]
            else:
                data = self.df[gender_mask & age_mask].iloc[:, idx]
            
            param_data[param_name] = pd.to_numeric(data, errors='coerce').dropna()
        
        return param_data

# Example usage:
analyzer = MorphologyAnalyzer(
    base_dir=Path('data/output/ofCases'),
    fitted_dist_file='data/processed/fitted_distributions.json',
    data_path='data/input/aaa_data.xlsx'
)

print("Created analyzer")
morphed_cases = analyzer.get_morphed_cases()
print(f"Found {len(morphed_cases)} statistical variants")
for var, cases in morphed_cases.items():
    print(f"Variant {var}: {len(cases)} cases")

print("\nGenerating plots...")
analyzer.analyze_distributions(all_together=False)
print("Done!")