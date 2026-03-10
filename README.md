# POME Evaluation

A Python package for analyzing treatment recommendations in survival data using k-nearest neighbors and statistical testing.

## Overview

This project implements a collaborative filtering-inspired approach to treatment recommendations based on patient survival outcomes. For each patient sample, the package:

1. Identifies k-nearest neighbors based on feature similarity (with temporal constraints)
2. Groups neighbors by treatment modality
3. Calculates survival fractions for each modality among neighbors
4. Recommends the modality with the highest survival fraction
5. Evaluates recommendation adherence against actual treatment

## Installation

### Using conda

```bash
conda env create -f environment.yml
conda activate hancock_survival
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Quick Start

Generate the analysis figure from data files:

```bash
python scripts/generate_survival_figure.py
python scripts/generate_imputation_figure.py
python scripts/generate_imputation_binning_figure.py
```

This will:
- Process all HANCOCK sample files from the `data/` directory
- Compute k-NN graphs with year-based temporal constraints
- Generate treatment recommendations
- Perform paired Wilcoxon signed-rank statistical tests
- Save manuscript-ready PDFs to `output/`

## Project Structure

```
hancock_survival/
├── src/hancock_survival/        # Main package
│   ├── survival_data_processing.py  # Data loading and k-NN graph construction
│   ├── survival_analysis.py         # Treatment recommendation logic
│   ├── survival_statistics.py       # Statistical testing (Wilcoxon)
│   ├── survival_plotting.py         # Survival figure generation
│   ├── imputation_analysis.py       # Imputation rank/distribution analysis
│   └── imputation_plotting.py       # Imputation figure generation
├── scripts/
│   ├── generate_survival_figure.py
│   ├── generate_imputation_figure.py
│   ├── generate_imputation_binning_figure.py
│   ├── generate_survival_results.py
│   └── README.md                 # Script documentation
├── data/                         # HANCOCK sample data files
├── output/                       # Output CSVs and figures (PDF)
├── environment.yml               # Conda environment specification
└── README.md                     # This file
```

## Usage

### As a module

```python
from hancock_survival.survival_analysis import process_multi_file_analysis
from hancock_survival.survival_statistics import perform_wilcoxon_tests
from hancock_survival.survival_plotting import create_survival_figure

# Process data files
results_df, proportions_df = process_multi_file_analysis(
    glob_pattern='data/HANCOCK_samples_*.tsv',
    feature_cols=[f'dim_{i}' for i in range(16)],
    year_filter=2019,
    n_neighbors=10
)

# Run statistical tests
wilcoxon_results = perform_wilcoxon_tests(results_df)

# Generate figure
fig = create_survival_figure(
    results_df, 
    proportions_df, 
    wilcoxon_results,
    output_path='output/analysis.pdf'
)
```

## Dependencies

- **Python** ≥ 3.11
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing (k-NN, statistical tests)
- **scikit-learn**: Machine learning (k-NN distance computation)
- **Matplotlib**: Plotting
- **Seaborn**: Statistical data visualization
- **JupyterLab**: Interactive notebooks (optional)

## Methods

### k-NN Graph with Temporal Constraints

Only samples with an earlier `year_of_initial_diagnosis` can serve as neighbors, preserving temporal ordering in the analysis.

### Treatment Recommendation

For each sample, recommendations are based on the survival fraction of neighbors grouped by treatment modality:
- **Modalities**: none, systemic_therapy, radiotherapy, both
- **Selection**: The modality with the highest survival fraction among neighbors is recommended

### Statistical Testing

Paired Wilcoxon signed-rank test comparing survival fractions between samples that followed vs. did not follow the recommendation, with pairing by data file.

## License

[Add your license here]

## Contact

For questions or contributions, please contact the BioNetS Lab.
