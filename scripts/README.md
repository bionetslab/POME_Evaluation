# Survival Analysis Figure Generation

This directory contains scripts to generate the survival analysis figure from the notebook analysis.

## Usage

From the project root directory, run:

```bash
python scripts/generate_figure.py
```

This will:
1. Load all HANCOCK sample files from the `data/` directory
2. Compute k-NN graphs with year-based constraints
3. Generate treatment recommendations based on neighbor survival outcomes
4. Perform paired Wilcoxon signed-rank tests
5. Generate a manuscript-ready PDF figure with the following panels:
   - **Panel A**: Proportion of samples following the recommendation
   - **Panel B**: Survivor fractions by embedding size with p-values annotated

The output PDF is saved to `figures/survival_analysis.pdf`.

## Configuration

To modify analysis parameters, edit `scripts/generate_figure.py`:
- `data_glob`: Pattern to match data files (default: `'data/HANCOCK_samples_*.tsv'`)
- `feature_cols`: Feature columns to use for k-NN (default: `dim_0` through `dim_15`)
- `n_neighbors`: Number of neighbors in k-NN graph (default: 10)
- `year_filter`: Year to filter data on (default: 2019)
- `output_pdf`: Path for output PDF (default: `'figures/survival_analysis.pdf'`)

## Modules

The analysis is organized into the following modules in `src/hancock_survival/`:

- **data_processing.py**: Data loading, modality computation, k-NN graph construction
- **analysis.py**: Treatment recommendation logic and multi-file analysis
- **statistics.py**: Wilcoxon signed-rank statistical tests
- **plotting.py**: Figure generation with matplotlib
