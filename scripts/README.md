# Figure Generation Scripts

This directory contains scripts to generate the manuscript figures.

## Usage

From the project root directory, run:

```bash
python scripts/generate_survival_figure.py
```

or for the imputation benchmarking figure:

```bash
python scripts/generate_imputation_figure.py
python scripts/generate_imputation_figure.py --dim 64
```

or for the 12-panel POME binning-effects figure:

```bash
python scripts/generate_imputation_binning_figure.py
```

or for the 2x3 POME embedding-size figure (fixed to z-score binning, 15 bins):

```bash
python scripts/generate_imputation_dim_figure.py
```

`generate_survival_figure.py` will:
1. Load all HANCOCK sample files from the `data/` directory
2. Compute k-NN graphs with year-based constraints
3. Generate treatment recommendations based on neighbor survival outcomes
4. Perform paired Wilcoxon signed-rank tests
5. Generate a manuscript-ready PDF figure with the following panels:
   - **Panel A**: Proportion of samples following the recommendation
   - **Panel B**: Survivor fractions by embedding size with p-values annotated

The output PDF is saved to `output/survival_analysis.pdf`.

`generate_imputation_figure.py` will:
1. Load imputation benchmark CSVs for HANCOCK, MIMIC IV, and TCGA-LUAD
2. Compute rank tables for categorical accuracy (`acc_cat`) and numeric MAE (`mae_cont`)
3. Build long-form metric distributions across all missingness ratios
4. Generate the 8-panel imputation figure

The output PDF is saved to `output/imputation_ranks_with_competitors_<dim>.pdf`.

`generate_imputation_binning_figure.py` will:
1. Load imputation benchmark CSVs for HANCOCK, MIMIC-IV, and TCGA-LUAD
2. Keep only rows for `tool == 'POME'`
3. Aggregate across all embedding dimensions (`dim`)
4. Generate a 12-panel (3×4) violin-plot figure across binning strategies and bin counts

The output PDF is saved to `output/imputation_binning_effects_pome.pdf`.

`generate_imputation_dim_figure.py` will:
1. Load `*_imputation_z_bins_15.csv` benchmark files for HANCOCK, MIMIC IV, and TCGA-LUAD
2. Keep only rows with `tool == 'POME'`
3. Build a 2x3 violin-plot figure comparing embedding sizes (`dim` = 16, 32, 64)
4. Match categorical and numeric panel styles to the main imputation figure

The output PDF is saved to `output/imputation_dims_pome_z15.pdf`.

## Configuration

To modify survival-analysis parameters, edit `scripts/generate_survival_figure.py`:
- `data_glob`: Pattern to match data files (default: `'data/HANCOCK_samples_*.tsv'`)
- `feature_cols`: Feature columns to use for k-NN (default: `dim_0` through `dim_15`)
- `n_neighbors`: Number of neighbors in k-NN graph (default: 10)
- `year_filter`: Year to filter data on (default: 2019)
- `output_pdf`: Path for output PDF (default: `'output/survival_analysis.pdf'`)

For the imputation benchmarking figure, use:
- `--dim`: embedding dimension (`16`, `32`, or `64`; default: `32`)

## Modules

The analysis is organized into the following modules in `src/hancock_survival/`:

- **survival_data_processing.py**: Data loading, modality computation, k-NN graph construction
- **survival_analysis.py**: Treatment recommendation logic and multi-file analysis
- **imputation_analysis.py**: Imputation rank-table and distribution data preparation
- **imputation_plotting.py**: Imputation figure plotting
- **survival_statistics.py**: Wilcoxon signed-rank statistical tests
- **survival_plotting.py**: Figure generation with matplotlib
