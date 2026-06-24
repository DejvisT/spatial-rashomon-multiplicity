# Rashomon Multiplicity — Spatial Analysis of Predictive Multiplicity

MSc thesis project investigating **predictive multiplicity** in Rashomon sets using spatial statistics.

## Overview

Many datasets admit multiple near-optimal models that disagree on individual predictions. This project uses observation-wise predictive variance and conflict ratio together with spatial autocorrelation methods (Moran's I and LISA) on kNN feature-space graphs to identify localized multiplicity hotspot regions.

The final analyses cover benchmark datasets, null testing, hotspot stability, global vs. per-family Rashomon sets, robustness checks, hyperparameter and family importance, synthetic validation, interpretable rules, decision-boundary proximity, and subgroup exposure diagnostics.

## Datasets

- **COMPAS** (recidivism prediction; local CSV in `data/`)
- **German Credit** (`credit-g`, loaded from OpenML)
- **Adult** (income prediction, loaded from OpenML)
- **Synthetic single-island, three-islands, and structural-exception datasets** for controlled validation

## Pipeline

```text
1. Train candidate models        -> run_training_pipeline.py
2. Train fixed-test runs         -> run_training_pipeline_fixed_test.py
3. Select Rashomon sets          -> analysis/run_analysis.py
4. Compute spatial metrics       -> analysis/run_analysis.py
5. Run thesis analysis notebooks -> notebooks/01-10
```

The main thesis setting uses `K = 25`, `R_null = 100`, 10 outer seeds, and 50 candidate configurations per model family.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activation is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want to execute notebooks from the command line, also install Jupyter:

```bash
pip install jupyter nbconvert ipykernel
```

## Pre-computed artifacts (optional download)

The submitted repository contains the source code, notebooks, thesis-facing generated tables/figures, and summary outputs. Full per-seed training artifacts are not included in the repository because of size; they can be regenerated with the training scripts or downloaded from the linked archive.

Training all benchmark runs locally is time-consuming. Pre-computed artifacts are available via [LRZ Sync+Share](https://syncandshare.lrz.de/getlink/fiKs14bmixnPFPx5Mfem5i/) (password: `multiplicity`).

After downloading, place the folders as follows in the repository root:

```text
spatial-rashomon-multiplicity/
├── results/              # from the shared link → repository root
├── results_fixed_test/   # from the shared link → repository root
└── thesis_outputs/
    └── cache/            # the shared `cache` folder → thesis_outputs/cache/
```

The shared archive contains:

- **`results/`** — main training artifacts (`P_test.npy`, `meta.csv`, splits, etc.) for COMPAS, German Credit, Adult, and the three synthetic designs used in notebook 08
- **`results_fixed_test/`** — fixed-test training artifacts for cross-seed hotspot stability (notebook 03)
- **`cache/`** — notebook parquet caches (under `thesis_outputs/cache/` in the repo) so notebooks can skip expensive recomputation when `FORCE_RECOMPUTE = False`

With these in place you can skip sections 1–2 below and run the notebooks directly. To regenerate everything from scratch, omit the download and run the training scripts instead.

## How to run the analysis

### 1. Train the main candidate model pools

Run the standard 60/20/20 train/validation/test pipeline for all benchmark datasets:

```bash
python run_training_pipeline.py --dataset compas --n_outer 10 --n_candidates 50 --out_dir results
python run_training_pipeline.py --dataset german --n_outer 10 --n_candidates 50 --out_dir results
python run_training_pipeline.py --dataset adult  --n_outer 10 --n_candidates 50 --out_dir results
```

PowerShell equivalent:

```powershell
foreach ($d in "compas", "german", "adult") {
    python run_training_pipeline.py --dataset $d --n_outer 10 --n_candidates 50 --out_dir results
}
```

This creates:

```text
results/compas/seed=0 ... seed=9
results/german/seed=0 ... seed=9
results/adult/seed=0  ... seed=9
```

Each run stores validation/test prediction matrices, candidate metadata, split indices, and configuration.

### 2. Train fixed-test runs for hotspot stability

Notebook 03 uses a fixed test set so that HH masks can be compared pointwise across seeds. Generate those artifacts with:

```bash
python run_training_pipeline_fixed_test.py --dataset compas --n_outer 10 --n_candidates 50 --out_dir results_fixed_test
python run_training_pipeline_fixed_test.py --dataset german --n_outer 10 --n_candidates 50 --out_dir results_fixed_test
python run_training_pipeline_fixed_test.py --dataset adult  --n_outer 10 --n_candidates 50 --out_dir results_fixed_test
```

PowerShell equivalent:

```powershell
foreach ($d in "compas", "german", "adult") {
    python run_training_pipeline_fixed_test.py --dataset $d --n_outer 10 --n_candidates 50 --out_dir results_fixed_test
}
```

This creates:

```text
results_fixed_test/compas/seed=0 ... seed=9
results_fixed_test/german/seed=0 ... seed=9
results_fixed_test/adult/seed=0  ... seed=9
```

### 3. Run the thesis notebooks

Run the notebooks in numeric order:

```text
01_main_multiplicity_summary.ipynb
02_null_experiment.ipynb
03_spatial_patterns.ipynb
04_sensitivity_K.ipynb
05_sensitivity_kNN.ipynb
06_hyperparameter_analysis.ipynb
07_calibration_robustness.ipynb
08_synthetic_multiplicity.ipynb
09_interpretable_rules.ipynb
10_robustness_and_fairness.ipynb
```

Recommended during final thesis updates: open them in Jupyter/VS Code and run them one by one, checking each output before moving on.

**Important for Notebook 03:**  
`03_spatial_patterns.ipynb` is dataset-specific. Run it once for each benchmark dataset by setting the dataset variable in the notebook to:

```text
compas
german
adult
```

This ensures that the Notebook 03 outputs are regenerated for all datasets, including the dataset-specific HH component summaries and spatial-pattern figures/tables.

Command-line execution is possible with:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_main_multiplicity_summary.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/02_null_experiment.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/03_spatial_patterns.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/04_sensitivity_K.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/05_sensitivity_kNN.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/06_hyperparameter_analysis.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/07_calibration_robustness.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/08_synthetic_multiplicity.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/09_interpretable_rules.ipynb --inplace --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute notebooks/10_robustness_and_fairness.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

For Notebook 03, command-line execution only runs the dataset currently configured inside the notebook. To regenerate all Notebook 03 outputs, change the configured dataset and rerun it once for each dataset.

### 4. Export thesis assets

The notebooks write thesis-facing figures and tables into `thesis_outputs/`. After rerunning notebooks, export the LaTeX-ready thesis assets with:

```bash
python scripts/export_thesis_assets.py
```

Optional flags:

- `--copy-all-figures` — copy every PDF from `thesis_outputs/figures/`, not only those referenced in the thesis `.tex` files
- `--prune-presentation-figs` — remove PDFs under `presentation_assets/fig/` that are not referenced in the thesis `.tex` files

The export script writes Overleaf-ready assets into:

```text
overleaf_bundle/presentation_assets/fig/
overleaf_bundle/presentation_assets/tab/
```

## Expected outputs

After a full run, the important generated outputs are organized under:

```text
results/                    # main training artifacts
results_fixed_test/          # fixed-test artifacts for stability analysis
thesis_outputs/figures/      # thesis-facing figures
thesis_outputs/tables/       # thesis-facing CSV/LaTeX tables
thesis_outputs/cache/        # optional notebook parquet caches (see pre-computed download)
overleaf_bundle/             # Overleaf-ready thesis bundle and exported assets
```


## Project structure

| Directory / file | Contents |
|---|---|
| `analysis/` | Reusable analysis modules for Rashomon selection, spatial statistics, stability, calibration, rules, and hyperparameter/family analyses |
| `src/` | Data loading, preprocessing, training pipeline, and synthetic data generation |
| `notebooks/` | Thesis analysis notebooks `01`-`10` |
| `data/` | Local raw data files, mainly COMPAS |
| `scripts/` | Thesis asset export scripts |
| `results/` | Main training artifacts; usually generated locally or downloaded from Sync+Share |
| `results_fixed_test/` | Fixed-test training artifacts for cross-seed hotspot stability |
| `thesis_outputs/` | Generated figures and tables used by the thesis |
| `thesis_outputs/cache/` | Notebook parquet caches (`notebooks/` subfolder inside `cache/`) |
| `overleaf_bundle/` | Overleaf-compatible thesis bundle and exported presentation assets |
| `run_training_pipeline.py` | Main candidate-training script |
| `run_training_pipeline_fixed_test.py` | Fixed-test candidate-training script |

## Notebook guide

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_main_multiplicity_summary.ipynb` | Default multiplicity and spatial summary tables/figures (aggregates saved training artifacts; no model training) |
| 02 | `02_null_experiment.ipynb` | Prediction-matrix permutation null for Moran's I and HH counts |
| 03 | `03_spatial_patterns.ipynb` | HH hotspot stability, Jaccard overlap, connected components, and PCA visualizations; run once per dataset |
| 04 | `04_sensitivity_K.ipynb` | Sensitivity to Rashomon set size `K` |
| 05 | `05_sensitivity_kNN.ipynb` | Sensitivity to kNN neighborhood size `k` |
| 06 | `06_hyperparameter_analysis.ipynb` | Family importance, within-family hyperparameter importance, hotspot-specific shifts, and meta-model diagnostics |
| 07 | `07_calibration_robustness.ipynb` | Platt/isotonic calibration robustness |
| 08 | `08_synthetic_multiplicity.ipynb` | Single-island, three-islands, and structural-exception synthetic validation |
| 09 | `09_interpretable_rules.ipynb` | Global and component-level rule extraction for high-multiplicity regions |
| 10 | `10_robustness_and_fairness.ipynb` | Alternative graph construction, decision-boundary proximity, and subgroup exposure diagnostics |

