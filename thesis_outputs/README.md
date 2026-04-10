# Thesis outputs layout

This folder holds **derived** tables and figures produced by the numbered analysis notebooks. It answers: *which notebook produced this file?*

## Raw results (not stored here)

Training and experiment-runner artifacts live under `results/` (gitignored by default):

| Content | Path pattern |
|--------|----------------|
| Saved predictions, meta, splits | `results/<dataset>/seed=<n>/` (`P_test.npy`, `meta.csv`, `split.npz`, …) |
| Per-run analysis after the runner | `results/<dataset>/summary_per_run.csv`, `results/<dataset>/seed=<n>/per_point/` (`quadrant_summary.csv` per seed), `results/<dataset>/quadrant_breakdown_aggregated.csv` (mean ± std over seeds for Table 3–style quadrants), `quadrant_thresholds_per_run.csv`, `per_family_spatial_per_run.csv`, `per_family_spatial_aggregated.csv` (per-family Rashomon mean ± std; thesis Table 6 uses COMPAS) |
| Optional fixed-test replication | `results_fixed_test/<dataset>/seed=<n>/` (see notebook 03) |
| Optional bootstrap / alternate training | `results_fixed/` (see notebook 10) |

**Produce raw runs:** `python run_training_pipeline.py --dataset <name>` then run the experiment pipeline from notebook 01 (or `analysis.experiment_runner`).

## Derived outputs (this folder)

Subfolders are named after the notebook id (`nb01` … `nb10`), matching `notebooks/NN_*.ipynb`.

| Notebook | Tables | Figures |
|----------|--------|---------|
| 01 Primary experiment | `tables/nb01/` (`dataset_summary*.csv`, …) | `figures/nb01/` |
| 02 Null model | `tables/nb02/` | `figures/nb02/` |
| 03 Spatial patterns | `tables/nb03/` | `figures/nb03/` |
| 04 Sensitivity to K | — | `figures/nb04/` |
| 05 Sensitivity to kNN | — | `figures/nb05/` |
| 06 Hyperparameter analysis | `tables/nb06/` | `figures/nb06/` |
| 07 Calibration | `tables/nb07/` (`calibration_per_run.csv`, `calibration_summary.csv`); per-dataset files still under `results/<dataset>/` from the calibration module | `figures/nb07/` |
| 08 Synthetic | (no standard exports) | — |
| 09 Interpretable rules | `tables/nb09/` | `figures/nb09/` |
| 10 Robustness & fairness | `tables/nb10/` | `figures/nb10/` |

Notebook 10 **reads** rule-related CSVs from notebook 09 (`thesis_outputs/tables/nb09/`); run 09 before sections that need those tables.

## Code

Paths are defined in `thesis_layout.py` at the repo root (`thesis_table_dir`, `thesis_figure_dir`, `resolve_csv`, `RAW_RESULTS`).

## Legacy paths

Older runs may still have files under top-level `tables/` and `figures/`. `resolve_csv()` and export scripts check those after the new paths.
