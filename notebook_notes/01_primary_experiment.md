# 01 — Primary experiment

**Key idea:** Load all runs, perform global Rashomon selection (K=25), compute multiplicity and spatial metrics, aggregate across seeds, and produce thesis-ready summary tables and dashboard plots.

## Loads

- `results/{dataset}/seed=*/meta.csv` — model metadata (loss, family)
- `results/{dataset}/seed=*/P_test.npy` — test-set predictions
- `results/{dataset}/seed=*/split.npz` — train/val/test indices
- `results/{dataset}/seed=*/config.json` — run configuration
- Raw datasets via `load_dataset` (COMPAS, German Credit, Breast Cancer)
- `summary_per_run.csv` per dataset (generated or loaded)

## Produces

- `results/{dataset}/summary_per_run.csv` — per-run diagnostics (25 columns)
- `results/{dataset}/seed=*/per_point/var_p.npy`, `conflict.npy`, `var_hard.npy`
- `results/_thesis_tables/dataset_summary.csv` — consolidated aggregate
- `results/_thesis_tables/dataset_summary_formatted.csv` — mean ± std for thesis
- `results/_thesis_tables/per_family_summary.csv`
- `results/_thesis_tables/global_vs_perfamily_comparison.csv`
- `figures/nb01/dataset_comparison_bars.pdf` — 2×2 bar charts
- Dashboard plots: performance boxplots, predictive-multiplicity bars, spatial dashboard (Moran bar, LISA HH/LL, neighborhood agreement, LCAE)

## Parameters

- K = 25, k_nn = 30, R_null = 100, epsilon = 0.05, seed = 42
- outer_seed = 0..9 (10 runs per dataset)
- Datasets: compas, german, breast_cancer

## Key functions called

- `run_dataset_experiment`, `run_all_experiments` — experiment orchestration
- `select_rashomon_global`, `select_rashomon_per_family` — Rashomon selection
- `run_multiplicity`, `run_spatial`, `run_null` — metric computation
- `pointwise_variance`, `spatial_analysis`, `quadrant_analysis`
- `compute_multiplicity_metrics` — ambiguity, disagreement rate, discrepancy
- `brier_score_loss` (sklearn)

## Core objects (shapes)

- `P_test`: (n_candidates, n_test) — raw test predictions
- `P_sel`: (K, n_test) — selected Rashomon predictions
- `var_p`: (n_test,) — pointwise variance
- `conflict`: (n_test,) — pointwise conflict
- `HH_mask`: (n_test,) boolean — LISA High–High hotspots
- `agg_df`: (3, ~31) — one row per dataset
- `per_run` (COMPAS example): (10, 25) — 10 seeds × 25 columns
- `df_runs`: (30, 18) — dashboard: all runs × 18 metrics

## Main results (numbers)

- **COMPAS:** mean_variance 0.0013 ± 0.0003, Moran's I 0.1993 ± 0.0772, n_HH 553.80 ± 43.07, accuracy 0.6796 ± 0.0080
- **German:** mean_variance 0.0050 ± 0.0021, Moran's I 0.1042 ± 0.0397, accuracy 0.7491 ± 0.0276
- **Breast Cancer:** mean_variance 0.0032 ± 0.0026, Moran's I 0.0315 ± 0.0207, accuracy 0.9732 ± 0.0079
- frac_significant_hh: COMPAS 1.0, German 0.7, Breast Cancer 0.3

## One-liner interpretation

COMPAS shows highest spatial clustering of multiplicity (Moran's I always significant); Breast Cancer has highest accuracy but least spatial structure; mean variance is low across all datasets but spatially concentrated.

## Open questions / TODO

- Consider adding per-dataset dashboard breakdowns (currently pooled across datasets in dashboard section)
- Investigate why Breast Cancer has low HH significance despite non-trivial mean variance
