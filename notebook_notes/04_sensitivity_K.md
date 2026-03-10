# 04 — Sensitivity to Rashomon size K

**Key idea:** Sweep Rashomon set size K ∈ {5, 10, ..., 50} and measure how mean variance, Moran's I, and HH count stabilize — confirming that K = 25 is past the elbow.

## Loads

- Run directories from `results/`
- Per-run artifacts via `get_transformed_test_features`, `load_meta`
- Datasets: compas, german, breast_cancer

## Produces

- `figures/sensitivity_K_curves_{dataset}.pdf` — mean variance, Moran's I, n_HH vs K
- `figures/sensitivity_K_hh_jaccard_{dataset}.pdf` — pairwise Jaccard of HH masks across K values
- `hh_masks_by_run` dict: `{(dataset, seed): {K: HH_mask}}`

## Parameters

- K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
- k_nn = 30
- Datasets: compas, german, breast_cancer

## Key functions called

- `run_multiplicity` — mean variance, conflict metrics per K
- `run_spatial` — Moran's I, LISA HH/LL per K
- `get_transformed_test_features`, `_get_run_dirs`

## Core objects (shapes)

- `results_by_K`: list of dicts with `mean_variance`, `moran_i`, `n_hh` per (dataset, seed, K)
- Aggregation: mean ± std across seeds per K value

## Main results (numbers)

- Elbow near K ≈ 20–25 for mean variance and Moran's I across all datasets
- Beyond K = 25, metrics stabilize (diminishing change)
- Jaccard overlap of HH masks between adjacent K values is high after K ≥ 20
- Candidate pool check: n_candidates ≥ 50 in all runs

## One-liner interpretation

K = 25 sits past the stabilization elbow for all key metrics, confirming it as a well-justified default for Rashomon set size.

## Open questions / TODO

- Consider log-scale K sweep for larger candidate pools
- Report stabilization formally (e.g., relative change < threshold)
