# 05 — Sensitivity to kNN parameter k

**Key idea:** Sweep the kNN graph parameter k ∈ {10, 20, 30, 50} and measure how Moran's I, HH count, and graph connectivity respond — confirming k = 30 avoids fragmentation while preserving spatial signal.

## Loads

- Run directories from `results/`
- Per-run artifacts via `get_transformed_test_features`, `load_meta`
- Datasets: compas, german, breast_cancer

## Produces

- `figures/sensitivity_kNN_curves_{dataset}.pdf` — metrics vs k
- `figures/sensitivity_kNN_connectivity_{dataset}.pdf` — number of components vs k
- `figures/sensitivity_kNN_pca_hh_{dataset}.pdf` — PCA with HH highlights per k

## Parameters

- K = 25 (fixed Rashomon size)
- K_NN_LIST = [10, 20, 30, 50]
- Datasets: compas, german, breast_cancer

## Key functions called

- `run_multiplicity`, `run_spatial`
- `get_transformed_test_features`, `_get_run_dirs`
- PCA for visualization of HH locations

## Core objects (shapes)

- `df_knn`: DataFrame with dataset, seed, k_nn, mean_variance, moran_i, n_hh, n_components
- Connectivity: `comp_mean` (mean component count), `frac_mean` (largest component fraction)

## Main results (numbers)

- Small k (10): graph may fragment (multiple components), inflated Moran's I
- k = 30: single connected component in most runs, stable Moran's I
- k = 50: similar to k = 30 but smoother; minor differences
- HH counts decrease slightly with larger k (smoother neighborhoods)

## One-liner interpretation

k = 30 provides a well-connected graph without fragmentation, giving stable and interpretable Moran's I and LISA results; smaller k risks graph disconnection.

## Open questions / TODO

- Test even larger k values (e.g., 75, 100) for completeness
- Report median connectivity alongside mean
