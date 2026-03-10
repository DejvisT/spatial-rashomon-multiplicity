# 03 — Spatial patterns

**Key idea:** Analyze stability and structure of LISA HH hotspots across seeds — frequency histograms, pairwise Jaccard, connected components, family-level vs global spatial metrics, and variance-vs-conflict overlap.

## Loads

- Run directories via `_get_run_dirs`
- Per-run artifacts: split, meta, P_test, transformed test features
- Dataset: **compas** (single dataset, `DATASET = "compas"`)

## Produces

- `figures/hh_freq_histogram_{dataset}.pdf` — how often each point is HH across seeds
- `figures/hh_stability_freq_{dataset}.pdf` — stability frequency distribution
- `figures/hh_jaccard_heatmap_{dataset}.pdf` — pairwise seed Jaccard of HH sets
- `tables/hh_component_summary_{dataset}.csv` — connected component sizes
- `figures/hh_location_{dataset}.pdf`, `figures/hh_moran_per_run_{dataset}.pdf`
- `figures/variance_vs_moran_{dataset}.pdf` — mean variance vs Moran's I scatter
- `figures/family_vs_global_spatial_{dataset}.pdf`
- `figures/var_vs_conflict_hh_{dataset}.pdf` — overlap of variance and conflict hotspots
- `figures/hh_jaccard_distributions_{dataset}.pdf`

## Parameters

- K = 25, k_nn = 30, USE_FIXED_TEST = True
- Dataset: compas

## Key functions called

- `run_spatial`, `run_spatial_per_family`, `select_rashomon_global`
- `pointwise_variance`, `spatial_analysis`
- `scipy.sparse.csgraph.connected_components` — for hotspot region analysis

## Core objects (shapes)

- `spatial_by_run`: list of dicts with `HH_mask`, `moran_i`, `n_hh`, `W`
- `HH_mask`: (n_test,) boolean per run
- `HH_freq`: (n_test,) — count of HH membership across seeds
- `jaccard_matrix`: (n_seeds, n_seeds) — pairwise Jaccard of HH sets
- `stable_id`: original row indices for cross-seed comparison

## Main results (numbers)

- HH hotspot pairwise Jaccard across seeds: moderate overlap, indicating partial stability
- Variance vs conflict hotspots: substantial but not perfect overlap
- Per-family spatial metrics differ from global (MLP highest, LogReg lowest)
- Connected components identify 2–4 persistent spatial clusters in COMPAS

## One-liner interpretation

LISA HH hotspots in COMPAS are partially stable across seeds, spatially clustered into a few connected regions, and arise more from flexible families (MLP, GBM) than simple ones (LogReg).

## Open questions / TODO

- Extend analysis to German and Breast Cancer datasets
- Quantify region persistence more formally (e.g., matching across seeds)
