# 02 — Null experiment

**Key idea:** Test whether observed spatial clustering of multiplicity (Moran's I, LISA HH counts) exceeds what would be expected under a label-permutation null.

## Loads

- `results/{dataset}/seed=*/` run artifacts: meta, P_test, split, config
- Transformed test features via `get_transformed_test_features`
- Datasets: compas, german, breast_cancer

## Produces

- `figures/null_moran_{dataset}_seed{seed}.pdf` — observed vs null Moran's I histograms
- `figures/null_conflict_moran_{dataset}_seed{seed}.pdf` — conflict Moran's I null
- `tables/null_significance_summary.csv` — aggregated significance results
- `figures/null_summary_across_seeds.pdf` — cross-seed null summary

## Parameters

- K = 25, k_nn = 30, R_null = 100 (permutations)
- Datasets: compas, german, breast_cancer
- Seeds: 0..9

## Key functions called

- `run_spatial` — compute Moran's I and LISA on real predictions
- `run_null` — permutation null (shuffle predictions, recompute Moran's I and HH)
- `get_transformed_test_features` — preprocessed test features for kNN graph

## Core objects (shapes)

- `P_test`: (n_candidates, n_test)
- `X_test`: (n_test, n_features)
- `null_moran`: (R_null,) — null distribution of Moran's I
- `null_n_hh`: (R_null,) — null distribution of HH count
- `HH_mask`: (n_test,) boolean
- `spatial["moran_i"]`: scalar per run

## Main results (numbers)

- **COMPAS:** Moran's I always above null (p < 0.01 in 10/10 runs)
- **German:** Moran's I significant in 7/10 runs
- **Breast Cancer:** Moran's I significant in 3/10 runs
- Z-scores aggregated across seeds confirm COMPAS strongest, Breast Cancer weakest

## One-liner interpretation

Spatial clustering of multiplicity in COMPAS is robust and far exceeds chance; German is moderate; Breast Cancer is marginal, consistent with its high accuracy and low multiplicity.

## Open questions / TODO

- Consider alternative null strategies (e.g., permute features instead of labels, or bootstrap models)
- Report Z-score distributions alongside p-values
