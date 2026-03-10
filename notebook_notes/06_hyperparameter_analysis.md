# 06 — Hyperparameter analysis

**Key idea:** Quantify which hyperparameters drive within-family multiplicity (V_m) using a between-group variance decomposition, and assess rank stability across seeds.

## Loads

- Run directories via `_get_run_dirs`
- `run_hp_importance_all_seeds` from `analysis.hp_analysis`
- Datasets: compas, german, breast_cancer

## Produces

- `tables/hp_importance_per_seed_{dataset}.csv` — per-seed HP importance
- `tables/hp_importance_agg_{dataset}.csv` — aggregated HP importance
- `figures/hp_importance_bar_{dataset}_{family}.pdf` — bar charts (mean ± std)
- `figures/hp_rank_stability_{dataset}_{family}.pdf` — rank stability heatmaps
- `figures/hp_marginal_effect_{dataset}_{family}_{hp}.pdf` — marginal V_m vs HP value

## Parameters

- K = 25 (within-family)
- TOP_HP = 10
- Datasets: compas, german, breast_cancer

## Key functions called

- `run_hp_importance_all_seeds`, `select_rashomon_family`
- `compute_Vm` — per-model deviation from family ensemble mean
- `hp_importance_Vm` — between-group variance ratio
- `marginal_Vm_by_hp` — marginal effect of each HP on V_m
- `aggregate_hp_importance`

## Core objects (shapes)

- `V_m`: (n_models,) — per-model deviation from family ensemble mean
- `ratio_of_sums`: scalar per (family, HP) — between-group variance ratio
- `df_per_seed`: rows per (dataset, seed, family, hp_name) with ratio_of_sums, n_values, n_models

## Main results (numbers)

- **GBM:** top HPs are subsample, max_depth, learning_rate
- **RF:** min_samples_leaf, max_depth, n_estimators
- **LogReg:** C, penalty type
- Rank stability: top 2–3 HPs are consistent across seeds; lower ranks fluctuate
- Marginal effect plots show non-monotonic V_m for some HPs

## One-liner interpretation

A small number of hyperparameters (2–3 per family) explain most within-family multiplicity; the rankings are stable across seeds, supporting interpretable driver identification.

## Open questions / TODO

- Consider interaction effects between HPs (currently univariate decomposition)
- Extend to joint HP importance (e.g., ANOVA-style)
