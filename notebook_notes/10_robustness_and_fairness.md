# 10 — Robustness and fairness

**Key idea:** Aggregate multiplicity drivers across seeds (family importance, HP importance, rule features), test variance-vs-margin proximity to decision boundaries, assess fairness via subgroup HH exposure disparity, and compare alternative kNN graph constructions.

## Loads

- `results/{dataset}/seed={0..9}/` for compas, german, breast_cancer
- Per-run artifacts (meta, P_test, split, transformed features)
- Rule stability tables from notebook 09 (e.g., `tables/final_rules_*.csv`)
- Raw COMPAS data for protected attributes (race, sex)

## Produces

- `tables/family_importance_aggregated.csv`
- `tables/hp_importance_aggregated_top5.csv`
- `tables/rule_feature_frequency_compas.csv`
- `tables/variance_vs_margin_summary.csv` — variance–margin correlations
- `tables/margin_hh_wilcoxon.csv` — Wilcoxon test (HH vs non-HH margin)
- `tables/fairness_subgroup_rates_compas.csv` — HH/HV rates by race and sex
- `tables/fairness_permutation_test_compas.csv` — stratified permutation test
- `tables/knn_excl_protected_compas.csv`, `tables/alternative_knn_comparison.csv`
- Figures: `family_importance_aggregated.pdf`, `variance_vs_margin_*.pdf`, `fairness_hh_rate_by_race_compas.pdf`, `fairness_hh_rate_by_sex_compas.pdf`, `alternative_knn_comparison.pdf`

## Parameters

- K = 25, K_NN = 30, SEEDS = [0..9]
- Datasets: compas, german, breast_cancer
- MIN_GROUP_N = 30 (minimum subgroup size for significance)

## Key functions called

- `compute_family_importance`, `compute_within_family_hp_importance`
- `spatial_analysis`, `pointwise_variance`, `select_rashomon_per_family_k_each`
- `stats.pearsonr` — variance vs margin correlation
- `stats.wilcoxon` / `stats.mannwhitneyu` — HH vs non-HH margin test
- Stratified permutation test for subgroup disparity

## Core objects (shapes)

- `df_fam_imp`: (n_datasets × n_seeds,) — family importance per (dataset, seed)
- `df_hp_imp`: rows per (dataset, seed, family, hp) — HP importance
- `df_fair`: rows per (seed, group_col, group_val) — HH rate, HV rate, mean variance
- `margin = |p_mean − 0.5|`: (n_test,) — distance to decision boundary

## Main results (numbers)

- **Family importance:** Breast Cancer ~0.62 (all), ~0.69 (HH); COMPAS ~0.41 / ~0.52
- **Variance vs margin:** weak negative correlation; HH points closer to boundary
- **Wilcoxon:** HH has significantly lower margin than non-HH in COMPAS (p < 0.05)
- **Fairness (COMPAS):** HH rate African-American ~7.95%, Caucasian ~3.18%
- **kNN robustness:** Moran's I stable across Euclidean, PCA, cosine constructions

## One-liner interpretation

Multiplicity concentrates near decision boundaries and disproportionately affects African-Americans in COMPAS; spatial structure is robust to kNN graph construction choices.

## Open questions / TODO

- Deeper intersectional fairness analysis (race × sex)
- Causal analysis of whether disparity is driven by feature distributions vs model behavior
