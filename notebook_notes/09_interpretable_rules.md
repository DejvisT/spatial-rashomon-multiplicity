# 09 — Interpretable rules

**Key idea:** Extract human-readable rules that describe high-variance / HH regions using four methods (decision tree, L1 logistic regression, cluster-then-describe, beam-search conjunctions), then validate via bootstrap OOB and permutation tests.

## Loads

- `results/{dataset}/seed={SEED}/` run artifacts (P_test, split, meta)
- Transformed test features, raw dataset
- Datasets: primarily compas

## Produces

- `tables/rules_summary_{dataset}.csv` — rule text with support, purity, recall, lift
- `tables/rules_oob_summary_{dataset}.csv` — OOB robustness metrics
- `tables/rules_permutation_pvals_{dataset}.csv` — enrichment p-values
- `tables/rule_feature_stability_{dataset}.csv` — feature stability across seeds
- `tables/final_rules_{dataset}.csv` — selected rules for thesis
- `figures/rules_support_purity_{dataset}.pdf`
- `figures/pca_hv_hh_{dataset}.pdf` — PCA scatter with HV/HH highlighted

## Parameters

- dataset_name = "compas", SEED = 0, q = 10 (top 10% for HV_q), K = 25, k_nn = 30
- Tree: max_depths (2,3,4), min_leaf (5,10,20), ccp (0.0, 0.01)
- L1 LogReg: C=0.1, l1_ratio=1.0
- Beam search: min_support=10, max_conditions=4, beam_width=20, top_k=5

## Key functions called

- `fit_tree_surrogate` (Method 1), `method2_l1_logreg` (Method 2)
- `method3_cluster_describe` (Method 3), `beam_search_rules` (Method 4)
- `run_all_methods` — orchestrates all four methods
- `rule_metrics` — support, purity, recall, lift per rule
- Bootstrap OOB and permutation enrichment test

## Core objects (shapes)

- `X_test`: (n_test, n_features) — preprocessed test features
- `variance`: (n_test,) — pointwise variance
- `HH_mask`: (n_test,) boolean
- Labels: HV_q (top 10% variance), HH, HV_only, HH_only
- Rules: (rule_text, mask, support, purity, recall, lift) tuples

## Main results (numbers)

- PR-AUC (descriptive): HV_q tree 0.63, HH tree 0.82
- Beam search and tree give highest OOB purity/lift for HH
- Permutation tests confirm enrichment above random
- Feature stability: top rule features consistent across seeds

## One-liner interpretation

Beam-search conjunctions and shallow decision trees produce interpretable rules that reliably identify HH/high-variance regions, with enrichment confirmed by permutation tests and OOB validation.

## Open questions / TODO

- Extend rule extraction to German and Breast Cancer datasets
- Consider ensemble of rules (voting across methods)
