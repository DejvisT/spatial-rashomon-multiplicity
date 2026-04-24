# 09 — Interpretable rules (HH, trees only)

**Key idea:** After LISA marks **HH** hotspots, summarize them with **shallow decision trees** (positive-leaf if–then rules). Optional bootstrap summaries describe **feature stability** in those rules. Descriptive only.

## Code

- **`analysis/nb09_interpretable_rules.py`** — tree fitting, rule tables, cross-seed helpers, OOB / permutation / stability / final selection (single module for this notebook).

## Loads

- `results/{dataset}/seed=*` (preprocessed test features, spatial HH mask)

## Produces

- `tables/rules_summary_{dataset}.csv`, `rules_oob_summary_*`, `rules_permutation_pvals_*`, `rule_feature_stability_*`, `final_rules_*`, `rules_oob_bootstrap_long_*`, `rule_feature_frequency_*`, `rules_recurring_across_seeds_*`, `rule_stability_*`
- `figures/rules_support_purity_{dataset}.pdf`, `figures/pca_hh_{dataset}.pdf`

## Parameters

- `OUTER_SEEDS`, `SEED_FOR_FIGURES`, `dataset_name`, `K`
- Tree: `max_depth=3`, `min_samples_leaf=10`, `ccp_alpha=0`

## One-liner

HH hotspots are described by short tree rules; resampling checks how stable those descriptions are across bootstrap draws.
