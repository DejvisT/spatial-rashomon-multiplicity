# Notebook Guide — Rashomon Multiplicity Analysis

This document explains every notebook in the project: what analyses are performed, what code does, what plots and tables are produced, and how to interpret the results.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Code Architecture](#code-architecture)
3. [Key Concepts and Metrics](#key-concepts-and-metrics)
4. [Notebook 01 — Primary Experiment](#notebook-01--primary-experiment)
5. [Notebook 02 — Null Experiment](#notebook-02--null-experiment)
6. [Notebook 03 — Spatial Patterns](#notebook-03--spatial-patterns)
7. [Notebook 04 — Sensitivity to K](#notebook-04--sensitivity-to-k)
8. [Notebook 05 — Sensitivity to kNN](#notebook-05--sensitivity-to-knn)
9. [Notebook 06 — Hyperparameter Analysis](#notebook-06--hyperparameter-analysis)
10. [Notebook 07 — Calibration Robustness](#notebook-07--calibration-robustness)
11. [Notebook 08 — Metrics Dashboard](#notebook-08--metrics-dashboard)
12. [Notebook 10 — Synthetic Multiplicity](#notebook-10--synthetic-multiplicity)
13. [Notebook 11 — Interpretable Rules](#notebook-11--interpretable-rules)
14. [Notebook 12 — Robustness and Fairness](#notebook-12--robustness-and-fairness)

---

## Project Overview

This project investigates **predictive multiplicity** in machine learning — the phenomenon where many models achieve near-identical performance but make different predictions for the same individuals. The central contribution is treating multiplicity as a **spatially structured** phenomenon: instead of reporting a single global disagreement number, we use spatial statistics (Moran's I, LISA) to identify *where* in feature space models disagree most.

**Datasets:** COMPAS (recidivism), German Credit, Breast Cancer (UCI), plus synthetic datasets.

**Pipeline:**

1. Train a pool of candidate models (5 families: LogReg, kNN, RF, GBM, MLP; 50 candidates each).
2. Select a **Rashomon set** — the top-K models by validation Brier score.
3. Compute **pointwise predictive variance** across Rashomon models for each test point.
4. Build a **kNN graph** over the test set in feature space.
5. Run **Moran's I** (global spatial autocorrelation of variance) and **LISA** (local clusters: HH = high-variance-surrounded-by-high-variance, LL = low-low).
6. Analyze what *drives* the hotspots: model family, hyperparameters, decision boundary proximity, fairness implications.

---

## Code Architecture

### `src/` — Core utilities


| Module                 | Purpose                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `data.py`              | Dataset loading (`load_dataset`), train/val/test splitting (`make_split`, `make_split_with_fixed_test`), preprocessing (`make_preprocessor`) |
| `training_pipeline.py` | `run_one_training_run` — trains all candidate models for one seed, returns split, metadata, P_val, P_test                                    |
| `synthetic_data.py`    | Three synthetic dataset generators: single island, three islands + outliers, graduated ambiguity                                             |
| `plots.py`             | Shared plotting utilities                                                                                                                    |


### `analysis/` — Reusable analysis functions


| Module                 | Purpose                                                                                                                                                                                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_analysis.py`      | Central module: `load_meta`, `load_P_test`, `load_split`, `select_rashomon_global`, `select_rashomon_per_family_totalK`, `select_rashomon_per_family_k_each`, `pointwise_variance`, `spatial_analysis` (Moran + LISA), `run_spatial`, `run_null`, `compute_multiplicity_metrics` |
| `experiment_runner.py` | `run_dataset_experiment`, `run_all_experiments` — orchestrates loading + analysis for all seeds of a dataset                                                                                                                                                                     |
| `preprocessing.py`     | `get_transformed_test_features` — loads the fitted preprocessor and transforms X_test to match training                                                                                                                                                                          |
| `spatial.py`           | Hand-rolled kNN graph and LISA (older code), `extract_hh_components` for connected-component analysis                                                                                                                                                                            |
| `stability.py`         | `hh_selection_frequency`, `hh_jaccard_matrix`, `jaccard_index`, `summarize_hh_stability`                                                                                                                                                                                         |
| `hyperparams.py`       | Variance decomposition: `compute_family_importance`, `compute_within_family_hp_importance`, `compute_within_family_hp_importance_on_subset`, `hyperparameter_profiling`                                                                                                          |
| `calibration.py`       | `run_calibration_experiment` — Platt scaling on validation set, recomputes metrics on calibrated predictions                                                                                                                                                                     |
| `rules.py`             | Rule extraction helpers for notebook 11                                                                                                                                                                                                                                          |


---

## Key Concepts and Metrics

### Rashomon Set Selection

- **Global (top-K):** Select the K models with the lowest validation Brier score from all families combined.
- **Per-family total-K:** Distribute K models across families (roughly equal shares). Used for fair global-vs-family comparisons.
- **Per-family K-each:** Select K models *within each family* (up to 5K total). Used for within-family hyperparameter analysis.

### Predictive Multiplicity Metrics


| Metric                 | Definition                                                               | Interpretation                                                                      |
| ---------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Pointwise variance** | `Var(P_m(x))` across Rashomon models m, per test point x                 | How much models disagree on this specific individual                                |
| **Mean variance**      | Average of pointwise variance over all test points                       | Overall level of disagreement                                                       |
| **Ambiguity**          | Mean over observations of `max_m P_m(x) - min_m P_m(x)`                  | Width of the prediction range per individual (Rudin's probability-space definition) |
| **Disagreement rate**  | Fraction of test points where any two models differ by more than epsilon | Share of population affected by meaningful disagreement                             |
| **Discrepancy**        | Maximum over model pairs of mean absolute prediction difference          | Worst-case pairwise disagreement                                                    |


### Spatial Metrics


| Metric                     | Definition                                                                               | Interpretation                                                                             |
| -------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Moran's I**              | Global spatial autocorrelation of pointwise variance on the kNN graph                    | Positive = high-variance points cluster near each other; 0 = random; negative = dispersion |
| **LISA (Local Moran)**     | Per-point local indicator of spatial association                                         | Identifies local clusters: HH, LL, HL, LH                                                  |
| **HH (High-High)**         | Points with high variance whose neighbors also have high variance (after FDR correction) | *Multiplicity hotspots* — regions where models systematically disagree                     |
| **LL (Low-Low)**           | Points with low variance whose neighbors also have low variance                          | *Stable regions* — models agree                                                            |
| **Neighborhood agreement** | Mean fraction of neighbors with the same LISA label                                      | How coherent are the spatial clusters                                                      |
| **LCAE**                   | Local Calibration-Agreement Error — measures how well local predictions agree            | Complementary to variance                                                                  |


### Recovery Metrics (Synthetic Only)


| Metric        | Definition                                                                    |
| ------------- | ----------------------------------------------------------------------------- |
| **Precision** | HH ∩ island / HH — what fraction of detected hotspots are truly ambiguous     |
| **Recall**    | HH ∩ island / island — what fraction of the true ambiguous region is detected |
| **Jaccard**   | HH ∩ island / (HH ∪ island) — overall overlap                                 |


---

## Notebook 01 — Primary Experiment

**File:** `notebooks/01_primary_experiment.ipynb`

### Purpose

Load all training runs for each dataset, perform global Rashomon selection (K=25), compute multiplicity and spatial metrics, aggregate across seeds (mean +/- std), and produce summary tables.

### What the Code Does

1. **Experiment runner** (`run_all_experiments`): For each dataset (compas, german, breast_cancer), iterates over `results/{dataset}/seed=*/`, loads P_test and meta, selects Rashomon set, calls `run_spatial` and `run_null`, writes `summary_per_run.csv`.
2. **Per-run diagnostics**: Displays a table of per-seed metrics (mean_variance, moran_i, n_hh, p_empirical, etc.).
3. **Aggregation**: Computes mean +/- std across seeds and formats as "0.1234 +/- 0.0056" for thesis tables.
4. **Per-family comparison**: Calls `run_spatial_per_family` to compute Moran's I and HH count for each model family separately, then compares to the global Rashomon set.

### Plots

- **Global vs Per-family HH counts** (bar chart, 3 panels for compas/german/breast_cancer): Each panel shows per-family HH counts as bars with error bars, plus a red dashed line for the global HH count.

### Output Files

- `results/_thesis_tables/dataset_summary.csv` — raw aggregate numbers
- `results/_thesis_tables/dataset_summary_formatted.csv` — formatted mean +/- std
- `results/_thesis_tables/per_family_summary.csv` — per-family metrics
- `results/_thesis_tables/global_vs_perfamily_comparison.csv`
- `results/_thesis_tables/global_vs_perfamily_hh.pdf`

### How to Interpret

- **High mean_variance** = more disagreement among Rashomon models.
- **High Moran's I** (positive, significant) = disagreement is spatially clustered, not random.
- **n_hh** = number of individuals in multiplicity hotspots.
- Compare global vs per-family: if per-family HH counts are much lower, inter-family diversity drives hotspots.

---

## Notebook 02 — Null Experiment

**File:** `notebooks/02_null_experiment.ipynb`

### Purpose

Test the statistical significance of spatial clustering. Under the null hypothesis (no spatial structure), Moran's I should be near zero. We permute model predictions R=100 times, recompute Moran's I each time, and compare the observed value to this null distribution.

### What the Code Does

1. For each dataset and seed: load run, compute `run_spatial` (observed Moran's I) and `run_null` (permuted Moran's I distribution).
2. Compute empirical p-value: `p = (1 + count(null >= observed)) / (R + 1)`.
3. A run is "significant" if `p < 0.05`.
4. Aggregate: fraction of significant runs per dataset.

### Plots

- **Per-run histogram**: For each (dataset, seed), a histogram of null Moran's I values with a vertical dashed line at the observed value. Saved as `figures/null_moran_{dataset}_seed{seed}.pdf`.

### Output Files

- `tables/null_significance_summary.csv` — dataset, n_runs, frac_significant, mean_moran +/- std, mean_p

### How to Interpret

- If **frac_significant = 1.0** (all runs significant): spatial clustering of multiplicity is robust and not an artifact.
- If the observed line is far to the right of the null histogram: very strong evidence of spatial structure.
- COMPAS and German typically show 10/10 significant; Breast Cancer is weaker (7/10).

---

## Notebook 03 — Spatial Patterns

**File:** `notebooks/03_spatial_patterns.ipynb`

### Purpose

Visualize HH and LL clusters, examine HH frequency and stability across seeds, and run connected-component analysis. **Requires fixed-test-set results** (`results_fixed_test/`) for valid observation-level cross-seed comparisons.

### What the Code Does

1. **Load fixed-test runs**: Uses `results_fixed_test/{dataset}/seed=*/` where all runs share the same test set indices. Validates that test indices are identical across runs.
2. **Spatial analysis per run**: For each seed, calls `run_spatial` to get HH/LL masks, Moran's I, pointwise variance.
3. **HH stability**:
  - `hh_selection_frequency`: For each test point, counts how many runs flag it as HH (0 to n_runs).
  - `hh_jaccard_matrix`: Pairwise Jaccard similarity of HH masks across seeds.
  - `summarize_hh_stability`: Summary statistics (mean Jaccard, etc.).
4. **Connected components**: Uses `extract_hh_components` to find spatially contiguous HH clusters in the kNN graph.
5. **Correlation analysis**: Scatter plot of mean variance vs Moran's I across runs (Pearson r).
6. **Family-wise comparison**: Per-family Moran's I and HH counts.

### Plots

- **HH frequency scatter**: Each test point colored by how often it appears in HH across seeds.
- **Jaccard heatmap**: n_runs x n_runs matrix of pairwise HH overlap.
- **HH locations in PCA feature space**: 2D projection showing where HH points cluster.
- **HH count and Moran's I per run**: Bar charts.
- **Mean variance vs Moran's I scatter**: One point per run, with Pearson r annotation.

### Output Files

- `figures/hh_frequency_{dataset}.pdf`, `hh_jaccard_{dataset}.pdf`, `hh_location_{dataset}.pdf`
- `figures/hh_moran_per_run_{dataset}.pdf`, `figures/variance_vs_moran_{dataset}.pdf`
- `tables/hh_component_summary_{dataset}.csv`

### How to Interpret

- **HH frequency**: Points flagged as HH in 8+/10 runs are reliably unstable individuals.
- **High mean Jaccard** (e.g., 0.5+): HH regions are stable across random seeds. Low Jaccard (< 0.3): HH regions shift substantially with different train/val splits.
- **Positive correlation** (variance vs Moran's I): More multiplicity tends to concentrate spatially.
- **Connected components**: Large components = big contiguous hotspot regions; many small components = fragmented instability.

---

## Notebook 04 — Sensitivity to K

**File:** `notebooks/04_sensitivity_K.ipynb`

### Purpose

Test how sensitive the spatial and multiplicity metrics are to the Rashomon set size K. Vary K in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}.

### What the Code Does

For each dataset, run, and K value: call `run_multiplicity` and `run_spatial` to get mean_variance, moran_i, and n_hh. Aggregate across seeds (mean +/- std per K).

### Plots

- **Stabilization curves**: Three-panel figure (one per dataset) showing mean_variance, Moran's I, and HH count as a function of K, with error bars.

### How to Interpret

- If metrics **stabilize** for K >= 20-25: the choice of K=25 is justified and results are robust to this parameter.
- If metrics keep changing at K=50: the Rashomon set definition is sensitive, and the choice of K matters more.
- COMPAS typically stabilizes around K=15-20. German and Breast Cancer stabilize even earlier.

---

## Notebook 05 — Sensitivity to kNN

**File:** `notebooks/05_sensitivity_kNN.ipynb`

### Purpose

Test sensitivity to the kNN graph's neighborhood size k. Vary k in {10, 20, 30, 50}.

### What the Code Does

For each dataset, run, and k value: call `run_spatial` with different k values. Aggregate across seeds.

### Plots

- **Elbow curves**: Two panels — HH count vs k and Moran's I vs k, one line per dataset with error bars.

### How to Interpret

- If metrics change little for k >= 20-30: the kNN graph is stable and the spatial results are robust.
- `mean_variance` is invariant to k (it doesn't use the spatial weights).
- Large k smooths out the spatial weights, potentially merging distinct clusters. Small k creates a noisier graph.
- k=30 is typically a good middle ground.

---

## Notebook 06 — Hyperparameter Analysis

**File:** `notebooks/06_hyperparameter_analysis.ipynb`

### Purpose

Decompose predictive variance into **between-family** and **within-family** components. Identify which hyperparameters drive disagreement, both globally and specifically within HH hotspots.

### What the Code Does

1. **Rashomon selection**: Uses `select_rashomon_per_family_k_each` (K models *per family*) to preserve enough within-family variation for hyperparameter analysis.
2. **Family importance** (`compute_family_importance`):
  - For each test point: compute `Var_between(family) / Var_total`.
  - `ratio_of_sums` = sum of between-family variance / sum of total variance across all test points.
  - High ratio = model family choice explains most of the disagreement.
3. **Within-family HP importance** (`compute_within_family_hp_importance`):
  - For each hyperparameter within each family: group models by HP value, compute between-group variance / total variance.
  - `ratio_of_sums` ranks hyperparameters by how much they explain within-family disagreement.
4. **HH-specific drivers**: Same analysis restricted to LISA HH points only. Computes delta (HH importance - global importance) to identify HPs that matter *more* in hotspots.
5. **Family effect by subset**: Compare family importance on ALL vs HH vs non-HH points.
6. **Multi-seed aggregation**: Repeat for all seeds and average.

### Plots

- **Between-family ratio histogram**: Distribution of `Var_between(family) / Var_total` per observation.
- **HP importance bar charts**: Horizontal bars of `ratio_of_sums` for top HPs per family.
- **Delta bar charts**: `ratio_of_sums(HH) - ratio_of_sums(all)` — positive means the HP is more important in hotspots.
- **Family importance by subset**: Grouped bars comparing family importance on ALL/HH/non-HH.

### Output Files

- `tables/family_importance_{dataset}_seed{seed}.csv`
- `tables/hp_importance_{dataset}_seed{seed}_{family}.csv`
- `tables/hp_importance_HH_{dataset}_seed{seed}_{family}.csv`
- `figures/family_between_ratio_hist_*.pdf`, `within_family_hp_importance_*.pdf`, `hh_delta_hp_importance_*.pdf`

### How to Interpret

- **High family importance** (ratio_of_sums > 0.5): Most disagreement comes from choosing different model families. The pipeline's diversity drives multiplicity.
- **High HP importance for a specific parameter**: That hyperparameter creates significant variance within its family. For example, `n_estimators` in RF or `C` in LogReg.
- **Positive delta in HH**: That HP is *more* important in hotspot regions than globally — it's a driver of localized instability.
- **Low family importance in HH but high globally** (or vice versa): Hotspot drivers differ from population-level drivers.

---

## Notebook 07 — Calibration Robustness

**File:** `notebooks/07_calibration_robustness.ipynb`

### Purpose

Test whether spatial multiplicity is an artifact of **probability miscalibration** across model families. If different families produce systematically shifted probabilities (e.g., RF always predicts higher), that alone could create variance without genuine disagreement.

### What the Code Does

1. **Platt scaling**: For each model in the Rashomon set, fit `LogisticRegression(P_val_predictions)` on the *validation set* only, then transform test predictions.
2. **Recompute metrics**: Mean variance, ambiguity, disagreement rate, discrepancy, Moran's I, HH count — all on the calibrated predictions.
3. **Compare**: Before vs after calibration. Compute Jaccard(HH_before, HH_after).

### Plots

- **Moran's I before vs after** (grouped bar chart per run)
- **HH count before vs after** (grouped bar chart per run)
- **Jaccard overlap** (bar chart per run with mean line)

### Output Files

- `results/{dataset}/calibration_summary_per_run.csv`
- `results/{dataset}/calibration_summary.csv`

### How to Interpret

- **Small drops in Moran's I and HH count after calibration + high Jaccard (> 0.7)**: Multiplicity is *structural*, not driven by miscalibration. The spatial patterns persist even after removing probability shifts.
- **Large drops + low Jaccard**: Multiplicity was partly an artifact of miscalibration. The HH regions change significantly after calibration.
- In practice, COMPAS shows that spatial structure largely survives calibration, confirming that the hotspots reflect genuine model disagreement.

---

## Notebook 08 — Metrics Dashboard

**File:** `notebooks/08_metrics_dashboard.ipynb`

### Purpose

Consolidated dashboard of all metrics — performance, predictive multiplicity, and spatial multiplicity — aggregated across runs. Intended for quick reference and thesis reporting.

### What the Code Does

1. For each dataset and seed: load run, select Rashomon (K=25), compute performance (accuracy, Brier), multiplicity (mean_variance, ambiguity, disagreement_rate, discrepancy), and spatial metrics (Moran's I, n_hh, n_ll, neighborhood_agreement, LCAE).
2. Aggregate by dataset: mean +/- std.
3. Optionally compute sensitivity to K.

### Plots

- **Performance box plots**: Accuracy and Brier score distributions.
- **Predictive multiplicity bar chart**: Mean variance, ambiguity, disagreement rate, discrepancy.
- **Multiplicity vs K** (4-panel): How each metric changes with Rashomon set size.
- **Spatial multiplicity**: Moran's I bar, HH/LL bar, neighborhood agreement box plot, pointwise variance and LCAE box plots.

### Output

- Summary table with all metrics formatted for the thesis.

### How to Interpret

This is the "one-stop shop" for seeing all metrics side by side. Use it to:

- Verify that models perform well (high accuracy, low Brier).
- Compare multiplicity across datasets (COMPAS typically has more HH than Breast Cancer).
- Check that spatial metrics are consistent with the null experiment (notebook 02).

---

## Notebook 10 — Synthetic Multiplicity

**File:** `notebooks/10_synthetic_multiplicity.ipynb`

### Purpose

Validate the entire spatial multiplicity pipeline on synthetic data with **known ground truth**. If the method works, LISA HH hotspots should recover the planted ambiguous regions.

### What the Code Does

**Part A — Single Island:**

1. Generate data: two well-separated Gaussian blobs (stable) + one uniform disk at the origin (ambiguous island) with a weak XOR probability pattern (p = 0.5 +/- delta).
2. Train 250 candidate models (5 families x 50 each), save artifacts.
3. Select Rashomon set (K=25), run spatial analysis.
4. **Recovery metrics**: Compare HH mask to ground-truth island mask — Precision, Recall, Jaccard.
5. **FDR sensitivity**: Vary alpha in {0.01, 0.05, 0.10, 0.20}; report recovery metrics and decision tree metrics at each level.
6. **Decision tree**: Fit a shallow tree to predict HH vs non-HH — shows what simple rules describe the hotspot.
7. **Null experiment**: Permute predictions, compare observed Moran's I to null distribution.

**Part B — Three Islands + Outliers:**
Same pipeline on a harder dataset with three ambiguous islands and scattered outliers (p=0.5, isolated). Tests whether LISA correctly finds multiple islands and ignores outliers.

**Part C — Graduated Ambiguity:**
Two islands with different XOR signal strengths (delta=0.35 "strong" and delta=0.20 "moderate") plus isolated outliers. Demonstrates that:

- Strong island: clear HH recovery.
- Moderate island: weaker/fewer HH but still detectable.
- Outliers: high per-point variance but NOT flagged as HH (spatially isolated).

### Key Design Insight

The XOR pattern (p = 0.5 +/- delta based on sign(x1*x2)) is clever because:

- **Tree-based models** (RF, GBM) can learn the XOR boundary and predict 0.5 +/- delta.
- **Linear models** (LogReg) cannot learn XOR and predict ~0.5 everywhere.
- This creates genuine inter-model disagreement in the island, producing high pointwise variance.
- The `delta` parameter controls the disagreement magnitude: higher delta = larger gap between model types = more variance = easier HH detection.

### Plots

- Data scatter colored by y.
- Variance scatter with ground-truth island circle.
- HH hotspot scatter.
- FDR sensitivity curves (#HH and Jaccard vs alpha; Precision and Recall vs alpha).
- Decision tree visualization.
- Null Moran's I histogram.
- Per-region variance bar chart (graduated dataset).

### How to Interpret

- **Precision near 1.0**: Almost all points LISA flags as HH are genuinely in the ambiguous region.
- **Recall near 1.0**: Almost all truly ambiguous points are detected.
- **Jaccard near 1.0**: Strong overall overlap.
- **Outliers with 0% HH rate**: The method correctly ignores isolated high-variance points — it detects *structured, spatially concentrated* ambiguity, not isolated noise.

---

## Notebook 11 — Interpretable Rules

**File:** `notebooks/11_interpretable_rules.ipynb`

### Purpose

Find interpretable feature-based rules that **describe** HH and high-variance regions. The goal is not prediction but explanation: "what features characterize the individuals where models disagree most?"

### What the Code Does

1. **Labels**: Define HV_q (top q% by variance), HH (LISA hotspot), HV_only (high variance but not HH), HH_only (HH but not top-q% variance).
2. **Four extraction methods**:
  - **Shallow decision tree** (depth 2-4): Fit `DecisionTreeClassifier` to predict HH vs non-HH using test features. Extract rules from leaf nodes.
  - **L1 logistic regression**: Fit sparse LogReg; features with non-zero coefficients form the rule.
  - **Cluster-describe**: PCA + KMeans on HV points to find sub-clusters, then describe each cluster with a tiny decision tree.
  - **Beam search (subgroup discovery)**: Iteratively build conjunctions of feature thresholds that maximize purity (P(HH=1 | rule)).
3. **Rule metrics**: purity = P(label=1 | rule region), support = #points in region, recall = coverage of positives, lift = purity / base_rate.
4. **Robustness**:
  - **Bootstrap OOB**: Resample B=50 times, evaluate rules on out-of-bag points.
  - **Permutation enrichment**: Shuffle labels N=500 times, check if observed purity exceeds the null.
  - **Feature stability**: Across B=50 bootstrap samples, how often does each feature appear in the extracted rules.

### Plots

- **PCA scatter**: HV_only, HH_only, Other points in 2D PCA space.
- **Support vs purity scatter**: Each rule as a point, colored by method and label.

### Output Files

- `tables/rules_summary_{dataset}.csv` — all extracted rules with metrics
- `tables/rules_oob_summary_{dataset}.csv` — bootstrap OOB evaluation
- `tables/rules_permutation_pvals_{dataset}.csv` — permutation p-values
- `tables/rule_feature_stability_{dataset}.csv` — feature appearance frequency
- `tables/final_rules_{dataset}.csv` — filtered rules (support >= 30, conditions <= 3)

### How to Interpret

- **High lift** (e.g., 15x): The rule identifies a subgroup where HH rates are 15 times higher than the base rate. Very informative.
- **Low permutation p-value** (e.g., < 0.01): The rule's purity is unlikely under random labeling — it's a genuine pattern.
- **Stable features** (appear in > 80% of bootstrap samples): Robust indicators of hotspot membership.
- For COMPAS: `num__priors_count` and `num__age` are the most stable hotspot descriptors — individuals with high prior count and high age are systematically in the multiplicity hotspot.

---

## Notebook 12 — Robustness and Fairness

**File:** `notebooks/12_robustness_and_fairness.ipynb`

### Purpose

Four add-on studies that extend the core analysis with robustness checks and fairness considerations.

### Section 1: Aggregated Drivers Across Seeds

**What it does:**

- Aggregates family importance (between-family variance ratio) across all 10 seeds per dataset.
- Aggregates within-family HP importance across seeds (top 5 HPs per family).
- Extracts feature frequencies from interpretable rules across seeds.

**Output:** `tables/family_importance_aggregated.csv`, `tables/hp_importance_aggregated_top5.csv`, `tables/rule_feature_frequency_compas.csv`

**How to interpret:** Shows whether the drivers identified in notebook 06 (for a single seed) are stable across seeds. If the same HPs and families dominate across seeds, the findings are robust.

### Section 2: Decision Boundary Analysis (Variance vs Margin)

**What it does:**

- For each test point: compute `p_mean = mean(P_m(x))` across Rashomon models, then `margin = |p_mean - 0.5|`.
- Points near the decision boundary have low margin; points far from it have high margin.
- Correlate margin with pointwise variance (Pearson, Spearman).
- Mann-Whitney U test: compare margin distributions of HH vs non-HH points.

**Plots:** Scatter plot of variance vs margin (COMPAS seed=0), colored by HH/HV/other.

**Output:** `tables/variance_vs_margin_summary.csv`, `tables/margin_hh_wilcoxon.csv`

**How to interpret:**

- **Negative correlation** (variance high when margin is low): Hotspots are near the decision boundary — models disagree because the point is inherently ambiguous.
- **Weak or no correlation**: Hotspots are not simply "boundary points" — the spatial structure captures something beyond proximity to p=0.5.
- For COMPAS: correlation is weak, meaning HH hotspots are not just boundary effects. The spatial analysis provides genuinely new information.

### Section 3: Fairness / Subgroup Exposure (COMPAS)

**What it does:**

- For each seed: compute HH rate, HV rate, and mean variance per demographic group (race, sex).
- Aggregate across seeds with bootstrap CIs.
- **Stratified permutation test**: Shuffle group labels within each seed, recompute group disparities, build null distribution. If observed disparity > 95% of null, it's significant.
- **Protected-attribute kNN robustness**: Rebuild the kNN graph *excluding* race and sex features; check if HH patterns persist (Jaccard overlap with original).

**Plots:** Bar charts of HH rate by race and sex with error bars.

**Output:** `tables/fairness_subgroup_rates_compas.csv`, `tables/fairness_permutation_test_compas.csv`, `tables/knn_excl_protected_compas.csv`

**How to interpret:**

- **Unequal HH rates across groups** (e.g., African-American > Caucasian): Multiplicity hotspots disproportionately affect certain demographics — a fairness concern.
- **Significant permutation p-value**: The disparity is unlikely due to chance.
- **High Jaccard with protected-excluded kNN**: HH patterns are not driven by race/sex features in the kNN graph — they persist even without protected attributes, suggesting they reflect genuine structural features of the prediction problem.

### Section 4: Alternative kNN Graph Constructions

**What it does:**

- Build three kNN graphs: Euclidean (baseline), PCA-reduced (15 components), cosine distance.
- For each: recompute Moran's I and HH count.
- Compare across methods.

**Plots:** Grouped bar plot of Moran's I and HH count by method and dataset.

**Output:** `tables/alternative_knn_comparison.csv`

**How to interpret:**

- **Similar results across graph types**: Spatial patterns are robust to the distance metric.
- **Cosine tends to yield higher Moran's I**: The angular distance captures inter-point relationships differently; more HH may be detected.
- **PCA close to Euclidean**: Dimensionality reduction doesn't fundamentally change the spatial structure.
- If results differ dramatically: the spatial analysis is sensitive to the graph construction, and this should be noted as a limitation.

---

## General Workflow

To reproduce all results from scratch:

1. **Train models**: `python run_training_pipeline.py` (10 seeds x 3 datasets)
2. **Train fixed-test models**: `python run_training_pipeline_fixed_test.py` (for notebook 03)
3. **Run notebooks in order**: 01 through 12
4. **Notebook 10** (synthetic) is self-contained — it trains its own models inline
5. **Check thesis tables**: `results/_thesis_tables/` and `tables/`
6. **Check figures**: `figures/`

