# Thesis Context: Variance-based Analysis of Model Multiplicity in the Rashomon Set

This document provides the context needed to assist with thesis code development and experiments. It summarizes the thesis proposal, supervisor feedback, current implementation state, and the implementation roadmap.

---

## 1. Thesis Summary

### Title
**Variance-based analysis of model multiplicity in the Rashomon set**

### Motivation
Modern ML focuses on selecting a single "best" model. However, many datasets admit **Rashomon sets**: collections of models with similarly high predictive performance but qualitatively different behaviors (**predictive multiplicity**). This undermines claims about model stability, fairness, and interpretability. Prior work has shown multiplicity at the model and global explanation level, but less is known about **where** multiplicity manifests in feature space and whether it forms coherent, interpretable regions.

### Research Questions
1. Does observation-wise predictive variance exhibit **spatial autocorrelation** in feature space?
2. Are there **localized regions** (clusters) where predictive multiplicity is systematically higher?
3. Can such regions be characterized by **simple, interpretable rules**?
4. How **consistent** are these patterns across datasets (COMPAS, German Credit, Breast Cancer)?

### Methodology (overview)
- **Datasets**: COMPAS, German Credit, Breast Cancer (70/30 train/test)
- **Rashomon set**: Models with AUC ≤ 1% worse than best (ε = 0.01)
- **Models**: Logistic Regression, Random Forest, XGBoost (sklearn GBM), MLP (4 families × 25 HP configs × 3 seeds = 300 models per dataset)
- **Observation-wise variance**: `Var_m(p_hat_mi)` per observation across Rashomon models
- **Spatial structure**:
  1. kNN graph (k=20) on standardized features
  2. **Moran's I**: global spatial autocorrelation
  3. **LISA**: Local Moran → High-High (HH) regions
  4. **Connected components**: Extract coherent HH regions from kNN graph
- **Interpretability**: Shallow decision trees to describe HH components (precision/recall, out-of-sample)

---

## 2. Supervisor Feedback (Action Items)

### Overall Storyline

Find multiplicity hotspots statistically in feature space; show with null models and robustness tests that they are not random or artefactual (including analyses within single model families and across families, plus distinction from ensemble/bootstrap uncertainty); then describe hotspots with simple, out-of-sample validated rules.

---

### Specific Feedback Items

| # | Feedback | Status | Notes |
|---|----------|--------|-------|
| 1 | **Moran/LISA: within vs. across families** | Partially done | Compute Moran/LISA **separately within each model class** (only LR, only XGB, etc.) AND **globally over all classes**. Goal: exclude "family-mismatch" as explanation. |
| 2 | **Hyperparameter analysis** | MISSING | Optional block after baseline controls: Which HPs drive hotspots? Variance decomposition. |
| 3 | **Central null experiment** | Implemented | **Model-wise permutation** of predictions (per model separately) → recompute variance → Moran/LISA. Signal should vanish. Goal: Hotspots are not pipeline artefacts. |
| 4 | **Robustness / stability** | Partially done | Need systematic checks: (1) HH point set over runs/splits ("how often is point x in HH"), (2) stability of hotspot regions/connected components (same regions reappear? cluster metrics), (3) sensitivity to ε, k, n_models (elbow/stabilization). Goal: Hotspots are reproducible. |
| 5 | **Calibration** | MISSING | Consider **CalibratedClassifierCV** (Platt / isotonic). Calibrate only on Train/Val, not Test. Compare Moran/LISA/hotspots before vs. after calibration to rule out "different families → different scaling → variance increases" as artefact. |

---

## 3. Current Project State

### File Structure
```
rashomon-multiplicity/
├── analysis/
│   ├── run_analysis.py    # Rashomon selection, spatial (Moran/LISA), null, multiplicity metrics
│   ├── experiment_runner.py # run_dataset_experiment, run_all_experiments
│   ├── preprocessing.py   # get_transformed_test_features (for spatial/null)
│   ├── spatial.py         # kNN graph, LISA, HH component extraction, regionality metrics
│   ├── stability.py       # hh_selection_frequency, jaccard, region-level stability
│   ├── hyperparams.py     # family/HP importance, variance decomposition
│   ├── calibration.py     # Platt + isotonic scaling, calibration robustness
│   ├── rules.py           # interpretable rules for HH components
│   └── bootstrap_ablation.py  # bootstrap ablation infrastructure for robustness testing
├── src/
│   ├── data.py            # load_dataset, make_split, make_preprocessor, make_split_with_fixed_test
│   ├── training_pipeline.py # run_one_training_run, TRAINING_MODEL_CONFIGS
│   ├── synthetic_data.py  # synthetic dataset generators
│   └── plots.py
├── notebooks/
│   ├── 01_primary_experiment.ipynb
│   ├── 02_null_experiment.ipynb
│   ├── 03_spatial_patterns.ipynb
│   ├── 04_sensitivity_K.ipynb, 05_sensitivity_kNN.ipynb
│   ├── 06_hyperparameter_analysis.ipynb, 07_calibration_robustness.ipynb
│   ├── 08_synthetic_multiplicity.ipynb
│   ├── 09_interpretable_rules.ipynb
│   └── 10_robustness_and_fairness.ipynb
├── run_training_pipeline.py      # Main training (10 outer runs, 50 candidates/family)
├── run_training_pipeline_fixed_test.py  # Same with fixed test set (for notebook 03)
├── run_experiments.py            # Analysis over results (Rashomon, spatial, null)
└── results/
    └── {dataset}/
        └── seed={N}/   # split.npz, meta.csv, P_val.npy, P_test.npy, config.json
```

### Key Data Artifacts
- `P_test.npy`: shape `(n_candidates, n_test)` – predicted probabilities per candidate per test observation
- `P_val.npy`: same for validation set
- `meta.csv`: model_name, val_brier, etc. per candidate
- `split.npz`: train, val, test indices; config.json: dataset, outer_seed, n_candidates, etc.
- Preprocessed test features are obtained via `analysis.preprocessing.get_transformed_test_features` (fit on train).

### Implemented Functionality
- **Spatial**: kNN graph, Moran's I, LISA (with FDR), HH component extraction
- **Nulls**: `permute_predictions` (model-wise), `run_null_experiment`, `run_null_experiments`
- **Stability**: `hh_selection_frequency`, `jaccard_index`, `hh_jaccard_matrix`, `summarize_hh_stability`
- **Single-family**: Notebook 05 compares global vs. per-family (RF, LR, GBM, MLP) Moran/LISA

### Gaps (last reviewed 2026-03-13)
- **Interpretability / rules**: ✓ Implemented in `analysis/rules.py` and `09_interpretable_rules.ipynb`
- **Hyperparameter analysis**: ✓ Implemented in `analysis/hyperparams.py` and `06_hyperparameter_analysis.ipynb`
- **Stability**: ✓ Pointwise HH stability in `03_spatial_patterns.ipynb`; region-level stability metrics added to `analysis/stability.py`
- **Sensitivity (K, kNN)**: ✓ Implemented in `04_sensitivity_K.ipynb` and `05_sensitivity_kNN.ipynb`
- **Calibration**: ✓ Platt + isotonic scaling in `analysis/calibration.py` and `07_calibration_robustness.ipynb`
- **Multi-dataset**: ✓ COMPAS, German Credit, Breast Cancer all running
- **Region-level metrics**: ✓ Regionality ratio, HH/HL fractions in `analysis/spatial.py`
- **Within-hotspot performance**: ✓ HH vs non-HH accuracy/Brier comparison in `10_robustness_and_fairness.ipynb`
- **Getis-Ord Gi***: ✓ Cleaner hotspot definition via Gi* statistic in `01_primary_experiment.ipynb`
- **Soft Rashomon set**: ✓ Weighted model selection (Brier-based weights) in `01_primary_experiment.ipynb`
- **Spatial correlogram**: ✓ Distance-decay analysis of spatial autocorrelation in `01_primary_experiment.ipynb`
- **Cross-family HH overlap**: ✓ Overlap matrix and variance correlation across families in `10_robustness_and_fairness.ipynb`
- **HP diversity/entropy/prediction-difference**: ✓ Diversity indices and entropy analysis in `06_hyperparameter_analysis.ipynb`
- **Calibration quadrant movement**: ✓ Tracking how points move between LISA quadrants after calibration in `10_robustness_and_fairness.ipynb`
- **Systematic driver analysis**: ✓ L1 logistic + decision tree for HH vs non-HH characterization in `09_interpretable_rules.ipynb`
- **Bootstrap ablation infrastructure**: ✓ Implemented in `analysis/bootstrap_ablation.py` and `10_robustness_and_fairness.ipynb`
- **Gower distance and PCA-based kNN**: ✓ Mixed-type feature support via Gower distance and PCA-reduced kNN in `10_robustness_and_fairness.ipynb`

---

## 4. Implementation Roadmap

### Priority 1: Core Storyline (Must Have)
1. **Moran/LISA within vs. across families**
   - Extend notebook 05 to report Moran I and HH counts for each family (LR, RF, GBM, MLP) + global
   - Ensure "family-mismatch" is explicitly addressed in text
2. **Central null experiment**
   - Ensure notebook 03 clearly shows: real Moran I >> null Moran I; HH count real >> null
   - Possibly add visualization: histogram of null Moran I vs. real
3. **Simple, out-of-sample rules**
   - Implement: for each HH component, fit shallow DecisionTreeClassifier (HH vs. rest) on train, evaluate on test
   - Report precision, recall, rule text (sklearn `export_text` or similar)

### Priority 2: Robustness
4. **HH point stability**
   - Notebook 04: extend to multiple seeds/splits (need more runs); report `hh_selection_frequency` per point
5. **HH region stability**
   - Add: compare connected components across runs (e.g., Jaccard at component level, or cluster overlap metrics)
6. **Sensitivity analysis**
   - New notebook or section: vary ε (e.g. 0.005, 0.01, 0.02), k (5, 10, 20, 50), n_models
   - Report: Moran I, HH count, component sizes as function of these; look for elbow/stabilization

### Priority 3: Optional Extensions
7. **Hyperparameter block**
   - Which HPs correlate with observation-wise variance? Variance decomposition by HP
   - Which HPs drive hotspots? (e.g., high-variance regions correlated with certain HP configs)
8. **Calibration**
   - Add `CalibratedClassifierCV` to pipeline (fit on train/val only)
   - Rerun Rashomon with calibrated models; compare Moran/LISA/hotspots before vs. after
9. **Multi-dataset**
   - Run experiments for German Credit, Breast Cancer; ensure all analyses generalize

---

## 5. Technical Notes (for Code Development)

### Rashomon Pipeline
- **Training:** `run_training_pipeline.py` trains 50 candidates per family per run (10 outer seeds), saves P_val, P_test, meta, split. Rashomon set is **not** chosen at training time; it is selected at analysis time by top-K validation Brier.
- **Analysis:** `run_experiments.py` uses `analysis.experiment_runner` and `analysis.run_analysis`: for each run dir, load artifacts, select Rashomon (global top-K or per-family), compute multiplicity and spatial metrics.
- **Selection:** `select_rashomon_global(run_dir, K=25)`, `select_rashomon_per_family_totalK`, `select_rashomon_per_family_k_each` in `run_analysis.py`.
- Output: per-run summary in `results/{dataset}/summary_per_run.csv`; per-point arrays in `results/{dataset}/seed=N/per_point/`.

### Spatial Analysis
- In `run_analysis.py`: `spatial_analysis(v, X_test, k=30)` uses PySAL (KNN, Moran, Moran_Local), FDR correction; returns Moran's I, HH_mask, LL_mask, etc.
- `run_spatial`, `run_spatial_per_family`, `run_null` wrap loading + Rashomon selection + spatial/null.
- In `analysis/spatial.py`: `extract_hh_components` for connected-component analysis of HH subgraph.

### Null Model
- In `run_analysis.py`: `permute_predictions_independent(P, seed)` permutes each model's predictions independently; `null_experiment(...)` runs R permutations, recomputes variance and Moran/LISA each time.
- `run_null(run_dir, X_test, K=25, R=100, ...)` loads run, selects Rashomon, runs null.
- Under null: Moran I ~ 0; HH count drops; empirical p-value computed as (1 + count(null >= observed)) / (R + 1).

### Calibration (sklearn)
- `CalibratedClassifierCV(estimator, method='isotonic'|'sigmoid', cv=5)`
- Must be fit on train/val only; predict on test for evaluation
- Wrap base model in pipeline: `preprocess → model → calibrate`

### Interpretability Rules
- `DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)` on binary target: in_component vs. not
- Train on observations with known component membership (from LISA)
- Use `train` indices for fitting, `test` for evaluation
- `tree.export_text()` or `export_graphviz` for rule extraction

---

## 6. Terminology

| Term | Definition |
|------|------------|
| **Rashomon set** | Models with validation loss ≤ best + ε |
| **Predictive multiplicity** | Disagreement among near-optimal models |
| **Observation-wise variance** | `v_i = Var_m(p_hat_mi)` |
| **HH (High-High)** | LISA cluster: high variance + high-variance neighbors |
| **Hotspot** | Region of high multiplicity (HH component) |
| **Family-mismatch** | Variance driven by different model families, not genuine multiplicity |
| **ε (epsilon)** | Rashomon tolerance (e.g. 0.01 = 1% AUC difference) |

---

## 7. References (from Proposal)

- Predictive multiplicity, Rashomon sets
- Moran's I, LISA (Local Indicators of Spatial Association)
- COMPAS, German Credit, Breast Cancer benchmarks
- Related: PDP instability, model disagreement, bootstrap/ensemble uncertainty (to be distinguished)

---

*Last updated: 2026-03-13. Use this document when implementing thesis code and experiments.*
