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
│   ├── nulls.py        # Model-wise permutation, run_null_experiment
│   ├── rules.py        # extract_component_rules, rules_summary_df (interpretable rules)
│   ├── spatial.py      # build_knn_graph, moran_global, lisa_local, extract_hh_components
│   └── stability.py     # hh_selection_frequency, jaccard_index, hh_jaccard_matrix
├── src/
│   ├── data.py         # load_dataset, make_split, make_preprocessor
│   ├── metrics.py      # prediction_variance, flip_instability, etc.
│   ├── rashomon.py     # build_rashomon_set (global, per_family, single-family)
│   └── plots.py
├── notebooks/
│   ├── 01_load_and_sanity_check.ipynb
│   ├── 02_spatial_hotspots.ipynb      # Moran I, LISA, HH components
│   ├── 03_null_models.ipynb          # Permutation null
│   ├── 04_stability.ipynb            # HH stability across runs
│   ├── 05_single_family_rashomon.ipynb  # Within-family Moran/LISA
│   └── 06_interpretable_rules.ipynb   # Describe HH components with decision-tree rules
├── run_rashomon_experiment.py
├── data/
│   └── compas-scores-two-years.csv
└── results/
    └── compas/
        ├── seed=42_eps=0.01/         # Global Rashomon
        ├── family=LogReg/, family=RF/, family=GBM/, family=MLP/
        └── ...
```

### Key Data Artifacts
- `P_test.npy`: shape `(n_models, n_obs)` – predicted probabilities per model per observation
- `metrics.npz`: `variance`, `flip_instability`, etc.
- `X_test.csv`, `y_test.npy`: test features and labels
- `config.json`: ε, seed, n_models_rashomon, etc.

### Implemented Functionality
- **Spatial**: kNN graph, Moran's I, LISA (with FDR), HH component extraction
- **Nulls**: `permute_predictions` (model-wise), `run_null_experiment`, `run_null_experiments`
- **Stability**: `hh_selection_frequency`, `jaccard_index`, `hh_jaccard_matrix`, `summarize_hh_stability`
- **Single-family**: Notebook 05 compares global vs. per-family (RF, LR, GBM, MLP) Moran/LISA

### Gaps
- **Interpretability / rules**: ✓ Implemented in `analysis/rules.py` and `06_interpretable_rules.ipynb` (decision trees, precision/recall, out-of-sample when possible)
- **Hyperparameter analysis**: No variance decomposition, no HP-vs-hotspot analysis
- **Stability (3)**: No sensitivity analysis for ε, k, n_models
- **Stability (2)**: No cluster-level stability (e.g., same regions across runs)
- **Calibration**: No CalibratedClassifierCV integration
- **Multi-dataset**: Only COMPAS results present; German Credit, Breast Cancer need runs

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
- `run_rashomon_experiment.py` uses `src/rashomon.py`
- `build_rashomon_set` supports `selection_mode`: `"global"` or `"per_family"`
- Single-family: `--family LogReg` (or RF, GBM, MLP)
- Output: `P_test`, `metrics`, `meta`, `X_test`, `y_test`

### Spatial Analysis
- `build_knn_graph(X, k)` – standardize=True by default
- `moran_global(v, W)` – returns `{"I": float, "p_value": float}`
- `lisa_local(v, W)` – returns DataFrame with `cluster` (HH, HL, LH, LL, NS)
- `extract_hh_components(lisa_df, W, min_size=5)` – returns `comp_id`, `components`

### Null Model
- `permute_predictions(P)` – permutes each model's predictions independently
- `run_null_experiment(P, X_knn, k=10)` – returns `v_perm`, `moran`, `lisa`
- Under null: Moran I should be ~0; HH count should drop

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

*Last updated: 2025-02-08. Use this document when implementing thesis code and experiments.*
