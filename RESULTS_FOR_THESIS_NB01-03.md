# Results from Notebooks 01–03 for Thesis (Ch. 6)

**Purpose:** Drop-in text and placeholders for Prism. Sections map to thesis Ch. 6 (Results). Use placeholders where figures should go.

**Notebook mapping (current codebase):** Results below refer to the current notebooks: **01_primary_experiment**, **02_null_experiment**, **03_spatial_patterns** (replacing the earlier 01_load_and_sanity_check, 02_spatial_hotspots, 03_null_models).

---

## 6.1 Rashomon set composition

**Source: Notebook 01 (load and sanity check)**

For the COMPAS experiment (seed=42, ε=0.01, single split), the pipeline produced a prediction matrix **P** of shape **(24, 1443)**: 24 models in the Rashomon set and 1443 test observations. The Rashomon set is not evenly distributed across families: **MLP contributes 14 models**, **GBM 9**, and **RF 1**; Logistic Regression contributes none under this run’s threshold. Validation losses of the selected models are tightly clustered (mean ≈ 0.601, std ≈ 0.003), confirming that all selected models lie within the ε-band of the best model.

**Interpretation:** The dominance of MLP and GBM in this run illustrates that multiplicity is present even when the set is restricted to near-optimal performance. The small number of families represented also implies that observation-wise variance is driven by both hyperparameter and architectural differences within and across these families.

---

## 6.2 Observation-wise predictive variance

**Source: Notebook 01**

Observation-wise predictive variance **v** (variance of predicted probabilities across the 24 Rashomon models for each test point) is right-skewed: most observations have low variance, and a minority have high variance. There are **no points with near-zero variance** (v < 1e-6), and **14 points** lie above the 99th percentile of variance—i.e. a small subset of individuals exhibits very high prediction instability. The **fraction of points with high flip instability** (proportion of models that predict the opposite class at threshold 0.5) above 0.9 is about **5.6%**, indicating that for roughly one in eighteen test points, the majority of near-optimal models disagree on the binary decision.

A scatter of **mean predicted probability vs. variance** shows that variance tends to be elevated for points whose mean prediction is near 0.5; this is consistent with decision-boundary effects, where small probability differences across models more easily flip the classification.

**[FIGURE: Distribution of observation-wise predictive variance (histogram). Caption: Distribution of predictive variance across test observations; most points have low variance, with a long right tail.]**

**[FIGURE: Mean predicted probability vs. variance (scatter). Caption: Variance is concentrated for observations with mean prediction near 0.5.]**

**[FIGURE: Distribution of flip instability (histogram). Caption: Fraction of models disagreeing at threshold 0.5; a minority of points show very high flip instability.]**

**Interpretation:** The variance-based metric distinguishes individuals who are stable across the Rashomon set from those who are not. The concentration of high variance near the decision boundary supports the interpretation that multiplicity is especially relevant for “borderline” cases, with direct implications for fairness and reliability of automated decisions.

---

## 6.3 Spatial structure of instability

**Source: Notebook 02 (spatial hotspots)**

Spatial analysis uses a **k-NN graph** (k=10) on standardized numeric features of the test set. **Global Moran’s I** for the vector of observation-wise variances is **I ≈ 0.44** with **p_value ≈ 0** (permutation test), indicating strong positive spatial autocorrelation: observations with similar predictive variance tend to be neighbors in feature space. Thus, instability is not scattered at random; it clusters.

**Local Indicators of Spatial Association (LISA)** assign each observation to a cluster type (HH, HL, LH, LL, or non-significant). For this run there are **46 High–High (HH)** observations—points with high variance surrounded by high-variance neighbors. These form **3 connected components** (min size 5) in the k-NN graph restricted to HH nodes. The **largest component** contains 30 points. Mean variance among HH points is substantially higher than among non-HH points, and HH points appear as coherent regions in a PCA projection of the feature space.

**[FIGURE: Variance distribution with HH points highlighted (e.g. vertical line at HH mean). Caption: HH observations have systematically higher predictive variance than the rest.]**

**[FIGURE: HH points in feature space (PCA projection, HH in red). Caption: High–High multiplicity points form localized regions in feature space.]**

**[FIGURE: HH component size distribution (bar chart). Caption: Sizes of connected HH components; largest component has 30 points.]**

**Cross-dataset summary (from multiplicity_regions_summary.csv):**

| Dataset        | N    | HH count | HH % | Moran's I | Moran p-value | Components | Largest comp |
|----------------|------|----------|------|-----------|---------------|------------|--------------|
| Compas         | 2165 | 152      | 7.0% | 0.2912    | 0.0003        | 11         | 30           |
| German Credit  | 300  | 12       | 4.0% | 0.1735    | 0.0003        | 1          | 11           |
| Breast Cancer  | 171  | 14       | 8.2% | 0.1821    | 0.0003        | 1          | 14           |

**Interpretation:** Positive Moran’s I and localized HH regions across all three datasets show that predictive multiplicity has **spatial structure** in feature space. The hotspots are interpretable as subgroups where model-selection uncertainty concentrates, rather than isolated outliers. The number and size of components vary by dataset (COMPAS shows more and larger components), consistent with differences in sample size and feature structure.

---

## 6.3 (continued) Null model validation

**Source: Notebook 03 (null models)**

To check that the detected spatial structure is not an artefact of the variance distribution alone, **model-wise permutation** is applied: for each model, predictions are permuted across observations (preserving each model’s marginal distribution but breaking any alignment with feature space). Under this null, observation-wise variance is recomputed and Moran’s I and HH counts are obtained for each null replicate.

**Results:** For the **global** Rashomon run, **observed Moran’s I ≈ 0.44** lies far above the null distribution (e.g. null max ≈ 0.08), giving a **ratio of about 5.2×** (real to null maximum). **Observed HH count (46)** is much larger than under the null (null mean and max are near 0–2). Empirical p-values (fraction of null replicates ≥ observed) are effectively zero for both Moran’s I and HH count. The same pattern holds for **per-family** runs: real Moran’s I and n_HH exceed the null in all cases, with the largest ratios for Global, GBM, and MLP (e.g. GBM ≈ 3.5×, MLP ≈ 4.1×). LogReg shows the smallest ratio (~1.1×), consistent with less spatial structure when the Rashomon set is restricted to a single, relatively stable family.

**[FIGURE: Moran’s I — observed vs null. Histogram of null Moran’s I with a vertical line at the observed value. Caption: Under model-wise permutation, Moran’s I collapses to near zero; the observed value is far above the null.]**

**[FIGURE: HH count — real vs null (bar chart by run: Global, LogReg, RF, GBM, MLP). Caption: Number of High–High points under the real data vs mean under the null; real counts are much higher.]**

**Interpretation:** The spatial pattern of predictive variance is **not** explained by the marginal distribution of variances or by the geometry of the k-NN graph alone. When the link between predictions and feature space is broken by permutation, spatial autocorrelation and HH counts drop to levels consistent with chance. This supports the conclusion that **multiplicity hotspots are a genuine feature of the prediction–feature relationship** in the Rashomon set, rather than a pipeline artefact.

---

## Suggested figure list for Prism

1. **Fig (6.1):** Histogram of observation-wise predictive variance.
2. **Fig (6.2):** Scatter: mean predicted probability vs variance.
3. **Fig (6.3):** Histogram of flip instability.
4. **Fig (6.4):** Variance distribution with HH mean indicated (or HH vs non-HH).
5. **Fig (6.5):** HH points in feature space (PCA projection).
6. **Fig (6.6):** HH component size distribution.
7. **Fig (6.7):** Moran’s I — observed vs null (histogram + vertical line).
8. **Fig (6.8):** HH count — real vs null mean by run (bar chart).

(Table: Multi-dataset spatial summary can stay as in the CSV or be formatted as a single table in the thesis.)
