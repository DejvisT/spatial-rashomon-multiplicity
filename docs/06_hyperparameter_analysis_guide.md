# Guide: Notebook 06 — Hyperparameter Analysis

This document explains the **06_hyperparameter_analysis** notebook in detail: how predictive multiplicity variance is decomposed, what each formula and metric means, and how importance is computed for model families and hyperparameters.

---

## 1. Purpose and workflow (Option A: family-first)

The notebook answers: *What drives predictive multiplicity in the Rashomon set?*

To avoid confounding (e.g. “missing hyperparameter” acting as a proxy for model family), the analysis uses a **family-first** workflow:

1. **Step 1 — Family importance:** Decompose variance by **model family** (`model_name`). This quantifies how much multiplicity is explained by “which algorithm family you chose” (e.g. GBM vs MLP vs RF).
2. **Step 2 — Within-family HP importance:** Within each family, decompose variance by **hyperparameter** (e.g. `max_depth`, `learning_rate`). Missing HP values are **dropped** so “nan” is not treated as a group.
3. **Step 3 — Hotspot-specific drivers:** Repeat the within-family HP decomposition **only on HH (high–high) hotspot observations** from LISA, and compare to “all observations” via **Δ ratio_of_sums**.

All computations use **population variances** (ddof=0) and operate on saved artifacts (`meta.csv`, `P_test.npy`) without retraining.

---

## 2. Variance decomposition: the exact identity

We have:

- **P:** prediction matrix of shape `(n_models, n_obs)`. Row `m` = predictions of model `m` on all test points.
- **G:** a categorical factor (e.g. model family or HP value) with one label per model.

For each observation \(i\), the **total predictive variance** across models is:

\[
\text{Var}_{\text{total}}(i) = \text{Var}_m\bigl(P[m,i]\bigr)
\]

We decompose this using the **exact population identity** (law of total variance):

\[
\text{Var}(P) = \text{Var}\bigl(\mathbb{E}[P \mid G]\bigr) + \mathbb{E}\bigl[\text{Var}(P \mid G)\bigr]
\]

In our notation:

- **Var_between(i)** = variance of the **group means** (how much the average prediction per group differs from the overall mean at observation \(i\)).
- **Var_within(i)** = **average within-group variance** (how much models within the same group still disagree at observation \(i\)).

So:

\[
\text{Var}_{\text{total}}(i) = \text{Var}_{\text{between}}(i) + \text{Var}_{\text{within}}(i).
\]

No approximation: this holds exactly for the population variances we use.

---

## 3. Implementation: per-observation decomposition

In code (`analysis/hyperparams.py`, `variance_decomposition_by_groups`):

- **Inputs:** `preds` shape `(n_models, n_obs)`, and `group_keys` length `n_models` (one group label per model).
- Groups are filtered: groups with fewer than `min_group_size` models are dropped; only models in retained groups are used.
- For each observation \(i\):

  - \(\bar{p}_i\) = overall mean over models (on kept models).
  - \(\text{Var}_{\text{total}}(i) = \frac{1}{n}\sum_m (P[m,i] - \bar{p}_i)^2\) (population variance, ddof=0).

  For each group \(g\) with index set \(I_g\) and size \(n_g\):

  - \(\bar{p}_{g,i}\) = mean of \(P[m,i]\) for \(m \in I_g\).
  - \(p_g = n_g / n_{\text{models}}\) (fraction of models in group \(g\)).

  Then:

  \[
  \text{Var}_{\text{between}}(i) = \sum_g p_g\, (\bar{p}_{g,i} - \bar{p}_i)^2,
  \]
  \[
  \text{Var}_{\text{within}}(i) = \sum_g p_g\, \text{Var}(P[I_g, i]).
  \]

- **Ratio (per observation):**  
  \[
  \text{ratio}(i) = \frac{\text{Var}_{\text{between}}(i)}{\max(\text{Var}_{\text{total}}(i), \varepsilon)}.
  \]  
  Default \(\varepsilon = 10^{-12}\) to avoid division by zero. So **ratio** is the fraction of total variance at observation \(i\) explained by the factor (family or HP).

---

## 4. Summary statistics from the decomposition

After computing `var_total`, `var_between`, `var_within`, and `ratio` for each observation, we summarize (see `_summarize_decomposition` in `hyperparams.py`):

| Quantity | Definition | Meaning |
|----------|------------|--------|
| **mean_ratio** | \(\frac{1}{n_{\text{obs}}}\sum_i \text{ratio}(i)\) | Average over observations of “fraction of variance explained by factor”. |
| **median_ratio** | Median of \(\text{ratio}(i)\) | Robust center of that distribution. |
| **p90_ratio** | 90th percentile of \(\text{ratio}(i)\) | Detects **localized** effects (e.g. factor matters a lot for some points). |
| **ratio_of_sums** | \(\frac{\sum_i \text{Var}_{\text{between}}(i)}{\sum_i \text{Var}_{\text{total}}(i)}\) | **Global** fraction of total variance (over all observations) explained by the factor. More stable than mean_ratio when variance is concentrated in few points. |

Denominator in **ratio_of_sums** is clamped by `EPS_RATIO` so it is never zero.

In tables you also see:

- **mean_var_between_ratio** — same as **mean_ratio** (mean of per-observation ratios).
- **mean_var_within_ratio** — \(\frac{1}{n}\sum_i \frac{\text{Var}_{\text{within}}(i)}{\max(\text{Var}_{\text{total}}(i), \varepsilon)}\). So mean fraction of variance that is *within* groups; it equals `1 - mean_ratio` when the decomposition is exact.

---

## 5. Family importance

**Function:** `compute_family_importance(meta, preds, family_col="model_name", obs_mask=None)`  

**What it does:**

- Uses **model family** (`model_name`) as the group label \(G\).
- Calls `variance_decomposition_by_groups(preds, group_keys)` with `group_keys = meta[family_col]`.
- Returns a **one-row DataFrame** with:
  - `factor` = `"model_name"`,
  - `n_models_total`, `n_models_used`, `n_groups`,
  - `mean_ratio`, `median_ratio`, `p90_ratio`, **ratio_of_sums**.

**Interpretation:**  
A high **ratio_of_sums** (or mean_ratio) for family means: “a large share of predictive multiplicity is explained by which algorithm family you chose,” rather than by hyperparameters within a family.

Optional **obs_mask**: restrict to a subset of observations (e.g. HH hotspots); then decomposition and summaries are only over that subset.

---

## 6. Within-family hyperparameter importance

**Function:** `compute_within_family_hp_importance(meta, preds, hp_cols=None, family_col="model_name", obs_mask=None, dropna=True, ...)`  

**What it does:**

- For **each model family** (e.g. GBM, MLP, RF):
  - Restrict to models in that family: `P_f`, `meta_f`.
  - For **each HP column** (e.g. `hp_max_depth`, `hp_learning_rate`):
    - **Group key** = HP value (via `make_hp_key` for stable string representation).
    - If **dropna=True** (default): drop models where this HP is missing, so “nan” is not a group.
    - Require at least **min_groups** distinct values and **min_group_size** models per group.
    - Run `variance_decomposition_by_groups(P_hp, keys_hp)` and then `_summarize_decomposition`.
  - One row per (family, hp) with: `family`, `hp`, `n_models_used`, `n_groups`, `mean_ratio`, `median_ratio`, `p90_ratio`, **ratio_of_sums**.

**Interpretation:**  
Within that family, **ratio_of_sums** for an HP answers: “what fraction of (that family’s) predictive variance is explained by this hyperparameter?”  
High **ratio_of_sums** → that HP is an important driver of multiplicity within the family.

**Backward-compatible API:** `compute_hp_importance(P, meta, model_family=None, dropna=True, ...)`  
- If `model_family` is set, restricts to that family then does the same per-HP decomposition.  
- Returns columns: `hyperparameter`, `mean_var_between_ratio`, **ratio_of_sums**, `median_ratio`, `p90_ratio`, `mean_var_within_ratio`, `n_values`, `n_groups`, `n_models_used`, etc.

---

## 7. Hotspot-specific importance (HH subset)

**Idea:** Multiplicity may be driven by different factors **inside spatial hotspots** (HH from LISA) than on the rest of the test set.

**Steps in the notebook:**

1. Get **pointwise variance** for the selected Rashomon set: \(v_i = \text{Var}_m(P[m,i])\).
2. Run **LISA** (Local Moran) on \(v\) with kNN graph on test features; FDR-corrected significance gives **HH_mask** (high–high hotspots).
3. **Family importance on HH:**  
   `compute_family_importance(meta, P, obs_mask=HH_mask)`  
   → ratio_of_sums and mean_ratio **restricted to HH points**.
4. **Within-family HP importance on HH:**  
   `compute_within_family_hp_importance_on_subset(meta_f, P_f, obs_mask=HH_mask, ...)`  
   → same decomposition, but only over observations where `HH_mask` is True.

So you get:

- **ratio_of_sums_all** — factor importance on all test points.
- **ratio_of_sums_HH** — factor importance on HH points only.

**Delta (hotspot-specific driver):**

\[
\Delta\,\text{ratio\_of\_sums} = \text{ratio\_of\_sums}_{\text{HH}} - \text{ratio\_of\_sums}_{\text{all}}.
\]

- **Positive Δ:** this factor explains **more** variance in hotspots than on average → **hotspot-specific driver**.
- **Negative Δ:** this factor explains **less** variance in hotspots.

The notebook plots **Δ ratio_of_sums** per HP (and optionally per family) to highlight which HPs (or families) are especially important in HH regions.

---

## 8. Hyperparameter profiling (local Rashomon sets)

**Function:** `hyperparameter_profiling(meta, hp_name, epsilon, model_family=None, loss_col=None, dropna=False)`  

**What it does:** For each **value** \(h\) of the given hyperparameter:

1. Take all models with that HP value: \(h\).
2. **Best loss** at \(h\): \(L^*(h) = \min\{\text{val\_loss among models with HP}=h\}\).
3. **Local Rashomon set** at \(h\): models with HP\(=h\) and loss \(\le L^*(h) + \varepsilon\).
4. Record: `hp_value`, `best_loss`, `local_rashomon_size`, `total_models`, **local_rashomon_fraction** = size of local set / total models at \(h\).

**Interpretation:**  
High **local_rashomon_fraction** at a value \(h\) means: among models with that HP value, a large fraction are in the “good” (Rashomon) region. So that value is “safe” in terms of multiplicity of near-optimal models.  
This uses the **candidate pool in `meta`** (e.g. full candidate set or pre-filtered Rashomon set, depending on what you pass).

---

## 9. Outputs produced by the notebook

**Tables:**

- `family_importance_{dataset}_seed{seed}.csv` — one row: family factor summary (mean_ratio, ratio_of_sums, etc.).
- `hp_importance_within_family_{dataset}_seed{seed}.csv` — within-family HP importance (all observations).
- `hp_importance_within_family_HH_{dataset}_seed{seed}.csv` — within-family HP importance on HH only (or per-family files depending on flow).
- `family_importance_subsets_{dataset}_seed{seed}.csv` — family importance for subsets: all / HH / non-HH.

**Figures:**

- `family_between_ratio_hist_{dataset}_seed{seed}.pdf` — histogram of per-observation **ratio** (Var_between / Var_total) for family.
- `within_family_hp_importance_{dataset}_seed{seed}_{family}.pdf` — bar chart of **ratio_of_sums** per HP within that family.
- `hh_delta_hp_importance_{dataset}_seed{seed}_{family}.pdf` — bar chart of **Δ ratio_of_sums** (HH − all) per HP.
- `family_importance_by_subset_{dataset}_seed{seed}.pdf` — ratio_of_sums for family on subsets (all / HH / non-HH).

---

## 10. Quick reference: main formulas

| Symbol / name | Formula |
|---------------|--------|
| Var_total(i) | \(\text{Var}_m(P[m,i])\) (population, ddof=0) |
| Var_between(i) | \(\sum_g p_g\, (\bar{p}_{g,i} - \bar{p}_i)^2\) |
| Var_within(i) | \(\sum_g p_g\, \text{Var}(P[I_g,i])\) |
| ratio(i) | \(\text{Var}_{\text{between}}(i) \big/ \max(\text{Var}_{\text{total}}(i), \varepsilon)\) |
| **ratio_of_sums** | \(\sum_i \text{Var}_{\text{between}}(i) \big/ \sum_i \text{Var}_{\text{total}}(i)\) |
| mean_ratio | \(\frac{1}{n}\sum_i \text{ratio}(i)\) |
| Δ ratio_of_sums | ratio_of_sums_HH − ratio_of_sums_all |

All variances in this module use **ddof=0** (population variance). Group weights \(p_g\) are proportions of (retained) models in each group.
