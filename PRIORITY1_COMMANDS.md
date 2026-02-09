# Priority 1: Commands and Interpretation Guide

This document lists all commands to run the Priority 1 experiments and how to interpret the results.

---

## 1. Prerequisites

Ensure you are in the project root:

```powershell
cd "c:\Users\dejvi\Documents\pythonProject\Rashomon Sets\rashomon-multiplicity"
$env:PYTHONPATH = ".\src"
```

---

## 2. Rashomon Training (required before analysis)

Run these commands **in order**. Each produces results in `results/compas/`.

### ⭐ Recommended: Cross-Validation Mode

**Use CV mode for more stable, less split-dependent Rashomon sets.** This is the recommended approach as it makes results more comparable to multiplicity papers that emphasize stability.

#### Global Rashomon with CV (all families)

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --use_cv --cv_method kfold --n_folds 5
```

#### Per-family Rashomon with CV

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family LogReg --use_cv --n_folds 5
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family RF --use_cv --n_folds 5
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family GBM --use_cv --n_folds 5
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family MLP --use_cv --n_folds 5
```

**CV Options:**
- `--use_cv`: Enable cross-validation mode
- `--cv_method kfold`: Use K-fold CV (recommended)
- `--cv_method repeated_holdout`: Use repeated random splits
- `--n_folds 5`: Number of folds for K-fold (default: 5)
- `--n_repeats 5`: Number of repeats for repeated holdout (default: 5)

**How CV works:**
- Test set (20%) is held out first
- Remaining 80% is split into K folds (or repeated holdouts)
- Each model's loss is averaged across all CV folds: `L(m) = mean(log_loss across folds)`
- Rashomon membership: `L(m) ≤ L* + ε` where L* is the best mean CV loss
- Final models are fit on all train+val data for test predictions

**Benefits:**
- Less dependent on a single lucky split
- More stable Rashomon set composition
- Comparable to practices in multiplicity literature
- Reduces variance in results

---

### Alternative: Single Split Mode (backward compatible)

For quick tests or comparison with old results, you can still use single train/val/test split:

```powershell
# Global Rashomon (single split)
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42

# Per-family Rashomon (single split)
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family LogReg
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family RF
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family GBM
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family MLP
```

**Expected output**: Each run creates (or overwrites) a folder under `results/compas/`:
- `seed=42_eps=0.01/` (global)
- `family=LogReg/seed=42_eps=0.01/`
- `family=RF/seed=42_eps=0.01/`
- `family=GBM/seed=42_eps=0.01/`
- `family=MLP/seed=42_eps=0.01/`

**Note:** When using CV, the `split.npz` file will contain CV metadata and `cv_splits.npz` will contain all fold splits.

---

## 3. Analysis Script (quick sanity check)

```powershell
python run_experiments.py --quick
```

**What it does**:
- Loads real results for global + each family
- Runs null experiments (5 runs, 99 permutations) for each
- Prints Moran I and n_HH (real vs null)
- Prints interpretable rules for Global HH components

**Note:** The analysis script works with both CV and single-split results. It automatically detects and loads the appropriate result files.

**Interpretation**:
- **Real Moran I >> null Moran I** → hotspots are not pipeline artefacts
- **Real n_HH >> null n_HH** → same conclusion
- If any family shows real >> null, within-family hotspots are real
- **CV results** should show similar or more stable patterns compared to single-split

---

## 4. Jupyter Notebooks (full analysis)

Run these **in order** from the project root (or from `notebooks/`):

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_load_and_sanity_check.ipynb` | Load data, check splits |
| 02 | `02_spatial_hotspots.ipynb` | Moran I, LISA, HH components (global) |
| 03 | `03_null_models.ipynb` | Real vs null Moran I, n_HH; histograms |
| 04 | `04_stability.ipynb` | HH stability across runs |
| 05 | `05_single_family_rashomon.ipynb` | Moran I + n_HH per family; family-mismatch |
| 06 | `06_interpretable_rules.ipynb` | Decision-tree rules for HH components |

---

## 5. How to Interpret Results

### Notebook 03 (Null models)

| Metric | Good outcome | Meaning |
|--------|--------------|---------|
| Moran I | Real >> null (e.g. real 0.4 vs null ~0.05) | Spatial clustering is not due to chance |
| n_HH | Real >> null (e.g. real 46 vs null 0–2) | HH regions are not artefacts |
| p_empirical | ≈ 0 | Real Moran I never exceeded by null |

**Conclusion**: If real >> null across Global and per-family, hotspots are **not** pipeline artefacts.

### Notebook 05 (Single-family)

| Set | Moran I | p | n_HH | Interpretation |
|-----|---------|---|------|----------------|
| Global | High (~0.44) | 0 | Many (~46) | Strong spatial clustering across all models |
| RF-only | Moderate (~0.15) | 0 | Some | Within-family hotspots exist |
| LogReg/GBM/MLP | Low | Often NS | Few | Weaker within-family signal |

**Family-mismatch**: If RF-only (or any single family) shows significant Moran I and n_HH > 0, then hotspots are **not** driven solely by mixing different model families. The global signal is partly cross-family, partly within-family.

### Notebook 06 (Interpretable rules)

| Metric | Target | Meaning |
|--------|--------|---------|
| precision_train | > 0.5 | Rule captures HH points reasonably |
| recall_train | > 0.3 | Rule does not miss too many HH points |
| precision_eval | Similar to train | Rule generalizes out-of-sample |
| rule_text | Readable | e.g. "age > 25 AND priors > 2" |

**Conclusion**: Simple rules can describe hotspot regions; useful for thesis narrative.

---

## 6. What to Do Next (after running)

1. **Check null experiments**: In notebook 03, ensure histograms show observed Moran I well above the null distribution for Global and (ideally) RF.
2. **Check single-family**: In notebook 05, ensure the Moran I + n_HH table is populated; confirm family-mismatch interpretation.
3. **Check rules**: In notebook 06, ensure at least one component has interpretable rules with reasonable precision/recall.
4. **Write up**: Use these results to support the thesis claim that multiplicity hotspots are real, not artefacts, and can be described with simple rules.

---

## 7. CV vs Single Split: When to Use What?

| Scenario | Recommendation | Command |
|----------|---------------|---------|
| **Thesis/publication** | Use CV (more robust) | `--use_cv --cv_method kfold --n_folds 5` |
| **Quick testing** | Single split (faster) | (no CV flags) |
| **Comparison study** | Run both, compare | Run with and without `--use_cv` |
| **Stability analysis** | Use CV (required) | `--use_cv --cv_method kfold --n_folds 5` |

**Key differences:**
- **CV mode**: Rashomon membership based on mean CV loss (averaged across folds)
- **Single split**: Rashomon membership based on single validation loss
- **Training time**: CV is ~K times slower (K = number of folds) but more stable
- **Results**: CV results are less split-dependent and more reproducible

---

## 8. Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing results at ...` | Run Rashomon training first (section 2) |
| `ModuleNotFoundError: analysis` | Set `PYTHONPATH=.\src` and run from project root |
| `ImportError: cannot import name 'load_dataset'` | Script now auto-adds `src/` to path; ensure you're in project root |
| LISA slow | In nulls, reduce `permutations` to 99 (or 999 for publication) |
| Empty HH components | Try smaller `min_size` in `extract_hh_components` |
| CV training very slow | Reduce `--n_folds` to 3 for faster testing, or use `--n_models 10` for quick checks |

---

## 9. Progress Tracking

The script now prints detailed progress:
- Configuration summary at start
- Progress for each model family (every 10 models)
- Rashomon selection summary (best loss, threshold, counts)
- Final predictions shape

Example output:
```
============================================================
Rashomon Experiment Configuration
============================================================
Dataset: compas
Mode: global
CV: kfold (5 folds/repeats)
...

[LogReg] Starting training (30 models)...
  [LogReg] Progress: 10/30 models trained
  [LogReg] Progress: 20/30 models trained
  [LogReg] Progress: 30/30 models trained
[LogReg] ✓ Finished training 30 models
...
```

---

*Last updated: 2025-02-08 (Added CV support)*
