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

### Global Rashomon (all families)

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42
```

### Per-family Rashomon (LogReg, RF, GBM, MLP)

```powershell
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

**Interpretation**:
- **Real Moran I >> null Moran I** → hotspots are not pipeline artefacts
- **Real n_HH >> null n_HH** → same conclusion
- If any family shows real >> null, within-family hotspots are real

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

## 7. Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing results at ...` | Run Rashomon training first (section 2) |
| `ModuleNotFoundError: analysis` | Set `PYTHONPATH=.\src` and run from project root |
| LISA slow | In nulls, reduce `permutations` to 99 (or 999 for publication) |
| Empty HH components | Try smaller `min_size` in `extract_hh_components` |

---

*Last updated: 2025-02-08*
