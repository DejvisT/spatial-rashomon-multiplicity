# Priority 1: Commands and Interpretation Guide

This document lists all commands to run the Priority 1 experiments and how to interpret the results. It reflects the **current pipeline**: training with `run_training_pipeline.py`, analysis with `run_experiments.py`, and the notebooks in `notebooks/`.

---

## 1. Prerequisites

Ensure you are in the project root:

```powershell
cd "c:\Users\dejvi\Documents\pythonProject\Rashomon Sets\rashomon-multiplicity"
# Scripts add src/ to path automatically when run from project root
```

---

## 2. Training (required before analysis)

Run **one** of these. Each produces result directories under `results/` (or `results_fixed_test/` for fixed-test).

### Recommended: Standard training (10 outer runs per dataset)

Produces `results/{dataset}/seed=0/`, `seed=1/`, … `seed=9/` with 60/20/20 stratified splits, 50 candidates per family per run.

```powershell
python run_training_pipeline.py --dataset compas
python run_training_pipeline.py --dataset german
python run_training_pipeline.py --dataset breast_cancer
python run_training_pipeline.py --dataset adult
```

Options:
- `--out_dir results` (default)
- `--n_outer 10` — number of outer seeds
- `--n_candidates 50` — candidates per model family per run
- `--save_models` — also save trained pipeline pickles

### Optional: Fixed test set (for notebook 03 cross-seed stability)

Use when you need **the same test set across all seeds** (e.g. HH mask overlap, point-level stability). Writes to `results_fixed_test/{dataset}/seed=*/`.

```powershell
python run_training_pipeline_fixed_test.py --dataset compas
```

---

## 3. Analysis script (aggregate metrics over runs)

After training, run the experiment runner to compute Rashomon selection, multiplicity, spatial (Moran's I, HH count), and null (empirical p-value) per run, and aggregate to per-dataset summaries. Writes `results/{dataset}/summary_per_run.csv` and per-run `per_point/` outputs.

```powershell
# Single dataset
python run_experiments.py --results_dir results --dataset compas

# All datasets under results/
python run_experiments.py --results_dir results
```

Options:
- `--results_dir results` (default) — base directory containing dataset folders
- `--dataset compas` — run only this dataset; omit to run all
- `--K 25` — Rashomon top-K (default 25)
- `--R_null 100` — null permutations per run (default 100)

**Interpretation:**
- **Real Moran I >> null** → hotspots are not pipeline artefacts
- **Real n_HH >> null** → same conclusion
- Output is printed as a summary table; check `results/{dataset}/summary_per_run.csv` for per-seed diagnostics

---

## 4. Jupyter notebooks (full analysis)

Run in order from the project root (or from `notebooks/`). See **NOTEBOOK_GUIDE.md** for detailed descriptions.

| #   | Notebook                    | Purpose |
|-----|-----------------------------|---------|
| 01  | `01_primary_experiment.ipynb`   | Load runs, Rashomon selection, multiplicity + spatial, aggregate tables |
| 02  | `02_null_experiment.ipynb`     | Null permutation, empirical p-value, histograms |
| 03  | `03_spatial_patterns.ipynb`    | HH stability, Jaccard, components (uses fixed-test runs if available) |
| 04  | `04_sensitivity_K.ipynb`       | Sensitivity to Rashomon K |
| 05  | `05_sensitivity_kNN.ipynb`     | Sensitivity to kNN k |
| 06  | `06_hyperparameter_analysis.ipynb` | Family/HP importance, HH drivers |
| 07  | `07_calibration_robustness.ipynb`  | Platt scaling, before/after Moran/HH |
| 08  | `08_metrics_dashboard.ipynb`   | Consolidated metrics dashboard |
| 10  | `10_synthetic_multiplicity.ipynb` | Synthetic validation (self-contained) |
| 11  | `11_interpretable_rules.ipynb` | Rules for HH/high-variance regions |
| 12  | `12_robustness_and_fairness.ipynb` | Aggregated drivers, boundary analysis, fairness |

---

## 5. How to interpret results

### Null experiment (notebook 02)

| Metric      | Good outcome              | Meaning |
|------------|----------------------------|--------|
| Moran I    | Real >> null               | Spatial clustering is not due to chance |
| n_HH       | Real >> null               | HH regions are not artefacts |
| p_empirical| ≈ 0                        | Real Moran I not exceeded by null |

### Single-family / global (notebook 01, 05)

Compare global vs per-family Moran I and n_HH. If a single family (e.g. RF) shows significant Moran I and n_HH > 0, hotspots are not driven solely by mixing families.

### Interpretable rules (notebook 11)

- **precision_train** > 0.5 — rule captures HH reasonably
- **recall_train** > 0.3 — does not miss too many HH
- **precision_eval** similar to train — rule generalizes

---

## 6. What to do next

1. Run null experiment (notebook 02): confirm observed Moran I above null distribution.
2. Run single-family comparison (notebook 01/05): confirm Moran I + n_HH table and family-mismatch interpretation.
3. Run rules (notebook 11): get at least one component with interpretable rules and reasonable precision/recall.
4. Use results for thesis: multiplicity hotspots are real, not artefacts, and can be described with simple rules.

---

## 7. Troubleshooting

| Error | Fix |
|-------|-----|
| Missing results at … | Run training first (section 2) |
| `ModuleNotFoundError: analysis` or `ImportError: ... load_dataset` | Run from project root; scripts add `src/` to path |
| LISA slow | In analysis, reduce `R_null` (e.g. 99) or permutations in notebooks |
| Empty HH components | Try smaller `min_size` in `extract_hh_components` (notebook 03) |

---

## 8. Python files you need

| File | Purpose |
|------|--------|
| `run_training_pipeline.py` | Main training: 10 outer runs, 50 candidates/family, saves artifacts to `results/{dataset}/seed=N/` |
| `run_training_pipeline_fixed_test.py` | Same but fixed test set across seeds → `results_fixed_test/` (for notebook 03) |
| `run_experiments.py` | Analysis over existing runs: Rashomon selection, multiplicity, spatial, null; writes summary tables |
| `src/training_pipeline.py` | Core training logic |
| `src/data.py` | Datasets, splits, preprocessing |
| `analysis/run_analysis.py` | Rashomon selection, spatial, null, multiplicity metrics |
| `analysis/experiment_runner.py` | Orchestrates run_experiments per dataset |
| `analysis/preprocessing.py` | Transformed test features for spatial/null |

---

*Last updated to match current pipeline (run_training_pipeline, run_experiments, notebooks 01_primary_experiment etc.).*
