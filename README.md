# Rashomon Multiplicity — Spatial Analysis of Predictive Multiplicity

MSc thesis project investigating **predictive multiplicity** in Rashomon sets using spatial statistics.

## Overview

Many datasets admit multiple near-optimal models that disagree on individual predictions. This project uses spatial autocorrelation (Moran's I, LISA, Getis-Ord Gi*) on kNN feature-space graphs to identify **where** models disagree — revealing coherent *multiplicity hotspot regions* rather than just global disagreement metrics. Extended analyses include soft (weighted) Rashomon sets, spatial correlograms for distance-decay structure, Gower-distance and PCA-based kNN for mixed-type features, and bootstrap ablation for robustness testing.

## Datasets

- **COMPAS** (recidivism prediction)
- **German Credit** (UCI)
- **Breast Cancer** (sklearn / UCI)
- Synthetic datasets with planted ambiguous regions (for validation)

## Pipeline

```
1. Train candidate models    →  run_training_pipeline.py
2. Select Rashomon sets      →  analysis/run_analysis.py (hard threshold & soft/weighted)
3. Compute spatial metrics   →  analysis/spatial.py, analysis/run_analysis.py (Moran, LISA, Gi*)
4. Robustness testing        →  analysis/bootstrap_ablation.py
5. Analyze in notebooks      →  notebooks/01–10
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Train models (10 seeds x 3 datasets, 5 families x 50 candidates each)
python run_training_pipeline.py

# Train with fixed test set (for cross-seed stability in notebook 03)
python run_training_pipeline_fixed_test.py

# Run notebooks 01–10 in order
```

## Project Structure

| Directory     | Contents                                                      |
|---------------|---------------------------------------------------------------|
| `analysis/`   | Reusable analysis modules (spatial, stability, calibration, rules, hyperparams, bootstrap_ablation) |
| `src/`        | Core utilities (data loading, training pipeline, synthetic data) |
| `notebooks/`  | Analysis notebooks 01–10                                      |
| `data/`       | Raw datasets (COMPAS CSV; others fetched via sklearn)         |
| `scripts/`    | Thesis asset export scripts                                   |
| `results/`    | Training artifacts (gitignored)                               |

## Notebooks

| #  | Notebook                          | Purpose                                      |
|----|-----------------------------------|----------------------------------------------|
| 01 | Primary Experiment                | Rashomon selection, multiplicity + spatial metrics, summary tables |
| 02 | Null Experiment                   | Permutation test for spatial significance     |
| 03 | Spatial Patterns                  | HH stability across seeds, connected components |
| 04 | Sensitivity to K                  | Rashomon set size sensitivity                 |
| 05 | Sensitivity to kNN                | Neighborhood size sensitivity                 |
| 06 | Hyperparameter Analysis           | Variance decomposition by family and HP       |
| 07 | Calibration Robustness            | Platt + isotonic scaling; spatial survival    |
| 08 | Synthetic Multiplicity            | Validation on synthetic data with ground truth |
| 09 | Interpretable Rules               | Rule extraction for hotspot description       |
| 10 | Robustness and Fairness           | Decision boundary, fairness, alternative kNN  |

See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for detailed documentation of each notebook.
