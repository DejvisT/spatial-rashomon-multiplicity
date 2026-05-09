# Rashomon Multiplicity — Spatial Analysis of Predictive Multiplicity

MSc thesis project investigating **predictive multiplicity** in Rashomon sets using spatial statistics.

## Overview

Many datasets admit multiple near-optimal models that disagree on individual predictions. This project uses observation-wise predictive variance and conflict ratio together with spatial autocorrelation methods (Moran's I and LISA) on kNN feature-space graphs to identify localized multiplicity hotspot regions. The final analyses cover benchmark datasets, null testing, hotspot stability, global vs per-family Rashomon sets, robustness checks, hyperparameter and family importance, synthetic validation, interpretable rules, decision-boundary proximity, and subgroup exposure diagnostics.

## Datasets

- COMPAS
- German Credit
- Adult
- Synthetic single-island and three-islands datasets

## Pipeline

```
1. Train candidate models    →  run_training_pipeline.py
2. Train fixed-test runs     →  run_training_pipeline_fixed_test.py
3. Select Rashomon sets      →  analysis/run_analysis.py
4. Compute spatial metrics   →  analysis/spatial.py, analysis/run_analysis.py (Moran, LISA)
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
| `analysis/`   | Reusable analysis modules (spatial, stability, calibration, rules, hyperparameter and family analyses) |
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
