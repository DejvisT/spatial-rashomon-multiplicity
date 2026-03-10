# 07 — Calibration robustness check

**Key idea:** Apply Platt scaling to Rashomon-set predictions and check whether spatial multiplicity (Moran's I, HH count) changes substantially — distinguishing structural multiplicity from calibration artifacts.

## Loads

- Run directories, per-run artifacts (meta, P_test, split)
- `run_calibration_experiment`, `calibrate_predictions_for_run` from `analysis.calibration`
- All datasets with runs (skips synthetic)

## Produces

- `results/{dataset}/calibration_summary_per_run.csv`, `calibration_summary.csv`
- `tables/calibration_per_run.csv`, `tables/calibration_summary.csv`
- `figures/calibration_spatial_comparison_{dataset}.pdf`
- `figures/calibration_jaccard_{dataset}.pdf` — Jaccard HH before/after
- `figures/calibration_delta_metrics_{dataset}.pdf` — Δ metric distributions
- `figures/calibration_reliability_{dataset}.pdf` — reliability curves
- `figures/calibration_hh_switch_pca_{dataset}.pdf` — PCA of HH movers

## Parameters

- K = 25, k_nn = 30, epsilon = 0.05
- Calibration method: Platt scaling on validation set

## Key functions called

- `run_calibration_experiment`, `calibrate_predictions_for_run`
- `calibration_curve` (sklearn)
- `pointwise_variance`, `spatial_analysis` — recomputed on calibrated P

## Core objects (shapes)

- `P_test`: (n_models, n_test) → `P_calibrated`: (n_models, n_test)
- `per_run_df`: before/after metrics per run
- `jaccard_HH_before_after`: scalar per run — Jaccard of HH sets

## Main results (numbers)

- Δ Moran's I is small for COMPAS and German (< 0.02 on average)
- Jaccard of HH before/after calibration is high (> 0.8 for COMPAS)
- Reliability curves improve after Platt scaling, but spatial metrics barely change
- Breast Cancer shows larger relative Δ but from a low baseline

## One-liner interpretation

Spatial multiplicity is structural, not an artifact of miscalibration: Platt scaling improves reliability but barely changes Moran's I, HH counts, or HH locations.

## Open questions / TODO

- Try isotonic regression as an alternative calibration method
- Report ECE (expected calibration error) before/after
