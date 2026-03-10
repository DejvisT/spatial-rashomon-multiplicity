# 08 — Synthetic multiplicity

**Key idea:** Generate synthetic datasets with known ambiguity islands, train Rashomon-set models, and validate that the LISA HH pipeline recovers the planted ground-truth regions with high precision and recall.

## Loads

- No pre-existing results; generates synthetic datasets and trains models inside the notebook
- `make_synthetic_multiplicity_dataset`, `make_synth_three_islands_plus_outliers`, `make_synth_graduated_ambiguity` from `src.synthetic_data`
- `run_one_training_run`, `save_run_artifacts` from `src.training_pipeline`

## Produces

- Trained models and artifacts in `results/synthetic/seed=42/`, `results/synthetic_three_islands/seed=42/`, `results/synthetic_graduated/seed=42/`
- Variance scatter plots, HH overlays, FDR α sensitivity curves
- Null experiment histograms (observed vs permuted Moran's I)
- Decision tree explanations of HH vs non-HH

## Parameters

- SEED = 42, K = 25, K_NN = 30, epsilon = 0.05, R_null = 100
- **Dataset A (single island):** n_samples=3000, p_island=0.2, island_delta=0.30, island_radius=2.0
- **Dataset B (three islands):** n_samples=5000, p_islands=0.30, island_delta=0.30, n_outliers=100
- **Dataset C (graduated ambiguity):** n_samples=5000, strong delta=0.35, moderate delta=0.20
- FDR α values: {0.01, 0.05, 0.10, 0.20}

## Key functions called

- `make_synthetic_multiplicity_dataset`, `make_synth_three_islands_plus_outliers`, `make_synth_graduated_ambiguity`
- `run_one_training_run`, `save_run_artifacts`
- `select_rashomon_global`, `pointwise_variance`, `run_spatial`, `run_null`
- `compute_multiplicity_metrics`
- `DecisionTreeClassifier(max_depth=5)` — explainability of HH regions

## Core objects (shapes)

- `X`: (n_samples, 2) — synthetic features
- `y`: (n_samples,) — binary labels
- `gt`: SyntheticGroundTruth with `island_mask`, `stable_mask`, `outlier_mask`
- `P_test`: (n_models, n_test)
- `v` (pointwise variance): (n_test,)
- `HH_mask`: (n_test,) boolean

## Main results (numbers)

- **Single island:** Precision ~0.97, Recall ~0.80, Jaccard ~0.78 (TP=96, |HH|=99, |island|=120)
- **Three islands:** TP=260, |HH|=268, |island|=297 — high recovery
- **Graduated ambiguity:** strong island better recovered than moderate; outliers stay non-HH
- Null experiment: Moran's I significantly above permuted null (p < 0.05)

## One-liner interpretation

The LISA HH pipeline accurately recovers planted ambiguity islands in synthetic data with high precision, validating that it localizes genuine multiplicity rather than noise.

## Open questions / TODO

- Vary island size and delta to map out precision/recall trade-off curves
- Test with higher-dimensional synthetic features
