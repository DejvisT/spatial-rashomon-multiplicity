"""
Calibration robustness check: test whether spatial multiplicity is driven by
probability miscalibration. Uses only saved artifacts; does not retrain or
redefine Rashomon selection.

Platt scaling (LogisticRegression on predicted probabilities) fitted on
validation set only; applied to test predictions. Recomputes multiplicity and
spatial metrics on calibrated P_test and compares to uncalibrated.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from analysis.run_analysis import (
    load_meta,
    load_P_test,
    load_P_val,
    load_split,
    select_rashomon_global,
    mean_variance,
    pointwise_variance,
    pointwise_conflict,
    ambiguity,
    disagreement_rate,
    discrepancy,
    spatial_analysis,
)
from analysis.spatial import extract_hh_components
from analysis.preprocessing import get_transformed_test_features

PathLike = Union[str, Path]


def _get_run_dirs(dataset_dir: PathLike) -> List[Path]:
    """List run directories (seed=*) under dataset_dir, sorted by seed."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []
    run_dirs = []
    for p in dataset_dir.iterdir():
        if p.is_dir() and p.name.startswith("seed="):
            try:
                seed_val = int(p.name.split("=")[1])
                run_dirs.append((seed_val, p))
            except (IndexError, ValueError):
                continue
    run_dirs.sort(key=lambda x: x[0])
    return [p for _, p in run_dirs]


def _get_validation_labels(dataset_name: str, run_dir: PathLike) -> np.ndarray:
    """Load dataset and return y for validation indices of this run."""
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from data import load_dataset  # noqa: E402
    X, y, _ = load_dataset(dataset_name)
    split = load_split(run_dir)
    return np.asarray(y.iloc[split["val"]].values, dtype=np.float64)


def _get_test_labels(dataset_name: str, run_dir: PathLike) -> np.ndarray:
    """Load dataset and return y for test indices of this run."""
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from data import load_dataset  # noqa: E402
    X, y, _ = load_dataset(dataset_name)
    split = load_split(run_dir)
    return np.asarray(y.iloc[split["test"]].values, dtype=np.float64)


def platt_scale(
    p_val: np.ndarray,
    y_val: np.ndarray,
    p_test: np.ndarray,
) -> np.ndarray:
    """
    Fit Platt scaling (LogisticRegression on predicted probabilities) on
    (p_val, y_val) and apply to p_test. Returns calibrated probabilities.
    """
    if np.unique(y_val).size < 2:
        return p_test.copy()
    X_val = np.asarray(p_val, dtype=np.float64).reshape(-1, 1)
    X_test = np.asarray(p_test, dtype=np.float64).reshape(-1, 1)
    clf = LogisticRegression(C=1e10, max_iter=1000, random_state=42)
    clf.fit(X_val, y_val)
    return clf.predict_proba(X_test)[:, 1].ravel()


def calibrate_predictions_for_run(
    run_dir: PathLike,
    dataset_name: str,
    K: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one run: select Rashomon (top-K by val Brier), fit Platt scaling per
    model on validation predictions/labels, apply to test predictions.

    Uses only: meta.csv, P_val.npy, P_test.npy, split.npz; and dataset for y_val.

    Returns
    -------
    P_test_sel : (K, n_test) uncalibrated test predictions (Rashomon subset)
    P_test_calibrated : (K, n_test) calibrated test predictions
    """
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    P_val = load_P_val(run_dir)
    P_test = load_P_test(run_dir)
    n_cand = len(meta)
    K_actual = min(K, n_cand)
    idx = select_rashomon_global(run_dir, K=K_actual)
    P_val_sel = P_val[idx]   # (K, n_val)
    P_test_sel = P_test[idx]  # (K, n_test)
    y_val = _get_validation_labels(dataset_name, run_dir)
    n_val = len(y_val)
    n_test = P_test_sel.shape[1]
    if P_val_sel.shape[1] != n_val:
        raise ValueError(f"P_val columns {P_val_sel.shape[1]} != len(y_val) {n_val}")
    P_cal = np.empty_like(P_test_sel)
    for m in range(P_val_sel.shape[0]):
        P_cal[m] = platt_scale(P_val_sel[m], y_val, P_test_sel[m])
    return P_test_sel, P_cal


def _jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean arrays (same length)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 1.0


def run_calibration_experiment(
    dataset_dir: PathLike,
    dataset_name: Optional[str] = None,
    K: int = 25,
    k_nn: int = 30,
    epsilon: float = 0.05,
    permutations: int = 999,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run calibration for all outer runs in dataset_dir. Recompute multiplicity
    and spatial metrics on calibrated predictions; compare to uncalibrated.
    Save per-run and aggregated summaries to CSV.

    Returns
    -------
    per_run_df : per-run diagnostics (before/after, deltas, Jaccard)
    summary_df : aggregated mean ± std and mean Jaccard
    """
    dataset_dir = Path(dataset_dir)
    if dataset_name is None:
        dataset_name = dataset_dir.name
    run_dirs = _get_run_dirs(dataset_dir)
    if not run_dirs:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for run_dir in run_dirs:
        P_sel, P_cal = calibrate_predictions_for_run(run_dir, dataset_name, K=K)
        X_test = get_transformed_test_features(run_dir, dataset_name)
        n_test = P_sel.shape[1]
        if X_test.shape[0] != n_test:
            raise ValueError(f"X_test rows {X_test.shape[0]} != n_test {n_test}")

        # Uncalibrated metrics (Rashomon subset)
        v_before = pointwise_variance(P_sel, ddof=0)
        c_before = pointwise_conflict(P_sel)
        mult_before = {
            "mean_variance": mean_variance(P_sel, ddof=0),
            "mean_conflict": float(np.mean(c_before)),
            "ambiguity": ambiguity(P_sel),
            "disagreement_rate": disagreement_rate(P_sel, epsilon=epsilon),
            "discrepancy": discrepancy(P_sel),
        }
        spatial_before = spatial_analysis(
            v_before, X_test, k=k_nn, permutations=permutations,
            fdr_alpha=fdr_alpha, seed=seed,
        )
        HH_before = spatial_before["HH_mask"]
        n_hh_before = int(np.sum(HH_before))
        lisa_df_before = pd.DataFrame({"cluster": np.where(HH_before, "HH", "NS")})
        W = spatial_before["W"].to_sparse() if hasattr(spatial_before["W"], "to_sparse") else spatial_before["W"].sparse
        _, comp_before = extract_hh_components(lisa_df_before, W, min_size=5)
        comp_sizes_before = sorted((len(inds) for inds in comp_before.values()), reverse=True) if comp_before else []

        # Calibrated metrics
        v_after = pointwise_variance(P_cal, ddof=0)
        c_after = pointwise_conflict(P_cal)
        mult_after = {
            "mean_variance": mean_variance(P_cal, ddof=0),
            "mean_conflict": float(np.mean(c_after)),
            "ambiguity": ambiguity(P_cal),
            "disagreement_rate": disagreement_rate(P_cal, epsilon=epsilon),
            "discrepancy": discrepancy(P_cal),
        }
        spatial_after = spatial_analysis(
            v_after, X_test, k=k_nn, permutations=permutations,
            fdr_alpha=fdr_alpha, seed=seed,
        )
        HH_after = spatial_after["HH_mask"]
        n_hh_after = int(np.sum(HH_after))
        lisa_df_after = pd.DataFrame({"cluster": np.where(HH_after, "HH", "NS")})
        _, comp_after = extract_hh_components(lisa_df_after, W, min_size=5)
        comp_sizes_after = sorted((len(inds) for inds in comp_after.values()), reverse=True) if comp_after else []

        jaccard_hh = _jaccard_bool(HH_before, HH_after)

        y_test = _get_test_labels(dataset_name, run_dir)
        p_before = P_sel.mean(axis=0)
        p_after = P_cal.mean(axis=0)
        brier_before = float(np.mean((p_before - y_test) ** 2))
        brier_after = float(np.mean((p_after - y_test) ** 2))
        brier_improvement = brier_before - brier_after  # positive = calibration improved

        rows.append({
            "run": run_dir.name,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "brier_improvement": brier_improvement,
            "mean_variance_before": mult_before["mean_variance"],
            "mean_variance_after": mult_after["mean_variance"],
            "delta_mean_variance": mult_after["mean_variance"] - mult_before["mean_variance"],
            "mean_conflict_before": mult_before["mean_conflict"],
            "mean_conflict_after": mult_after["mean_conflict"],
            "delta_mean_conflict": mult_after["mean_conflict"] - mult_before["mean_conflict"],
            "moran_i_before": spatial_before["moran_i"],
            "moran_i_after": spatial_after["moran_i"],
            "delta_moran_i": spatial_after["moran_i"] - spatial_before["moran_i"],
            "n_HH_before": n_hh_before,
            "n_HH_after": n_hh_after,
            "delta_n_HH": n_hh_after - n_hh_before,
            "ambiguity_before": mult_before["ambiguity"],
            "ambiguity_after": mult_after["ambiguity"],
            "disagreement_rate_before": mult_before["disagreement_rate"],
            "disagreement_rate_after": mult_after["disagreement_rate"],
            "discrepancy_before": mult_before["discrepancy"],
            "discrepancy_after": mult_after["discrepancy"],
            "jaccard_HH_before_after": jaccard_hh,
            "n_components_before": len(comp_before),
            "n_components_after": len(comp_after),
            "max_component_size_before": comp_sizes_before[0] if comp_sizes_before else 0,
            "max_component_size_after": comp_sizes_after[0] if comp_sizes_after else 0,
        })

    per_run_df = pd.DataFrame(rows)

    # Aggregated summary (mean ± std over runs)
    agg = []
    for col in ["delta_mean_variance", "delta_moran_i", "delta_n_HH", "jaccard_HH_before_after", "brier_improvement"]:
        mean_val = per_run_df[col].mean()
        std_val = per_run_df[col].std()
        std_val = std_val if pd.notna(std_val) else 0.0
        agg.append({
            "metric": col,
            "mean": mean_val,
            "std": std_val,
            "mean_plus_minus_std": f"{mean_val:.4f} ± {std_val:.4f}",
        })
    summary_df = pd.DataFrame(agg)

    # Save CSVs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    per_run_df.to_csv(dataset_dir / "calibration_summary_per_run.csv", index=False)
    summary_df.to_csv(dataset_dir / "calibration_summary.csv", index=False)

    return per_run_df, summary_df
