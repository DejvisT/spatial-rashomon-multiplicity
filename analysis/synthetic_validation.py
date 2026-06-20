"""Synthetic validation helpers for LISA hotspot recovery on synthetic datasets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from analysis.run_analysis import spatial_analysis
from analysis.spatial import extract_hh_components


def _compute_fdr_sensitivity(
    v: np.ndarray,
    X_test: np.ndarray,
    X_test_2d: np.ndarray,
    island_test: np.ndarray,
    fdr_alphas: Sequence[float],
    *,
    k_nn: int,
    seed: int,
    min_component_size: int = 5,
    tree_max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> pd.DataFrame:
    """Run FDR alpha sensitivity for synthetic island recovery."""
    rows = []
    island_test = np.asarray(island_test, dtype=bool)

    for alpha in fdr_alphas:
        res = spatial_analysis(v, X_test, k=k_nn, fdr_alpha=alpha, seed=seed)
        HH_a = res["HH_mask"]
        n_hh = int(HH_a.sum())
        lisa_df = pd.DataFrame({"cluster": np.where(HH_a, "HH", "NS")})
        W_sparse = res["W"].to_sparse() if hasattr(res["W"], "to_sparse") else res["W"].sparse
        _, comps = extract_hh_components(lisa_df, W_sparse, min_size=min_component_size)
        n_comp = len(comps)
        max_size = max(len(inds) for inds in comps.values()) if comps else 0
        tp_gt = (HH_a & island_test).sum()
        pred_pos = HH_a.sum()
        true_pos = island_test.sum()
        precision_gt = tp_gt / pred_pos if pred_pos else np.nan
        recall_gt = tp_gt / true_pos if true_pos else np.nan
        union = (HH_a | island_test).sum()
        jaccard = tp_gt / union if union else np.nan
        dt = DecisionTreeClassifier(
            max_depth=tree_max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        dt.fit(X_test_2d, HH_a.astype(int))
        tree_pred = dt.predict(X_test_2d).astype(bool)
        dt_tp = (tree_pred & island_test).sum()
        dt_prec = dt_tp / tree_pred.sum() if tree_pred.sum() else np.nan
        dt_rec = dt_tp / true_pos if true_pos else np.nan
        rows.append({
            "α": alpha,
            "#HH": n_hh,
            "#components": n_comp,
            "max_component_size": max_size,
            "Jaccard": jaccard,
            "Precision": precision_gt,
            "Recall": recall_gt,
            "DT Precision": dt_prec,
            "DT Recall": dt_rec,
        })

    return pd.DataFrame(rows)


def compute_single_island_fdr_sensitivity(
    v: np.ndarray,
    X_test: np.ndarray,
    X_test_2d: np.ndarray,
    island_test: np.ndarray,
    fdr_alphas: Sequence[float],
    *,
    k_nn: int,
    seed: int,
    min_component_size: int = 5,
    tree_max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> pd.DataFrame:
    """FDR alpha sensitivity for the single-island synthetic design."""
    return _compute_fdr_sensitivity(
        v,
        X_test,
        X_test_2d,
        island_test,
        fdr_alphas,
        k_nn=k_nn,
        seed=seed,
        min_component_size=min_component_size,
        tree_max_depth=tree_max_depth,
        min_samples_leaf=min_samples_leaf,
    )


def compute_three_islands_fdr_sensitivity(
    v: np.ndarray,
    X_test: np.ndarray,
    X_test_2d: np.ndarray,
    island_test: np.ndarray,
    fdr_alphas: Sequence[float],
    *,
    k_nn: int,
    seed: int,
    min_component_size: int = 5,
    tree_max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> pd.DataFrame:
    """FDR alpha sensitivity for the three-islands synthetic design."""
    return _compute_fdr_sensitivity(
        v,
        X_test,
        X_test_2d,
        island_test,
        fdr_alphas,
        k_nn=k_nn,
        seed=seed,
        min_component_size=min_component_size,
        tree_max_depth=tree_max_depth,
        min_samples_leaf=min_samples_leaf,
    )


def compute_structural_exceptions_fdr_sensitivity(
    v: np.ndarray,
    X_test: np.ndarray,
    X_test_2d: np.ndarray,
    island_test: np.ndarray,
    fdr_alphas: Sequence[float],
    *,
    k_nn: int,
    seed: int,
    min_component_size: int = 5,
    tree_max_depth: int = 5,
    min_samples_leaf: int = 10,
) -> pd.DataFrame:
    """FDR alpha sensitivity for the structural-exception synthetic design."""
    return _compute_fdr_sensitivity(
        v,
        X_test,
        X_test_2d,
        island_test,
        fdr_alphas,
        k_nn=k_nn,
        seed=seed,
        min_component_size=min_component_size,
        tree_max_depth=tree_max_depth,
        min_samples_leaf=min_samples_leaf,
    )
