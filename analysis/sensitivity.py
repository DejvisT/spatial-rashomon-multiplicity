from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Sequence
from typing import Iterable, Mapping, Union

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components as sparse_connected_components

from analysis.run_analysis import load_meta, run_multiplicity, run_spatial
from analysis.preprocessing import get_transformed_test_features
from analysis.io_utils import get_run_dirs, PathLike


def run_one_k_sensitivity(
    run_dir: PathLike,
    dataset_name: str,
    K: int,
    k_nn: int,
) -> dict | None:
    """Compute multiplicity and spatial metrics for one run and one Rashomon-set size.

    Returns None when K > n_candidates so callers can skip.
    """
    run_dir = Path(run_dir)
    n_cand = len(load_meta(run_dir))

    if K > n_cand:
        return None

    K_actual = min(K, n_cand)
    X_test = get_transformed_test_features(run_dir, dataset_name)

    mult = run_multiplicity(run_dir, K=K_actual)
    spatial = run_spatial(run_dir, X_test, K=K_actual, k=k_nn)

    return {
        "mean_variance": mult["mean_variance"],
        "mean_conflict": mult["mean_conflict"],
        "moran_i": spatial["moran_i"],
        "n_hh": int(np.sum(spatial["HH_mask"])),
        "conflict_moran_i": spatial.get("conflict_moran_i", np.nan),
        "conflict_n_hh": spatial.get("conflict_n_hh", 0),
        "n_candidates": n_cand,
        "K_actual": K_actual,
    }


def compute_k_sensitivity(
    results_dir: PathLike,
    *,
    datasets: Iterable[str],
    K_list: Iterable[int],
    k_nn_by_dataset: Mapping[str, int],
) -> pd.DataFrame:
    """Compute K-sensitivity results across datasets, seeds, and K values."""
    results_dir = Path(results_dir)
    rows: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        dataset_dir = results_dir / dataset_name
        if not dataset_dir.is_dir():
            continue

        run_dirs = get_run_dirs(dataset_dir)
        if not run_dirs:
            continue

        k_nn = k_nn_by_dataset[dataset_name]

        for K in K_list:
            for run_dir in run_dirs:
                res = run_one_k_sensitivity(run_dir, dataset_name, K, k_nn)
                if res is None:
                    continue

                rows.append({
                    "dataset": dataset_name,
                    "seed": run_dir.name,
                    "K": K,
                    **res,
                })

    return pd.DataFrame(rows)


def run_one_knn_sensitivity(run_dir: Path, dataset_name: str, K: int, k_nn: int) -> Dict[str, Any]:
    """Compute kNN-sensitivity metrics for one run directory.

    Returns a dict with the same keys/columns as the original notebook.
    """
    n_cand = len(load_meta(run_dir))
    K_actual = min(K, n_cand)
    mult = run_multiplicity(run_dir, K=K_actual)
    X_test = get_transformed_test_features(run_dir, dataset_name)
    spatial = run_spatial(run_dir, X_test, K=K_actual, k=k_nn)

    W = spatial["W"]
    W_sparse = W.to_sparse() if hasattr(W, "to_sparse") else W.sparse
    W_sym = W_sparse + W_sparse.T
    W_sym = (W_sym > 0).astype(int)
    n_comp, comp_labels = sparse_connected_components(W_sym, directed=False)
    comp_sizes = np.bincount(comp_labels)
    largest_frac = int(comp_sizes.max()) / W_sym.shape[0]

    return {
        "mean_variance": mult["mean_variance"],
        "mean_conflict": mult["mean_conflict"],
        "moran_i": spatial["moran_i"],
        "n_hh": int(np.sum(spatial["HH_mask"])),
        "conflict_moran_i": spatial.get("conflict_moran_i", np.nan),
        "conflict_n_hh": spatial.get("conflict_n_hh", 0),
        "n_components": int(n_comp),
        "largest_component_frac": float(largest_frac),
    }


def compute_knn_sensitivity(K: int, K_NN_LIST: Sequence[int], dataset_dirs: Sequence[Path]) -> pd.DataFrame:
    """Compute sensitivity table across datasets, seeds, and k_nn values.

    Matches the output and column names used in notebooks/05_sensitivity_kNN.ipynb.
    """
    results_knn: List[Dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        run_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed=")]
        run_dirs.sort(key=lambda p: int(p.name.split("=")[1]) if "=" in p.name else 0)
        for k_nn_val in K_NN_LIST:
            for run_dir in run_dirs:
                res = run_one_knn_sensitivity(run_dir, dataset_name, K, k_nn_val)
                results_knn.append({
                    "dataset": dataset_name,
                    "seed": run_dir.name,
                    "k_nn": int(k_nn_val),
                    **res,
                })
    return pd.DataFrame(results_knn)


def run_one_knn_masks(run_dir: Path, X_test: np.ndarray, k_nn: int) -> np.ndarray:
    """Return boolean HH mask array for `run_dir` given precomputed `X_test` and `k_nn`.

    Matches the notebook's `run_one_knn_masks` signature.
    """
    n_cand = len(load_meta(run_dir))
    # Use module-level override if set by compute_knn_hh_overlay, otherwise default to 25
    K_val = globals().get("_HH_OVERLAY_K", 25)
    K_actual = min(K_val, n_cand)
    spatial = run_spatial(run_dir, X_test, K=K_actual, k=k_nn)
    hh_mask = np.asarray(spatial["HH_mask"], dtype=bool)
    return hh_mask


def compute_knn_hh_overlay(K: int, K_NN_LIST: Sequence[int], dataset_dirs: Sequence[Path]) -> pd.DataFrame:
    """Compute HH-overlay table across datasets, seeds, and k_nn values.

    Returns DataFrame with columns: dataset, seed, k_nn, point_idx, is_hh.
    """
    # expose K to run_one_knn_masks via module-level variable so the simple
    # run_one_knn_masks signature matches the notebook's expectations
    globals()["_HH_OVERLAY_K"] = K
    rows: List[Dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        for k_nn_val in K_NN_LIST:
            for run_dir in get_run_dirs(dataset_dir):
                X_test = get_transformed_test_features(run_dir, dataset_name)
                hh_mask = run_one_knn_masks(run_dir, X_test, k_nn_val)
                for point_idx, is_hh in enumerate(hh_mask):
                    rows.append({
                        "dataset": dataset_name,
                        "seed": run_dir.name,
                        "k_nn": int(k_nn_val),
                        "point_idx": int(point_idx),
                        "is_hh": bool(is_hh),
                    })
    # cleanup module-level override
    if "_HH_OVERLAY_K" in globals():
        del globals()["_HH_OVERLAY_K"]
    return pd.DataFrame(rows)