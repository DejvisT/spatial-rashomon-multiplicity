from typing import Dict, List
import numpy as np
import pandas as pd

from analysis.spatial import build_knn_graph, moran_global, lisa_local
from src.metrics import prediction_variance


# ---------------------------------------------------------------------
# Model-wise permutation of predictions
# ---------------------------------------------------------------------

def permute_predictions(
    P: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Permute predictions model-wise (row-wise).

    Parameters
    ----------
    P : array of shape (n_models, n_obs)
    seed : random seed

    Returns
    -------
    P_perm : array of same shape as P
    """
    rng = np.random.RandomState(seed)
    P_perm = np.empty_like(P)

    for m in range(P.shape[0]):
        P_perm[m] = rng.permutation(P[m])

    return P_perm


# ---------------------------------------------------------------------
# Run a single null experiment
# ---------------------------------------------------------------------

def run_null_experiment(
    P: np.ndarray,
    X_knn,
    *,
    k: int = 10,
    permutations: int = 999,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Run a single null experiment:
    - permute predictions
    - recompute variance
    - recompute Moran's I and LISA
    """

    # Permute predictions
    P_perm = permute_predictions(P, seed=seed)

    # Recompute variance
    v_perm = prediction_variance(P_perm)

    # Spatial graph
    W = build_knn_graph(X_knn, k=k)

    # Global Moran
    moran_res = moran_global(v_perm, W, permutations=permutations, seed=seed)

    # Local Moran
    lisa_df = lisa_local(v_perm, W, permutations=permutations, seed=seed)

    return {
        "v_perm": v_perm,
        "moran": moran_res,
        "lisa": lisa_df,
    }


# ---------------------------------------------------------------------
# Multiple null runs
# ---------------------------------------------------------------------

def run_null_experiments(
    P: np.ndarray,
    X_knn,
    *,
    n_runs: int = 50,
    k: int = 10,
    permutations: int = 999,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Run multiple null experiments and collect Moran's I statistics.

    Returns
    -------
    DataFrame with columns: run, I, p_value
    """
    records = []

    for r in range(n_runs):
        print(f"Running null experiment {r+1} of {n_runs}")
        seed = base_seed + r
        res = run_null_experiment(
            P,
            X_knn,
            k=k,
            permutations=permutations,
            seed=seed,
        )
        records.append({
            "run": r,
            "I": res["moran"]["I"],
            "p_value": res["moran"]["p_value"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Null experiments with HH count (for comparison tables)
# ---------------------------------------------------------------------

def run_null_experiments_with_hh(
    P: np.ndarray,
    X_knn,
    *,
    n_runs: int = 50,
    k: int = 10,
    permutations: int = 999,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Run multiple null experiments and collect Moran's I and HH count.

    Returns
    -------
    DataFrame with columns: run, I, p_value, n_hh
    """
    records = []

    for r in range(n_runs):
        print(f"  Null run {r+1}/{n_runs}")
        seed = base_seed + r
        res = run_null_experiment(
            P,
            X_knn,
            k=k,
            permutations=permutations,
            seed=seed,
        )
        n_hh = (res["lisa"]["cluster"] == "HH").sum()
        records.append({
            "run": r,
            "I": res["moran"]["I"],
            "p_value": res["moran"]["p_value"],
            "n_hh": int(n_hh),
        })

    return pd.DataFrame(records)
