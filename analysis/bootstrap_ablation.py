"""
Bootstrap vs fixed-train ablation: compare disagreement/hotspots with and
without bootstrap-resampled training sets.

This isolates whether multiplicity comes from hyperparameter diversity alone
(fixed train) or also from data resampling (bootstrap).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def run_bootstrap_ablation(
    run_dir_fixed: PathLike,
    run_dir_bootstrap: PathLike,
    dataset_name: str,
    K: int = 25,
    k_nn: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare spatial multiplicity between fixed-train and bootstrap-train runs.
    
    Parameters
    ----------
    run_dir_fixed : path to a run with fixed training set
    run_dir_bootstrap : path to a run with bootstrap-resampled training set
    dataset_name : name of the dataset
    K : Rashomon set size
    k_nn : kNN graph neighborhood size
    seed : random seed for spatial analysis
    
    Returns
    -------
    dict with before/after metrics for comparison
    """
    from analysis.run_analysis import (
        load_P_test, select_rashomon_global,
        pointwise_variance, mean_variance, spatial_analysis,
    )
    from analysis.preprocessing import get_transformed_test_features
    
    results = {}
    
    for label, run_dir in [("fixed", run_dir_fixed), ("bootstrap", run_dir_bootstrap)]:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            results[label] = {"error": f"run_dir not found: {run_dir}"}
            continue
        
        P_test = load_P_test(run_dir)
        idx = select_rashomon_global(run_dir, K=K)
        P_sel = P_test[idx]
        
        X_test = get_transformed_test_features(run_dir, dataset_name)
        v = pointwise_variance(P_sel, ddof=0)
        
        spatial_res = spatial_analysis(v, X_test, k=k_nn, seed=seed)
        
        results[label] = {
            "mean_variance": mean_variance(P_sel, ddof=0),
            "moran_i": spatial_res["moran_i"],
            "n_hh": int(np.sum(spatial_res["HH_mask"])),
            "n_ll": int(np.sum(spatial_res["LL_mask"])),
        }
    
    return results


def run_bootstrap_ablation_multi_seed(
    results_dir_fixed: PathLike,
    results_dir_bootstrap: PathLike,
    dataset_name: str,
    K: int = 25,
    k_nn: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run bootstrap ablation across all seeds and return a comparison DataFrame.
    """
    results_dir_fixed = Path(results_dir_fixed)
    results_dir_bootstrap = Path(results_dir_bootstrap)
    
    rows = []
    for fixed_dir in sorted(results_dir_fixed.glob("seed=*")):
        seed_val = fixed_dir.name
        bootstrap_dir = results_dir_bootstrap / seed_val
        
        if not bootstrap_dir.exists():
            continue
        
        res = run_bootstrap_ablation(
            fixed_dir, bootstrap_dir, dataset_name,
            K=K, k_nn=k_nn, seed=seed,
        )
        
        for condition in ["fixed", "bootstrap"]:
            if "error" in res.get(condition, {}):
                continue
            row = {"seed": seed_val, "condition": condition}
            row.update(res[condition])
            rows.append(row)
    
    return pd.DataFrame(rows)
