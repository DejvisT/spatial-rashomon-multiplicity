"""Fairness and protected-group analysis functions for Notebook 10."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from libpysal.weights import KNN as PySAL_KNN
from esda.moran import Moran, Moran_Local

from analysis.run_analysis import (
    load_P_test,
    load_split,
    select_rashomon_global,
    pointwise_variance,
    spatial_analysis,
    fdr_benjamini_hochberg,
)
from analysis.preprocessing import get_transformed_test_features
from analysis.knn_defaults import K_NN_BY_DATASET
from src.data import load_dataset, make_preprocessor


def _seed_group_rates(hh_mask: np.ndarray, group_vals: np.ndarray, min_group_n: int) -> Dict[Any, float]:
    """Compute HH rate for each group in a seed."""
    rates = {}
    groups, counts = np.unique(group_vals, return_counts=True)
    for g, n in zip(groups, counts):
        if n >= min_group_n:
            g_mask = group_vals == g
            rates[g] = float(hh_mask[g_mask].mean())
    return rates


def _aggregate_group_means(
    per_seed_rates: Dict[int, Dict[Any, float]], min_seeds_per_group: int
) -> Dict[Any, float]:
    """Aggregate group means across seeds."""
    group_to_vals = {}
    for rates in per_seed_rates.values():
        for g, v in rates.items():
            group_to_vals.setdefault(g, []).append(v)
    return {
        g: float(np.mean(vs))
        for g, vs in group_to_vals.items()
        if len(vs) >= min_seeds_per_group
    }


def compute_fairness_and_perm(
    results_dir: Path,
    datasets: List[str],
    seeds: List[int],
    K: int,
    cache_dir: Path,
    cache_version: str,
    min_group_n: int = 30,
    min_seeds_per_group: int = 5,
    n_perm: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fairness analysis and permutation test for COMPAS protected attributes."""
    X_raw, y_raw, _ = load_dataset("compas")
    
    fairness_rows = []
    fairness_cache = {}
    for seed in seeds:
        run_dir = results_dir / "compas" / f"seed={seed}"
        try:
            split = load_split(run_dir)
            test_idx = split["test"]
            X_test_raw_i = X_raw.iloc[test_idx]
            P_test = load_P_test(run_dir)
            idx = select_rashomon_global(run_dir, K=K)
            P_sel = P_test[idx]
            v = pointwise_variance(P_sel)
            X_test_proc = get_transformed_test_features(run_dir, "compas")
            sp = spatial_analysis(v, X_test_proc, k=K_NN_BY_DATASET["compas"], permutations=999)
            hh_mask = sp["HH_mask"]
            fairness_cache[seed] = (hh_mask, X_test_raw_i)
            for group_col in ["race", "sex"]:
                group_vals = X_test_raw_i[group_col].values
                for g in np.unique(group_vals):
                    g_mask = group_vals == g
                    n_group = int(g_mask.sum())
                    fairness_rows.append({
                        "seed": seed,
                        "group_col": group_col,
                        "group_val": g,
                        "n_group": n_group,
                        "n_test": len(v),
                        "eligible_sig": n_group >= min_group_n,
                        "hh_rate": hh_mask[g_mask].mean() if n_group > 0 else np.nan,
                        "mean_variance": v[g_mask].mean() if n_group > 0 else np.nan,
                    })
        except Exception as e:
            print(f"  seed={seed}: SKIP ({e})")

    perm_rows = []
    rng = np.random.RandomState(42)
    for group_col in ["race", "sex"]:
        per_seed_rates = {}
        for seed in seeds:
            if seed not in fairness_cache:
                continue
            hh_mask, X_test_raw_i = fairness_cache[seed]
            group_vals = X_test_raw_i[group_col].values
            per_seed_rates[seed] = _seed_group_rates(hh_mask, group_vals, min_group_n)
        obs_group_means = _aggregate_group_means(per_seed_rates, min_seeds_per_group)
        if len(obs_group_means) < 2:
            print(f"  {group_col}: SKIP (fewer than 2 eligible groups)")
            continue

        obs_range = max(obs_group_means.values()) - min(obs_group_means.values())
        null_ranges = []
        for _ in range(n_perm):
            perm_seed_rates = {}
            for seed in per_seed_rates.keys():
                hh_mask, X_test_raw_i = fairness_cache[seed]
                shuffled_vals = rng.permutation(X_test_raw_i[group_col].values)
                perm_seed_rates[seed] = _seed_group_rates(hh_mask, shuffled_vals, min_group_n)
            perm_group_means = _aggregate_group_means(perm_seed_rates, min_seeds_per_group)
            if len(perm_group_means) >= 2:
                null_ranges.append(max(perm_group_means.values()) - min(perm_group_means.values()))
        null_ranges = np.asarray(null_ranges, dtype=float)
        p_val = (1 + (null_ranges >= obs_range).sum()) / (len(null_ranges) + 1)
        perm_rows.append({
            "group_col": group_col,
            "obs_range_stratified": float(obs_range),
            "null_range_mean": float(null_ranges.mean()) if len(null_ranges) else np.nan,
            "null_range_std": float(null_ranges.std()) if len(null_ranges) else np.nan,
            "null_ci_low": float(np.quantile(null_ranges, 0.025)) if len(null_ranges) else np.nan,
            "null_ci_high": float(np.quantile(null_ranges, 0.975)) if len(null_ranges) else np.nan,
            "p_value_stratified": float(p_val),
            "n_perm_valid": int(len(null_ranges)),
            "n_groups_eligible": int(len(obs_group_means)),
            "eligible_groups": ", ".join(sorted(obs_group_means.keys())),
            "min_group_n": int(min_group_n),
            "min_seeds_per_group": int(min_seeds_per_group),
            "mean_p": float(p_val),
            "obs_range_mean": float(obs_range),
        })

    fair = pd.DataFrame(fairness_rows)
    perm = pd.DataFrame(perm_rows)
    fair.to_parquet(cache_dir / f"nb10_fairness_{cache_version}.parquet", index=False)
    perm.to_parquet(cache_dir / f"nb10_fairness_perm_{cache_version}.parquet", index=False)
    return fair, perm


def load_fairness_and_perm(
    results_dir: Path,
    datasets: List[str],
    seeds: List[int],
    K: int,
    cache_dir: Path,
    cache_version: str,
    force_recompute: bool = False,
    min_group_n: int = 30,
    min_seeds_per_group: int = 5,
    n_perm: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load or compute fairness and permutation test results."""
    f_path = cache_dir / f"nb10_fairness_{cache_version}.parquet"
    p_path = cache_dir / f"nb10_fairness_perm_{cache_version}.parquet"
    if not force_recompute and f_path.is_file() and p_path.is_file():
        return pd.read_parquet(f_path), pd.read_parquet(p_path)
    return compute_fairness_and_perm(
        results_dir, datasets, seeds, K, cache_dir, cache_version,
        min_group_n, min_seeds_per_group, n_perm
    )


def _get_non_protected_cols_for_run(run_dir: Path) -> Tuple[List[int], List[str]]:
    """Return preprocessed column indices that are NOT race/sex for this run."""
    X_compas, y_compas, fi_compas = load_dataset("compas")
    split = load_split(run_dir)
    pre = make_preprocessor(fi_compas, scale_numeric=True)
    pre.fit(X_compas.iloc[split["train"]], y_compas.iloc[split["train"]])
    feat_names = list(pre.get_feature_names_out())
    keep = [
        i for i, n in enumerate(feat_names)
        if ("cat__sex_" not in n and "cat__race_" not in n)
    ]
    return keep, feat_names


def compute_excl_protected(
    results_dir: Path,
    datasets: List[str],
    seeds: List[int],
    K: int,
    cache_dir: Path,
    cache_version: str,
) -> pd.DataFrame:
    """Compute spatial analysis excluding race/sex features (COMPAS)."""
    excl_rows = []
    for seed in seeds:
        run_dir = results_dir / "compas" / f"seed={seed}"
        try:
            P_test = load_P_test(run_dir)
            idx = select_rashomon_global(run_dir, K=K)
            P_sel = P_test[idx]
            v = pointwise_variance(P_sel)
            X_test_full = get_transformed_test_features(run_dir, "compas")
            if hasattr(X_test_full, "toarray"):
                X_test_full = X_test_full.toarray()
            X_test_full = np.asarray(X_test_full, dtype=float)
            non_prot_cols, feat_names = _get_non_protected_cols_for_run(run_dir)

            if X_test_full.shape[1] != len(feat_names):
                raise ValueError(
                    f"Feature mismatch: transformed has {X_test_full.shape[1]} cols, "
                    f"preprocessor reports {len(feat_names)}"
                )
            sp_full = spatial_analysis(v, X_test_full, k=K_NN_BY_DATASET["compas"], permutations=999)
            X_no_prot = X_test_full[:, non_prot_cols]
            W_np = PySAL_KNN.from_array(X_no_prot, k=K_NN_BY_DATASET["compas"]).symmetrize(inplace=False)
            W_np.transform = "r"
            np.random.seed(42)
            moran_np = Moran(v, W_np, permutations=999)
            lm_np = Moran_Local(v, W_np, transformation="r", permutations=999, seed=42)
            p_sim_np = np.asarray(lm_np.p_sim).flatten()
            q_np = np.asarray(lm_np.q).flatten()
            sig_np = fdr_benjamini_hochberg(p_sim_np, alpha=0.05)
            hh_np = (q_np == 1) & sig_np
            hh_full = sp_full["HH_mask"]
            inter = np.logical_and(hh_full, hh_np).sum()
            union = np.logical_or(hh_full, hh_np).sum()
            jaccard = inter / union if union > 0 else 1.0
            excl_rows.append({
                "seed": seed,
                "moran_full": sp_full["moran_i"],
                "moran_excl_prot": float(moran_np.I),
                "n_hh_full": int(hh_full.sum()),
                "n_hh_excl_prot": int(hh_np.sum()),
                "jaccard_hh": jaccard,
            })
        except Exception as e:
            print(f"  seed={seed}: SKIP ({e})")
    return pd.DataFrame(excl_rows)


def bootstrap_mean_ci(values, n_boot=2000, seed=42):
    """Bootstrap CI for the mean from per-seed subgroup rates."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.RandomState(seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))