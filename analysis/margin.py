"""Margin vs variance analysis functions for Notebook 10."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.run_analysis import (
    load_P_test,
    select_rashomon_global,
    pointwise_variance,
    spatial_analysis,
)
from analysis.preprocessing import get_transformed_test_features
from analysis.knn_defaults import K_NN_BY_DATASET


def compute_margin_and_wilcoxon(
    results_dir: Path,
    datasets: List[str],
    seeds: List[int],
    K: int,
    cache_dir: Path,
    cache_version: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute margin vs variance analysis and Mann-Whitney U test.

    Returns:
        Tuple of (margin_df, mannwhitney_df) with same columns as notebook.
    """
    margin_rows = []
    wilcoxon_rows = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = results_dir / dataset / f"seed={seed}"
            try:
                P_test = load_P_test(run_dir)
                idx_global = select_rashomon_global(run_dir, K=K)
                P_sel = P_test[idx_global]
                p_mean = P_sel.mean(axis=0)
                margin = np.abs(p_mean - 0.5)
                v = pointwise_variance(P_sel)
                q90 = np.quantile(v, 0.90)
                hv_mask = v >= q90
                X_test = get_transformed_test_features(run_dir, dataset)
                sp = spatial_analysis(
                    v, X_test, k=K_NN_BY_DATASET[dataset], permutations=999,
                )
                hh_mask = sp["HH_mask"]
                r_pearson, p_pearson = stats.pearsonr(v, margin)
                r_spearman, p_spearman = stats.spearmanr(v, margin)
                margin_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "pearson_r": r_pearson,
                    "pearson_p": p_pearson,
                    "spearman_r": r_spearman,
                    "spearman_p": p_spearman,
                    "margin_mean_hh": margin[hh_mask].mean() if hh_mask.sum() > 0 else np.nan,
                    "margin_mean_non_hh": margin[~hh_mask].mean(),
                    "margin_mean_hv": margin[hv_mask].mean(),
                    "margin_mean_non_hv": margin[~hv_mask].mean(),
                    "n_hh": int(hh_mask.sum()),
                    "n_hv": int(hv_mask.sum()),
                    "n_test": len(v),
                })
                if hh_mask.sum() >= 3 and (~hh_mask).sum() >= 3:
                    stat, pval = stats.mannwhitneyu(
                        margin[hh_mask], margin[~hh_mask], alternative="less"
                    )
                    wilcoxon_rows.append({
                        "dataset": dataset, "seed": seed,
                        "U_stat": stat, "p_value": pval,
                        "margin_hh": margin[hh_mask].mean(),
                        "margin_non_hh": margin[~hh_mask].mean(),
                    })
            except Exception as e:
                print(f"  {dataset} seed={seed}: SKIP ({e})")
    m = pd.DataFrame(margin_rows)
    w = pd.DataFrame(wilcoxon_rows)
    m.to_parquet(cache_dir / f"nb10_margin_{cache_version}.parquet", index=False)
    w.to_parquet(cache_dir / f"nb10_wilcoxon_{cache_version}.parquet", index=False)
    return m, w


def load_margin_and_wilcoxon(
    results_dir: Path,
    datasets: List[str],
    seeds: List[int],
    K: int,
    cache_dir: Path,
    cache_version: str,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load or compute margin and Wilcoxon test results."""
    m_path = cache_dir / f"nb10_margin_{cache_version}.parquet"
    w_path = cache_dir / f"nb10_wilcoxon_{cache_version}.parquet"
    if not force_recompute and m_path.is_file() and w_path.is_file():
        return pd.read_parquet(m_path), pd.read_parquet(w_path)
    return compute_margin_and_wilcoxon(
        results_dir, datasets, seeds, K, cache_dir, cache_version
    )
