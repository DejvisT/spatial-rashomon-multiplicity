"""
Per-family, multi-seed hyperparameter importance analysis for Rashomon sets.

Analyses HP associations among near-optimal models and their relationship
to model-level disagreement contribution V_m.

V_m = mean_x( (P[m,x] - p_bar(x))^2 )

measures how much model m's predictions deviate from the family ensemble
mean — its contribution to predictive multiplicity within that family.

Importance of an HP is quantified by the between-group variance of V_m when
models are grouped by unique HP values (a marginal association, not causal).
For the thesis pipeline this view is **secondary**; primary HP importance and
effect shapes are produced by ``analysis/hp_meta_model.py`` (descriptive
meta-models with out-of-sample checks).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from analysis.run_analysis import load_meta, select_rashomon_per_family_k_each
from analysis.hyperparams import make_hp_key

PathLike = Union[str, Path]

EPS = 1e-15

# Pool labels for downstream tables (extensible: full trained pool vs Rashomon subset).
POOL_TYPE_RASHOMON = "Rashomon"
POOL_TYPE_FULL_POOL = "full_pool"


def is_hp_numeric(hp_values: pd.Series) -> bool:
    """Check if HP values are numeric (can be converted to float)."""
    coerced = pd.to_numeric(hp_values, errors='coerce')
    return coerced.notna().any()


def determine_hp_grouping(hp_values: pd.Series) -> Tuple[str, Union[List[float], None]]:
    """
    Determine grouping strategy for HP values.
    
    Returns:
    - grouping_type: 'categorical_exact', 'numeric_exact', 'numeric_tertile'
    - bins: for tertiles, the bin edges (3 values for 2 bins), else None
    """
    unique_vals = hp_values.dropna().unique()
    
    if not is_hp_numeric(pd.Series(unique_vals)):
        # Categorical
        return 'categorical_exact', None
    
    # Numeric
    numeric_series = pd.to_numeric(pd.Series(unique_vals), errors='coerce').dropna()
    n_unique = len(numeric_series.unique())
    
    if n_unique <= 4:
        return 'numeric_exact', None
    else:
        # Tertiles
        quantiles = numeric_series.quantile([1/3, 2/3]).values
        return 'numeric_tertile', quantiles.tolist()


def apply_grouping(hp_values: pd.Series, grouping_type: str, bins: Optional[List[float]]) -> pd.Series:
    """Apply grouping to HP values."""
    if grouping_type == 'categorical_exact':
        return hp_values.astype(str)
    elif grouping_type == 'numeric_exact':
        return pd.to_numeric(hp_values, errors='coerce').astype(str)
    elif grouping_type == 'numeric_tertile':
        numeric_vals = pd.to_numeric(hp_values, errors='coerce')
        bins_full = [-np.inf] + bins + [np.inf]
        labels = ['low', 'medium', 'high']
        return pd.cut(numeric_vals, bins=bins_full, labels=labels, include_lowest=True).astype(str)
    else:
        raise ValueError(f"Unknown grouping_type: {grouping_type}")


def merge_small_groups(group_counts: Dict[str, int], min_size: int = 2) -> Dict[str, str]:
    """
    Merge small groups where reasonable.
    
    For tertiles, merge adjacent if small.
    For exact, don't merge, just mark to skip.
    """
    groups = list(group_counts.keys())
    counts = [group_counts[g] for g in groups]
    
    # If tertile, try merging adjacent
    if set(groups) == {'low', 'medium', 'high'}:
        # Merge low and medium if low < min_size
        if counts[0] < min_size and counts[1] >= min_size:
            return {'low': 'medium', 'medium': 'medium', 'high': 'high'}
        # Merge medium and high if high < min_size
        elif counts[2] < min_size and counts[1] >= min_size:
            return {'low': 'low', 'medium': 'medium', 'high': 'medium'}
        # If both ends small, merge all to medium
        elif counts[0] < min_size and counts[2] < min_size:
            return {'low': 'medium', 'medium': 'medium', 'high': 'medium'}
        else:
            return {g: g for g in groups}
    else:
        # For exact or categorical, no merging
        return {g: g for g in groups}


def get_valid_groups(group_counts: Dict[str, int], merge_map: Dict[str, str], min_size: int = 2) -> Dict[str, int]:
    """Get merged group counts, only including groups with >= min_size."""
    merged_counts = Counter()
    for g, count in group_counts.items():
        merged_g = merge_map.get(g, g)
        merged_counts[merged_g] += count
    
    return {g: c for g, c in merged_counts.items() if c >= min_size}


def select_pool_indices(
    run_dir: PathLike,
    *,
    pool_type: str,
    rashomon_k_each: int,
) -> np.ndarray:
    """
    Map pool_type to row indices into meta / P_test.

    * ``Rashomon`` — per-family top ``rashomon_k_each`` by validation Brier (same
      construction as ``select_rashomon_per_family_k_each``).
    * ``full_pool`` — all trained candidates in the run.
    """
    run_dir = Path(run_dir)
    if pool_type == POOL_TYPE_RASHOMON:
        return np.asarray(
            select_rashomon_per_family_k_each(run_dir, K_each=rashomon_k_each),
            dtype=int,
        )
    if pool_type == POOL_TYPE_FULL_POOL:
        meta = load_meta(run_dir)
        return np.arange(len(meta), dtype=int)
    raise ValueError(f"Unknown pool_type={pool_type!r}")


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def compute_Vm(
    P_sel: np.ndarray,
    obs_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Model-level disagreement contribution.

    For each model m in P_sel (K, n_obs):
        V_m = mean_x( (P_sel[m, x] - p_bar(x))^2 )
    where p_bar(x) = mean_m P_sel[m, x].

    If ``obs_mask`` is provided, means are taken only over the masked test points
    (e.g. HH vs non-HH subsets).
    """
    if obs_mask is None:
        P = P_sel
    else:
        P = P_sel[:, np.asarray(obs_mask, dtype=bool)]
    if P.size == 0 or P.shape[1] == 0:
        return np.zeros(P_sel.shape[0], dtype=float)
    p_bar = P.mean(axis=0)
    return np.array([np.mean((P[m] - p_bar) ** 2) for m in range(P.shape[0])])


# ---------------------------------------------------------------------------
# HP importance via V_m (scalar per model)
# ---------------------------------------------------------------------------

def hp_importance_Vm(
    V_m: np.ndarray,
    meta_sel: pd.DataFrame,
    grouping_info: Optional[Dict[str, Tuple[str, Optional[List[float]]]]] = None,
    *,
    min_groups: int = 2,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    For each HP column in meta_sel, compute between-group / total variance
    of V_m using the specified grouping. Returns one row per HP that passes the filters.
    """
    hp_cols = [c for c in meta_sel.columns if c.startswith("hp_")]
    rows: List[Dict[str, Any]] = []

    for hp_col in hp_cols:
        hp_name = hp_col.replace("hp_", "")
        if grouping_info is not None and hp_name not in grouping_info:
            continue
        
        if grouping_info is not None:
            grouping_type, bins = grouping_info[hp_name]
            # Apply grouping
            grouped_keys = apply_grouping(meta_sel[hp_col], grouping_type, bins)
        else:
            # Old logic
            keys = np.array([make_hp_key(v) for v in meta_sel[hp_col].values], dtype=object)
            grouped_keys = pd.Series(keys)
            grouping_type = 'legacy_exact'
        
        valid = grouped_keys.notna() & (grouped_keys != 'nan')
        if valid.sum() < 3:
            continue
        V_valid = V_m[valid]
        keys_valid = grouped_keys[valid]
        
        if grouping_info is not None:
            group_counts = Counter(keys_valid)
            merge_map = merge_small_groups(group_counts, min_group_size)
            valid_groups = get_valid_groups(group_counts, merge_map, min_group_size)
            
            if len(valid_groups) < min_groups:
                continue
            
            # Apply merging
            keys_merged = keys_valid.map(lambda x: merge_map.get(x, x))
            keep_mask = keys_merged.isin(valid_groups.keys())
            V_use = V_valid[keep_mask]
            keys_use = keys_merged[keep_mask]
            
            n_groups = len(valid_groups)
            min_group_size_actual = min(valid_groups.values())
        else:
            # Old logic
            counts = Counter(keys_valid)
            keep_keys = {k for k, c in counts.items() if c >= min_group_size}
            if len(keep_keys) < min_groups:
                continue

            keep_mask = np.array([k in keep_keys for k in keys_valid])
            V_use = V_valid[keep_mask]
            keys_use = keys_valid[keep_mask]
            
            n_groups = len(keep_keys)
            min_group_size_actual = min(counts[k] for k in keep_keys)
        
        var_total = float(np.var(V_use))
        if var_total < EPS:
            continue

        overall_mean = np.mean(V_use)
        n_total = len(V_use)
        var_between = 0.0
        for k in np.unique(keys_use):
            group = V_use[keys_use == k]
            p_g = len(group) / n_total
            var_between += p_g * (np.mean(group) - overall_mean) ** 2

        rows.append({
            "hp_name": hp_name,
            "ratio_of_sums": var_between / var_total,
            "n_values": len(np.unique(keys_use)),
            "n_models": int(n_total),
            "mean_V_m": float(overall_mean),
            "grouping_type": grouping_type,
            "n_groups": n_groups,
            "min_group_size": min_group_size_actual,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ratio_of_sums", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_hp_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-seed HP importance across seeds.

    Adds rank_freq_top1 / top3: how often the HP appears in the top 1 / top 3
    within that seed+family.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    group_cols = ["dataset", "seed", "family"]
    if "pool_type" in df.columns:
        group_cols.append("pool_type")
    if "subset" in df.columns:
        group_cols.append("subset")
    df["rank"] = (
        df.groupby(group_cols)["ratio_of_sums"]
        .rank(ascending=False, method="min")
    )

    agg_keys = ["dataset", "family", "hp_name"]
    if "pool_type" in df.columns:
        agg_keys = ["dataset", "pool_type", "family", "hp_name"]
    if "subset" in df.columns:
        if "pool_type" in df.columns:
            agg_keys = ["dataset", "pool_type", "subset", "family", "hp_name"]
        else:
            agg_keys = ["dataset", "subset", "family", "hp_name"]

    agg = (
        df.groupby(agg_keys)
        .agg(
            mean_importance=("ratio_of_sums", "mean"),
            std_importance=("ratio_of_sums", "std"),
            n_seeds=("seed", "nunique"),
            mean_V_m=("mean_V_m", "mean"),
            mean_rank=("rank", "mean"),
            rank_freq_top1=("rank", lambda x: float((x == 1).mean())),
            rank_freq_top3=("rank", lambda x: float((x <= 3).mean())),
            grouping_type=("grouping_type", "first"),
            n_groups=("n_groups", "mean"),
            min_group_size=("min_group_size", "mean"),
        )
        .reset_index()
    )
    agg["std_importance"] = agg["std_importance"].fillna(0.0)
    sort_cols = [c for c in ["dataset", "pool_type", "subset", "family", "mean_importance"] if c in agg.columns]
    asc = [True] * (len(sort_cols) - 1) + [False]
    agg = agg.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    return agg
