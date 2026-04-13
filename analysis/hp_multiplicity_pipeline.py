"""
Multi-seed hyperparameter / family multiplicity tables (Rashomon vs full pool).

Builds tidy metric tables, model-level summaries, and aggregation helpers used
by notebook 06.  Plotting stays thin: consumers pass aggregated DataFrames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.hp_analysis import (
    POOL_TYPE_RASHOMON,
    compute_Vm,
    hp_importance_Vm,
    select_pool_indices,
)
from analysis.hyperparams import (
    compute_family_importance,
    compute_within_family_hp_importance,
    ensure_hp_columns,
)
from analysis.knn_defaults import K_NN_BY_DATASET
from analysis.preprocessing import get_transformed_test_features
from analysis.run_analysis import load_meta, load_P_test, pointwise_variance, spatial_analysis


def compute_lisa_hh_mask(
    run_dir: Any,
    dataset: str,
    P_sel: np.ndarray,
    *,
    k_nn: Optional[int] = None,
    permutations: int = 999,
) -> Tuple[np.ndarray, int]:
    """HH mask from pointwise variance on ``P_sel``; shape (n_test,), dtype bool."""
    k = int(k_nn if k_nn is not None else K_NN_BY_DATASET.get(dataset, 30))
    X_test = get_transformed_test_features(run_dir, dataset)
    v = pointwise_variance(P_sel, ddof=0)
    sp = spatial_analysis(v, X_test, k=k, permutations=permutations)
    hh = np.asarray(sp["HH_mask"], dtype=bool)
    return hh, int(hh.sum())


def per_seed_analysis_tables(
    run_dir: Any,
    dataset: str,
    seed: int,
    *,
    pool_type: str = POOL_TYPE_RASHOMON,
    rashomon_k_each: int = 25,
    min_hh_obs: int = 5,
    permutations: int = 999,
) -> Dict[str, pd.DataFrame]:
    """
    One seed: decomposition (family + within-family HP) on all/HH/non-HH,
    V_m-based HP importance on all/HH/non-HH, model-level V_m table.

    Returns keys: ``models``, ``metrics_long``, ``decomp_hp_wide``,
    ``vm_hp_wide`` (wide = seed-level frames suitable for aggregation).
    """
    run_dir = Path(run_dir)
    meta = ensure_hp_columns(load_meta(run_dir))
    P_test = load_P_test(run_dir)
    pool_idx = select_pool_indices(run_dir, pool_type=pool_type, rashomon_k_each=rashomon_k_each)
    if pool_idx.size == 0:
        return {
            "models": pd.DataFrame(),
            "metrics_long": pd.DataFrame(),
            "decomp_hp_wide": pd.DataFrame(),
            "vm_hp_wide": pd.DataFrame(),
        }

    meta_sel = meta.iloc[pool_idx].reset_index(drop=True)
    P_sel = P_test[pool_idx]

    hh_mask, n_hh = compute_lisa_hh_mask(
        run_dir, dataset, P_sel, k_nn=K_NN_BY_DATASET.get(dataset), permutations=permutations
    )
    non_mask = ~hh_mask
    n_non = int(non_mask.sum())

    long_rows: List[Dict[str, Any]] = []

    def add_family_decomp(subset: str, mask: Optional[np.ndarray]) -> None:
        fi = compute_family_importance(meta_sel, P_sel, obs_mask=mask)
        if fi.empty:
            return
        r = fi.iloc[0]
        long_rows.append({
            "dataset": dataset,
            "seed": seed,
            "pool_type": pool_type,
            "subset": subset,
            "family": "__global__",
            "hyperparameter": "__family__",
            "value_or_group": "model_name",
            "metric_name": "ratio_of_sums",
            "metric_value": float(r["ratio_of_sums"]),
        })
        long_rows.append({
            "dataset": dataset,
            "seed": seed,
            "pool_type": pool_type,
            "subset": subset,
            "family": "__global__",
            "hyperparameter": "__family__",
            "value_or_group": "model_name",
            "metric_name": "mean_ratio",
            "metric_value": float(r["mean_ratio"]),
        })

    add_family_decomp("all", None)
    if n_hh >= min_hh_obs:
        add_family_decomp("HH", hh_mask)
    if n_non >= min_hh_obs:
        add_family_decomp("non_HH", non_mask)

    decomp_rows: List[pd.DataFrame] = []

    def run_decomp_subset(subset: str, mask: Optional[np.ndarray]) -> None:
        hp_df = compute_within_family_hp_importance(meta_sel, P_sel, obs_mask=mask)
        if hp_df.empty:
            return
        h = hp_df.copy()
        h["dataset"] = dataset
        h["seed"] = seed
        h["pool_type"] = pool_type
        h["subset"] = subset
        decomp_rows.append(h)

    run_decomp_subset("all", None)
    if n_hh >= min_hh_obs:
        run_decomp_subset("HH", hh_mask)
    if n_non >= min_hh_obs:
        run_decomp_subset("non_HH", non_mask)

    decomp_hp_wide = pd.concat(decomp_rows, ignore_index=True) if decomp_rows else pd.DataFrame()

    vm_imp_rows: List[pd.DataFrame] = []

    def run_vm_subset(subset: str, mask: Optional[np.ndarray]) -> None:
        families = sorted(meta_sel["model_name"].unique())
        for family in families:
            fam_mask = (meta_sel["model_name"] == family).values
            if int(fam_mask.sum()) < 3:
                continue
            P_f = P_sel[fam_mask]
            meta_f = meta_sel.loc[fam_mask].reset_index(drop=True)
            V_m = compute_Vm(P_f, obs_mask=mask)
            imp = hp_importance_Vm(V_m, meta_f)
            if imp.empty:
                continue
            imp = imp.copy()
            imp["dataset"] = dataset
            imp["seed"] = seed
            imp["family"] = family
            imp["pool_type"] = pool_type
            imp["subset"] = subset
            imp["K_actual"] = int(fam_mask.sum())
            vm_imp_rows.append(imp)

    run_vm_subset("all", None)
    if n_hh >= min_hh_obs:
        run_vm_subset("HH", hh_mask)
    if n_non >= min_hh_obs:
        run_vm_subset("non_HH", non_mask)

    vm_hp_wide = pd.concat(vm_imp_rows, ignore_index=True) if vm_imp_rows else pd.DataFrame()

    # Long rows from decomposition HP
    if not decomp_hp_wide.empty:
        for _, row in decomp_hp_wide.iterrows():
            long_rows.append({
                "dataset": dataset,
                "seed": seed,
                "pool_type": pool_type,
                "subset": row["subset"],
                "family": row["family"],
                "hyperparameter": row["hp"],
                "value_or_group": "",
                "metric_name": "decomp_ratio_of_sums",
                "metric_value": float(row["ratio_of_sums"]),
            })

    if not vm_hp_wide.empty:
        for _, row in vm_hp_wide.iterrows():
            long_rows.append({
                "dataset": dataset,
                "seed": seed,
                "pool_type": pool_type,
                "subset": row["subset"],
                "family": row["family"],
                "hyperparameter": row["hp_name"],
                "value_or_group": "",
                "metric_name": "vm_between_group_ratio",
                "metric_value": float(row["ratio_of_sums"]),
            })

    metrics_long = pd.DataFrame(long_rows)

    # Model-level: V_m on all / HH / non-HH per family block
    model_blocks: List[pd.DataFrame] = []
    families = sorted(meta_sel["model_name"].unique())
    for family in families:
        fam_mask = (meta_sel["model_name"] == family).values
        if int(fam_mask.sum()) < 2:
            continue
        P_f = P_sel[fam_mask]
        meta_f = meta_sel.loc[fam_mask].reset_index(drop=True)
        idx_global = pool_idx[fam_mask]
        Vm_all = compute_Vm(P_f, None)
        Vm_hh = compute_Vm(P_f, hh_mask) if n_hh >= min_hh_obs else np.full(P_f.shape[0], np.nan)
        Vm_non = compute_Vm(P_f, non_mask) if n_non >= min_hh_obs else np.full(P_f.shape[0], np.nan)
        block = meta_f.copy()
        block["family"] = family  # alias for filtering/plots (matches ``model_name`` in this slice)
        block["dataset"] = dataset
        block["seed"] = seed
        block["pool_type"] = pool_type
        block["global_row_idx"] = idx_global
        block["V_m"] = Vm_all
        block["V_m_HH"] = Vm_hh
        block["V_m_nonHH"] = Vm_non
        block["val_brier"] = meta_f["val_brier"].values
        model_blocks.append(block)

    models = pd.concat(model_blocks, ignore_index=True) if model_blocks else pd.DataFrame()

    return {
        "models": models,
        "metrics_long": metrics_long,
        "decomp_hp_wide": decomp_hp_wide,
        "vm_hp_wide": vm_hp_wide,
    }


def run_dataset_all_seeds(
    dataset_dir: Any,
    dataset: str,
    *,
    pool_type: str = POOL_TYPE_RASHOMON,
    rashomon_k_each: int = 25,
    min_hh_obs: int = 5,
) -> Dict[str, pd.DataFrame]:
    """All seeds under ``dataset_dir`` for one dataset."""
    d = Path(dataset_dir)
    run_dirs = sorted(
        [p for p in d.iterdir() if p.is_dir() and p.name.startswith("seed=")],
        key=lambda p: int(p.name.split("=")[1]),
    )
    parts = [per_seed_analysis_tables(
        rd, dataset, int(rd.name.split("=")[1]),
        pool_type=pool_type,
        rashomon_k_each=rashomon_k_each,
        min_hh_obs=min_hh_obs,
    ) for rd in run_dirs]

    def cat(key: str) -> pd.DataFrame:
        dfs = [p[key] for p in parts if not p[key].empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return {
        "models": cat("models"),
        "metrics_long": cat("metrics_long"),
        "decomp_hp_wide": cat("decomp_hp_wide"),
        "vm_hp_wide": cat("vm_hp_wide"),
    }


def aggregate_decomposition_hp(
    df: pd.DataFrame,
    *,
    ratio_col: str = "ratio_of_sums",
    hp_col: str = "hp",
) -> pd.DataFrame:
    """Mean ± std, mean rank, top-k frequency per (dataset, pool_type, subset, family, hp)."""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    gcols = ["dataset", "seed", "family"]
    for c in ("pool_type", "subset"):
        if c in d.columns:
            gcols.append(c)
    d["rank"] = d.groupby(gcols)[ratio_col].rank(ascending=False, method="min")
    agg_keys = [k for k in ("dataset", "pool_type", "subset", "family", hp_col) if k in d.columns]
    agg = (
        d.groupby(agg_keys, dropna=False)
        .agg(
            mean_importance=(ratio_col, "mean"),
            std_importance=(ratio_col, "std"),
            n_seeds=("seed", "nunique"),
            mean_rank=("rank", "mean"),
            rank_freq_top1=("rank", lambda x: float((x == 1).mean())),
            rank_freq_top3=("rank", lambda x: float((x <= 3).mean())),
        )
        .reset_index()
    )
    agg["std_importance"] = agg["std_importance"].fillna(0.0)
    sort_cols = [c for c in ("dataset", "pool_type", "subset", "family", "mean_importance") if c in agg.columns]
    if not sort_cols:
        return agg.reset_index(drop=True)
    asc = [True] * (len(sort_cols) - 1) + [False]
    return agg.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def hotspot_delta_decomp(
    decomp_hp_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Per seed/family/hp: ratio_HH - ratio_all (decomposition), then seed-level rows."""
    if decomp_hp_wide.empty or "subset" not in decomp_hp_wide.columns:
        return pd.DataFrame()
    sub = decomp_hp_wide[decomp_hp_wide["subset"].isin(["all", "HH"])].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index=[c for c in ("dataset", "seed", "pool_type", "family", "hp") if c in sub.columns],
        columns="subset",
        values="ratio_of_sums",
        aggfunc="first",
    )
    if "HH" not in pivot.columns or "all" not in pivot.columns:
        return pd.DataFrame()
    pivot["delta_HH_minus_all"] = pivot["HH"] - pivot["all"]
    out = pivot.reset_index().rename_axis(None, axis=1)
    return out


def aggregate_hotspot_delta(out_per_seed: pd.DataFrame) -> pd.DataFrame:
    if out_per_seed.empty or "delta_HH_minus_all" not in out_per_seed.columns:
        return pd.DataFrame()
    keys = [c for c in ("dataset", "pool_type", "family", "hp") if c in out_per_seed.columns]
    return (
        out_per_seed.groupby(keys)["delta_HH_minus_all"]
        .agg(mean_delta="mean", std_delta="std", n_seeds="count")
        .reset_index()
    )


def aggregate_family_importance_long(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds for global family factor (ratio_of_sums)."""
    if metrics_long.empty:
        return pd.DataFrame()
    sub = metrics_long[
        (metrics_long["family"] == "__global__")
        & (metrics_long["hyperparameter"] == "__family__")
        & (metrics_long["metric_name"] == "ratio_of_sums")
    ]
    if sub.empty:
        return pd.DataFrame()
    keys = [c for c in ("dataset", "pool_type", "subset") if c in sub.columns]
    return (
        sub.groupby(keys, dropna=False)
        .agg(mean_ratio=("metric_value", "mean"), std_ratio=("metric_value", "std"), n_seeds=("seed", "nunique"))
        .reset_index()
    )


def plot_family_importance_bars(
    fam_agg: pd.DataFrame,
    *,
    fig_path: Optional[Path] = None,
) -> None:
    """One row of panels: one subplot per dataset; bars = subsets with error bars."""
    import matplotlib.pyplot as plt

    if fam_agg.empty:
        return
    ds_list = list(fam_agg["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4 * len(ds_list), 3.5), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        g = fam_agg[fam_agg["dataset"] == ds]
        subs = g["subset"].astype(str).tolist()
        x = np.arange(len(subs))
        ax.bar(x, g["mean_ratio"], yerr=g["std_ratio"], capsize=3, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=20, ha="right")
        ax.set_ylabel("Family importance (ratio of sums)")
        ax.set_title(str(ds))
    fig.suptitle("Global family factor (mean ± std over seeds)", fontsize=12)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_decomp_hp_bars(
    df_agg: pd.DataFrame,
    *,
    dataset: str,
    subset: str,
    family: str,
    top_n: int = 10,
    fig_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart from ``aggregate_decomposition_hp`` output."""
    import matplotlib.pyplot as plt

    col = "hp" if "hp" in df_agg.columns else "hyperparameter"
    sub = df_agg[(df_agg["dataset"] == dataset) & (df_agg["family"] == family)]
    if "subset" in df_agg.columns:
        sub = sub[sub["subset"] == subset]
    sub = sub.sort_values("mean_importance", ascending=False).head(top_n)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_importance"], xerr=sub["std_importance"], capsize=2, color="darkseagreen")
    ax.set_yticks(y)
    ax.set_yticklabels(sub[col].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Within-family HP importance (decomposition ratio)")
    ax.set_title(f"{dataset} — {family} — subset={subset}")
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_vm_hp_bars(
    df_agg: pd.DataFrame,
    *,
    dataset: str,
    family: str,
    subset: str = "all",
    pool_type: Optional[str] = None,
    top_n: int = 10,
    fig_path: Optional[Path] = None,
) -> None:
    """Horizontal bars from ``aggregate_hp_importance`` (V_m-based ratios)."""
    import matplotlib.pyplot as plt

    sub = df_agg[(df_agg["dataset"] == dataset) & (df_agg["family"] == family)]
    if "subset" in df_agg.columns:
        sub = sub[sub["subset"] == subset]
    if pool_type is not None and "pool_type" in df_agg.columns:
        sub = sub[sub["pool_type"] == pool_type]
    sub = sub.sort_values("mean_importance", ascending=False).head(top_n)
    if sub.empty or "hp_name" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_importance"], xerr=sub["std_importance"], capsize=2, color="steelblue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["hp_name"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("V_m-based HP importance (mean ± std over seeds)")
    ax.set_title(f"{dataset} — {family} — subset={subset}")
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()


def plot_hotspot_hp_delta(
    delta_agg: pd.DataFrame,
    *,
    dataset: str,
    family: str,
    top_n: int = 12,
    fig_path: Optional[Path] = None,
) -> None:
    import matplotlib.pyplot as plt

    sub = delta_agg[delta_agg["dataset"] == dataset]
    if "family" in delta_agg.columns:
        sub = sub[sub["family"] == family]
    sub = sub.assign(_abs=np.abs(sub["mean_delta"])).sort_values("_abs", ascending=False).drop(columns="_abs").head(top_n)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_delta"], xerr=sub["std_delta"], capsize=2, color="coral")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["hp"].astype(str))
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Δ importance (HH − all), mean ± std over seeds")
    ax.set_title(f"{dataset} — {family}")
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()


__all__ = [
    "POOL_TYPE_RASHOMON",
    "aggregate_decomposition_hp",
    "aggregate_family_importance_long",
    "aggregate_hotspot_delta",
    "compute_lisa_hh_mask",
    "hotspot_delta_decomp",
    "per_seed_analysis_tables",
    "plot_decomp_hp_bars",
    "plot_family_importance_bars",
    "plot_vm_hp_bars",
    "plot_hotspot_hp_delta",
    "run_dataset_all_seeds",
]
