"""
Descriptive / exploratory meta-models for model-level disagreement score ``V_m``.

**This is the **main descriptive** tool for within-family hyperparameter 
*importance* and *range-style* summaries, conditional on validation Brier.
**Family decomposition** (between vs within family on ``P``) lives in 
``hp_results.py`` / ``hp_decomposition.py`` and is the main *structural* analysis.
**Unique-value HP variance decompositions** on ``P`` are secondary / appendix
material.

RandomForest fits are **not** causal or fully calibrated predictors. We report
leave-one-(outer-)seed-out metrics when possible (else a labeled row-level KFold
fallback). ``run_hp_meta_model_suite`` writes compact thesis-facing summary CSVs
only (no per-group importance dumps).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.hp_analysis import POOL_TYPE_RASHOMON

MIN_GROUP_ROWS = 20
MIN_SEEDS_LOSO = 3
MIN_TARGET_UNIQUE = 2
MIN_TRAIN_ROWS_FOLD = 10
MIN_TEST_ROWS_FOLD = 1
N_STABILITY_REPS_DEFAULT = 25
N_GRID_1D = 24
N_GRID_2D = 10
MAX_BACKGROUND_PDP = 100


def _safe_obj_to_str(v: Any) -> Any:
    if v is None:
        return None
    try:
        is_missing = pd.isna(v)
        if isinstance(is_missing, (bool, type(pd.NA))) and bool(is_missing):
            return None
    except Exception:
        pass
    return repr(v)


def _resolve_perf_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("validation_brier", "val_brier", "brier_val", "brier_score"):
        if c in df.columns:
            return c
    return None


def unify_validation_brier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "validation_brier" not in out.columns:
        alt = _resolve_perf_column(out)
        if alt is not None:
            out = out.rename(columns={alt: "validation_brier"})
    return out


def resolve_outer_seed_column(
    df: pd.DataFrame,
    preference: Sequence[str] = ("outer_seed", "seed"),
) -> str:
    """
    Column used for leave-one-seed-out validation and seed-level bootstrap.

    Prefer ``outer_seed`` (training outer split / run id) when present, else ``seed``.
    """
    for c in preference:
        if c in df.columns and df[c].notna().any():
            return c
    return "seed"


def collapse_feature_name(name: str, perf_col: str, hp_cols: Sequence[str]) -> str:
    if "__" in name:
        name = name.split("__", 1)[1]
    if name == perf_col:
        return "validation_brier"
    if name in hp_cols:
        return name
    matches = [hp for hp in hp_cols if name.startswith(hp + "_")]
    if matches:
        return max(matches, key=len)
    return name


def prepare_xy(
    grp: pd.DataFrame,
    hp_cols: Sequence[str],
    target_col: str,
    seed_col: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], List[str], Optional[np.ndarray]]:
    """Build X, y, hp list, and optional per-row seed array (aligned, reset index)."""
    perf_src = _resolve_perf_column(grp)
    if perf_src is None or target_col not in grp.columns:
        return None, None, [], None
    extra = [seed_col] if seed_col and seed_col in grp.columns else []
    use_cols = [perf_src] + list(hp_cols) + [target_col] + extra
    g = grp[[c for c in use_cols if c in grp.columns]].copy()
    g = g.dropna(subset=[target_col])
    if len(g) < MIN_GROUP_ROWS or g[target_col].nunique() < MIN_TARGET_UNIQUE:
        return None, None, [], None

    seeds = None
    if seed_col and seed_col in grp.columns:
        seeds = g[seed_col].values

    y = g[target_col].values.astype(float)
    X = g[[perf_src] + [c for c in hp_cols if c in g.columns]].copy()
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return None, None, [], None

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map(_safe_obj_to_str)

    keep_cols = [c for c in X.columns if X[c].dropna().nunique() > 1]
    X = X[keep_cols]
    if X.shape[1] == 0:
        return None, None, [], None

    if perf_src in X.columns and perf_src != "validation_brier":
        X = X.rename(columns={perf_src: "validation_brier"})

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    if seeds is not None:
        seeds = np.asarray(seeds)

    hp_in = [c for c in hp_cols if c in X.columns]
    return X, y, hp_in, seeds


def _build_preprocessors(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_rf_pipeline(X: pd.DataFrame, random_state: int, *, n_estimators: int = 150) -> Pipeline:
    pre = _build_preprocessors(X)
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=1,
    )
    return Pipeline([("prep", pre), ("model", model)])


def grouped_importances_from_pipe(
    pipe: Pipeline,
    X: pd.DataFrame,
    perf_col: str,
    hp_cols: Sequence[str],
) -> pd.DataFrame:
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df["feature_group"] = imp_df["feature"].map(
        lambda n: collapse_feature_name(str(n), perf_col, list(hp_cols))
    )
    return (
        imp_df.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.nanstd(y_true) < 1e-15:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def leave_one_seed_out_scores(
    X: pd.DataFrame,
    y: np.ndarray,
    seeds: Optional[np.ndarray],
    random_state: int,
) -> Tuple[str, List[float], List[float], List[float], int, int]:
    r2_list: List[float] = []
    rmse_list: List[float] = []
    mae_list: List[float] = []

    if seeds is not None and len(seeds) == len(y):
        uniq = pd.unique(pd.Series(seeds).dropna())
        uniq = np.array([s for s in uniq if str(s) != "nan"])
        n_seeds = int(len(uniq))
    else:
        uniq = np.array([])
        n_seeds = 0

    if n_seeds >= MIN_SEEDS_LOSO:
        scheme = "leave_one_seed_out"
        for s in uniq:
            train_mask = seeds != s
            test_mask = seeds == s
            if int(train_mask.sum()) < MIN_TRAIN_ROWS_FOLD or int(test_mask.sum()) < MIN_TEST_ROWS_FOLD:
                continue
            X_tr, y_tr = X.loc[train_mask].copy(), y[np.asarray(train_mask, dtype=bool)]
            X_te, y_te = X.loc[test_mask].copy(), y[np.asarray(test_mask, dtype=bool)]
            X_tr = X_tr.loc[:, X_tr.columns[X_tr.nunique(dropna=False) > 1]]
            common = [c for c in X_tr.columns if c in X_te.columns]
            if not common:
                continue
            X_tr, X_te = X_tr[common], X_te[common]
            if X_tr.shape[1] == 0:
                continue
            pipe = make_rf_pipeline(X_tr, random_state=random_state, n_estimators=80)
            try:
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_te)
            except Exception:
                continue
            r2v = _r2_safe(y_te, pred)
            if not np.isnan(r2v):
                r2_list.append(r2v)
            rmse_list.append(float(np.sqrt(mean_squared_error(y_te, pred))))
            mae_list.append(float(mean_absolute_error(y_te, pred)))
        return scheme, r2_list, rmse_list, mae_list, len(r2_list), n_seeds

    scheme = "kfold_row_level_fallback"
    n_splits = min(5, max(2, len(y) // max(MIN_TRAIN_ROWS_FOLD // 2, 2)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in kf.split(X):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx]
        X_te, y_te = X.iloc[te_idx].copy(), y[te_idx]
        X_tr = X_tr.loc[:, X_tr.columns[X_tr.nunique(dropna=False) > 1]]
        common = [c for c in X_tr.columns if c in X_te.columns]
        if not common:
            continue
        X_tr, X_te = X_tr[common], X_te[common]
        pipe = make_rf_pipeline(X_tr, random_state=random_state + int(tr_idx[0]), n_estimators=80)
        try:
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
        except Exception:
            continue
        r2v = _r2_safe(y_te, pred)
        if not np.isnan(r2v):
            r2_list.append(r2v)
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        mae_list.append(float(mean_absolute_error(y_te, pred)))
    n_seeds_report = int(pd.Series(seeds).nunique()) if seeds is not None and len(seeds) == len(y) else 0
    return scheme, r2_list, rmse_list, mae_list, len(r2_list), n_seeds_report


def meta_model_target_specs(meta_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """(target_column, filename_suffix) pairs for pooled meta-models."""
    specs: List[Tuple[str, str]] = [("V_m", "")]
    for alt, tag in (("V_m_HH", "_VmHH"), ("V_m_nonHH", "_VmNonHH")):
        if alt in meta_df.columns and int(meta_df[alt].notna().sum()) >= MIN_GROUP_ROWS:
            specs.append((alt, tag))
    return specs


def write_hp_top2_driver_summary(meta_summary: pd.DataFrame, path: Path) -> None:
    """Thesis export expects columns: dataset, family, Top-1 driver, Top-2 driver."""
    if meta_summary.empty:
        return
    if "pool_type" in meta_summary.columns:
        sub = meta_summary[meta_summary["pool_type"] == POOL_TYPE_RASHOMON].copy()
    else:
        sub = meta_summary.copy()
    if "target_vm" in sub.columns:
        sub = sub[sub["target_vm"] == "V_m"].copy()
    if sub.empty:
        sub = meta_summary.copy()
    rows = []
    for _, r in sub.iterrows():
        d1 = r.get("top_feature_1")
        i1 = r.get("top_feature_1_importance")
        d2 = r.get("top_feature_2")
        i2 = r.get("top_feature_2_importance")
        oos = r.get("r2_oos_mean")
        t1 = ""
        if pd.notna(d1):
            imp_s = f"{float(i1):.4f}" if pd.notna(i1) else "nan"
            oos_s = f"{float(oos):.4f}" if pd.notna(oos) else "nan"
            t1 = f"{d1} (grouped imp={imp_s}; OOS R² mean={oos_s})"
        t2 = f"{d2} (grouped imp={float(i2):.4f})" if pd.notna(d2) and pd.notna(i2) else ("" if not pd.notna(d2) else str(d2))
        rows.append({"dataset": r["dataset"], "family": r["family"], "Top-1 driver": t1, "Top-2 driver": t2})
    pd.DataFrame(rows).to_csv(path, index=False)


def run_hp_meta_model_suite(
    df_models: pd.DataFrame,
    *,
    table_dir: Path,
    fig_dir: Optional[Path] = None,
    random_state: int = 0,
    seed_col: Optional[str] = None,
) -> None:
    """
    Fit descriptive meta-models per (dataset, family, pool_type) and optional
    target columns V_m / V_m_HH / V_m_nonHH; write compact summary CSVs.

    If ``seed_col`` is None, uses ``outer_seed`` when available else ``seed`` for LOSO / bootstrap.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    if df_models.empty:
        return

    meta_df = unify_validation_brier(df_models.copy())
    if "validation_brier" not in meta_df.columns or "V_m" not in meta_df.columns:
        return

    seed_col_eff = seed_col or resolve_outer_seed_column(meta_df)

    hp_cols_all = sorted([c for c in meta_df.columns if c.startswith("hp_")])
    if not hp_cols_all:
        return

    group_cols = [c for c in ("dataset", "family", "pool_type") if c in meta_df.columns]
    if not group_cols:
        return

    target_specs = meta_model_target_specs(meta_df)

    summary_rows: List[Dict[str, Any]] = []
    compact_meta_rows: List[Dict[str, Any]] = []

    for keys, grp in meta_df.groupby(group_cols):
        if isinstance(keys, tuple):
            key_map = dict(zip(group_cols, keys))
        else:
            key_map = {group_cols[0]: keys}
        ds = key_map.get("dataset", "unknown")
        fam = key_map.get("family", "unknown")
        pt = key_map.get("pool_type", "unknown")

        for target_col, _target_suffix in target_specs:
            g = grp.dropna(subset=[target_col]).copy()
            if len(g) < MIN_GROUP_ROWS or g[target_col].nunique() < MIN_TARGET_UNIQUE:
                continue

            seed_for_xy = seed_col_eff if seed_col_eff in g.columns else None
            X, y, hp_use, seeds_arr = prepare_xy(g, hp_cols_all, target_col, seed_for_xy)
            if X is None or y is None:
                continue

            perf_col = "validation_brier"
            scheme, r2_l, rmse_l, mae_l, n_folds, n_seeds = leave_one_seed_out_scores(
                X, y, seeds_arr, random_state
            )

            pipe_full = make_rf_pipeline(X, random_state=random_state)
            try:
                pipe_full.fit(X, y)
            except Exception:
                continue
            y_hat_in = pipe_full.predict(X)
            r2_in = float(r2_score(y, y_hat_in)) if np.var(y) > 1e-15 else float("nan")

            imp_grouped = grouped_importances_from_pipe(pipe_full, X, perf_col, hp_use)
            imp_ranked = imp_grouped.sort_values("importance", ascending=False).reset_index(drop=True)
            imp_ranked["importance_rank"] = np.arange(1, len(imp_ranked) + 1)

            r2_mean = float(np.mean(r2_l)) if r2_l else float("nan")
            r2_std = float(np.std(r2_l, ddof=1)) if len(r2_l) > 1 else (0.0 if r2_l else float("nan"))
            rmse_mean = float(np.mean(rmse_l)) if rmse_l else float("nan")
            mae_mean = float(np.mean(mae_l)) if mae_l else float("nan")

            top_row = imp_ranked.iloc[0] if len(imp_ranked) else None
            vb_sub = imp_ranked[imp_ranked["feature_group"] == "validation_brier"]
            non_perf = imp_ranked[imp_ranked["feature_group"] != "validation_brier"]

            summary_rows.append(
                {
                    "dataset": ds,
                    "family": fam,
                    "pool_type": pt,
                    "target_vm": target_col,
                    "n_models": int(len(g)),
                    "n_seeds": int(n_seeds),
                    "validation_scheme": scheme,
                    "n_oos_folds_used": int(n_folds),
                    "r2_oos_mean": r2_mean,
                    "r2_oos_std": r2_std,
                    "rmse_oos_mean": rmse_mean,
                    "mae_oos_mean": mae_mean,
                    "r2_in_sample": r2_in,
                    "n_predictors_raw": int(X.shape[1]),
                    "top_feature_1": imp_ranked.iloc[0]["feature_group"] if len(imp_ranked) > 0 else None,
                    "top_feature_1_importance": float(imp_ranked.iloc[0]["importance"]) if len(imp_ranked) > 0 else None,
                    "top_feature_2": imp_ranked.iloc[1]["feature_group"] if len(imp_ranked) > 1 else None,
                    "top_feature_2_importance": float(imp_ranked.iloc[1]["importance"]) if len(imp_ranked) > 1 else None,
                    "top_feature_3": imp_ranked.iloc[2]["feature_group"] if len(imp_ranked) > 2 else None,
                    "top_feature_3_importance": float(imp_ranked.iloc[2]["importance"]) if len(imp_ranked) > 2 else None,
                }
            )

            if target_col == "V_m":
                compact_meta_rows.append(
                    {
                        "dataset": ds,
                        "family": fam,
                        "pool_type": pt,
                        "top_meta_feature": top_row["feature_group"] if top_row is not None else None,
                        "top_meta_feature_importance": float(top_row["importance"]) if top_row is not None else float("nan"),
                        "validation_brier_importance": float(vb_sub["importance"].iloc[0]) if len(vb_sub) else float("nan"),
                        "validation_brier_rank": int(vb_sub["importance_rank"].iloc[0]) if len(vb_sub) else pd.NA,
                        "top_non_performance_feature_importance": float(non_perf.iloc[0]["importance"]) if len(non_perf) else float("nan"),
                    }
                )

    if summary_rows:
        meta_summary = pd.DataFrame(summary_rows).sort_values(["dataset", "family", "pool_type", "target_vm"])
        meta_summary.to_csv(table_dir / "hp_meta_model_summary.csv", index=False)
        write_hp_top2_driver_summary(meta_summary, table_dir / "hp_top2_driver_summary.csv")

        if compact_meta_rows:
            hp_meta_compact = pd.DataFrame(compact_meta_rows).sort_values(["dataset", "family", "pool_type"])
            _meta_cols = [
                "dataset",
                "family",
                "pool_type",
                "top_meta_feature",
                "top_meta_feature_importance",
                "validation_brier_importance",
                "validation_brier_rank",
            ]
            hp_meta_compact[_meta_cols].to_csv(table_dir / "hp_meta_summary.csv", index=False)
            cmp = hp_meta_compact.assign(val_brier_is_rank1=hp_meta_compact["validation_brier_rank"] == 1)
            pool_cmp = (
                cmp.groupby("pool_type", dropna=False)
                .agg(
                    n_dataset_family_groups=("dataset", "count"),
                    n_val_brier_rank_1=("val_brier_is_rank1", "sum"),
                    mean_validation_brier_importance=("validation_brier_importance", "mean"),
                    mean_top_non_performance_feature_importance=(
                        "top_non_performance_feature_importance",
                        "mean",
                    ),
                )
                .reset_index()
            )
            pool_cmp.to_csv(table_dir / "hp_meta_summary_by_pool.csv", index=False)


__all__ = [
    "grouped_importances_from_pipe",
    "leave_one_seed_out_scores",
    "meta_model_target_specs",
    "prepare_xy",
    "resolve_outer_seed_column",
    "run_hp_meta_model_suite",
    "unify_validation_brier",
    "collapse_feature_name",
]
