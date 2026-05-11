"""
Hyperparameter multiplicity decomposition analysis.

This module analyzes *predictive multiplicity* within a Rashomon set of models.
It operates only on saved artifacts (meta.csv, P_test.npy) and does not retrain models.

Key idea (variance decomposition)
---------------------------------
For a set of model predictions P with shape (n_models, n_obs), define for each
observation i the total predictive variance across models:

    Var_total(i) = Var_m( P[m, i] )

To explain where this variance comes from, we decompose it with respect to a
categorical factor (e.g. model family, or hyperparameter value):

    Var_total(i) = Var_between(i) + Var_within(i)

using the exact identity:

    Var(P) = Var( E[P | G] ) + E[ Var(P | G) ]

where G is the group label.

Important: "Option A" (family-first)
------------------------------------
Global hyperparameter importance across mixed model families is confounded
because "missing hyperparameter" is effectively a proxy for family membership.
This module therefore supports a hierarchical workflow:

1) quantify between-family effects (model_name)
2) quantify within-family hyperparameter effects (dropping missing values)

The notebook should use `compute_family_importance(...)` followed by
`compute_within_family_hp_importance(...)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import ast
import json
import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Numerical stability (ratios / denominators)
EPS_RATIO = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hp_key(v: Any) -> str:
    """
    Convert an arbitrary HP value to a consistent, hashable string key.

    This avoids mixed-type grouping bugs (e.g. floats + 'nan' strings) and
    makes grouping stable across numpy/pandas.
    """
    # Missing
    if v is None:
        return "nan"
    # pandas / numpy NaN
    try:
        if isinstance(v, float) and np.isnan(v):
            return "nan"
        # numpy scalar NaN
        if isinstance(v, (np.floating,)) and np.isnan(v):
            return "nan"
    except Exception:
        pass

    # Collections -> stable tuple string
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return str(tuple(v))
        except Exception:
            return str(v)

    # Dict -> stable json string (sorted keys)
    if isinstance(v, dict):
        try:
            return json.dumps(v, sort_keys=True)
        except Exception:
            return str(v)

    return str(v)


def _subset_observations(P: np.ndarray, obs_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Subset predictions by observations.

    obs_mask may be:
      - None (no subsetting)
      - boolean mask of shape (n_obs,)
      - integer indices of shape (n_subset,)
    """
    if obs_mask is None:
        return P
    obs_mask = np.asarray(obs_mask)
    if obs_mask.dtype == bool:
        return P[:, obs_mask]
    return P[:, obs_mask.astype(int)]


def _choose_loss_col(meta: pd.DataFrame, loss_col: Optional[str] = None) -> str:
    """Pick an appropriate validation loss column for profiling."""
    if loss_col is not None:
        if loss_col not in meta.columns:
            raise ValueError(f"loss_col='{loss_col}' not found in meta columns")
        return loss_col
    for c in ("val_brier", "val_loss", "val_error", "loss", "score"):
        if c in meta.columns:
            return c
    raise ValueError(
        "Could not infer a validation loss column. Expected one of "
        "val_brier / val_loss / val_error / loss / score."
    )


def _resolve_hp_column(meta: pd.DataFrame, hp_name_or_col: str) -> Optional[str]:
    """
    Resolve a hyperparameter name to the column name in meta.

    Accepts either:
      - "max_depth" (will resolve to "hp_max_depth" if present)
      - "hp_max_depth" (direct)
    """
    if hp_name_or_col in meta.columns:
        return hp_name_or_col
    cand = f"hp_{hp_name_or_col}"
    if cand in meta.columns:
        return cand
    return None


def ensure_hp_columns(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `meta` contains expanded hyperparameter columns (hp_*).

    If meta has an 'hp' column (dict or dict-like string), it is expanded to hp_*
    columns via :func:`parse_hyperparameters`. This is done even if some hp_*
    columns already exist, so the expanded representation is consistent.

    If meta has no 'hp' column, this is a no-op.
    """
    if meta is None:
        raise ValueError("meta must not be None")

    if "hp" in meta.columns:
        return parse_hyperparameters(meta)

    return meta


# ---------------------------------------------------------------------------
# Hyperparameter parsing
# ---------------------------------------------------------------------------

def _parse_hp_value(hp_val: Any) -> Dict[str, Any]:
    """
    Parse a single hp value from meta['hp'] into a dict safely (no eval()).
    """
    if isinstance(hp_val, dict):
        return hp_val
    if hp_val is None or (isinstance(hp_val, float) and np.isnan(hp_val)):
        return {}
    if not isinstance(hp_val, str):
        return {}

    s = hp_val.strip()
    if not s:
        return {}

    # Try Python literal (handles single quotes, tuples, None)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try JSON (handles null/true/false)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}


def parse_hyperparameters(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Expand meta['hp'] (dict-like) into individual hp_* columns.

    Parameters
    ----------
    meta : DataFrame
        Expected to contain an 'hp' column with dicts or dict-like strings.

    Returns
    -------
    meta_expanded : DataFrame
        Same meta but without the 'hp' column and with hp_* columns added.
    """
    meta = meta.copy()

    if "hp" not in meta.columns:
        return meta

    hp_dicts = [_parse_hp_value(v) for v in meta["hp"].values]

    # Union of keys (stable order)
    all_keys: List[str] = sorted({k for d in hp_dicts for k in d.keys()})

    hp_rows: List[Dict[str, Any]] = []
    for d in hp_dicts:
        row = {k: d.get(k, np.nan) for k in all_keys}
        hp_rows.append(row)

    hp_df = pd.DataFrame(hp_rows, index=meta.index)

    meta_expanded = meta.drop(columns=["hp"]).copy()
    for col in hp_df.columns:
        meta_expanded[f"hp_{col}"] = hp_df[col]

    return meta_expanded


# ---------------------------------------------------------------------------
# Core: variance decomposition by groups (exact population identity)
# ---------------------------------------------------------------------------

def variance_decomposition_by_groups(
    preds: np.ndarray,
    group_keys: Sequence[Any],
    *,
    min_group_size: int = 1,
    epsilon: float = EPS_RATIO,
) -> Dict[str, Any]:
    """
    Per-observation variance decomposition by group (e.g. family or HP value).

    Uses the exact identity:
      Var(P) = Var(E[P|G]) + E[Var(P|G)]
    computed with population variances (ddof=0).

    Parameters
    ----------
    preds : array
        Shape (n_models, n_obs) or (n_obs, n_models). Axis 0 must correspond to models.
    group_keys : sequence length n_models
        Group label per model (family, hp value, etc.). Values are converted via make_hp_key.
    min_group_size : int
        Groups smaller than this are dropped (and their models removed).
    epsilon : float
        Lower bound on Var_total in the ratio denominator.

    Returns
    -------
    dict with keys:
      - var_total: (n_obs,)
      - var_between: (n_obs,)
      - var_within: (n_obs,)
      - ratio: (n_obs,)  (= var_between / max(var_total, epsilon))
      - group_counts: dict key->count after filtering
      - n_models_used: int
      - n_groups: int
    """
    preds = np.asarray(preds)

    group_keys = np.asarray(list(group_keys), dtype=object)
    if preds.ndim != 2:
        raise ValueError("preds must be a 2D array of shape (n_models, n_obs)")

    # Ensure axis 0 is models
    if preds.shape[0] != len(group_keys):
        if preds.shape[1] == len(group_keys):
            preds = preds.T
        else:
            raise ValueError(
                "preds first axis or second axis must match len(group_keys)"
            )

    keys_str = np.array([make_hp_key(k) for k in group_keys], dtype=object)
    unique_keys = pd.unique(keys_str)
    group_to_idx = {k: np.where(keys_str == k)[0] for k in unique_keys}

    # Drop small groups
    group_to_idx = {k: idx for k, idx in group_to_idx.items() if len(idx) >= min_group_size}

    if not group_to_idx:
        n_obs = preds.shape[1]
        var_total = np.var(preds, axis=0, ddof=0)
        return {
            "var_total": var_total,
            "var_between": np.zeros(n_obs, dtype=float),
            "var_within": var_total.copy(),
            "ratio": np.zeros(n_obs, dtype=float),
            "group_counts": {},
            "n_models_used": int(preds.shape[0]),
            "n_groups": 0,
        }

    # Keep only models in retained groups (recompute weights on kept models)
    keep_idx = np.sort(np.concatenate(list(group_to_idx.values())))
    preds = preds[keep_idx]
    keys_str = keys_str[keep_idx]

    n_models, n_obs = preds.shape
    unique_keys_kept = pd.unique(keys_str)
    n_groups = len(unique_keys_kept)

    # If only one group, between=0 and within=total
    if n_groups <= 1:
        var_total = np.var(preds, axis=0, ddof=0)
        return {
            "var_total": var_total,
            "var_between": np.zeros(n_obs, dtype=float),
            "var_within": var_total.copy(),
            "ratio": np.zeros(n_obs, dtype=float),
            "group_counts": {k: int(np.sum(keys_str == k)) for k in unique_keys_kept},
            "n_models_used": int(n_models),
            "n_groups": int(n_groups),
        }

    overall_mean = np.mean(preds, axis=0)
    var_total = np.var(preds, axis=0, ddof=0)

    var_between = np.zeros(n_obs, dtype=float)
    var_within = np.zeros(n_obs, dtype=float)

    for k in unique_keys_kept:
        idx = np.where(keys_str == k)[0]
        p_g = len(idx) / n_models
        group_preds = preds[idx]
        group_mean = np.mean(group_preds, axis=0)
        group_var = np.var(group_preds, axis=0, ddof=0)
        var_between += p_g * (group_mean - overall_mean) ** 2
        var_within += p_g * group_var

    denom = np.maximum(var_total, epsilon)
    ratio = np.where(denom > 0, var_between / denom, 0.0)

    return {
        "var_total": var_total,
        "var_between": var_between,
        "var_within": var_within,
        "ratio": ratio,
        "group_counts": {k: int(np.sum(keys_str == k)) for k in unique_keys_kept},
        "n_models_used": int(n_models),
        "n_groups": int(n_groups),
    }


def _summarize_decomposition(out: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary stats for a decomposition result dict."""
    var_total = np.asarray(out["var_total"], dtype=float)
    var_between = np.asarray(out["var_between"], dtype=float)
    ratio = np.asarray(out["ratio"], dtype=float)

    sum_total = float(np.sum(var_total))
    sum_between = float(np.sum(var_between))

    return {
        "mean_ratio": float(np.nanmean(ratio)) if ratio.size else 0.0,
        "median_ratio": float(np.nanmedian(ratio)) if ratio.size else 0.0,
        "p90_ratio": float(np.nanpercentile(ratio, 90)) if ratio.size else 0.0,
        "ratio_of_sums": float(sum_between / max(sum_total, EPS_RATIO)),
    }


# ---------------------------------------------------------------------------
# Option A: family-first importance + within-family HP importance
# ---------------------------------------------------------------------------

def compute_family_importance(
    meta: pd.DataFrame,
    preds: np.ndarray,
    *,
    family_col: str = "model_name",
    obs_mask: Optional[np.ndarray] = None,
    min_group_size: int = 1,
) -> pd.DataFrame:
    """
    Importance of model family (single factor), using variance decomposition.

    Returns a one-row DataFrame with summary statistics for the *factor*.
    """
    if family_col not in meta.columns:
        raise ValueError(f"family_col='{family_col}' not found in meta columns")

    preds = np.asarray(preds)
    if preds.shape[0] != len(meta):
        raise ValueError("preds axis 0 must match len(meta)")

    preds_use = _subset_observations(preds, obs_mask)
    group_keys = meta[family_col].astype(str).values

    out = variance_decomposition_by_groups(preds_use, group_keys, min_group_size=min_group_size)
    summ = _summarize_decomposition(out)

    return pd.DataFrame([{
        "factor": family_col,
        "n_models_total": int(preds.shape[0]),
        "n_models_used": int(out["n_models_used"]),
        "n_groups": int(out["n_groups"]),
        **summ,
    }])


def compute_within_family_hp_importance(
    meta: pd.DataFrame,
    preds: np.ndarray,
    hp_cols: Optional[Sequence[str]] = None,
    *,
    family_col: str = "model_name",
    obs_mask: Optional[np.ndarray] = None,
    min_models_per_family: int = 3,
    min_groups: int = 2,
    min_group_size: int = 1,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Within each family, compute hyperparameter importance (variance decomposition).

    This is the core of "Option A": first condition on family, then analyze HPs.
    Missing values (NaN) for a given HP are excluded by default (dropna=True),
    so "nan" does not become a spurious group.

    Parameters
    ----------
    meta : DataFrame
        Must contain family_col and hp_* columns (or an 'hp' column to be expanded).
    preds : ndarray
        Shape (n_models, n_obs), aligned with meta rows.
    hp_cols : list of str, optional
        Hyperparameter names or columns. If None, use all hp_* columns.
    family_col : str
        Column name for family grouping (default: model_name).
    obs_mask : optional
        Restrict decomposition to subset of observations (e.g., HH points).
    min_models_per_family : int
        Skip families with fewer models than this.
    min_groups : int
        Require at least this many unique HP values to compute importance.
    min_group_size : int
        Drop HP value groups with fewer models than this.
    dropna : bool
        If True, exclude missing HP values.

    Returns
    -------
    DataFrame with one row per (family, hp) and summary stats.
    """
    meta = ensure_hp_columns(meta)
    preds = np.asarray(preds)

    if preds.shape[0] != len(meta):
        raise ValueError("preds axis 0 must match len(meta)")

    if family_col not in meta.columns:
        raise ValueError(f"family_col='{family_col}' not found in meta columns")

    preds_use = _subset_observations(preds, obs_mask)

    # Determine hp columns
    if hp_cols is None:
        hp_cols_resolved = [c for c in meta.columns if c.startswith("hp_")]
        hp_names = [c.replace("hp_", "") for c in hp_cols_resolved]
    else:
        hp_cols_resolved = []
        hp_names = []
        for h in hp_cols:
            col = _resolve_hp_column(meta, str(h))
            if col is None:
                continue
            hp_cols_resolved.append(col)
            hp_names.append(col.replace("hp_", "") if col.startswith("hp_") else col)

    if not hp_cols_resolved:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    families = meta[family_col].dropna().unique()

    for fam in families:
        mask_f = (meta[family_col] == fam).values
        if int(mask_f.sum()) < min_models_per_family:
            continue

        P_f = preds_use[mask_f]
        meta_f = meta.loc[mask_f].reset_index(drop=True)

        for hp_name, hp_col in zip(hp_names, hp_cols_resolved):
            if hp_col not in meta_f.columns:
                continue

            keys = np.array([make_hp_key(v) for v in meta_f[hp_col].values], dtype=object)

            if dropna:
                valid_models = keys != "nan"
                if int(valid_models.sum()) < min_models_per_family:
                    continue
                P_hp = P_f[valid_models]
                keys_hp = keys[valid_models]
            else:
                P_hp = P_f
                keys_hp = keys

            unique_keys = pd.unique(keys_hp)
            if len(unique_keys) < min_groups:
                continue

            out = variance_decomposition_by_groups(
                P_hp,
                keys_hp,
                min_group_size=min_group_size,
            )

            if out["n_groups"] < min_groups:
                continue

            summ = _summarize_decomposition(out)
            rows.append({
                "family": fam,
                "hp": hp_name,
                "hp_col": hp_col,
                "n_models_total_family": int(P_f.shape[0]),
                "n_models_used": int(out["n_models_used"]),
                "n_groups": int(out["n_groups"]),
                **summ,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["family", "ratio_of_sums"], ascending=[True, False]).reset_index(drop=True)
    return df