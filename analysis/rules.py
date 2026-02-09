"""
Interpretable rule extraction for HH (High-High) hotspot components.

For each HH component, fits a shallow decision tree to distinguish component
members from the rest of the data. Reports precision, recall, and rule text.
Supports out-of-sample evaluation via train/eval split.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Prepare features for rule fitting
# ---------------------------------------------------------------------

def prepare_X_for_rules(
    X: pd.DataFrame,
    max_categories: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare feature matrix for decision tree with interpretable column names.

    Converts categorical columns to one-hot encoding. Drops columns with
    too many categories to keep rules interpretable.

    Parameters
    ----------
    X : raw feature DataFrame
    max_categories : max one-hot columns per original categorical (default 10)

    Returns
    -------
    X_arr : array of shape (n_obs, n_features)
    feature_names : list of feature names
    """
    X = X.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    parts = [X[numeric_cols].astype(float)]
    names = list(numeric_cols)

    for col in cat_cols:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
        if dummies.shape[1] > max_categories:
            # Keep only top categories by frequency
            top = X[col].value_counts().head(max_categories).index.tolist()
            mapped = X[col].apply(lambda x: x if x in top else "_other_")
            dummies = pd.get_dummies(mapped, prefix=col)
        parts.append(dummies)
        names.extend(dummies.columns.tolist())

    X_arr = np.hstack([p.values for p in parts])
    # Handle any NaN
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return X_arr, names


# ---------------------------------------------------------------------
# Fit rule for a single HH component
# ---------------------------------------------------------------------

def fit_component_rule(
    X: np.ndarray,
    feature_names: List[str],
    in_component: np.ndarray,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    test_size: float = 0.3,
    min_samples_for_split: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Fit a shallow decision tree to describe an HH component.

    Uses train/eval split when possible for out-of-sample precision/recall.
    Falls back to in-sample metrics when too few points.

    Parameters
    ----------
    X : feature matrix (n_obs, n_features)
    feature_names : list of feature names
    in_component : boolean mask of shape (n_obs,) for component membership
    max_depth : max tree depth
    min_samples_leaf : min samples per leaf
    test_size : fraction for eval set (for out-of-sample)
    min_samples_for_split : need at least this many HH points to split
    seed : random seed

    Returns
    -------
    dict with keys:
        - tree: fitted DecisionTreeClassifier
        - rule_text: str (exported tree)
        - precision_train, recall_train
        - precision_eval, recall_eval (or None if no split)
        - n_train, n_eval
        - out_of_sample: bool
    """
    y = in_component.astype(int)
    n_hh = y.sum()
    n_total = len(y)

    if n_hh < 2:
        return {
            "tree": None,
            "rule_text": "(no rule: component too small)",
            "precision_train": 0.0,
            "recall_train": 0.0,
            "precision_eval": None,
            "recall_eval": None,
            "n_train": 0,
            "n_eval": 0,
            "out_of_sample": False,
        }

    # Decide whether to split for out-of-sample
    can_split = n_hh >= min_samples_for_split or n_hh >= 10

    if can_split and n_hh >= 5:
        # Stratified split to preserve HH ratio
        idx = np.arange(n_total)
        idx_train, idx_eval = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        X_train, X_eval = X[idx_train], X[idx_eval]
        y_train, y_eval = y[idx_train], y[idx_eval]
        out_of_sample = True
    else:
        idx_train = np.arange(n_total)
        idx_eval = None
        X_train, y_train = X, y
        X_eval, y_eval = None, None
        out_of_sample = False

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        class_weight="balanced",
    )
    tree.fit(X_train, y_train)

    # Metrics on train
    train_pred = tree.predict(X_train)
    train_recall = _recall(y_train, train_pred)
    train_precision = _precision(y_train, train_pred)

    # Metrics on eval (if split)
    eval_precision, eval_recall = None, None
    if out_of_sample and X_eval is not None:
        eval_pred = tree.predict(X_eval)
        eval_recall = _recall(y_eval, eval_pred)
        eval_precision = _precision(y_eval, eval_pred)

    # Rule text
    try:
        rule_text = export_text(
            tree,
            feature_names=feature_names,
            max_depth=max_depth,
            decimals=2,
        )
    except Exception:
        rule_text = "(could not export)"

    return {
        "tree": tree,
        "rule_text": rule_text,
        "precision_train": float(train_precision),
        "recall_train": float(train_recall),
        "precision_eval": float(eval_precision) if eval_precision is not None else None,
        "recall_eval": float(eval_recall) if eval_recall is not None else None,
        "n_train": len(idx_train),
        "n_eval": len(idx_eval) if idx_eval is not None else 0,
        "out_of_sample": out_of_sample,
    }


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_pos = y_pred.sum()
    if pred_pos == 0:
        return 0.0
    return (y_true & (y_pred.astype(bool))).sum() / pred_pos


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_pos = y_true.sum()
    if true_pos == 0:
        return 0.0
    return (y_true & (y_pred.astype(bool))).sum() / true_pos


# ---------------------------------------------------------------------
# Extract rules for all HH components
# ---------------------------------------------------------------------

def extract_component_rules(
    X: pd.DataFrame,
    components: Dict[int, np.ndarray],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    test_size: float = 0.3,
    seed: int = 42,
) -> Dict[int, Dict]:
    """
    For each HH component, fit an interpretable rule and return metrics.

    Parameters
    ----------
    X : raw feature DataFrame (same rows as LISA)
    components : dict mapping comp_id -> array of indices
    max_depth, min_samples_leaf, test_size, seed : rule hyperparameters

    Returns
    -------
    rules : dict mapping comp_id -> rule result dict
    """
    X_arr, feature_names = prepare_X_for_rules(X)
    n = X_arr.shape[0]

    rules = {}
    for cid, indices in components.items():
        in_comp = np.zeros(n, dtype=bool)
        in_comp[indices] = True

        res = fit_component_rule(
            X_arr,
            feature_names,
            in_comp,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            test_size=test_size,
            seed=seed,
        )
        res["n_component"] = len(indices)
        res["comp_id"] = cid
        rules[cid] = res

    return rules


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def rules_summary_df(rules: Dict[int, Dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame of rule quality per component.
    """
    if not rules:
        return pd.DataFrame(
            columns=[
                "component", "n_hh", "precision_train", "recall_train",
                "precision_eval", "recall_eval", "out_of_sample",
            ]
        )
    rows = []
    for cid, r in rules.items():
        rows.append({
            "component": cid,
            "n_hh": r["n_component"],
            "precision_train": r["precision_train"],
            "recall_train": r["recall_train"],
            "precision_eval": r["precision_eval"],
            "recall_eval": r["recall_eval"],
            "out_of_sample": r["out_of_sample"],
        })
    return pd.DataFrame(rows)
