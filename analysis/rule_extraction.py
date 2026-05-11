"""
Interpretable rule extraction for HH (High-High) hotspot components.

Module provides:
1. Global HH-vs-non-HH rule extraction on raw/readable features
2. Component-level one-vs-rest rules for connected HH components
3. OOB bootstrap evaluation for rule stability
4. Permutation tests for statistical significance
5. Feature frequency summaries
6. Output table generation

All rules are learned on raw (non-scaled) features for interpretability.
Spatial analysis (LISA, connectivity) uses transformed features.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# Ensure project root on path
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from data import load_dataset  # noqa: E402
from analysis.run_analysis import load_split  # noqa: E402


# =====================================================================
# Load raw features for rule learning
# =====================================================================

def get_raw_test_features(run_dir: Path, dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load raw (non-scaled) test features and feature names.
    
    Returns
    -------
    X_test_raw : pd.DataFrame with raw feature columns
    feature_names : list of feature names (readable format)
    """
    run_dir = Path(run_dir)
    split = load_split(run_dir)
    X, y, feature_info = load_dataset(dataset_name)
    
    test_idx = split["test"]
    X_test_raw = X.iloc[test_idx].reset_index(drop=True)
    feature_names = list(X_test_raw.columns)
    
    return X_test_raw, feature_names


def prepare_X_for_rules(
    X: pd.DataFrame,
    max_categories: int = 10,
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare raw feature matrix for decision tree training.

    Numeric columns are preserved and categorical columns are one-hot encoded.
    """
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    parts = [X[numeric_cols].astype(float)]
    names = list(numeric_cols)

    for col in cat_cols:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
        if dummies.shape[1] > max_categories:
            top = X[col].value_counts().head(max_categories).index.tolist()
            mapped = X[col].apply(lambda x: x if x in top else "_other_")
            dummies = pd.get_dummies(mapped, prefix=col)
        parts.append(dummies)
        names.extend(dummies.columns.tolist())

    X_arr = np.hstack([part.values for part in parts]) if parts else np.empty((len(X), 0))
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return X_arr, names


# =====================================================================
# Decision tree rules extraction
# =====================================================================

def _format_threshold(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return str(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-8:
        return str(int(rounded))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_rules_from_tree_paths(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
) -> Dict[int, Dict[str, Any]]:
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right

    paths: Dict[int, Dict[str, Any]] = {}

    def recurse(node: int, conditions: List[str]):
        if children_left[node] == children_right[node]:
            rule_text = " AND ".join(conditions) if conditions else "ALL"
            features_used = [cond.split()[0] for cond in conditions]
            features_used = list(dict.fromkeys(features_used))
            paths[node] = {
                "rule_text": rule_text,
                "features_used": features_used,
            }
            return

        feat_idx = feature[node]
        if feat_idx < 0 or feat_idx >= len(feature_names):
            recurse(children_left[node], conditions)
            recurse(children_right[node], conditions)
            return

        name = feature_names[int(feat_idx)]
        thresh = _format_threshold(threshold[node])
        left_cond = f"{name} <= {thresh}"
        right_cond = f"{name} > {thresh}"

        recurse(children_left[node], conditions + [left_cond])
        recurse(children_right[node], conditions + [right_cond])

    recurse(0, [])
    return paths


def extract_rules_from_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    X: pd.DataFrame,
    y: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Extract positive-leaf rules from a decision tree with metrics.
    
    For each positive leaf (class=1), evaluate on training data
    and compute support, purity, recall, lift.
    """
    rules = []
    
    if len(X) == 0 or len(y) == 0:
        return rules
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    leaf_id = tree.apply(X_arr)
    
    n_pos = int(y.sum())
    n_total = len(y)
    base_rate = n_pos / n_total if n_total > 0 else 0.0
    
    rule_paths = _extract_rules_from_tree_paths(tree, feature_names)
    
    unique_leaves = np.unique(leaf_id)
    
    for leaf in unique_leaves:
        leaf_mask = (leaf_id == leaf)
        support = int(leaf_mask.sum())
        support_in_pos = int(y[leaf_mask].sum())
        
        if support_in_pos == 0:
            continue
        
        purity = support_in_pos / support if support > 0 else 0.0
        recall = support_in_pos / n_pos if n_pos > 0 else 0.0
        lift = purity / base_rate if base_rate > 0 else 0.0

        path_info = rule_paths.get(leaf, {})
        rule_text = path_info.get("rule_text", "ALL")
        features_used = path_info.get("features_used", [])
        if not features_used:
            # Fallback to feature importances if no explicit path features
            feat_imp = tree.feature_importances_
            features_used = [
                feature_names[i]
                for i in np.argsort(feat_imp)[::-1]
                if i < len(feature_names) and feat_imp[i] > 1e-6
            ]
        rule_features = ", ".join(features_used)

        rules.append({
            "rule_text": rule_text,
            "features_used": features_used,
            "rule_features": rule_features,
            "support": support,
            "support_frac": support / n_total if n_total > 0 else 0.0,
            "purity": float(purity),
            "recall": float(recall),
            "lift": float(lift),
            "base_rate": float(base_rate),
            "n_hh_total": int(n_pos),
            "n_total": n_total,
        })
    
    rules = sorted(rules, key=lambda x: x["support"], reverse=True)
    return rules


def fit_tree_and_extract_rules(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 10,
    seed: int = 42,
) -> Tuple[DecisionTreeClassifier, List[Dict[str, Any]]]:
    """
    Fit decision tree on raw features and extract positive-leaf rules.
    
    Parameters
    ----------
    X_raw : raw feature DataFrame
    y : binary target (0/1)
    feature_names : feature names
    max_depth : tree max depth
    min_samples_leaf : minimum samples per leaf
    seed : random seed
    
    Returns
    -------
    tree : fitted DecisionTreeClassifier
    rules : list of rule dicts with metrics
    """
    X_processed, prepared_feature_names = prepare_X_for_rules(X_raw)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    tree.fit(X_processed, y)
    
    # Extract rules with metrics
    X_processed_df = pd.DataFrame(X_processed, columns=prepared_feature_names)
    rules = extract_rules_from_tree(tree, prepared_feature_names, X_processed_df, y)
    
    return tree, rules


# =====================================================================
# Global HH-vs-non-HH analysis
# =====================================================================

def analyze_hh_global(
    X_raw: pd.DataFrame,
    hh_mask: np.ndarray,
    feature_names: List[str],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Global analysis: distinguish HH from non-HH using decision tree.
    
    Returns
    -------
    rules_df : DataFrame with rule information
    metadata : dict with context info
    """
    y = hh_mask.astype(int)
    n_hh = y.sum()
    
    tree, rules = fit_tree_and_extract_rules(
        X_raw, y, feature_names,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    
    # Convert rules to DataFrame
    rows = []
    for rank, rule_dict in enumerate(rules):
        row = {
            "rule_rank": rank,
            "method": "tree",
            "label": "HH",
        }
        row.update(rule_dict)
        rows.append(row)
    
    rules_df = pd.DataFrame(rows)
    
    metadata = {
        "tree": tree,
        "hh_rate": y.mean(),
        "n_hh": n_hh,
        "n_total": len(y),
    }
    
    return rules_df, metadata


# =====================================================================
# Component-level analysis
# =====================================================================

def analyze_component(
    X_raw: pd.DataFrame,
    feature_names: List[str],
    component_mask: np.ndarray,
    component_size: int,
    all_hh_mask: np.ndarray,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Analyze a single HH component with one-vs-rest classification.
    
    Parameters
    ----------
    X_raw : raw feature DataFrame
    feature_names : feature names
    component_mask : boolean mask for this component
    component_size : size of component
    all_hh_mask : mask for all HH points (used for metrics)
    max_depth, min_samples_leaf, seed : tree hyperparameters
    
    Returns
    -------
    rules_df : DataFrame with rule information or None
    metadata : dict with context info
    """
    y = component_mask.astype(int)
    n_component = y.sum()
    
    if n_component < 2:
        return None, {"error": "component too small", "n_component": n_component}
    
    tree, rules = fit_tree_and_extract_rules(
        X_raw, y, feature_names,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    
    # Compute additional metrics
    n_total = len(y)
    n_all_hh = all_hh_mask.sum()
    component_base_rate = n_component / n_total if n_total > 0 else 0.0
    hh_base_rate = n_all_hh / n_total if n_total > 0 else 0.0
    
    rows = []
    for rank, rule_dict in enumerate(rules):
        row = {
            "rule_rank": rank,
            "component_support": rule_dict.get("support", 0),
            "component_purity": rule_dict.get("purity", 0.0),
            "component_recall": rule_dict.get("recall", 0.0),
            "component_lift": rule_dict.get("lift", 0.0),
            "component_base_rate": component_base_rate,
        }
        
        # Preserve existing rule metrics too
        row.update(rule_dict)
        rows.append(row)

    rules_df = pd.DataFrame(rows) if rows else None
    
    metadata = {
        "tree": tree,
        "component_size": n_component,
        "n_test": n_total,
        "component_rate": component_base_rate,
        "n_all_hh": n_all_hh,
    }
    
    return rules_df, metadata

# =====================================================================
# Multi-seed aggregation
# =====================================================================

def concat_with_seed(
    list_of_dfs: List[pd.DataFrame],
    seed_list: List[int],
) -> pd.DataFrame:
    """
    Concatenate list of DataFrames and add outer_seed column.
    """
    if not list_of_dfs:
        return pd.DataFrame()
    
    dfs_with_seed = []
    for df, seed in zip(list_of_dfs, seed_list):
        if df is not None and len(df) > 0:
            df_copy = df.copy()
            df_copy["outer_seed"] = seed
            dfs_with_seed.append(df_copy)
    
    if dfs_with_seed:
        return pd.concat(dfs_with_seed, ignore_index=True)
    else:
        return pd.DataFrame()


def recurring_rules_by_method_label(
    rules_summary: pd.DataFrame,
    min_seeds: int = 3,
) -> pd.DataFrame:
    """
    Find rules that appear across multiple seeds.
    """
    if "rule_text" not in rules_summary.columns:
        return pd.DataFrame()
    
    rule_counts = rules_summary.groupby("rule_text").agg({
        "outer_seed": "nunique",
        "support": ["min", "mean", "max"],
        "purity": ["min", "mean", "max"],
    }).reset_index()
    
    rule_counts.columns = ["rule_text", "n_seeds", "support_min", "support_mean", "support_max",
                           "purity_min", "purity_mean", "purity_max"]
    
    recurring = rule_counts[rule_counts["n_seeds"] >= min_seeds]
    recurring = recurring.sort_values("n_seeds", ascending=False)
    
    return recurring


def rule_feature_frequency_across_seeds(
    rules_summary: pd.DataFrame,
    known_feature_names: Optional[List[str]] = None,
    min_purity: float = 0.5,
) -> pd.DataFrame:
    """
    Compute feature frequency across seeds and rules.
    """
    if "features_used" not in rules_summary.columns or len(rules_summary) == 0:
        return pd.DataFrame()
    
    feature_stats = {}
    
    for idx, row in rules_summary.iterrows():
        features = row.get("features_used", [])
        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]
        else:
            features = list(features) if features else []
        
        purity = float(row.get("purity", 0.0))
        lift = float(row.get("lift", 0.0)) if "lift" in row else 0.0
        
        for feat in features:
            if feat not in feature_stats:
                feature_stats[feat] = {
                    "n_rules": 0,
                    "n_seeds": set(),
                    "purities": [],
                    "lifts": [],
                }
            feature_stats[feat]["n_rules"] += 1
            feature_stats[feat]["n_seeds"].add(row.get("outer_seed", 0))
            feature_stats[feat]["purities"].append(purity)
            feature_stats[feat]["lifts"].append(lift)
    
    rows = []
    for feat, stats in sorted(feature_stats.items()):
        rows.append({
            "feature": feat,
            "n_rules_with_feature": stats["n_rules"],
            "n_seeds_with_feature": len(stats["n_seeds"]),
            "mean_purity_when_used": float(np.mean(stats["purities"])) if stats["purities"] else 0.0,
            "mean_lift_when_used": float(np.mean(stats["lifts"])) if stats["lifts"] else 0.0,
        })
    
    return pd.DataFrame(rows).sort_values(
        ["n_seeds_with_feature", "n_rules_with_feature", "mean_purity_when_used"],
        ascending=False,
    )