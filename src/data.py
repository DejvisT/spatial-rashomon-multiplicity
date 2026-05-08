from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Data directory relative to project root (src/data.py -> parent.parent/data)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml


# ---------------------------------------------------------------------
# Splits (index-based)
# ---------------------------------------------------------------------

def make_split(
    n_samples: int,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,  # fraction of FULL dataset
    seed: int = 42,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Create a train/val/test split using indices only.

    val_size is relative to the FULL dataset.
    """

    if stratify is not None:
        stratify = np.asarray(stratify)

    indices = np.arange(n_samples)

    # First split: test vs rest
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # Convert val_size to be relative to train_val
    val_size_rel = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_rel,
        random_state=seed,
        stratify=stratify[train_val_idx] if stratify is not None else None,
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "seed": seed,
    }


def make_split_with_fixed_test(
    n_samples: int,
    *,
    fixed_test_idx: np.ndarray,
    val_size: float = 0.2,  # fraction of FULL dataset
    seed: int = 42,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Create a train/val/test split where the test set is fixed.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    fixed_test_idx : array-like
        Absolute indices of the test set to keep fixed across runs.
    val_size : float
        Validation fraction relative to the full dataset.
    seed : int
        Random seed used only for train/val split.
    stratify : Optional[np.ndarray]
        Optional labels for stratified train/val split.
    """
    indices = np.arange(n_samples)
    fixed_test_idx = np.asarray(fixed_test_idx, dtype=int)
    fixed_test_idx = np.unique(fixed_test_idx)

    if np.any(fixed_test_idx < 0) or np.any(fixed_test_idx >= n_samples):
        raise ValueError("fixed_test_idx contains out-of-range indices")
    if len(fixed_test_idx) == 0:
        raise ValueError("fixed_test_idx must not be empty")

    if stratify is not None:
        stratify = np.asarray(stratify)

    train_val_idx = np.setdiff1d(indices, fixed_test_idx, assume_unique=False)
    if len(train_val_idx) == 0:
        raise ValueError("No samples left for train/val after fixing test set")

    # Convert full-dataset val fraction to fraction of train_val pool
    val_size_rel = val_size * n_samples / len(train_val_idx)
    val_size_rel = float(np.clip(val_size_rel, 1.0 / len(train_val_idx), 1.0 - 1.0 / len(train_val_idx)))

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_rel,
        random_state=seed,
        stratify=stratify[train_val_idx] if stratify is not None else None,
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": fixed_test_idx,
        "seed": seed,
    }



# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_compas() -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Load COMPAS dataset with a fixed feature set.
    """
    df = pd.read_csv(_DATA_DIR / "compas-scores-two-years.csv")

    feature_info = {
        "numeric": ["age", "priors_count"],
        "categorical": ["sex", "race", "c_charge_degree"],
    }

    X = df[feature_info["numeric"] + feature_info["categorical"]].copy()
    y = df["two_year_recid"].astype(int)

    return X, y, feature_info


def load_german_credit() -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Load German Credit dataset from OpenML (credit-g).
    """

    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.frame.copy()

    # Target column in OpenML version is "class"
    # Values are "good" and "bad"
    df["credit_risk"] = (df["class"] == "good").astype(int)
    df = df.drop(columns=["class"])

    target = "credit_risk"

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in df.columns if c not in categorical + [target]]

    feature_info = {
        "numeric": numeric,
        "categorical": categorical,
    }

    X = df[numeric + categorical].copy()
    y = df[target].astype(int)

    return X, y, feature_info


def load_adult() -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Load Adult (Census Income) dataset from OpenML (adult, data_id=1590).
    Target: income >50K (1) vs <=50K (0).
    """
    data = fetch_openml(data_id=1590, as_frame=True)
    df = data.frame.copy()

    # Target may be in frame as 'class' or 'income', or in data.target
    target_series = None
    if "class" in df.columns:
        target_series = (df["class"].astype(str).str.strip() == ">50K").astype(int)
        df = df.drop(columns=["class"])
    elif "income" in df.columns:
        target_series = (df["income"].astype(str).str.strip() == ">50K").astype(int)
        df = df.drop(columns=["income"])
    if target_series is None and hasattr(data, "target") and data.target is not None:
        target_series = (pd.Series(data.target).astype(str).str.strip() == ">50K").astype(int)
    if target_series is None:
        raise ValueError("Could not find target column (class/income) in Adult dataset")

    y = target_series.astype(int)
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in df.columns if c not in categorical]

    feature_info = {
        "numeric": numeric,
        "categorical": categorical,
    }

    X = df[numeric + categorical].copy()
    return X, y, feature_info


def load_dataset(
    name: str,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Unified dataset loader.
    """
    name = name.lower()

    if name == "compas":
        return load_compas()
    elif name == "german":
        return load_german_credit()
    elif name == "adult":
        return load_adult()
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------
# Preprocessing (definition only — no fitting)
# ---------------------------------------------------------------------

def make_preprocessor(
    feature_info: Dict[str, List[str]],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Define a preprocessing transformer.

    IMPORTANT:
    - This function only DEFINES the transformer.
    - It must be fit on TRAIN DATA ONLY.
    """

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_info["numeric"]),
            ("cat", categorical_transformer, feature_info["categorical"]),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------
# Utility: apply split safely
# ---------------------------------------------------------------------

def apply_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: Dict[str, np.ndarray],
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Apply an index-based split to X and y.

    Returns dict with keys: train / val / test
    """
    return {
        "train": (X.iloc[split["train"]], y.iloc[split["train"]]),
        "val": (X.iloc[split["val"]], y.iloc[split["val"]]),
        "test": (X.iloc[split["test"]], y.iloc[split["test"]]),
    }


# ---------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------

def make_cv_splits(
    n_samples: int,
    *,
    cv_method: str = "kfold",  # "kfold" or "repeated_holdout"
    n_folds: int = 5,
    n_repeats: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
    """
    Create CV splits for train/val (excluding test set).
    
    Returns:
        - test_split: Dict with "test" indices (held out)
        - cv_splits: List of dicts, each with "train" and "val" indices for CV
    
    cv_method:
        - "kfold": K-fold cross-validation on train+val data
        - "repeated_holdout": n_repeats random train/val splits
    """
    if stratify is not None:
        stratify = np.asarray(stratify)
    
    indices = np.arange(n_samples)
    
    # First, hold out test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    
    test_split = {"test": test_idx, "seed": seed}
    
    # Create CV splits on train_val data
    cv_splits = []
    
    if cv_method == "kfold":
        if stratify is not None:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = cv.split(train_val_idx, stratify[train_val_idx])
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = cv.split(train_val_idx)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            cv_splits.append({
                "train": train_val_idx[train_idx],
                "val": train_val_idx[val_idx],
                "fold": fold_idx,
            })
    
    elif cv_method == "repeated_holdout":
        val_size_rel = 0.2  # 20% of train_val for validation in each repeat
        rng = np.random.RandomState(seed)
        
        for repeat_idx in range(n_repeats):
            if stratify is not None:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_rel,
                    random_state=rng.randint(0, 2**31),
                    stratify=stratify[train_val_idx],
                )
            else:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_rel,
                    random_state=rng.randint(0, 2**31),
                )
            
            cv_splits.append({
                "train": train_idx,
                "val": val_idx,
                "fold": repeat_idx,
            })
    
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}. Use 'kfold' or 'repeated_holdout'")
    
    return test_split, cv_splits
