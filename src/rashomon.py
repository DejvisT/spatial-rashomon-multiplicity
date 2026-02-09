"""
Rashomon set construction and analysis.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------
# Default model configurations
# ---------------------------------------------------------------------

DEFAULT_MODEL_CONFIGS = {
    'LogReg': {
        'class': LogisticRegression,
        'params': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [2000]
        }
    },
    'RF': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'n_jobs': [-1]
        }
    },
    'GBM': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.6, 0.8, 1.0]
        }
    },
    'MLP': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50), (200, 100)],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-4, 1e-3, 1e-2],
            'activation': ['relu', 'tanh'],
            'max_iter': [1000]
        }
    }
}

SCALE_BY_DEFAULT = {
    "LogReg": True,
    "MLP": True,
    "RF": False,
    "GBM": False,
}


# ---------------------------------------------------------------------
# Fit + evaluate a single model
# ---------------------------------------------------------------------

def fit_and_eval_model(
    *,
    model_name: str,
    model_cfg: Dict[str, Any],
    hp: Dict[str, Any],
    preprocessor,
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int,
) -> Dict[str, Any]:
    """
    Fit a model on TRAIN and evaluate on VAL.
    """

    ModelClass = model_cfg["class"]

    model = ModelClass(**hp)
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_loss = log_loss(y_val, val_probs)

    return {
        "model_name": model_name,
        "hp": hp,
        "seed": seed,
        "val_loss": val_loss,
        "pipeline": pipeline,
    }


def fit_and_eval_model_cv(
    *,
    model_name: str,
    model_cfg: Dict[str, Any],
    hp: Dict[str, Any],
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List[Dict[str, np.ndarray]],
    seed: int,
) -> Tuple[float, Pipeline]:
    """
    Fit a model using cross-validation and return mean CV loss.
    
    cv_splits should contain train/val indices from the train+val portion (test excluded).
    
    Returns:
        - mean_cv_loss: Mean log loss across all CV folds
        - pipeline: Pipeline fit on all train+val data (for final predictions)
    """
    ModelClass = model_cfg["class"]
    
    cv_losses = []
    
    # Compute CV loss
    for cv_split in cv_splits:
        X_train_cv = X.iloc[cv_split["train"]]
        y_train_cv = y.iloc[cv_split["train"]]
        X_val_cv = X.iloc[cv_split["val"]]
        y_val_cv = y.iloc[cv_split["val"]]
        
        model = ModelClass(**hp)
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)
        
        pipeline_cv = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        
        pipeline_cv.fit(X_train_cv, y_train_cv)
        val_probs = pipeline_cv.predict_proba(X_val_cv)[:, 1]
        cv_loss = log_loss(y_val_cv, val_probs)
        cv_losses.append(cv_loss)
    
    mean_cv_loss = np.mean(cv_losses)
    
    # Fit final pipeline on all train+val data (for test predictions)
    # Combine all train and val indices from CV splits
    all_train_val_indices = []
    for cv_split in cv_splits:
        all_train_val_indices.extend(cv_split["train"].tolist())
        all_train_val_indices.extend(cv_split["val"].tolist())
    train_val_indices = np.unique(all_train_val_indices)
    
    X_train_val = X.iloc[train_val_indices]
    y_train_val = y.iloc[train_val_indices]
    
    model_final = ModelClass(**hp)
    if "random_state" in model_final.get_params():
        model_final.set_params(random_state=seed)
    
    pipeline_final = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model_final),
        ]
    )
    
    pipeline_final.fit(X_train_val, y_train_val)
    
    return mean_cv_loss, pipeline_final


# ---------------------------------------------------------------------
# Build Rashomon set
# ---------------------------------------------------------------------

def build_rashomon_set(
    *,
    X,
    y,
    split: Optional[Dict[str, np.ndarray]] = None,
    test_split: Optional[Dict[str, np.ndarray]] = None,
    cv_splits: Optional[List[Dict[str, np.ndarray]]] = None,
    base_preprocessor,
    epsilon: float,
    model_configs: Dict[str, Dict[str, Any]] = DEFAULT_MODEL_CONFIGS,
    n_samples_per_model: int = 30,
    seed: int = 42,
    selection_mode: str = "global",
    family: str | None = None,
    model_seeds: Optional[List[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Train many models and return Rashomon predictions on TEST.

    Either `split` (single train/val/test) or (`test_split` + `cv_splits`) must be provided.
    If `cv_splits` is provided, uses mean CV loss for Rashomon membership.

    selection_mode:
        - "global": single threshold across all models
        - "per_family": separate threshold per model family
    family:
        - if not None, restrict Rashomon set to a single family
    model_seeds:
        - if not None (and family is set), use these seeds for each model (cycling)
    """

    use_cv = cv_splits is not None

    if use_cv:
        if test_split is None:
            raise ValueError("When using CV, 'test_split' must be provided")
        X_test = X.iloc[test_split["test"]]
    else:
        if split is None:
            raise ValueError("When not using CV, 'split' must be provided")
        X_train, y_train = X.iloc[split["train"]], y.iloc[split["train"]]
        X_val, y_val = X.iloc[split["val"]], y.iloc[split["val"]]
        X_test = X.iloc[split["test"]]

    rng = np.random.RandomState(seed)

    results = []
    model_idx = 0

    # ------------------------------------------------------------
    # Train all candidate models (or only the specified family)
    # ------------------------------------------------------------
    configs_to_train = (
        [(family, model_configs[family])] if family is not None
        else list(model_configs.items())
    )

    for model_name, cfg in configs_to_train:
        hp_samples = list(
            ParameterSampler(
                cfg["params"],
                n_iter=n_samples_per_model,
                random_state=rng,
            )
        )

        for hp in hp_samples:
            if model_seeds is not None:
                model_seed = model_seeds[model_idx % len(model_seeds)]
            else:
                model_seed = seed
            model_idx += 1

            if use_cv:
                mean_cv_loss, pipeline = fit_and_eval_model_cv(
                    model_name=model_name,
                    model_cfg=cfg,
                    hp=hp,
                    preprocessor=base_preprocessor,
                    X=X,
                    y=y,
                    cv_splits=cv_splits,
                    seed=model_seed,
                )
                results.append({
                    "model_name": model_name,
                    "hp": hp,
                    "seed": model_seed,
                    "val_loss": mean_cv_loss,  # Using CV loss
                    "pipeline": pipeline,
                })
            else:
                res = fit_and_eval_model(
                    model_name=model_name,
                    model_cfg=cfg,
                    hp=hp,
                    preprocessor=base_preprocessor,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    seed=model_seed,
                )
                results.append(res)

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------
    # Rashomon selection logic
    # ------------------------------------------------------------
    if family is not None:
        # Family-specific Rashomon
        df_fam = results_df[results_df["model_name"] == family]
        if len(df_fam) == 0:
            raise ValueError(f"No models found for family={family}")

        best_loss = df_fam["val_loss"].min()
        rashomon_df = df_fam[
            df_fam["val_loss"] <= best_loss + epsilon
        ]

    elif selection_mode == "global":
        # Global Rashomon
        best_loss = results_df["val_loss"].min()
        rashomon_df = results_df[
            results_df["val_loss"] <= best_loss + epsilon
        ]

    elif selection_mode == "per_family":
        # One Rashomon threshold per family
        parts = []
        for model_name, group in results_df.groupby("model_name"):
            best_loss_fam = group["val_loss"].min()
            parts.append(
                group[group["val_loss"] <= best_loss_fam + epsilon]
            )
        rashomon_df = pd.concat(parts, ignore_index=True)

    else:
        raise ValueError("selection_mode must be 'global' or 'per_family'")

    rashomon_df = rashomon_df.reset_index(drop=True)

    # ------------------------------------------------------------
    # Predict on TEST
    # ------------------------------------------------------------
    P_test = np.vstack(
        [row.pipeline.predict_proba(X_test)[:, 1]
         for row in rashomon_df.itertuples()]
    )

    meta = rashomon_df.drop(columns="pipeline")

    return P_test, meta
