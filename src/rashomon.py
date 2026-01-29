"""
Rashomon set construction and analysis.
"""
from typing import Dict, List, Tuple, Any
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


# ---------------------------------------------------------------------
# Build Rashomon set
# ---------------------------------------------------------------------

def build_rashomon_set(
    *,
    X,
    y,
    split: Dict[str, np.ndarray],
    base_preprocessor,
    epsilon: float,
    model_configs: Dict[str, Dict[str, Any]] = DEFAULT_MODEL_CONFIGS,
    n_samples_per_model: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Train many models and return Rashomon predictions on TEST.
    """

    rng = np.random.RandomState(seed)

    X_train, y_train = X.iloc[split["train"]], y.iloc[split["train"]]
    X_val, y_val = X.iloc[split["val"]], y.iloc[split["val"]]
    X_test = X.iloc[split["test"]]

    results = []

    for model_name, cfg in model_configs.items():
        hp_samples = list(
            ParameterSampler(
                cfg["params"],
                n_iter=n_samples_per_model,
                random_state=rng,
            )
        )

        for hp in hp_samples:
            res = fit_and_eval_model(
                model_name=model_name,
                model_cfg=cfg,
                hp=hp,
                preprocessor=base_preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                seed=seed,
            )
            results.append(res)

    results_df = pd.DataFrame(results)

    best_loss = results_df["val_loss"].min()
    rashomon_df = results_df[
        results_df["val_loss"] <= best_loss + epsilon
    ].reset_index(drop=True)

    # Predict on TEST
    P_test = np.vstack(
        [row.pipeline.predict_proba(X_test)[:, 1] for row in rashomon_df.itertuples()]
    )

    meta = rashomon_df.drop(columns="pipeline")

    return P_test, meta