"""
Main training pipeline: 60/20/20 stratified splits, 50 candidates per family,
deterministic seeds, preprocessing fit on train only. No Rashomon selection.

LogReg uses elastic net (penalty='elasticnet', solver='saga', l1_ratio). HP
samples are deduplicated per family so multiplicity is not artificially
reduced for LogReg/kNN; if fewer than 50 unique configs, we resample until
we have 50.

Saves per run: split indices, hyperparameters, validation Brier, validation
and test predicted probabilities (P_val, P_test), model family label;
optionally trained pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Model configs aligned with experimental details
# LogReg: elastic net (saga, l1_ratio); RF: bootstrap True; MLP: adam, early_stopping; GBM: subsample; kNN: standard
# ---------------------------------------------------------------------------

TRAINING_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LogReg": {
        "class": LogisticRegression,
        "params": {
            "C": np.logspace(-5, 5, 20).tolist(),
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "max_iter": [2000],
            "tol": [1e-4],
        },
    },
    "kNN": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": [3, 5, 7, 10, 15, 20, 25, 30, 50, 70, 90, 100, 150],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    },
    "RF": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True],
            "n_jobs": [-1],
        },
    },
    "GBM": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.6, 0.8, 1.0],
        },
    },
    "MLP": {
        "class": MLPClassifier,
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (200, 100)],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 1e-3, 1e-2],
            "activation": ["relu", "tanh"],
            "solver": ["adam"],
            "early_stopping": [True],
            "n_iter_no_change": [10],
            "max_iter": [1000],
        },
    },
}

def _fit_and_evaluate_candidate(
    *,
    model_name: str,
    model_cfg: Dict[str, Any],
    hp: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    candidate_seed: int,
) -> Tuple[Pipeline, float, np.ndarray, np.ndarray]:
    """
    Fit one candidate on train; return pipeline, validation Brier, test probs.
    Preprocessor is already fit on train; we clone it per pipeline so each
    saved pipeline is self-contained (clone is fit on same train data).
    """
    from sklearn.base import clone

    ModelClass = model_cfg["class"]
    model = ModelClass(**hp)
    if "random_state" in model.get_params():
        model.set_params(random_state=candidate_seed)

    # Clone preprocessor so each pipeline is self-contained when saved
    pipe = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_val)[:, 1]
    val_brier = brier_score_loss(y_val, val_probs)

    test_probs = pipe.predict_proba(X_test)[:, 1]
    return pipe, val_brier, val_probs, test_probs


def run_one_training_run(
    X: pd.DataFrame,
    y: pd.Series,
    feature_info: Dict[str, List[str]],
    preprocessor_factory,
    *,
    outer_seed: int,
    n_candidates_per_family: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.2,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    families: Optional[List[str]] = None,
    save_pipelines: bool = False,
    verbose: int = 1,
    split: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, np.ndarray, np.ndarray, Optional[List[Pipeline]]]:
    """
    Run one outer run: one 60/20/20 stratified split, 50 candidates per family,
    deterministic seeds (outer_seed + candidate_id). Preprocessing fit on train only.

    Returns
    -------
    split : dict with keys "train", "val", "test" (indices)
    meta : DataFrame with columns model_name, candidate_id, candidate_seed, hp, val_brier, outer_seed
    P_val : array (n_candidates_total, n_val) validation predicted probabilities
    P_test : array (n_candidates_total, n_test) test predicted probabilities
    pipelines : list of fitted pipelines or None if save_pipelines=False
    """
    from data import make_split

    if model_configs is None:
        model_configs = TRAINING_MODEL_CONFIGS
    if families is None:
        families = list(model_configs.keys())

    if split is None:
        split = make_split(
            n_samples=len(X),
            test_size=test_size,
            val_size=val_size,
            seed=outer_seed,
            stratify=y.values,
        )
    else:
        required = {"train", "val", "test"}
        if not required.issubset(set(split.keys())):
            raise ValueError(f"Provided split must contain keys {required}")
        split = {
            "train": np.asarray(split["train"], dtype=int),
            "val": np.asarray(split["val"], dtype=int),
            "test": np.asarray(split["test"], dtype=int),
            "seed": int(split.get("seed", outer_seed)),
        }
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test = X.iloc[test_idx]

    # Fit preprocessor on train only
    preprocessor = preprocessor_factory(feature_info)
    preprocessor.fit(X_train, y_train)

    rng = np.random.RandomState(outer_seed)
    rows: List[Dict[str, Any]] = []
    P_val_list: List[np.ndarray] = []
    P_test_list: List[np.ndarray] = []
    pipelines_list: List[Pipeline] = [] if save_pipelines else []

    for model_name in families:
        if model_name not in model_configs:
            continue
        cfg = model_configs[model_name]
        # Sample and deduplicate so multiplicity is not artificially reduced (LogReg/kNN)
        hp_samples = list(
            ParameterSampler(
                cfg["params"],
                n_iter=n_candidates_per_family,
                random_state=rng,
            )
        )
        hp_samples = list(
            {json.dumps(_hp_to_json_serializable(hp), sort_keys=True): hp for hp in hp_samples}.values()
        )
        # Resample until we have at least n_candidates_per_family unique configs (with safeguard)
        seen = {json.dumps(_hp_to_json_serializable(hp), sort_keys=True) for hp in hp_samples}
        max_attempts = 10
        attempt = 0
        while len(hp_samples) < n_candidates_per_family and attempt < max_attempts:
            extra = list(
                ParameterSampler(
                    cfg["params"],
                    n_iter=n_candidates_per_family,
                    random_state=rng,
                )
            )
            for hp in extra:
                key = json.dumps(_hp_to_json_serializable(hp), sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    hp_samples.append(hp)
                    if len(hp_samples) >= n_candidates_per_family:
                        break
            attempt += 1
        if len(hp_samples) < n_candidates_per_family:
            raise ValueError(
                f"[{model_name}] After {max_attempts} resample attempts only {len(hp_samples)} unique "
                f"configs (need {n_candidates_per_family}). Enlarge the hyperparameter grid."
            )
        hp_samples = hp_samples[:n_candidates_per_family]

        for candidate_id, hp in enumerate(hp_samples):
            candidate_seed = outer_seed + candidate_id
            pipe, val_brier, val_probs, test_probs = _fit_and_evaluate_candidate(
                model_name=model_name,
                model_cfg=cfg,
                hp=hp,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                candidate_seed=candidate_seed,
            )
            rows.append({
                "model_name": model_name,
                "candidate_id": candidate_id,
                "candidate_seed": candidate_seed,
                "hp": hp,
                "val_brier": val_brier,
            })
            P_val_list.append(val_probs)
            P_test_list.append(test_probs)
            if save_pipelines:
                pipelines_list.append(pipe)
        if verbose >= 1:
            print(f"  [{model_name}] {len(hp_samples)} candidates trained")

    meta = pd.DataFrame(rows)
    meta["outer_seed"] = outer_seed
    P_val = np.vstack(P_val_list)
    P_test = np.vstack(P_test_list)
    pipelines = pipelines_list if save_pipelines else None

    return split, meta, P_val, P_test, pipelines


def _hp_to_json_serializable(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hp dict for JSON (e.g. tuple -> list, np types -> native)."""
    out = {}
    for k, v in hp.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def save_run_artifacts(
    out_dir: Path,
    split: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    P_val: np.ndarray,
    P_test: np.ndarray,
    pipelines: Optional[List[Pipeline]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save artifacts for one run to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train/val/test indices
    np.savez(
        out_dir / "split.npz",
        train=split["train"],
        val=split["val"],
        test=split["test"],
        seed=int(split["seed"]),
    )

    # Meta: model_name, candidate_id, candidate_seed, hp (as JSON string), val_brier, outer_seed
    meta_out = meta.copy()
    meta_out["hp"] = meta_out["hp"].apply(
        lambda d: json.dumps(_hp_to_json_serializable(d))
    )
    meta_out.to_csv(out_dir / "meta.csv", index=False)

    # Validation and test predicted probabilities (n_candidates, n_val) and (n_candidates, n_test)
    np.save(out_dir / "P_val.npy", P_val)
    np.save(out_dir / "P_test.npy", P_test)

    # Optional: trained pipelines
    if pipelines is not None and len(pipelines) > 0:
        import pickle
        pipelines_dir = out_dir / "pipelines"
        pipelines_dir.mkdir(exist_ok=True)
        for i, pipe in enumerate(pipelines):
            with open(pipelines_dir / f"{i}.pkl", "wb") as f:
                pickle.dump(pipe, f)

    if config is not None:
        config_ser = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, type(None)))}
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_ser, f, indent=2)
