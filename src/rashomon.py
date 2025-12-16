"""
Rashomon set construction and analysis.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
import joblib


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


def save_rashomon_result(rashomon_result: Dict[str, Any], path: str) -> str:
    """
    Save a Rashomon set result (including fitted sklearn models) to disk via joblib.

    Parameters
    ----------
    rashomon_result : dict
        Output of build_rashomon_set()
    path : str
        Output filepath, e.g. "artifacts/rashomon_compas.joblib"

    Returns
    -------
    str
        The resolved path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rashomon_result, p)
    return str(p)


def load_rashomon_result(path: str) -> Dict[str, Any]:
    """
    Load a saved Rashomon set result from disk (joblib).
    """
    return joblib.load(path)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_configs: Dict = None,
    seeds: List[int] = [0, 1, 2],
    metric: str = 'auc',
    verbose: bool = True,
    n_iter: int = 25,
    search_random_state: int = 0,
) -> Dict:
    """
    Train all model configurations and compute performance scores.
    
    Parameters
    ----------
    X_train, y_train : training data (should be scaled)
    X_test, y_test : test data (should be scaled)
    model_configs : dict of model configurations (uses defaults if None)
    seeds : random seeds to try for each configuration
    metric : 'auc' or 'accuracy'
    verbose : print progress
    n_iter : number of random hyperparameter samples per model type
    search_random_state : seed for hyperparameter sampling (does not affect model seeds)
    
    Returns
    -------
    dict with keys:
        - models: list of trained models
        - scores: list of performance scores
        - params: list of parameter dictionaries
    """
    configs = model_configs or DEFAULT_MODEL_CONFIGS
    
    all_models = []
    all_scores = []
    all_params = []
    
    for model_name, config in configs.items():
        if verbose:
            print(f"   Training {model_name}...")
        
        model_class = config['class']
        param_grid = config['params']

        # Random search: sample hyperparameter configs instead of full cartesian product
        sampled_params = ParameterSampler(
            param_grid,
            n_iter=int(n_iter),
            random_state=int(search_random_state),
        )

        for param_dict in sampled_params:
            
            for seed in seeds:
                try:
                    model = model_class(random_state=seed, **param_dict)
                    model.fit(X_train, y_train)
                    
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    if metric == 'auc':
                        score = roc_auc_score(y_test, y_prob)
                    else:
                        score = (y_prob.round() == y_test).mean()
                    
                    all_models.append(model)
                    all_scores.append(score)
                    all_params.append({
                        'model_type': model_name,
                        'seed': seed,
                        **param_dict
                    })
                except Exception:
                    continue
    
    if verbose:
        print(f"   ✅ Trained {len(all_models)} models")
    
    return {
        'models': all_models,
        'scores': np.array(all_scores),
        'params': all_params
    }


def build_rashomon_set(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epsilon: float = 0.01,
    model_configs: Dict = None,
    seeds: List[int] = [0, 1, 2],
    metric: str = 'auc',
    scale: bool = True,
    verbose: bool = True,
    n_iter: int = 25,
    search_random_state: int = 0,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train multiple models and build Rashomon set.
    
    The Rashomon set contains all models whose performance is within
    epsilon of the best model:
    
        R_ε = {f ∈ F : score(f) ≥ score(f*) - ε}
    
    Parameters
    ----------
    X_train, X_test : feature matrices
    y_train, y_test : target arrays
    epsilon : performance tolerance for Rashomon set
    model_configs : dict of model configurations (uses defaults if None)
    seeds : random seeds to try for each configuration
    metric : 'auc' or 'accuracy'
    scale : whether to apply StandardScaler to features
    verbose : print progress
    n_iter : number of random hyperparameter samples PER model type
    search_random_state : RNG seed for hyperparameter sampling
    save_path : if provided, save the Rashomon result (including models) to this joblib file
    
    Returns
    -------
    dict with keys:
        - models: list of trained models in Rashomon set
        - params: list of parameter dicts for Rashomon models
        - scores: array of performance scores for Rashomon models
        - pred_matrix: (n_models, n_samples) prediction probability matrix
        - best_score: best model score
        - threshold: Rashomon threshold (best_score - epsilon)
        - epsilon: epsilon value used
        - scaler: fitted StandardScaler (or None if scale=False)
        - n_total_models: total number of models trained
    """
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = None
        X_train_s = X_train
        X_test_s = X_test
    
    # Train all models
    if verbose:
        print(f"📦 Building Rashomon set (ε = {epsilon})...")
    
    training_result = train_all_models(
        X_train_s, y_train, X_test_s, y_test,
        model_configs=model_configs,
        seeds=seeds,
        metric=metric,
        verbose=verbose,
        n_iter=n_iter,
        search_random_state=search_random_state,
    )
    
    all_models = training_result['models']
    all_scores = training_result['scores']
    all_params = training_result['params']
    
    # Compute Rashomon threshold
    best_score = all_scores.max()
    threshold = best_score - epsilon
    
    # Filter to Rashomon set
    rashomon_idx = np.where(all_scores >= threshold)[0]
    rashomon_models = [all_models[i] for i in rashomon_idx]
    rashomon_params = [all_params[i] for i in rashomon_idx]
    rashomon_scores = all_scores[rashomon_idx]
    
    # Build prediction matrix for Rashomon models
    n_models = len(rashomon_models)
    n_samples = len(X_test)
    pred_matrix = np.zeros((n_models, n_samples))
    
    for i, model in enumerate(rashomon_models):
        pred_matrix[i] = model.predict_proba(X_test_s)[:, 1]
    
    if verbose:
        print(f"   ✅ Rashomon set: {n_models}/{len(all_models)} models")
        print(f"      Best {metric}: {best_score:.4f}")
        print(f"      Threshold: {threshold:.4f}")
        
        # Model type breakdown
        types = {}
        for p in rashomon_params:
            t = p['model_type']
            types[t] = types.get(t, 0) + 1
        print(f"      Composition: {types}")
    
    result = {
        'models': rashomon_models,
        'params': rashomon_params,
        'scores': rashomon_scores,
        'pred_matrix': pred_matrix,
        'best_score': best_score,
        'threshold': threshold,
        'epsilon': epsilon,
        'scaler': scaler,
        'n_total_models': len(all_models),
        'X_test_scaled': X_test_s
    }

    if save_path is not None:
        written = save_rashomon_result(result, save_path)
        if verbose:
            print(f"   💾 Saved Rashomon set to: {written}")

    return result


def get_model_composition(rashomon_result: Dict) -> Dict[str, int]:
    """Get count of each model type in the Rashomon set."""
    types = {}
    for p in rashomon_result['params']:
        t = p['model_type']
        types[t] = types.get(t, 0) + 1
    return types


def compare_epsilon_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epsilons: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05],
    **kwargs
) -> Dict[float, Dict]:
    """
    Build Rashomon sets for multiple epsilon values.
    
    Parameters
    ----------
    X_train, X_test, y_train, y_test : data
    epsilons : list of epsilon values to try
    **kwargs : additional arguments passed to build_rashomon_set
    
    Returns
    -------
    dict mapping epsilon -> rashomon_result
    """
    results = {}
    
    # Train models once
    kwargs['verbose'] = False
    kwargs['scale'] = kwargs.get('scale', True)
    
    if kwargs['scale']:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test
        scaler = None
    
    training_result = train_all_models(
        X_train_s, y_train, X_test_s, y_test,
        model_configs=kwargs.get('model_configs'),
        seeds=kwargs.get('seeds', [0, 1, 2]),
        metric=kwargs.get('metric', 'auc'),
        verbose=True,
        n_iter=kwargs.get('n_iter', 25),
        search_random_state=kwargs.get('search_random_state', 0),
    )
    
    all_models = training_result['models']
    all_scores = training_result['scores']
    all_params = training_result['params']
    best_score = all_scores.max()
    
    print(f"\n📊 Comparing epsilon values...")
    print(f"   Best score: {best_score:.4f}")
    print(f"   {'ε':<10} {'Threshold':<12} {'Rashomon Size':<15}")
    print(f"   {'-'*37}")
    
    for eps in epsilons:
        threshold = best_score - eps
        rashomon_idx = np.where(all_scores >= threshold)[0]
        
        rashomon_models = [all_models[i] for i in rashomon_idx]
        rashomon_params = [all_params[i] for i in rashomon_idx]
        rashomon_scores = all_scores[rashomon_idx]
        
        # Build prediction matrix
        n_models = len(rashomon_models)
        pred_matrix = np.zeros((n_models, len(X_test)))
        for i, model in enumerate(rashomon_models):
            pred_matrix[i] = model.predict_proba(X_test_s)[:, 1]
        
        results[eps] = {
            'models': rashomon_models,
            'params': rashomon_params,
            'scores': rashomon_scores,
            'pred_matrix': pred_matrix,
            'best_score': best_score,
            'threshold': threshold,
            'epsilon': eps,
            'scaler': scaler,
            'n_total_models': len(all_models),
            'X_test_scaled': X_test_s
        }
        
        print(f"   {eps:<10} {threshold:<12.4f} {n_models:<15}")
    
    return results

