"""
Multiplicity metrics for Rashomon set analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
from sklearn.inspection import partial_dependence


# Observation-wise prediction variance

def compute_variance_metrics(pred_matrix: np.ndarray) -> Dict:
    """
    Compute variance metrics including RPIs and flip probability.
    
    Parameters
    ----------
    pred_matrix : np.ndarray of shape (n_models, n_samples)
        Prediction probabilities from each model
        
    Returns
    -------
    dict with keys:
        - variance: raw prediction variance
        - mean_pred: mean prediction
        - std_pred: standard deviation
        - rpi_lower_95: 2.5th percentile (lower bound of 95% RPI)
        - rpi_upper_95: 97.5th percentile (upper bound of 95% RPI)
        - rpi_width_95: width of 95% RPI
        - rpi_lower_50: 25th percentile (lower bound of 50% RPI)
        - rpi_upper_50: 75th percentile (upper bound of 50% RPI)
        - rpi_width_50: width of 50% RPI
        - normalized_variance: variance / max_possible_variance at that mean
        - normalized_variance_global: variance / 0.25 (global max)
        - flip_probability: fraction of models that disagree with majority
        - crosses_boundary: boolean mask where 95% RPI crosses 0.5
    """
    n_models, n_samples = pred_matrix.shape
    
    # Basic statistics
    mean_pred = np.mean(pred_matrix, axis=0)
    variance = np.var(pred_matrix, axis=0)
    std_pred = np.std(pred_matrix, axis=0)
    
    # Rashomon Prediction Intervals (95%)
    rpi_lower_95 = np.percentile(pred_matrix, 2.5, axis=0)
    rpi_upper_95 = np.percentile(pred_matrix, 97.5, axis=0)
    rpi_width_95 = rpi_upper_95 - rpi_lower_95
    
    # Rashomon Prediction Intervals (50% - IQR)
    rpi_lower_50 = np.percentile(pred_matrix, 25, axis=0)
    rpi_upper_50 = np.percentile(pred_matrix, 75, axis=0)
    rpi_width_50 = rpi_upper_50 - rpi_lower_50
    
    # Normalized variance (relative to max possible at that mean)
    # For Bernoulli, max variance at mean p is p*(1-p)
    max_possible_variance = mean_pred * (1 - mean_pred)
    max_possible_variance = np.maximum(max_possible_variance, 1e-10)  # avoid div by 0
    normalized_variance = variance / max_possible_variance
    
    # Global normalized variance (relative to absolute max = 0.25)
    normalized_variance_global = variance / 0.25
    
    # Flip probability: fraction of models that disagree with majority
    n_positive = np.sum(pred_matrix > 0.5, axis=0)
    n_negative = n_models - n_positive
    flip_probability = np.minimum(n_positive, n_negative) / n_models
    
    # Controversial cases: 95% RPI crosses decision boundary
    crosses_boundary = (rpi_lower_95 < 0.5) & (rpi_upper_95 > 0.5)
    
    return {
        'variance': variance,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'rpi_lower_95': rpi_lower_95,
        'rpi_upper_95': rpi_upper_95,
        'rpi_width_95': rpi_width_95,
        'rpi_lower_50': rpi_lower_50,
        'rpi_upper_50': rpi_upper_50,
        'rpi_width_50': rpi_width_50,
        'normalized_variance': normalized_variance,
        'normalized_variance_global': normalized_variance_global,
        'flip_probability': flip_probability,
        'crosses_boundary': crosses_boundary
    }


def metrics_summary(metrics: Dict, y_test: np.ndarray = None) -> None:
    """Print a summary of the computed metrics."""
    print("\n" + "=" * 60)
    print("           MULTIPLICITY METRICS SUMMARY")
    print("=" * 60)
    
    n_samples = len(metrics['variance'])
    
    print(f"\n📊 Observation-wise Variance")
    print(f"   Mean variance:     {metrics['variance'].mean():.6f}")
    print(f"   Max variance:      {metrics['variance'].max():.6f}")
    print(f"   High variance (>0.01): {(metrics['variance'] > 0.01).sum()} samples "
          f"({100*(metrics['variance'] > 0.01).mean():.1f}%)")
    
    print(f"\n📊 Rashomon Prediction Intervals")
    print(f"   Mean 95% width:    {metrics['rpi_width_95'].mean():.4f}")
    print(f"   Max 95% width:     {metrics['rpi_width_95'].max():.4f}")
    print(f"   Mean 50% width:    {metrics['rpi_width_50'].mean():.4f}")
    
    print(f"\n📊 Classification Instability")
    print(f"   Controversial cases: {metrics['crosses_boundary'].sum()} "
          f"({100*metrics['crosses_boundary'].mean():.1f}%)")
    print(f"   Mean flip prob:    {metrics['flip_probability'].mean():.4f}")
    print(f"   Max flip prob:     {metrics['flip_probability'].max():.4f}")
    
    print(f"\n📊 Normalized Variance")
    print(f"   Mean (local):      {metrics['normalized_variance'].mean():.4f}")
    print(f"   Mean (global):     {metrics['normalized_variance_global'].mean():.4f}")
    
    print("=" * 60)


def to_dataframe(
    metrics: Dict,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Convert metrics to a DataFrame for analysis.
    
    Parameters
    ----------
    metrics : dict from compute_variance_metrics
    X_test : optional feature matrix to include
    y_test : optional true labels
    feature_names : names for X_test columns
    
    Returns
    -------
    pd.DataFrame with all metrics per sample
    """
    df = pd.DataFrame({
        'variance': metrics['variance'],
        'mean_pred': metrics['mean_pred'],
        'std_pred': metrics['std_pred'],
        'rpi_lower_95': metrics['rpi_lower_95'],
        'rpi_upper_95': metrics['rpi_upper_95'],
        'rpi_width_95': metrics['rpi_width_95'],
        'rpi_lower_50': metrics['rpi_lower_50'],
        'rpi_upper_50': metrics['rpi_upper_50'],
        'rpi_width_50': metrics['rpi_width_50'],
        'normalized_variance': metrics['normalized_variance'],
        'normalized_variance_global': metrics['normalized_variance_global'],
        'flip_probability': metrics['flip_probability'],
        'crosses_boundary': metrics['crosses_boundary']
    })
    
    if y_test is not None:
        df['y_true'] = y_test
    
    if X_test is not None and feature_names is not None:
        for i, name in enumerate(feature_names):
            df[name] = X_test[:, i]
    
    return df


# PDP variance - global feature effect instability


def compute_categorical_pdp_onehot(
    model,
    X: np.ndarray,
    category_to_index: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute PDP for a categorical feature represented by one-hot dummies.

    This enforces mutual exclusivity: for each category, set its dummy to 1 and all
    other dummies in the group to 0, then average the model predictions.

    Parameters
    ----------
    model
        Fitted classifier with predict_proba.
    X : np.ndarray
        Feature matrix used for averaging (typically scaled, shape (n_samples, n_features)).
    category_to_index : dict
        Mapping category label -> column index in X.

    Returns
    -------
    dict
        Mapping category label -> mean predicted probability for the positive class.
    """
    idxs = list(category_to_index.values())
    X0 = X.copy()
    X0[:, idxs] = 0
    pdp = {}
    for cat, j in category_to_index.items():
        X1 = X0.copy()
        X1[:, j] = 1
        pdp[cat] = float(model.predict_proba(X1)[:, 1].mean())
    return pdp


def compute_numeric_pdp(
    model,
    X: np.ndarray,
    feature_idx: int,
    grid_resolution: int = 50,
    scaler=None,
) -> Dict[str, np.ndarray]:
    """
    Compute 1D PDP for a numeric feature using sklearn.inspection.partial_dependence.

    Parameters
    ----------
    model
        Fitted classifier with predict_proba.
    X : np.ndarray
        Feature matrix used for averaging (typically scaled, shape (n_samples, n_features)).
    feature_idx : int
        Column index of the numeric feature in X.
    grid_resolution : int
        Number of grid points to evaluate.
    scaler
        Optional fitted StandardScaler. If provided, the returned `grid` is inverse-transformed
        to original units for plotting.

    Returns
    -------
    dict with keys:
        - grid_scaled: np.ndarray of grid points in X units
        - grid: np.ndarray of grid points in original units (if scaler provided)
        - pdp: np.ndarray PDP values at each grid point
    """
    res = partial_dependence(
        estimator=model,
        X=X,
        features=[feature_idx],
        kind="average",
        method="brute",
        response_method="predict_proba",
        percentiles=(0.0, 1.0),
        grid_resolution=grid_resolution,
    )
    avg = res["average"][0]
    grid_scaled = np.asarray(res["grid_values"][0])

    grid = grid_scaled
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        grid = grid_scaled * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]

    return {
        'grid_scaled': grid_scaled,
        'grid': grid,
        'pdp': np.asarray(avg),
    }


def compute_pdp_variance(
    models: List,
    X: np.ndarray,
    feature_names: List[str],
    categorical_groups: Optional[Dict[str, Dict[str, int]]] = None,
    grid_size: int = 50,
    scaler=None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Compute PDPs and their variance for all features.

    Parameters
    ----------
    models : list
        List of fitted models.
    X : np.ndarray
        Feature matrix used for averaging (typically scaled).
    feature_names : list of str
        Column names aligned with X.
    categorical_groups : dict, optional
        Mapping base categorical feature name: {category_label : column_index_in_X}.
        If provided, those features are treated as categoricals. All other columns are numeric.
    grid_size : int
        Grid resolution for numeric PDPs.
    scaler
        Optional fitted StandardScaler used to inverse-transform numeric grids for plotting.
    verbose : bool
        If True, show a progress bar.

    Returns
    -------
    dict
        Mapping feature name -> result dict with PDP matrix and variance summary.
    """
    results: Dict[str, Dict] = {}
    categorical_groups = categorical_groups or {}
    covered = {j for m in categorical_groups.values() for j in m.values()}

    # Categorical
    for base, cat_to_idx in categorical_groups.items():
        cats = list(cat_to_idx.keys())

        pdp_mat = []
        for m in models:
            pdp = compute_categorical_pdp_onehot(m, X, cat_to_idx)
            pdp_mat.append([pdp[c] for c in cats])
        pdp_mat = np.asarray(pdp_mat, float)

        results[base] = {
            "grid": np.array(cats, dtype=object),
            "pdp_matrix": pdp_mat,
            "mean_pdp": pdp_mat.mean(axis=0),
            "variance_pdp": pdp_mat.var(axis=0),
            "mean_variance_pdp": float(pdp_mat.var(axis=0).mean())
        }

    # Numeric
    iterator = list(enumerate(feature_names))
    if verbose:
        iterator = tqdm(iterator, desc="Computing PDP variance (numeric)")
    for j, name in iterator:
        if j in covered:
            continue
        grid_scaled = None
        grid = None
        pdps = []
        for i_m, m in enumerate(models):
            out = compute_numeric_pdp(m, X, j, grid_resolution=grid_size, scaler=scaler)
            if i_m == 0:
                grid_scaled = out["grid_scaled"]
                grid = out["grid"]
            pdps.append(out["pdp"])

        pdp_mat = np.vstack(pdps)
        results[name] = {
            "grid_scaled": grid_scaled,
            "grid": grid,
            "pdp_matrix": pdp_mat,
            "mean_pdp": pdp_mat.mean(axis=0),
            "variance_pdp": pdp_mat.var(axis=0),
            "mean_variance_pdp": float(pdp_mat.var(axis=0).mean())
        }

    return results


def pdp_ranking(pdp_results: Dict[str, Dict]) -> pd.Series:
    """Get features ranked by mean PDP variance (most unstable first)."""
    ranking = {name: res['mean_variance_pdp'] for name, res in pdp_results.items()}
    return pd.Series(ranking).sort_values(ascending=False)


# FME variance - local feature effect instability

def compute_fme_variance_all(
    models: List,
    X: np.ndarray,
    feature_names: List[str],
    categorical_groups: Optional[Dict[str, Dict[str, int]]] = None,
    h: float = 0.1,
    categorical_reference: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Compute forward marginal effect (fME) variance across models
    for continuous and categorical (one-hot) features.
    """
    results = {}

    categorical_groups = categorical_groups or {}
    categorical_reference = categorical_reference or {}

    # Indices covered by categorical groups
    covered = {j for g in categorical_groups.values() for j in g.values()}

    # (feature_name, feature_idx or None)
    items = (
        [(name, None) for name in categorical_groups.keys()] +
        [(name, j) for j, name in enumerate(feature_names) if j not in covered]
    )

    if verbose:
        items = tqdm(items, desc="Computing FME variance", total=len(items))

    for name, idx in items:

        # categorical (one-hot)
        if idx is None:
            group = categorical_groups[name]
            categories = list(group.keys())
            group_idx = list(group.values())

            ref_cat = categorical_reference.get(name)
            ref_i = categories.index(ref_cat) if ref_cat in categories else 0

            other_i = [i for i in range(len(categories)) if i != ref_i]
            other_cats = [categories[i] for i in other_i]

            n_models, n_samples, n_cats = len(models), len(X), len(other_i)
            fme = np.zeros((n_models, n_samples, n_cats))

            X_ref = X.copy()
            X_ref[:, group_idx] = 0.0
            X_ref[:, group_idx[ref_i]] = 1.0

            for m, model in enumerate(models):
                pred_ref = model.predict_proba(X_ref)[:, 1]

                for k, ci in enumerate(other_i):
                    X_tmp = X_ref.copy()
                    X_tmp[:, group_idx] = 0.0
                    X_tmp[:, group_idx[ci]] = 1.0
                    fme[m, :, k] = model.predict_proba(X_tmp)[:, 1] - pred_ref

            results[name] = {
                "reference_category": categories[ref_i],
                "categories": other_cats,
                "fme_matrix": fme,
                "mean_fme": fme.mean(axis=0),
                "variance": fme.var(axis=0),
                "mean_variance": float(fme.var(axis=0).mean()),
                "is_categorical": True,
            }
            continue

        # continuous
        n_models, n_samples = len(models), len(X)
        fme = np.zeros((n_models, n_samples))

        X_plus = X.copy()
        X_plus[:, idx] += h

        for m, model in enumerate(models):
            p0 = model.predict_proba(X)[:, 1]
            p1 = model.predict_proba(X_plus)[:, 1]
            fme[m] = p1 - p0

        results[name] = {
            "fme_matrix": fme,
            "mean_fme": fme.mean(axis=0),
            "variance": fme.var(axis=0),
            "mean_variance": float(fme.var(axis=0).mean()),
        }

    return results



def fme_ranking(fme_results: Dict[str, Dict]) -> pd.Series:
    """Get features ranked by mean FME variance (most unstable first)."""
    ranking = {name: res['mean_variance'] for name, res in fme_results.items()}
    return pd.Series(ranking).sort_values(ascending=False)


# Full analysis

def full_multiplicity_analysis(
    rashomon_result: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    compute_pdp: bool = True,
    compute_fme: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete multiplicity analysis on a Rashomon set.
    
    Parameters
    ----------
    rashomon_result : dict from build_rashomon_set
    X_test : test features (unscaled)
    y_test : test labels
    feature_names : feature names
    compute_pdp : whether to compute PDP variance
    compute_fme : whether to compute FME variance
    verbose : print progress
    
    Returns
    -------
    dict with keys:
        - metrics: enhanced metrics dict
        - metrics_df: DataFrame with per-sample results
        - pdp_results: dict of PDP results per feature (if computed)
        - pdp_ranking: feature ranking by mean PDP variance (if computed)
        - fme_results: dict of FME results per feature (if computed)
        - fme_ranking: feature ranking by mean FME variance (if computed)
    """
    pred_matrix = rashomon_result['pred_matrix']
    models = rashomon_result['models']
    X_scaled = rashomon_result['X_test_scaled']
    scaler = rashomon_result.get('scaler')
    
    if verbose:
        print("\n📊 Running full multiplicity analysis...")
    
    # Observation-wise metrics
    if verbose:
        print("   Computing observation-wise metrics...")
    metrics = compute_variance_metrics(pred_matrix)
    metrics_df = to_dataframe(metrics, X_test, y_test, feature_names)
    
    result = {
        'metrics': metrics,
        'metrics_df': metrics_df
    }
    
    # PDP variance
    if compute_pdp:
        if verbose:
            print("   Computing PDP variance...")
        pdp_results = compute_pdp_variance(
            models,
            X_scaled,
            feature_names,
            categorical_groups=rashomon_result.get("categorical_groups"),
            verbose=verbose,
            scaler=scaler,
        )
        result['pdp_results'] = pdp_results
        result['pdp_ranking'] = pdp_ranking(pdp_results)
    
    # FME variance
    if compute_fme:
        if verbose:
            print("   Computing FME variance...")
        fme_results = compute_fme_variance_all(
            models,
            X_scaled,
            feature_names,
            categorical_groups=rashomon_result.get('categorical_groups'),
            verbose=verbose,
        )
        result['fme_results'] = fme_results
        result['fme_ranking'] = fme_ranking(fme_results)
    
    if verbose:
        metrics_summary(metrics, y_test)
        
        if compute_pdp:
            print(f"\n📊 Top 5 Unstable Features (PDP):")
            for i, (name, val) in enumerate(result['pdp_ranking'].head().items()):
                print(f"   {i+1}. {name}: {val:.6f}")
        
        if compute_fme:
            print(f"\n📊 Top 5 Unstable Features (FME):")
            for i, (name, val) in enumerate(result['fme_ranking'].head().items()):
                print(f"   {i+1}. {name}: {val:.6f}")
    
    return result

