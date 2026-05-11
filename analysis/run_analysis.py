"""
Analysis layer over training artifacts. Operates only on saved artifacts;
does not retrain models or modify training/data code.

Expects run directories: results/{dataset}/seed={outer_seed}/
containing: split.npz, meta.csv, P_val.npy, P_test.npy, config.json.

Spatial and null functions require X_test (preprocessed test features) to be
provided by the caller (e.g. from data load + split + same preprocessing as training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# PySAL for spatial analysis (optional at import; required for spatial_* and null_*)
try:
    from libpysal.weights import KNN
    from esda.moran import Moran, Moran_Local
    _HAS_PYSAL = True
except ImportError:
    _HAS_PYSAL = False

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Loading artifacts (paths to run directory)
# ---------------------------------------------------------------------------

def load_meta(run_dir: PathLike) -> pd.DataFrame:
    """Load meta.csv from a run directory."""
    run_dir = Path(run_dir)
    return pd.read_csv(run_dir / "meta.csv")


def load_P_val(run_dir: PathLike) -> np.ndarray:
    """Load P_val.npy (n_candidates, n_val)."""
    return np.load(Path(run_dir) / "P_val.npy")


def load_P_test(run_dir: PathLike) -> np.ndarray:
    """Load P_test.npy (n_candidates, n_test)."""
    return np.load(Path(run_dir) / "P_test.npy")


def load_split(run_dir: PathLike) -> Dict[str, np.ndarray]:
    """Load split.npz; return dict with train, val, test, seed."""
    data = np.load(Path(run_dir) / "split.npz")
    return {k: data[k] for k in data.files}


def load_config(run_dir: PathLike) -> Dict[str, Any]:
    """Load config.json from a run directory."""
    import json
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Rashomon selection (global, top-K by validation Brier)
# ---------------------------------------------------------------------------

def select_rashomon_global(
    run_dir: PathLike,
    K: int = 25,
) -> np.ndarray:
    """
    Select top-K models by validation Brier (pooled across families).
    Returns indices into the candidate list (rows of meta / P_test).
    """
    meta = load_meta(run_dir)
    if len(meta) < K:
        raise ValueError(f"Run has {len(meta)} candidates, requested K={K}")
    order = meta["val_brier"].values.argsort()
    return order[:K]


def select_rashomon_per_family_totalK(
    run_dir: PathLike,
    K: int = 25,
) -> np.ndarray:
    """
    Select top models per family such that the TOTAL count is K.

    With 5 families and K=25, each family contributes 5 models.
    Any remainder from integer division is assigned to families with the
    best (lowest) family-wise validation Brier.

    Returns indices into the candidate list (rows of meta / P_test).
    """
    meta = load_meta(run_dir)
    families = sorted(meta["model_name"].unique())
    if len(families) == 0:
        return np.array([], dtype=int)

    n_families = len(families)
    base_per_fam = K // n_families
    remainder = K % n_families

    # Rank families by best validation Brier (lowest first) for remainder
    fam_best = []
    for fam in families:
        best = meta.loc[meta["model_name"] == fam, "val_brier"].min()
        fam_best.append((best, fam))
    fam_best.sort()
    fam_rank = {fam: rank for rank, (_, fam) in enumerate(fam_best)}

    idx_list = []
    for family in families:
        mask = meta["model_name"] == family
        fam_indices = np.where(mask)[0]
        fam_brier = meta.loc[mask, "val_brier"].values
        n_take = base_per_fam + (1 if fam_rank[family] < remainder else 0)
        n_take = min(n_take, len(fam_indices))
        order = np.argsort(fam_brier)[:n_take]
        idx_list.extend(fam_indices[order].tolist())
    return np.array(idx_list, dtype=int)


def select_rashomon_per_family_k_each(
    run_dir: PathLike,
    K_each: int = 25,
) -> np.ndarray:
    """
    Select top-K_each models *within each family*.

    This is the appropriate selector for within-family decomposition analyses
    (e.g., hyperparameter drivers inside each family), where each family
    should retain the same number of candidates.

    Returns indices into the candidate list (rows of meta / P_test).
    """
    meta = load_meta(run_dir)
    families = sorted(meta["model_name"].unique())
    idx_list = []
    for family in families:
        mask = meta["model_name"] == family
        fam_indices = np.where(mask)[0]
        fam_brier = meta.loc[mask, "val_brier"].values
        n_take = min(K_each, len(fam_indices))
        order = np.argsort(fam_brier)[:n_take]
        idx_list.extend(fam_indices[order].tolist())
    return np.array(idx_list, dtype=int)


# ---------------------------------------------------------------------------
# 2. Predictive multiplicity metrics (on test set)
# ---------------------------------------------------------------------------

def pointwise_variance(P: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Population variance across models per observation. Shape (n_obs,)."""
    return np.var(P, axis=0, ddof=ddof)


def mean_variance(P: np.ndarray, ddof: int = 0) -> float:
    """Mean over observations of variance across models (population variance, ddof=0)."""
    return float(np.var(P, axis=0, ddof=ddof).mean())


def pointwise_conflict(P: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """
    Hard conflict ratio per observation.
    q_i = fraction of models predicting >= tau, conflict_i = min(q_i, 1 - q_i).
    Returns array of shape (n_obs,) in [0, 0.5].
    """
    q = np.mean(P >= tau, axis=0)
    return np.minimum(q, 1.0 - q)


def ambiguity(P: np.ndarray) -> float:
    """
    Probability-space ambiguity (Rudin et al., Resolving Predictive Multiplicity):
    mean over observations of (max_m p_m - min_m p_m).
    """
    return float(np.mean(np.max(P, axis=0) - np.min(P, axis=0)))


def disagreement_rate(P: np.ndarray, epsilon: float = 0.05) -> float:
    """
    Mean over model pairs of the fraction of test points
    where |f_m(x) - f_j(x)| > epsilon.
    P shape: (n_models, n_test)
    """
    n_models = P.shape[0]
    if n_models < 2:
        # No model pairs exist; disagreement is undefined but should be neutral.
        return 0.0

    pair_rates = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = np.abs(P[i] - P[j])
            pair_rates.append(np.mean(diff > epsilon))

    return float(np.mean(pair_rates)) if pair_rates else 0.0



def discrepancy(P: np.ndarray) -> float:
    """
    Probability-space discrepancy (Rudin et al., Resolving Predictive Multiplicity):
    maximum over model pairs of the mean absolute prediction difference.
    """
    n_models = P.shape[0]
    max_disc = 0.0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            d = np.mean(np.abs(P[i] - P[j]))
            if d > max_disc:
                max_disc = d
    return float(max_disc)



def compute_multiplicity_metrics(
    P: np.ndarray,
    epsilon: float = 0.05,
    ddof: int = 0,
    tau: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute predictive multiplicity metrics on predictions P of shape (n_models, n_obs).
    Returns dict with: mean_variance, ambiguity, disagreement_rate, discrepancy,
    pointwise_variance (array), pointwise_conflict (array),
    and summary conflict stats.
    """
    v = pointwise_variance(P, ddof=ddof)
    c = pointwise_conflict(P, tau=tau)
    return {
        "mean_variance": mean_variance(P, ddof=ddof),
        "ambiguity": ambiguity(P),
        "disagreement_rate": disagreement_rate(P, epsilon=epsilon),
        "discrepancy": discrepancy(P),
        "pointwise_variance": v,
        "pointwise_conflict": c,
        "mean_conflict": float(np.mean(c)),
        "frac_conflict_gt0": float(np.mean(c > 0)),
        "frac_conflict_ge025": float(np.mean(c >= 0.25)),
    }


# ---------------------------------------------------------------------------
# 3. Spatial analysis (kNN, PySAL Moran_Local, FDR, HH/LL masks)
# ---------------------------------------------------------------------------

def _fdr_benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. True = reject (significant)."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    thresh = alpha * (np.arange(1, n + 1, dtype=float) / n)
    passed = ranked_p <= thresh
    cutoff = np.max(ranked_p[passed]) if np.any(passed) else 0.0
    return p_values <= cutoff


def spatial_analysis(
    v: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    k: int = 30,
    permutations: int = 999,
    fdr_alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Build kNN graph (k=30, Euclidean), row-standardize weights, compute global
    Moran's I and LISA via PySAL; 999 permutations; FDR (Benjamini–Hochberg).
    Returns Moran's I, HH mask, LL mask, and LISA details.

    v : pointwise variance vector (n_test,).
    X_test : test set features for kNN (n_test, n_features); use preprocessed/
        standardized space as in training.
    """
    if not _HAS_PYSAL:
        raise ImportError("Spatial analysis requires libpysal and esda: pip install libpysal esda")

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.select_dtypes(include=[np.number]).values
    X_test = np.asarray(X_test, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(v) != X_test.shape[0]:
        raise ValueError("v length must match X_test number of rows")

    # kNN weights, Euclidean; row-standardize
    W = KNN.from_array(X_test, k=k)
    W.transform = "r"

    # Global Moran's I with permutation p-value
    # Moran (global) doesn't accept seed; seed numpy RNG manually for reproducibility.
    if seed is not None:
        np.random.seed(seed)
    moran_global = Moran(v, W, permutations=permutations)
    I_global = float(moran_global.I)

    # permutation p-value (two-sided, empirical)
    # PySAL stores it in .p_sim when permutations > 0
    p_value_global = float(moran_global.p_sim)

    # optionally also provide analytic/normal p-value
    p_value_norm = float(moran_global.p_norm)

    # Local Moran (LISA): 999 permutations, row-standardized
    if seed is not None:
        np.random.seed(seed)
    lm = Moran_Local(
        v,
        W,
        transformation="r",
        permutations=permutations,
    )
    # PySAL quadrant: 1=HH, 2=LH, 3=LL, 4=HL
    p_sim = np.asarray(lm.p_sim).flatten()
    q = np.asarray(lm.q).flatten()

    # FDR correction on LISA p-values
    sig = _fdr_benjamini_hochberg(p_sim, alpha=fdr_alpha)
    HH_mask = (q == 1) & sig
    LL_mask = (q == 3) & sig

    return {
        "moran_i": I_global,
        "moran_p_sim": p_value_global,
        "moran_p_norm": p_value_norm,
        "HH_mask": HH_mask,
        "LL_mask": LL_mask,
        "lisa_q": q,
        "lisa_p_sim": p_sim,
        "lisa_sig": sig,
        "W": W,
    }


# ---------------------------------------------------------------------------
# 3b. Quadrant analysis: soft variance vs hard conflict
# ---------------------------------------------------------------------------

def quadrant_analysis(
    var_p: np.ndarray,
    conflict: np.ndarray,
    *,
    var_q: float = 0.9,
    conflict_q: float = 0.9,
    var_thresh: Optional[float] = None,
    conflict_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Classify points into 4 quadrants based on soft variance and hard conflict,
    and compute summary statistics for each.

    Parameters
    ----------
    var_p : pointwise variance, shape (n_obs,)
    conflict : pointwise conflict ratio, shape (n_obs,)
    var_q, conflict_q : quantile thresholds (used when fixed thresholds are None)
    var_thresh, conflict_thresh : fixed thresholds (override quantile if given)

    Returns
    -------
    dict with 'thresholds', 'labels' (array of 'A','B','C','D'), and 'summary' DataFrame
    """
    n = len(var_p)
    vt = var_thresh if var_thresh is not None else float(np.quantile(var_p, var_q))
    ct = conflict_thresh if conflict_thresh is not None else float(np.quantile(conflict, conflict_q))

    high_v = var_p >= vt
    high_c = conflict >= ct

    labels = np.full(n, "D", dtype="U1")
    labels[high_v & high_c] = "A"
    labels[high_v & ~high_c] = "B"
    labels[~high_v & high_c] = "C"

    rows = []
    for grp in ["A", "B", "C", "D"]:
        mask = labels == grp
        cnt = int(mask.sum())
        row: Dict[str, Any] = {
            "quadrant": grp,
            "count": cnt,
            "fraction": cnt / n if n > 0 else 0.0,
            "mean_var_p": float(np.mean(var_p[mask])) if cnt > 0 else np.nan,
            "mean_conflict": float(np.mean(conflict[mask])) if cnt > 0 else np.nan,
        }

    return {
        "var_thresh": vt,
        "conflict_thresh": ct,
        "labels": labels,
        "summary": pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# 3c. Spatial HH overlap (Jaccard) between two variables
# ---------------------------------------------------------------------------

def spatial_hh_jaccard(
    result_a: Dict[str, Any],
    result_b: Dict[str, Any],
) -> float:
    """
    Jaccard index between HH masks of two spatial analysis results.
    """
    hh_a = np.asarray(result_a["HH_mask"]).astype(bool)
    hh_b = np.asarray(result_b["HH_mask"]).astype(bool)
    union = np.logical_or(hh_a, hh_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(hh_a, hh_b).sum() / union)


# ---------------------------------------------------------------------------
# 4. Null experiment (independent permutation per model, R=100)
# ---------------------------------------------------------------------------

def permute_predictions_independent(P: np.ndarray, seed: int) -> np.ndarray:
    """Permute each model's predictions independently across observations."""
    rng = np.random.RandomState(seed)
    P_perm = np.empty_like(P)
    for m in range(P.shape[0]):
        P_perm[m] = rng.permutation(P[m])
    return P_perm


def null_experiment(
    P: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    observed_I: Optional[float] = None,
    observed_n_hh: Optional[int] = None,
    R: int = 100,
    k: int = 30,
    seed: int = 42,
    lisa_permutations: int = 199,
    fdr_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Null experiment: independent permutation per model (R times).

    For each permutation:
      - Independently permute predictions per model
      - Recompute pointwise variance
      - Compute global Moran's I
      - Compute LISA (Moran_Local) and count HH/LL points after FDR

    Returns:
        null_moran_i: array of Moran's I under null (length R)
        null_n_hh: array of HH point counts under null (length R)
        null_n_ll: array of LL point counts under null (length R)
        null_mean: mean of null Moran's I
        null_std: std of null Moran's I
        p_empirical: empirical one-sided p-value for Moran's I (if observed_I provided)
        p_empirical_hh: empirical one-sided p-value for n_hh (if observed_n_hh provided)
        R: number of permutations
    """
    if not _HAS_PYSAL:
        raise ImportError("Null experiment requires libpysal and esda: pip install libpysal esda")

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.select_dtypes(include=[np.number]).values

    X_test = np.asarray(X_test, dtype=float)

    W = KNN.from_array(X_test, k=k)
    W.transform = "r"

    rng = np.random.RandomState(seed)
    null_moran_i = np.zeros(R)
    null_conflict_moran_i = np.zeros(R)
    null_n_hh = np.zeros(R, dtype=int)
    null_n_ll = np.zeros(R, dtype=int)

    for r in range(R):
        perm_seed = int(rng.randint(0, 2**31))
        P_perm = permute_predictions_independent(P, perm_seed)
        v_perm = pointwise_variance(P_perm, ddof=0)
        c_perm = pointwise_conflict(P_perm)

        moran_r = Moran(v_perm, W)
        null_moran_i[r] = moran_r.I

        if float(np.var(c_perm)) > 1e-15:
            moran_c = Moran(c_perm, W)
            null_conflict_moran_i[r] = moran_c.I
        else:
            null_conflict_moran_i[r] = 0.0

        lisa_seed = int(rng.randint(0, 2**31))
        np.random.seed(lisa_seed)
        lm = Moran_Local(
            v_perm, W,
            transformation="r",
            permutations=lisa_permutations,
        )
        p_sim = np.asarray(lm.p_sim).flatten()
        q = np.asarray(lm.q).flatten()
        sig = _fdr_benjamini_hochberg(p_sim, alpha=fdr_alpha)
        null_n_hh[r] = int(np.sum((q == 1) & sig))
        null_n_ll[r] = int(np.sum((q == 3) & sig))

    result = {
        "null_moran_i": null_moran_i,
        "null_conflict_moran_i": null_conflict_moran_i,
        "null_n_hh": null_n_hh,
        "null_n_ll": null_n_ll,
        "null_mean": float(np.mean(null_moran_i)),
        "null_std": float(np.std(null_moran_i, ddof=1)),
        "null_conflict_moran_mean": float(np.mean(null_conflict_moran_i)),
        "null_conflict_moran_std": float(np.std(null_conflict_moran_i, ddof=1)),
        "null_n_hh_mean": float(np.mean(null_n_hh)),
        "null_n_hh_std": float(np.std(null_n_hh, ddof=1)),
        "R": R,
    }

    if observed_I is not None:
        p_emp = (1 + np.sum(null_moran_i >= observed_I)) / (R + 1)
        result["observed_I"] = float(observed_I)
        result["p_empirical"] = float(p_emp)

    if observed_n_hh is not None:
        p_emp_hh = (1 + np.sum(null_n_hh >= observed_n_hh)) / (R + 1)
        result["observed_n_hh"] = int(observed_n_hh)
        result["p_empirical_hh"] = float(p_emp_hh)

    return result


# ---------------------------------------------------------------------------
# Convenience: run-level analysis using Rashomon selection
# ---------------------------------------------------------------------------

def run_multiplicity(
    run_dir: PathLike,
    K: int = 25,
    epsilon: float = 0.05,
) -> Dict[str, Any]:
    """
    Load run, select global Rashomon (top-K), compute multiplicity metrics on test.
    """
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    P_test = load_P_test(run_dir)
    idx = select_rashomon_global(run_dir, K=K)
    P_sel = P_test[idx]
    return compute_multiplicity_metrics(P_sel, epsilon=epsilon)


def run_spatial(
    run_dir: PathLike,
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    K: int = 25,
    k: int = 30,
    permutations: int = 999,
    fdr_alpha: float = 0.05,
    seed: Optional[int] = 42,
    selection: str = "global",
    tau: float = 0.5,
) -> Dict[str, Any]:
    """
    Load run, select Rashomon set (global or per-family total-K), compute pointwise
    variance on test, then spatial analysis (Moran's I, HH/LL masks, etc.).
    Also runs spatial analysis on conflict and reports HH Jaccard overlap.

    Parameters
    ----------
    selection : "global" or "per_family"
        - "global": top-K pooled across all families
        - "per_family": total-K distributed across families
    tau : decision threshold for hard conflict
    """
    valid_selection = {"global", "per_family"}
    if selection not in valid_selection:
        raise ValueError(
            f"selection must be one of {sorted(valid_selection)}, got '{selection}'"
        )

    run_dir = Path(run_dir)
    P_test = load_P_test(run_dir)
    if selection == "per_family":
        idx = select_rashomon_per_family_totalK(run_dir, K=K)
    else:
        idx = select_rashomon_global(run_dir, K=K)
    P_sel = P_test[idx]
    v = pointwise_variance(P_sel, ddof=0)
    c = pointwise_conflict(P_sel, tau=tau)

    out = spatial_analysis(
        v, X_test, k=k, permutations=permutations, fdr_alpha=fdr_alpha, seed=seed
    )
    out["n_ll"] = int(np.sum(out["LL_mask"]))

    # Spatial analysis on conflict
    conflict_has_variance = float(np.var(c)) > 1e-15
    if conflict_has_variance:
        spatial_conflict = spatial_analysis(
            c, X_test, k=k, permutations=permutations,
            fdr_alpha=fdr_alpha, seed=seed,
        )
        out["conflict_moran_i"] = spatial_conflict["moran_i"]
        out["conflict_n_hh"] = int(np.sum(spatial_conflict["HH_mask"]))
        out["conflict_n_ll"] = int(np.sum(spatial_conflict["LL_mask"]))
        out["conflict_HH_mask"] = spatial_conflict["HH_mask"]
        out["hh_jaccard_var_conflict"] = spatial_hh_jaccard(out, spatial_conflict)
    else:
        out["conflict_moran_i"] = np.nan
        out["conflict_n_hh"] = 0
        out["conflict_n_ll"] = 0
        out["conflict_HH_mask"] = np.zeros(len(v), dtype=bool)
        out["hh_jaccard_var_conflict"] = np.nan

    return out


def run_spatial_per_family(
    run_dir: PathLike,
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    K: int = 25,
    k: int = 30,
    permutations: int = 999,
    fdr_alpha: float = 0.05,
    seed: Optional[int] = 42,
    tau: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute spatial analysis separately for each model family's Rashomon set.
    Here, K is interpreted as K-per-family (K_each).

    Returns dict mapping family name -> spatial analysis result dict.
    """
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    P_test = load_P_test(run_dir)

    results = {}
    for family in sorted(meta["model_name"].unique()):
        mask = meta["model_name"] == family
        fam_indices = np.where(mask)[0]
        fam_brier = meta.loc[mask, "val_brier"].values
        n_take = min(K, len(fam_indices))
        order = np.argsort(fam_brier)[:n_take]
        idx = fam_indices[order]

        P_sel = P_test[idx]
        v = pointwise_variance(P_sel, ddof=0)
        c = pointwise_conflict(P_sel, tau=tau)
        out = spatial_analysis(
            v, X_test, k=k, permutations=permutations,
            fdr_alpha=fdr_alpha, seed=seed,
        )
        out["n_hh"] = int(np.sum(out["HH_mask"]))
        out["n_ll"] = int(np.sum(out["LL_mask"]))
        out["n_models"] = len(idx)
        out["mean_variance"] = float(np.mean(v))
        out["mean_conflict"] = float(np.mean(c))

        conflict_has_variance = float(np.var(c)) > 1e-15
        if conflict_has_variance:
            spatial_conflict = spatial_analysis(
                c, X_test, k=k, permutations=permutations,
                fdr_alpha=fdr_alpha, seed=seed,
            )
            out["conflict_moran_i"] = spatial_conflict["moran_i"]
            out["conflict_n_hh"] = int(np.sum(spatial_conflict["HH_mask"]))
            out["hh_jaccard_var_conflict"] = spatial_hh_jaccard(out, spatial_conflict)
        else:
            out["conflict_moran_i"] = np.nan
            out["conflict_n_hh"] = 0
            out["hh_jaccard_var_conflict"] = np.nan

        results[family] = out

    return results


def run_null(
    run_dir: PathLike,
    X_test: Union[np.ndarray, pd.DataFrame],
    *,
    K: int = 25,
    R: int = 100,
    k: int = 30,
    seed: int = 42,
    lisa_permutations: int = 199,
    fdr_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Load run, select global Rashomon (top-K), run null experiment (R permutations)
    and return null Moran's I and LISA distributions.
    """
    run_dir = Path(run_dir)
    P_test = load_P_test(run_dir)
    idx = select_rashomon_global(run_dir, K=K)
    P_sel = P_test[idx]
    return null_experiment(
        P_sel, X_test, R=R, k=k, seed=seed,
        lisa_permutations=lisa_permutations, fdr_alpha=fdr_alpha,
    )
