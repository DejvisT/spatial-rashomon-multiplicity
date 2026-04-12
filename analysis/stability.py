from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------
# HH selection stability (point-wise)
# ---------------------------------------------------------------------

def hh_point_level_support_metrics(freq: np.ndarray) -> Dict[str, float]:
    """
    Point-level HH support statistics for a fixed test set.

    Parameters
    ----------
    freq : array of shape (n_test,)
        Per-point fraction of runs in which the point is HH (e.g. from
        ``hh_selection_frequency``).

    Returns
    -------
    dict with keys:
        - hh_support_size: |{i : f_i > 0}| / N
        - hh_support_mean_f, hh_support_median_f: mean/median of f_i over f_i > 0
        - hh_support_bucket_00_02, _02_05, _05_08, _08_10: counts among f_i > 0
          for (0,0.2], (0.2,0.5], (0.5,0.8], (0.8,1]
        - hh_support_bucket_*_frac: each bucket count / |support|
        - hh_core_to_support_ratio: |{i : f_i >= 0.5}| / |{i : f_i > 0}|
    """
    f = np.asarray(freq, dtype=float).ravel()
    n = int(f.size)
    if n == 0:
        nan = float("nan")
        empty = {
            "hh_support_size": 0.0,
            "hh_support_mean_f": nan,
            "hh_support_median_f": nan,
            "hh_support_bucket_00_02": 0.0,
            "hh_support_bucket_02_05": 0.0,
            "hh_support_bucket_05_08": 0.0,
            "hh_support_bucket_08_10": 0.0,
            "hh_support_bucket_00_02_frac": nan,
            "hh_support_bucket_02_05_frac": nan,
            "hh_support_bucket_05_08_frac": nan,
            "hh_support_bucket_08_10_frac": nan,
            "hh_core_to_support_ratio": nan,
        }
        return empty

    support = f > 0
    n_support = int(np.sum(support))
    hh_support_size = n_support / n

    if n_support == 0:
        nan = float("nan")
        return {
            "hh_support_size": 0.0,
            "hh_support_mean_f": nan,
            "hh_support_median_f": nan,
            "hh_support_bucket_00_02": 0.0,
            "hh_support_bucket_02_05": 0.0,
            "hh_support_bucket_05_08": 0.0,
            "hh_support_bucket_08_10": 0.0,
            "hh_support_bucket_00_02_frac": nan,
            "hh_support_bucket_02_05_frac": nan,
            "hh_support_bucket_05_08_frac": nan,
            "hh_support_bucket_08_10_frac": nan,
            "hh_core_to_support_ratio": nan,
        }

    fs = f[support]
    n_core = int(np.sum(f >= 0.5))
    core_to_support = n_core / n_support

    b1 = int(np.sum(fs <= 0.2))
    b2 = int(np.sum((fs > 0.2) & (fs <= 0.5)))
    b3 = int(np.sum((fs > 0.5) & (fs <= 0.8)))
    b4 = int(np.sum(fs > 0.8))

    inv = 1.0 / n_support
    return {
        "hh_support_size": float(hh_support_size),
        "hh_support_mean_f": float(fs.mean()),
        "hh_support_median_f": float(np.median(fs)),
        "hh_support_bucket_00_02": float(b1),
        "hh_support_bucket_02_05": float(b2),
        "hh_support_bucket_05_08": float(b3),
        "hh_support_bucket_08_10": float(b4),
        "hh_support_bucket_00_02_frac": b1 * inv,
        "hh_support_bucket_02_05_frac": b2 * inv,
        "hh_support_bucket_05_08_frac": b3 * inv,
        "hh_support_bucket_08_10_frac": b4 * inv,
        "hh_core_to_support_ratio": float(core_to_support),
    }


def hh_selection_frequency(
    hh_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Compute how often each observation is classified as HH across runs.

    Parameters
    ----------
    hh_masks : list of boolean arrays of shape (n_obs,)
        Each array indicates HH membership for one run.

    Returns
    -------
    freq : array of shape (n_obs,)
        Fraction of runs in which each point is HH.
    """
    hh_stack = np.vstack(hh_masks)
    return hh_stack.mean(axis=0)


# ---------------------------------------------------------------------
# HH set overlap (run-to-run)
# ---------------------------------------------------------------------

def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Jaccard index between two HH masks.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    union = np.logical_or(a, b).sum()
    if union == 0:
        # By convention, two empty sets are perfectly identical.
        return 1.0

    intersection = np.logical_and(a, b).sum()
    return intersection / union


def hh_jaccard_matrix(
    hh_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Pairwise Jaccard overlap matrix between HH masks from different runs.
    """
    n = len(hh_masks)
    J = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            J[i, j] = jaccard_index(hh_masks[i], hh_masks[j])

    return J


# ---------------------------------------------------------------------
# Component-level stability
# ---------------------------------------------------------------------

def component_sizes(
    components: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Return sizes of hotspot components.
    """
    return np.array([len(v) for v in components.values()])


def largest_component_size(
    components: Dict[int, np.ndarray],
) -> int:
    """
    Size of the largest HH component.
    """
    if len(components) == 0:
        return 0
    return max(len(v) for v in components.values())


# ---------------------------------------------------------------------
# Stability summary helpers
# ---------------------------------------------------------------------

def summarize_hh_stability(
    hh_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Summarize HH stability across runs.

    Returns
    -------
    dict with keys:
        - mean_hh_fraction
        - mean_jaccard, min_jaccard, max_jaccard
        - hh_support_size, hh_support_mean_f, hh_support_median_f
        - hh_support_bucket_* (counts and fractions of support)
        - hh_core_to_support_ratio
        (see ``hh_point_level_support_metrics``).
    """
    freqs = hh_selection_frequency(hh_masks)
    J = hh_jaccard_matrix(hh_masks)

    # ignore diagonal
    off_diag = J[~np.eye(J.shape[0], dtype=bool)]

    base = {
        "mean_hh_fraction": float(freqs.mean()),
        "mean_jaccard": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "min_jaccard": float(off_diag.min()) if len(off_diag) > 0 else 0.0,
        "max_jaccard": float(off_diag.max()) if len(off_diag) > 0 else 0.0,
    }
    base.update(hh_point_level_support_metrics(freqs))
    return base


# ---------------------------------------------------------------------
# Region-level stability metrics
# ---------------------------------------------------------------------

def hh_frequency_mass(
    hh_freq: np.ndarray,
    components: Dict[int, np.ndarray],
    top_n: int = 1,
) -> float:
    """
    Proportion of total HH-frequency mass concentrated in the top-N components.

    Parameters
    ----------
    hh_freq : per-point HH selection frequency (from hh_selection_frequency)
    components : dict mapping comp_id -> array of point indices (from one seed)
    top_n : number of top components (by total frequency mass) to consider

    Returns
    -------
    Fraction of total HH frequency mass in the top-N components.
    """
    total_mass = float(hh_freq.sum())
    if total_mass < 1e-12 or len(components) == 0:
        return 0.0

    comp_masses = []
    for cid, indices in components.items():
        comp_masses.append(float(hh_freq[indices].sum()))
    comp_masses.sort(reverse=True)

    top_mass = sum(comp_masses[:top_n])
    return top_mass / total_mass


def component_centroid_stability(
    components_list: List[Dict[int, np.ndarray]],
    X_pca: np.ndarray,
    match_by: str = "largest",
) -> Dict[str, float]:
    """
    Measure stability of the largest HH component's centroid across seeds.

    Parameters
    ----------
    components_list : list of component dicts, one per seed
    X_pca : 2D PCA coordinates for all test points, shape (n_test, 2)
    match_by : "largest" matches the largest component from each seed

    Returns
    -------
    dict with mean_centroid_dist, std_centroid_dist, max_centroid_dist
    """
    centroids = []
    for components in components_list:
        if not components:
            continue
        largest_cid = max(components, key=lambda c: len(components[c]))
        indices = components[largest_cid]
        centroid = X_pca[indices].mean(axis=0)
        centroids.append(centroid)

    if len(centroids) < 2:
        return {
            "mean_centroid_dist": 0.0,
            "std_centroid_dist": 0.0,
            "max_centroid_dist": 0.0,
            "n_seeds_with_components": len(centroids),
        }

    centroids = np.array(centroids)
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    dists = np.array(dists)

    return {
        "mean_centroid_dist": float(dists.mean()),
        "std_centroid_dist": float(dists.std()),
        "max_centroid_dist": float(dists.max()),
        "n_seeds_with_components": len(centroids),
    }


def smoothed_overlap(
    hh_masks: List[np.ndarray],
    W: sparse.spmatrix,
    dilation_hops: int = 1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute Jaccard overlap of HH masks after neighborhood dilation.

    Dilation: a point is in the smoothed HH set if it OR any of its
    neighbors (within `dilation_hops` hops) is HH. This treats HH as
    a region rather than exact points.

    Parameters
    ----------
    hh_masks : list of boolean arrays (one per seed)
    W : sparse adjacency/weight matrix (n_test, n_test)
    dilation_hops : number of hops for dilation (1 = direct neighbors)

    Returns
    -------
    smoothed_jaccard_matrix : (n_seeds, n_seeds) pairwise Jaccard
    summary : dict with mean/min/max smoothed Jaccard
    """
    if not isinstance(W, sparse.csr_matrix):
        W = sparse.csr_matrix(W)
    W_binary = (W > 0).astype(float)

    dilated_masks = []
    for mask in hh_masks:
        dilated = mask.astype(float).copy()
        for _ in range(dilation_hops):
            neighbor_hh = W_binary @ dilated
            dilated = np.where((dilated > 0) | (neighbor_hh > 0), 1.0, 0.0)
        dilated_masks.append(dilated.astype(bool))

    n = len(dilated_masks)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(dilated_masks[i], dilated_masks[j]).sum()
            union = np.logical_or(dilated_masks[i], dilated_masks[j]).sum()
            J[i, j] = inter / union if union > 0 else 1.0

    off_diag = J[~np.eye(n, dtype=bool)]
    summary = {
        "mean_smoothed_jaccard": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "min_smoothed_jaccard": float(off_diag.min()) if len(off_diag) > 0 else 0.0,
        "max_smoothed_jaccard": float(off_diag.max()) if len(off_diag) > 0 else 0.0,
    }
    return J, summary


def summarize_region_stability(
    hh_masks: List[np.ndarray],
    components_list: List[Dict[int, np.ndarray]],
    X_pca: np.ndarray,
    W: sparse.spmatrix,
    dilation_hops: int = 1,
) -> Dict[str, float]:
    """
    Comprehensive region-level stability summary combining all three metrics.

    Returns a flat dict suitable for a thesis table row.
    """
    freq = hh_selection_frequency(hh_masks)

    freq_masses = []
    for comp in components_list:
        if comp:
            freq_masses.append(hh_frequency_mass(freq, comp, top_n=1))
    mean_freq_mass = float(np.mean(freq_masses)) if freq_masses else 0.0

    centroid = component_centroid_stability(components_list, X_pca)

    _, smooth = smoothed_overlap(hh_masks, W, dilation_hops=dilation_hops)

    point_summary = summarize_hh_stability(hh_masks)

    return {
        **point_summary,
        "mean_hh_frequency_mass_top1": mean_freq_mass,
        "mean_centroid_dist": centroid["mean_centroid_dist"],
        "max_centroid_dist": centroid["max_centroid_dist"],
        "n_seeds_with_components": centroid["n_seeds_with_components"],
        **smooth,
    }
