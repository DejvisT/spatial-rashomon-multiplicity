from typing import Dict, List
import numpy as np


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