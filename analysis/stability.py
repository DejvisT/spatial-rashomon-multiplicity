from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# HH selection stability (point-wise)
# ---------------------------------------------------------------------

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
        return 0.0

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
        - mean_jaccard
        - min_jaccard
        - max_jaccard
    """
    freqs = hh_selection_frequency(hh_masks)
    J = hh_jaccard_matrix(hh_masks)

    # ignore diagonal
    off_diag = J[~np.eye(J.shape[0], dtype=bool)]

    return {
        "mean_hh_fraction": float(freqs.mean()),
        "mean_jaccard": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "min_jaccard": float(off_diag.min()) if len(off_diag) > 0 else 0.0,
        "max_jaccard": float(off_diag.max()) if len(off_diag) > 0 else 0.0,
    }
