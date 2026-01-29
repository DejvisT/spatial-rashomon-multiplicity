from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.stats import zscore


# ---------------------------------------------------------------------
# kNN graph construction
# ---------------------------------------------------------------------

def build_knn_graph(
    X: pd.DataFrame | np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    standardize: bool = True,
) -> sparse.csr_matrix:
    """
    Build a row-normalized kNN weight matrix.

    Parameters
    ----------
    X : array-like of shape (n_obs, n_features)
        If DataFrame, only numeric columns are used
    k : number of neighbors (excluding self)
    metric : distance metric
    standardize : z-score features before kNN

    Returns
    -------
    W : sparse CSR matrix of shape (n_obs, n_obs)
        Row-normalized spatial weights
    """
    # If DataFrame, select only numeric columns
    if isinstance(X, pd.DataFrame):
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            raise ValueError(
                "X must contain at least one numeric column for kNN graph construction."
            )
        X = X_numeric
    
    X_arr = np.asarray(X)
    
    # Ensure we have numeric data (handle mixed-type arrays)
    if X_arr.dtype == object:
        raise ValueError(
            "X contains non-numeric data. "
            "If using a DataFrame, ensure it has numeric columns."
        )
    
    # Convert to float if needed for standardization
    if standardize and not np.issubdtype(X_arr.dtype, np.floating):
        X_arr = X_arr.astype(float)

    if standardize:
        X_arr = zscore(X_arr, axis=0, nan_policy='omit')

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X_arr)
    distances, indices = nn.kneighbors(X_arr)

    n = X_arr.shape[0]
    rows, cols, data = [], [], []

    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Row-normalize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = sparse.diags(1.0 / row_sums) @ W

    return W


# ---------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------

def moran_global(
    v: np.ndarray,
    W: sparse.csr_matrix,
    permutations: int = 999,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute global Moran's I with permutation test.
    """
    rng = np.random.RandomState(seed)

    v = np.asarray(v)
    v_centered = v - v.mean()

    num = v_centered @ (W @ v_centered)
    den = v_centered @ v_centered
    I = num / den

    permuted = []
    for _ in range(permutations):
        vp = rng.permutation(v_centered)
        permuted.append(vp @ (W @ vp) / den)

    permuted = np.asarray(permuted)
    p_value = (np.abs(permuted) >= np.abs(I)).mean()

    return {
        "I": float(I),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------

def lisa_local(
    v: np.ndarray,
    W: sparse.csr_matrix,
    permutations: int = 999,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute local Moran's I (LISA) with permutation tests.
    Returns cluster labels: HH, HL, LH, LL, NS.
    """
    rng = np.random.RandomState(seed)

    v = np.asarray(v)
    z = (v - v.mean()) / v.std(ddof=0)

    lag_z = W @ z
    Ii = z * lag_z

    # permutation test per observation
    p_vals = np.zeros_like(z)
    for i in range(len(z)):
        perm_stats = []
        for _ in range(permutations):
            zp = rng.permutation(z)
            perm_stats.append(z[i] * (W[i] @ zp))
        perm_stats = np.asarray(perm_stats)
        p_vals[i] = (np.abs(perm_stats) >= np.abs(Ii[i])).mean()

    if fdr:
        order = np.argsort(p_vals)
        ranked_p = p_vals[order]
        thresh = alpha * (np.arange(1, len(p_vals) + 1) / len(p_vals))
        passed = ranked_p <= thresh
        cutoff = ranked_p[passed].max() if np.any(passed) else 0.0
        sig = p_vals <= cutoff
    else:
        sig = p_vals <= alpha

    cluster = np.full(len(z), "NS", dtype=object)

    for i in range(len(z)):
        if not sig[i]:
            continue
        if z[i] > 0 and lag_z[i] > 0:
            cluster[i] = "HH"
        elif z[i] > 0 and lag_z[i] < 0:
            cluster[i] = "HL"
        elif z[i] < 0 and lag_z[i] > 0:
            cluster[i] = "LH"
        elif z[i] < 0 and lag_z[i] < 0:
            cluster[i] = "LL"

    return pd.DataFrame({
        "Ii": Ii,
        "p_value": p_vals,
        "cluster": cluster,
    })


# ---------------------------------------------------------------------
# Hotspot components (connected HH regions)
# ---------------------------------------------------------------------

def extract_hh_components(
    lisa_df: pd.DataFrame,
    W: sparse.csr_matrix,
    min_size: int = 5,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract connected components of HH points.

    Returns
    -------
    comp_id : array of shape (n_obs,), -1 for non-HH
    components : dict mapping comp_id -> indices
    """
    hh_mask = lisa_df["cluster"].values == "HH"
    n = len(hh_mask)

    visited = np.zeros(n, dtype=bool)
    comp_id = -np.ones(n, dtype=int)
    components = {}
    cid = 0

    for i in range(n):
        if not hh_mask[i] or visited[i]:
            continue

        stack = [i]
        members = []

        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            members.append(u)

            neighbors = W[u].indices
            for v in neighbors:
                if hh_mask[v] and not visited[v]:
                    stack.append(v)

        if len(members) >= min_size:
            components[cid] = np.array(members)
            for m in members:
                comp_id[m] = cid
            cid += 1

    return comp_id, components
