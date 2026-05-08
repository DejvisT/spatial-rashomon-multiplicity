from typing import Dict, Optional, Tuple
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
        # zscore leaves NaN where variance is 0; NearestNeighbors rejects NaN
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanitize any remaining NaN/inf (e.g. from input) so sklearn accepts X
    if not np.isfinite(X_arr).all():
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

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


def build_knn_graph_gower(
    X: pd.DataFrame,
    k: int = 10,
    cat_cols: Optional[list] = None,
) -> sparse.csr_matrix:
    """
    Build a kNN graph using Gower distance, which handles mixed numeric
    and categorical features. Useful for datasets like Adult with both types.
    
    Parameters
    ----------
    X : DataFrame with mixed types
    k : number of neighbors
    cat_cols : list of categorical column names (auto-detected if None)
    
    Returns
    -------
    W : sparse CSR matrix (row-normalized)
    """
    X = X.copy()
    
    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    n = len(X)
    
    # Compute Gower distance matrix
    dist_matrix = np.zeros((n, n))
    
    # Numeric: normalized Manhattan distance
    for col in num_cols:
        vals = X[col].values.astype(float)
        rng = vals.max() - vals.min()
        if rng > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    d = abs(vals[i] - vals[j]) / rng
                    dist_matrix[i, j] += d
                    dist_matrix[j, i] += d
    
    # Categorical: 0 if same, 1 if different
    for col in cat_cols:
        vals = X[col].values
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.0 if vals[i] == vals[j] else 1.0
                dist_matrix[i, j] += d
                dist_matrix[j, i] += d
    
    n_features = len(num_cols) + len(cat_cols)
    if n_features > 0:
        dist_matrix /= n_features
    
    # Build kNN from distance matrix
    rows_list, cols_list, data_list = [], [], []
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1:k+1]
        for j in neighbors:
            rows_list.append(i)
            cols_list.append(j)
            data_list.append(1.0)
    
    W = sparse.csr_matrix((data_list, (rows_list, cols_list)), shape=(n, n))
    
    # Row-normalize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = sparse.diags(1.0 / row_sums) @ W
    
    return W


def build_knn_graph_pca(
    X: pd.DataFrame | np.ndarray,
    k: int = 10,
    n_components: int = 15,
    standardize: bool = True,
) -> sparse.csr_matrix:
    """
    Build a kNN graph on PCA-reduced features.
    
    Parameters
    ----------
    X : feature array or DataFrame
    k : number of neighbors
    n_components : PCA dimensions to retain
    standardize : z-score before PCA
    
    Returns
    -------
    W : sparse CSR matrix (row-normalized)
    """
    from sklearn.decomposition import PCA
    
    if isinstance(X, pd.DataFrame):
        X_numeric = X.select_dtypes(include=[np.number])
        X_arr = X_numeric.values.astype(float)
    else:
        X_arr = np.asarray(X, dtype=float)
    
    if standardize:
        X_arr = zscore(X_arr, axis=0, nan_policy='omit')
    
    n_comp = min(n_components, X_arr.shape[1], X_arr.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_arr)
    
    return build_knn_graph(X_pca, k=k, standardize=False)


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
    if den < 1e-12:
        return {"I": 0.0, "p_value": 1.0}
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
    std = v.std(ddof=0)
    if std < 1e-12:
        return pd.DataFrame({
            "Ii": np.zeros(len(v)),
            "p_value": np.ones(len(v)),
            "cluster": np.full(len(v), "NS", dtype=object),
        })
    z = (v - v.mean()) / std

    lag_z = W @ z
    Ii = z * lag_z

    # Conditional permutation test per observation (hold z[i] fixed)
    p_vals = np.zeros_like(z)
    for i in range(len(z)):
        others = np.delete(z, i)
        perm_stats = []
        for _ in range(permutations):
            others_perm = rng.permutation(others)
            z_perm = np.insert(others_perm, i, z[i])
            perm_stats.append(z[i] * (W[i] @ z_perm))
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
    W: sparse.spmatrix,
    min_size: int = 5,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract connected components of HH points.
    W is converted to CSR internally if needed (e.g. PySAL may provide COO).

    Returns
    -------
    comp_id : array of shape (n_obs,), -1 for non-HH
    components : dict mapping comp_id -> indices
    """
    hh_mask = lisa_df["cluster"].values == "HH"
    n = len(hh_mask)

    # Row indexing W[u] requires CSR; PySAL may give COO
    if not isinstance(W, sparse.csr_matrix):
        if hasattr(W, "sparse"):
            W = W.sparse
        elif hasattr(W, "to_sparse"):
            W = W.to_sparse()
        else:
            W = sparse.csr_matrix(W)

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


# ---------------------------------------------------------------------
# Regionality metrics
# ---------------------------------------------------------------------

def compute_regionality_metrics(
    lisa_df: pd.DataFrame,
    n_total: int,
) -> Dict[str, float]:
    """
    Compute region-level summary metrics from LISA results.

    Returns
    -------
    dict with:
        frac_hh : |HH| / N
        frac_hl : |HL| / N
        frac_ll : |LL| / N
        frac_lh : |LH| / N
        regionality_ratio : |HH| / (|HH| + |HL|)
            High ratio => high-variance points form clusters, not outliers.
        n_hh, n_hl, n_ll, n_lh : raw counts
    """
    clusters = lisa_df["cluster"].values
    n_hh = int(np.sum(clusters == "HH"))
    n_hl = int(np.sum(clusters == "HL"))
    n_ll = int(np.sum(clusters == "LL"))
    n_lh = int(np.sum(clusters == "LH"))

    denom_regionality = n_hh + n_hl
    regionality_ratio = n_hh / denom_regionality if denom_regionality > 0 else float("nan")

    return {
        "n_hh": n_hh,
        "n_hl": n_hl,
        "n_ll": n_ll,
        "n_lh": n_lh,
        "frac_hh": n_hh / n_total if n_total > 0 else 0.0,
        "frac_hl": n_hl / n_total if n_total > 0 else 0.0,
        "frac_ll": n_ll / n_total if n_total > 0 else 0.0,
        "frac_lh": n_lh / n_total if n_total > 0 else 0.0,
        "regionality_ratio": regionality_ratio,
    }