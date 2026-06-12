from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from libpysal.weights import KNN as PySAL_KNN
from libpysal.weights import W as PySAL_W
from esda.moran import Moran, Moran_Local

from analysis.run_analysis import fdr_benjamini_hochberg


def build_cosine_knn_weights(X: np.ndarray, k: int = 30) -> PySAL_W:
    """Build a symmetrized row-standardized kNN graph using cosine distance."""
    X = np.asarray(X, dtype=float)
    X_norm = normalize(X, norm="l2")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X_norm)
    _, indices = nn.kneighbors(X_norm)

    neighbors = {}
    weights = {}

    for i in range(X.shape[0]):
        neighbors[i] = indices[i, 1:].tolist()
        w = np.ones(k, dtype=float)
        w = w / w.sum()
        weights[i] = w.tolist()

    W = PySAL_W(neighbors, weights)
    W = W.symmetrize(inplace=False)
    W.transform = "r"
    return W


def build_pca_knn_weights(
    X: np.ndarray,
    n_components: int = 5,
    k: int = 30,
) -> PySAL_KNN:
    """Build a symmetrized row-standardized kNN graph in PCA-reduced space."""
    X = np.asarray(X, dtype=float)
    n_components = min(n_components, X.shape[1], X.shape[0])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    W = PySAL_KNN.from_array(X_pca, k=k)
    W = W.symmetrize(inplace=False)
    W.transform = "r"
    return W


def spatial_with_custom_W(
    values: np.ndarray,
    W,
    *,
    permutations: int = 999,
    fdr_alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Compute Moran's I and HH LISA mask for a pre-built spatial weights matrix."""
    if seed is not None:
        np.random.seed(seed)

    values = np.asarray(values, dtype=float)

    moran_g = Moran(values, W, permutations=permutations)
    lm = Moran_Local(values, W, transformation="r", permutations=permutations, seed=seed)

    p_sim = np.asarray(lm.p_sim).flatten()
    q = np.asarray(lm.q).flatten()

    sig = fdr_benjamini_hochberg(p_sim, alpha=fdr_alpha)
    hh_mask = (q == 1) & sig

    return {
        "moran_i": float(moran_g.I),
        "moran_p": float(moran_g.p_sim),
        "n_hh": int(hh_mask.sum()),
        "HH_mask": hh_mask,
    }