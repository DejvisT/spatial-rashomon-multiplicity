from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from libpysal.weights import KNN as PySAL_KNN
from libpysal.weights import W as PySAL_W
from esda.moran import Moran, Moran_Local

from analysis.knn_defaults import K_NN_BY_DATASET
from analysis.preprocessing import get_transformed_test_features
from analysis.run_analysis import (
    fdr_benjamini_hochberg,
    load_P_test,
    pointwise_variance,
    select_rashomon_global,
    spatial_analysis,
)


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

    pca = PCA(n_components=n_components, svd_solver="full")
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


def compute_alt_graph(
    results_dir: Path | str,
    datasets: list[str],
    seeds: list[int],
    K: int,
) -> pd.DataFrame:
    """Compare Moran's I across alternative graph construction methods."""
    results_dir = Path(results_dir)

    alt_rows = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = results_dir / dataset / f"seed={seed}"
            try:
                P_test = load_P_test(run_dir)
                idx = select_rashomon_global(run_dir, K=K)
                P_sel = P_test[idx]
                v = pointwise_variance(P_sel)
                X_test = get_transformed_test_features(run_dir, dataset)
                k_ds = K_NN_BY_DATASET[dataset]
                sp_base = spatial_analysis(v, X_test, k=k_ds, permutations=999)
                alt_rows.append({
                    "dataset": dataset, "seed": seed, "method": "euclidean",
                    "moran_i": sp_base["moran_i"],
                    "n_hh": int(sp_base["HH_mask"].sum()),
                })
                W_pca = build_pca_knn_weights(X_test, n_components=5, k=k_ds)
                sp_pca = spatial_with_custom_W(v, W_pca)
                alt_rows.append({
                    "dataset": dataset, "seed": seed, "method": "pca_5",
                    "moran_i": sp_pca["moran_i"],
                    "n_hh": sp_pca["n_hh"],
                })
                W_cos = build_cosine_knn_weights(X_test, k=k_ds)
                sp_cos = spatial_with_custom_W(v, W_cos)
                alt_rows.append({
                    "dataset": dataset, "seed": seed, "method": "cosine",
                    "moran_i": sp_cos["moran_i"],
                    "n_hh": sp_cos["n_hh"],
                })
                if seed == 0:
                    print(f"  {dataset} seed={seed}: euc={sp_base['moran_i']:.3f}, "
                          f"pca={sp_pca['moran_i']:.3f}, cos={sp_cos['moran_i']:.3f}")
            except Exception as e:
                print(f"  {dataset} seed={seed}: SKIP ({e})")
    return pd.DataFrame(alt_rows)