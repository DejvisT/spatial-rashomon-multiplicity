from typing import Dict, Tuple
import numpy as np
import pandas as pd

from scipy import sparse

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