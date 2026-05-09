"""Synthetic binary classification datasets with known high-multiplicity (ambiguous) regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticGroundTruth:
    island_mask: np.ndarray      # True if point is in the ambiguous island region
    stable_mask: np.ndarray      # True if point is in the stable blob region
    xor_mask: np.ndarray         # True if x1*x2>0 inside the island (quadrants)
    p_true: np.ndarray           # True P(y=1|x) used to sample labels
    island_radius: float         # For plotting / reporting


def make_synthetic_multiplicity_dataset(
    *,
    n_samples: int = 2000,
    p_island: float = 0.20,           # fraction of points in ambiguous island
    stable_sep: float = 4.0,          # blob separation along x1
    stable_std: float = 0.8,          # blob spread
    island_radius: float = 2.0,       # island radius around origin
    island_delta: float = 0.30,       # island signal strength: p=0.5±delta (must be <0.5)
    random_state: int = 0,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, list], SyntheticGroundTruth]:
    """
    Synthetic binary classification dataset with a known high-multiplicity region.

    Stable region:
      - Two well-separated Gaussian blobs (dominant mass) -> models should agree.

    Ambiguous island:
      - Points uniformly in a disk around origin.
      - Labels follow a weak XOR-like probability pattern:
          p(y=1|x) = 0.5 + delta * sign(x1*x2)
        (so 0.5+delta in quadrants I&III, 0.5-delta in II&IV)
      - Tree-based models can learn the XOR boundary; linear models cannot,
        creating genuine inter-model disagreement and high predictive variance.

    Ground truth:
      - island_mask identifies where multiplicity should concentrate.
      - p_true gives the true conditional probability used for labels.
    """
    if not (0.0 < p_island < 1.0):
        raise ValueError("p_island must be in (0,1).")
    if not (0.0 < island_delta < 0.5):
        raise ValueError("island_delta must be in (0,0.5).")

    rng = np.random.default_rng(random_state)

    n_island = int(round(n_samples * p_island))
    n_stable = n_samples - n_island

    # ----- Stable region: two blobs -----
    n0 = n_stable // 2
    n1 = n_stable - n0

    X0 = rng.normal(loc=[-stable_sep, 0.0], scale=stable_std, size=(n0, 2))
    X1 = rng.normal(loc=[+stable_sep, 0.0], scale=stable_std, size=(n1, 2))
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)

    X_stable = np.vstack([X0, X1])
    y_stable = np.concatenate([y0, y1])

    # For stable region, treat p_true ~ 0 or 1 as "ground truth probability"
    p_true_stable = y_stable.astype(float)

    # ----- Ambiguous island: uniform disk around origin -----
    # Sample uniform in disk: radius = R*sqrt(U), angle=2πU
    U = rng.random(n_island)
    r = island_radius * np.sqrt(U)
    theta = 2.0 * np.pi * rng.random(n_island)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X_island = np.column_stack([x1, x2])

    xor = (X_island[:, 0] * X_island[:, 1] > 0.0)  # quadrants I&III -> True
    # p_true: 0.5 + delta if xor True else 0.5 - delta
    p_true_island = 0.5 + island_delta * (2.0 * xor.astype(float) - 1.0)
    y_island = rng.binomial(n=1, p=p_true_island).astype(int)

    # ----- Combine -----
    X_all = np.vstack([X_stable, X_island])
    y_all = np.concatenate([y_stable, y_island])
    p_true_all = np.concatenate([p_true_stable, p_true_island])

    island_mask = np.concatenate([np.zeros(n_stable, dtype=bool), np.ones(n_island, dtype=bool)])
    stable_mask = ~island_mask
    xor_mask = np.concatenate([np.zeros(n_stable, dtype=bool), xor])

    if shuffle:
        perm = rng.permutation(n_samples)
        X_all = X_all[perm]
        y_all = y_all[perm]
        p_true_all = p_true_all[perm]
        island_mask = island_mask[perm]
        stable_mask = stable_mask[perm]
        xor_mask = xor_mask[perm]

    X = pd.DataFrame(X_all, columns=["x1", "x2"])
    y = pd.Series(y_all, name="target")

    feature_info = {"numeric": ["x1", "x2"], "categorical": []}

    gt = SyntheticGroundTruth(
        island_mask=island_mask,
        stable_mask=stable_mask,
        xor_mask=xor_mask,
        p_true=p_true_all,
        island_radius=island_radius,
    )

    return X, y, feature_info, gt


# ---------------------------------------------------------------------------
# Three islands + outliers
# ---------------------------------------------------------------------------


@dataclass
class SyntheticGT:
    island_id: np.ndarray        # -1 = stable, 0/1/2 = island index, 99 = outlier
    island_mask: np.ndarray
    outlier_mask: np.ndarray
    stable_mask: np.ndarray
    p_true: np.ndarray           # true P(y=1|x)
    island_centers: List[Tuple[float, float]]
    island_radius: float


def _sample_uniform_disk(
    rng: np.random.Generator, n: int, radius: float, center: Tuple[float, float]
) -> np.ndarray:
    """Uniform points in a disk."""
    u = rng.random(n)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * rng.random(n)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.column_stack([x, y])


def _weak_xor_probability(X: np.ndarray, delta: float, center: Tuple[float, float]) -> np.ndarray:
    """
    Weak XOR-like signal around a center:
      p = 0.5 + delta if (x1-center1)*(x2-center2) > 0 else 0.5 - delta
    """
    z1 = X[:, 0] - center[0]
    z2 = X[:, 1] - center[1]
    xor = (z1 * z2 > 0.0).astype(float)
    return 0.5 + delta * (2.0 * xor - 1.0)


def _rejection_sample_outliers(
    rng: np.random.Generator,
    n: int,
    box: Tuple[float, float, float, float],
    *,
    min_dist_to_any: float,
    existing_points: np.ndarray,
    max_tries: int = 200000,
) -> np.ndarray:
    """Sample outliers uniformly in a box but far from existing points and from each other."""
    xmin, xmax, ymin, ymax = box
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        cand = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if existing_points.size > 0:
            d_exist = np.sqrt(((existing_points - cand) ** 2).sum(axis=1)).min()
            if d_exist < min_dist_to_any:
                continue
        if out:
            d_out = np.sqrt(((np.vstack(out) - cand) ** 2).sum(axis=1)).min()
            if d_out < min_dist_to_any:
                continue
        out.append(cand)
    if len(out) < n:
        raise RuntimeError(
            f"Could not place all outliers: placed {len(out)}/{n}. "
            "Try increasing box size or reducing min_dist_to_any."
        )
    return np.vstack(out)


def make_synth_three_islands_plus_outliers(
    *,
    n_samples: int = 5000,
    p_islands: float = 0.30,
    p_outliers: float = 0.02,
    stable_sep: float = 5.0,
    stable_std: float = 0.9,
    island_centers: Optional[List[Tuple[float, float]]] = None,
    island_radius: float = 2.0,
    island_delta: float = 0.30,
    outlier_box: Tuple[float, float, float, float] = (-20, 20, -15, 15),
    outlier_min_dist: float = 2.5,
    outlier_p: float = 0.5,
    random_state: int = 0,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, list], SyntheticGT]:
    """
    Synthetic dataset with: stable blobs, three ambiguous islands, isolated outliers.

    Islands use a weak XOR signal (p=0.5 +/- delta) that creates genuine
    model disagreement: tree-based models learn the XOR boundary while
    linear models predict ~0.5 everywhere in the island.

    Ground truth: island_mask (3 islands), outlier_mask, stable_mask, p_true.
    """
    if island_centers is None:
        island_centers = [(-6.0, 6.0), (6.0, 6.0), (0.0, -6.0)]
    if len(island_centers) != 3:
        raise ValueError("Provide exactly 3 island centers (or leave default).")
    if not (0.0 < island_delta < 0.5):
        raise ValueError("island_delta must be in (0, 0.5).")

    rng = np.random.default_rng(random_state)
    n_out = int(round(n_samples * p_outliers))
    n_islands_total = int(round(n_samples * p_islands))
    n_stable = n_samples - n_out - n_islands_total
    if n_stable <= 0:
        raise ValueError("Increase n_samples or reduce p_islands/p_outliers.")

    n0 = n_stable // 2
    n1 = n_stable - n0
    X0 = rng.normal(loc=[-stable_sep, 0.0], scale=stable_std, size=(n0, 2))
    X1 = rng.normal(loc=[+stable_sep, 0.0], scale=stable_std, size=(n1, 2))
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)
    X_stable = np.vstack([X0, X1])
    y_stable = np.concatenate([y0, y1])
    p_true_stable = y_stable.astype(float)

    n_each = [n_islands_total // 3] * 3
    n_each[-1] += n_islands_total - sum(n_each)
    X_islands = []
    y_islands = []
    p_true_islands = []
    island_id_islands = []
    for idx, (c, n_i) in enumerate(zip(island_centers, n_each)):
        Xi = _sample_uniform_disk(rng, n_i, island_radius, c)
        pi = _weak_xor_probability(Xi, island_delta, c)
        yi = rng.binomial(1, pi).astype(int)
        X_islands.append(Xi)
        y_islands.append(yi)
        p_true_islands.append(pi)
        island_id_islands.append(np.full(n_i, idx, dtype=int))

    X_islands = np.vstack(X_islands) if n_islands_total > 0 else np.zeros((0, 2))
    y_islands = np.concatenate(y_islands) if n_islands_total > 0 else np.zeros((0,), dtype=int)
    p_true_islands = np.concatenate(p_true_islands) if n_islands_total > 0 else np.zeros((0,), dtype=float)
    island_id_islands = np.concatenate(island_id_islands) if n_islands_total > 0 else np.zeros((0,), dtype=int)

    existing = np.vstack([X_stable, X_islands]) if (X_islands.size > 0) else X_stable
    X_out = _rejection_sample_outliers(
        rng, n_out, outlier_box, min_dist_to_any=outlier_min_dist, existing_points=existing
    )
    p_true_out = np.full(n_out, float(outlier_p))
    y_out = rng.binomial(1, p_true_out).astype(int)

    X_all = np.vstack([X_stable, X_islands, X_out])
    y_all = np.concatenate([y_stable, y_islands, y_out])
    p_true_all = np.concatenate([p_true_stable, p_true_islands, p_true_out])
    island_id = np.concatenate([
        np.full(n_stable, -1, dtype=int),
        island_id_islands,
        np.full(n_out, 99, dtype=int),
    ])
    island_mask = (island_id >= 0) & (island_id <= 2)
    outlier_mask = (island_id == 99)
    stable_mask = (island_id == -1)

    if shuffle:
        perm = rng.permutation(n_samples)
        X_all = X_all[perm]
        y_all = y_all[perm]
        p_true_all = p_true_all[perm]
        island_id = island_id[perm]
        island_mask = island_mask[perm]
        outlier_mask = outlier_mask[perm]
        stable_mask = stable_mask[perm]

    X = pd.DataFrame(X_all, columns=["x1", "x2"])
    y = pd.Series(y_all, name="target")
    feature_info = {"numeric": ["x1", "x2"], "categorical": []}
    gt = SyntheticGT(
        island_id=island_id,
        island_mask=island_mask,
        outlier_mask=outlier_mask,
        stable_mask=stable_mask,
        p_true=p_true_all,
        island_centers=island_centers,
        island_radius=island_radius,
    )
    return X, y, feature_info, gt