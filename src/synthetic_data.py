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
    outlier_type: Optional[np.ndarray] = None  # For three-islands-plus-outliers dataset: -1=not outlier, 0=low-var outlier, 1=high-var outlier


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
    outlier_type: Optional[np.ndarray] = None  # -1 = not outlier, 0 = low-var, 1 = high-var
    boundary_mask: Optional[np.ndarray] = None  # ordinary decision-boundary strip (structural design)


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


def make_synth_three_islands_plus_outliers(
    *,
    n_samples: int = 5000,
    p_islands: float = 0.30,
    p_outliers: float = 0.02,
    p_outliers_high_var: float = 0.50,
    stable_sep: float = 5.0,
    stable_std: float = 0.9,
    island_centers: Optional[List[Tuple[float, float]]] = None,
    island_radius: float = 2.0,
    island_delta: float = 0.30,
    outlier_radius: float = 1.0,
    outlier_centers_high_var: Optional[List[Tuple[float, float]]] = None,
    outlier_centers_low_var: Optional[List[Tuple[float, float]]] = None,
    random_state: int = 0,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, list], "SyntheticGT"]:
    """
    Synthetic dataset with stable blobs, three ambiguous islands, and two types of outliers:

    - low-variance outliers: isolated points with stable label probability
    - high-variance outliers: small ambiguous mini-clusters with weak XOR structure

    Islands and high-variance outliers use a weak XOR signal (p=0.5 +/- delta),
    which creates genuine model disagreement: tree-based models can learn the XOR
    boundary while linear models predict ~0.5 locally.

    Ground truth:
        island_mask
        outlier_mask
        stable_mask
        p_true
        outlier_type  (-1 = not outlier, 0 = low-var outlier, 1 = high-var outlier)
    """
    if island_centers is None:
        island_centers = [(-6.0, 6.0), (6.0, 6.0), (0.0, -6.0)]
    if len(island_centers) != 3:
        raise ValueError("Provide exactly 3 island centers (or leave default).")
    if not (0.0 < island_delta < 0.5):
        raise ValueError("island_delta must be in (0, 0.5).")
    if not (0.0 <= p_outliers_high_var <= 1.0):
        raise ValueError("p_outliers_high_var must be in [0, 1].")

    if outlier_centers_high_var is None:
        # Chosen far from the stable blobs/islands and with opposite-sign geometry
        # so they are visually distinct and easy to interpret.
        outlier_centers_high_var = [(-18.0, 10.0), (18.0, -10.0)]

    if outlier_centers_low_var is None:
        # Chosen far away and near "stable side" regions where the label is clearer.
        outlier_centers_low_var = [(-18.0, -10.0), (18.0, 10.0)]

    rng = np.random.default_rng(random_state)
    n_out = int(round(n_samples * p_outliers))
    n_out_high = int(round(n_out * p_outliers_high_var))
    n_out_low = n_out - n_out_high

    n_islands_total = int(round(n_samples * p_islands))
    n_stable = n_samples - n_out - n_islands_total
    if n_stable <= 0:
        raise ValueError("Increase n_samples or reduce p_islands/p_outliers.")

    # ------------------------------------------------------------------
    # Stable blobs
    # ------------------------------------------------------------------
    n0 = n_stable // 2
    n1 = n_stable - n0
    X0 = rng.normal(loc=[-stable_sep, 0.0], scale=stable_std, size=(n0, 2))
    X1 = rng.normal(loc=[+stable_sep, 0.0], scale=stable_std, size=(n1, 2))
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)
    X_stable = np.vstack([X0, X1])
    y_stable = np.concatenate([y0, y1])
    p_true_stable = y_stable.astype(float)

    # ------------------------------------------------------------------
    # Three ambiguous islands (weak XOR)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Outliers: LOW-VARIANCE type
    # ------------------------------------------------------------------
    # These are placed in small far-away regions and receive stable probabilities.
    # Left-side centers -> p_true = 0
    # Right-side centers -> p_true = 1
    # This makes it much clearer they should *not* be high-multiplicity hotspots.
    X_out_low = []
    p_true_out_low = []

    if n_out_low > 0:
        counts_low = [n_out_low // len(outlier_centers_low_var)] * len(outlier_centers_low_var)
        counts_low[-1] += n_out_low - sum(counts_low)

        for c, n_i in zip(outlier_centers_low_var, counts_low):
            Xi = _sample_uniform_disk(rng, n_i, outlier_radius, c)
            X_out_low.append(Xi)

            # Stable probability depending on side
            p_i = np.ones(n_i, dtype=float) if c[0] > 0 else np.zeros(n_i, dtype=float)
            p_true_out_low.append(p_i)

        X_out_low = np.vstack(X_out_low)
        p_true_out_low = np.concatenate(p_true_out_low)
        y_out_low = rng.binomial(1, p_true_out_low).astype(int)
    else:
        X_out_low = np.zeros((0, 2))
        p_true_out_low = np.zeros((0,), dtype=float)
        y_out_low = np.zeros((0,), dtype=int)

    # ------------------------------------------------------------------
    # Outliers: HIGH-VARIANCE type
    # ------------------------------------------------------------------
    # These are small ambiguous mini-clusters using the same weak XOR mechanism
    # as the islands, so they are intended to be true localized high-multiplicity areas.
    X_out_high = []
    p_true_out_high = []

    if n_out_high > 0:
        counts_high = [n_out_high // len(outlier_centers_high_var)] * len(outlier_centers_high_var)
        counts_high[-1] += n_out_high - sum(counts_high)

        for c, n_i in zip(outlier_centers_high_var, counts_high):
            Xi = _sample_uniform_disk(rng, n_i, outlier_radius, c)
            pi = _weak_xor_probability(Xi, island_delta, c)

            X_out_high.append(Xi)
            p_true_out_high.append(pi)

        X_out_high = np.vstack(X_out_high)
        p_true_out_high = np.concatenate(p_true_out_high)
        y_out_high = rng.binomial(1, p_true_out_high).astype(int)
    else:
        X_out_high = np.zeros((0, 2))
        p_true_out_high = np.zeros((0,), dtype=float)
        y_out_high = np.zeros((0,), dtype=int)

    # ------------------------------------------------------------------
    # Combine outliers
    # ------------------------------------------------------------------
    X_out = np.vstack([X_out_low, X_out_high])
    p_true_out = np.concatenate([p_true_out_low, p_true_out_high])
    y_out = np.concatenate([y_out_low, y_out_high])

    outlier_type = np.concatenate([
        np.full(n_stable + n_islands_total, -1, dtype=int),   # not outlier
        np.zeros(len(X_out_low), dtype=int),                  # low-var outlier
        np.ones(len(X_out_high), dtype=int),                  # high-var outlier
    ])

    # ------------------------------------------------------------------
    # Combine all data
    # ------------------------------------------------------------------
    X_all = np.vstack([X_stable, X_islands, X_out])
    y_all = np.concatenate([y_stable, y_islands, y_out])
    p_true_all = np.concatenate([p_true_stable, p_true_islands, p_true_out])

    island_id = np.concatenate([
        np.full(n_stable, -1, dtype=int),
        island_id_islands,
        np.full(len(X_out), 99, dtype=int),
    ])

    island_mask = (island_id >= 0) & (island_id <= 2)
    outlier_mask = (island_id == 99)
    stable_mask = (island_id == -1)

    if shuffle:
        perm = rng.permutation(len(X_all))
        X_all = X_all[perm]
        y_all = y_all[perm]
        p_true_all = p_true_all[perm]
        island_id = island_id[perm]
        island_mask = island_mask[perm]
        outlier_mask = outlier_mask[perm]
        stable_mask = stable_mask[perm]
        outlier_type = outlier_type[perm]

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
        outlier_type=outlier_type,
    )
    return X, y, feature_info, gt


# ---------------------------------------------------------------------------
# Non-boundary islands (COMPAS-like mechanism)
# ---------------------------------------------------------------------------


def make_synth_nonboundary_islands(
    *,
    n_stable_neg: int = 3000,
    n_stable_pos: int = 3000,
    n_boundary: int = 5000,
    n_each_exception: int = 300,
    stable_neg_center: Tuple[float, float] = (-3.0, 0.0),
    stable_pos_center: Tuple[float, float] = (3.0, 0.0),
    stable_std: float = 0.9,
    pos_exception_center: Tuple[float, float] = (-3.0, 3.0),
    neg_exception_center: Tuple[float, float] = (3.0, 3.0),
    exception_radius: float = 0.35,
    boundary_x1_std: float = 0.35,
    boundary_x2_range: Tuple[float, float] = (-2.0, 2.0),
    p_stable_neg: float = 0.05,
    p_stable_pos: float = 0.95,
    p_boundary: float = 0.50,
    random_state: int = 0,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, list], SyntheticGT]:
    """
    Synthetic dataset with stable blobs, a dense ordinary boundary strip, and
    two structural exception islands placed away from the ensemble decision boundary.

    Regions:
      - stable negative blob (p ≈ 0.05) and stable positive blob (p ≈ 0.95)
      - ordinary boundary strip around x1 = 0 (p = 0.50), dense aleatoric ambiguity
      - positive exception inside the negative side (p ≈ 0.95)
      - negative exception inside the positive side (p ≈ 0.05)

    The structural exception disks are the expected HH regions: localized multiplicity
    away from the ordinary x1 = 0 decision boundary (COMPAS-like mechanism).

    Ground truth:
        island_mask     — structural exception islands (expected HH)
        boundary_mask   — ordinary boundary strip (not the primary recovery target)
        stable_mask     — stable negative + stable positive blobs
        island_centers  — exception disk centers
        island_radius   — exception disk radius
    """
    if n_each_exception < 1:
        raise ValueError("n_each_exception must be >= 1.")
    if not (0.0 < p_stable_neg < p_stable_pos < 1.0):
        raise ValueError("Require 0 < p_stable_neg < p_stable_pos < 1.")

    rng = np.random.default_rng(random_state)

    X_neg = rng.normal(
        loc=stable_neg_center,
        scale=stable_std,
        size=(n_stable_neg, 2),
    )
    p_neg = np.full(n_stable_neg, p_stable_neg)

    X_pos = rng.normal(
        loc=stable_pos_center,
        scale=stable_std,
        size=(n_stable_pos, 2),
    )
    p_pos = np.full(n_stable_pos, p_stable_pos)

    X_boundary = np.column_stack([
        rng.normal(0.0, boundary_x1_std, n_boundary),
        rng.uniform(boundary_x2_range[0], boundary_x2_range[1], n_boundary),
    ])
    p_boundary_arr = np.full(n_boundary, p_boundary)

    X_pos_exception = _sample_uniform_disk(
        rng, n_each_exception, exception_radius, pos_exception_center
    )
    p_pos_exception = np.full(n_each_exception, p_stable_pos)

    X_neg_exception = _sample_uniform_disk(
        rng, n_each_exception, exception_radius, neg_exception_center
    )
    p_neg_exception = np.full(n_each_exception, p_stable_neg)

    X_all = np.vstack([
        X_neg,
        X_pos,
        X_boundary,
        X_pos_exception,
        X_neg_exception,
    ])
    p_true_all = np.concatenate([
        p_neg,
        p_pos,
        p_boundary_arr,
        p_pos_exception,
        p_neg_exception,
    ])
    y_all = rng.binomial(1, p_true_all).astype(int)

    n_total = len(y_all)
    island_id = np.concatenate([
        np.full(n_stable_neg + n_stable_pos, -1, dtype=int),
        np.full(n_boundary, 2, dtype=int),
        np.full(n_each_exception, 0, dtype=int),
        np.full(n_each_exception, 1, dtype=int),
    ])
    island_mask = np.isin(island_id, [0, 1])
    boundary_mask = island_id == 2
    stable_mask = island_id == -1
    outlier_mask = np.zeros(n_total, dtype=bool)

    if shuffle:
        perm = rng.permutation(n_total)
        X_all = X_all[perm]
        y_all = y_all[perm]
        p_true_all = p_true_all[perm]
        island_id = island_id[perm]
        island_mask = island_mask[perm]
        boundary_mask = boundary_mask[perm]
        stable_mask = stable_mask[perm]
        outlier_mask = outlier_mask[perm]

    X = pd.DataFrame(X_all, columns=["x1", "x2"])
    y = pd.Series(y_all, name="target")
    feature_info = {"numeric": ["x1", "x2"], "categorical": []}

    gt = SyntheticGT(
        island_id=island_id,
        island_mask=island_mask,
        outlier_mask=outlier_mask,
        stable_mask=stable_mask,
        p_true=p_true_all,
        island_centers=[pos_exception_center, neg_exception_center],
        island_radius=exception_radius,
        boundary_mask=boundary_mask,
    )
    return X, y, feature_info, gt