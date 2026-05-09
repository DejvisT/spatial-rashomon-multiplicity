"""
Default symmetrized kNN neighborhood sizes per dataset for spatial analysis.

Values are aligned with Notebook 05 (``05_sensitivity_kNN.ipynb``): the smallest
candidate *k* in the diagnostic grid such that the aggregated connectivity table
over outer runs on *transformed test* features has ``comp_max == 1`` and
``frac_min`` approximately ``1.0`` (fully connected graph in every run).

Defaults may be set slightly above that minimum when noted (e.g. COMPAS uses 30
while the minimum fully connected *k* is often 20, for consistency with sensitivity
analyses and stability).

Re-run Notebook 05 after new data or candidate *k* grids; update this module if the
printed ``chosen_k`` table changes.
"""
from __future__ import annotations

from typing import Dict

K_NN_BY_DATASET: Dict[str, int] = {
    "compas": 30,
    "german": 30,
    "adult": 60
}


def default_k_nn(dataset_name: str) -> int:
    """Return default *k* for spatial kNN weights; fallback 30 if unknown."""
    return K_NN_BY_DATASET.get(dataset_name, 30)
