"""Lightweight parquet file cache for notebook analysis results."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def load_or_compute_df(
    cache_path: Path,
    compute_fn: Callable[[], pd.DataFrame],
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load a cached DataFrame from ``cache_path`` if it exists and ``force`` is False.

    Otherwise run ``compute_fn()``, save the result as parquet, and return it.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and cache_path.is_file():
        return pd.read_parquet(cache_path)

    df = compute_fn()
    df.to_parquet(cache_path, index=False)
    return df
