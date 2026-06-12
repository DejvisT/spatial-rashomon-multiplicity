from __future__ import annotations

from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]


def get_run_dirs(dataset_dir: PathLike) -> List[Path]:
    """List run directories named seed=* under dataset_dir, sorted by seed."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []

    run_dirs = []
    for p in dataset_dir.iterdir():
        if p.is_dir() and p.name.startswith("seed="):
            try:
                seed_val = int(p.name.split("=")[1])
                run_dirs.append((seed_val, p))
            except (IndexError, ValueError):
                continue

    run_dirs.sort(key=lambda x: x[0])
    return [p for _, p in run_dirs]