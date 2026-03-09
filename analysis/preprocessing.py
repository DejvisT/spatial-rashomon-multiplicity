"""
Preprocessing for analysis: transform test set using the same logic as training.
Single source of truth so notebooks and experiment runner stay consistent with
the training pipeline (scale_numeric=True, one-hot, column ordering).
"""
from pathlib import Path
from typing import Union

import numpy as np

# Ensure project root and src on path for data loading
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from data import load_dataset, make_preprocessor  # noqa: E402

from analysis.run_analysis import load_split  # noqa: E402

PathLike = Union[str, Path]


def get_transformed_test_features(run_dir: PathLike, dataset_name: str) -> np.ndarray:
    """
    Return the preprocessed test feature matrix for a run.
    Uses the same preprocessing as the training pipeline:
    - load_dataset(dataset_name)
    - make_preprocessor(feature_info, scale_numeric=True)
    - fit on train only, transform test
    Column ordering and one-hot behavior match training.
    """
    run_dir = Path(run_dir)
    split = load_split(run_dir)
    X, y, feature_info = load_dataset(dataset_name)
    preprocessor = make_preprocessor(feature_info, scale_numeric=True)
    train_idx = split["train"]
    test_idx = split["test"]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    preprocessor.fit(X_train, y_train)
    X_test = X.iloc[test_idx]
    return preprocessor.transform(X_test)
