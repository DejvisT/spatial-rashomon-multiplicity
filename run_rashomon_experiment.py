import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import load_dataset, make_split, apply_split, make_preprocessor
from rashomon import build_rashomon_set
from metrics import compute_multiplicity_metrics


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run Rashomon experiment")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["compas", "german", "breast_cancer"])
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--n_models", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="results")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------------
    # Output directory
    # -------------------------------
    out_dir = Path(args.out_dir) / args.dataset / f"seed={args.seed}_eps={args.epsilon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Load data
    # -------------------------------
    X, y, feature_info = load_dataset(args.dataset)

    # -------------------------------
    # Split
    # -------------------------------
    split = make_split(
        n_samples=len(X),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        stratify=y.values,
    )

    splits_applied = apply_split(X, y, split)

    # -------------------------------
    # Preprocessor (definition only)
    # -------------------------------
    preprocessor = make_preprocessor(feature_info)

    # -------------------------------
    # Rashomon training
    # -------------------------------
    P_test, meta = build_rashomon_set(
        X=X,
        y=y,
        split=split,
        base_preprocessor=preprocessor,
        epsilon=args.epsilon,
        n_samples_per_model=args.n_models,
        seed=args.seed,
    )

    # -------------------------------
    # Multiplicity metrics
    # -------------------------------
    metrics = compute_multiplicity_metrics(P_test)

    # -------------------------------
    # Save artifacts
    # -------------------------------

    # Core predictions
    np.save(out_dir / "P_test.npy", P_test)

    # Metrics
    np.savez(out_dir / "metrics.npz", **metrics)

    # Model metadata
    meta.to_csv(out_dir / "meta.csv", index=False)

    # Split indices
    np.savez(
        out_dir / "split.npz",
        train=split["train"],
        val=split["val"],
        test=split["test"],
        seed=split["seed"],
    )

    # Feature space + labels for test
    X_test, y_test = splits_applied["test"]
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    np.save(out_dir / "y_test.npy", y_test.values)

    # Config for reproducibility
    config = vars(args)
    config["n_obs_test"] = len(X_test)
    config["n_models_rashomon"] = P_test.shape[0]

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✔ Results saved to {out_dir}")
    print(f"✔ Rashomon models: {P_test.shape[0]}")
    print(f"✔ Test observations: {P_test.shape[1]}")


if __name__ == "__main__":
    main()
