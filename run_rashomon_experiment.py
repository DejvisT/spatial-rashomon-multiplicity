# run_rashomon_experiment.py
# ---------------------------------------------------------------------
# Run Rashomon experiments (global / per-family / single-family)
# ---------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset, make_split, make_cv_splits, apply_split, make_preprocessor
from rashomon import build_rashomon_set
from metrics import compute_multiplicity_metrics


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["compas", "german", "breast_cancer"])
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--selection_mode", type=str, default="global",
                        choices=["global", "per_family"])

    parser.add_argument("--family", type=str, default="all",
                        help="Single family (LogReg, RF, GBM, MLP) or 'all'")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated model seeds for single-family (e.g. 42,1,2,3,100). Cycles over models.")

    parser.add_argument("--n_models", type=int, default=30)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)

    # Cross-validation options
    parser.add_argument("--use_cv", action="store_true",
                        help="Use cross-validation instead of single split")
    parser.add_argument("--cv_method", type=str, default="kfold",
                        choices=["kfold", "repeated_holdout"],
                        help="CV method: 'kfold' or 'repeated_holdout'")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for K-fold CV")
    parser.add_argument("--n_repeats", type=int, default=5,
                        help="Number of repeats for repeated holdout")

    parser.add_argument("--out_dir", type=str, default="results")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------------
    # Load data
    # -------------------------------
    X, y, feature_info = load_dataset(args.dataset)

    # -------------------------------
    # Create splits (CV or single split)
    # -------------------------------
    if args.use_cv:
        test_split, cv_splits = make_cv_splits(
            n_samples=len(X),
            cv_method=args.cv_method,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            test_size=args.test_size,
            seed=args.seed,
            stratify=y.values,
        )
        split = None  # Not used in CV mode
        splits = apply_split(X, y, {"test": test_split["test"]})
        X_test, y_test = splits["test"]
    else:
        split = make_split(
            n_samples=len(X),
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
            stratify=y.values,
        )
        test_split = None
        cv_splits = None
        splits = apply_split(X, y, split)
        X_test, y_test = splits["test"]

    preprocessor = make_preprocessor(feature_info)

    # -------------------------------
    # Determine run mode
    # -------------------------------
    if args.family != "all":
        run_label = f"family={args.family}"
        selection_mode = "global"   # ignored when family is set
        family = args.family
    else:
        run_label = args.selection_mode
        selection_mode = args.selection_mode
        family = None

    # -------------------------------
    # Output directory
    # -------------------------------
    out_dir = (
        Path(args.out_dir)
        / args.dataset
        / run_label
        / f"seed={args.seed}_eps={args.epsilon}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Run Rashomon
    # -------------------------------
    model_seeds = None
    if family is not None and args.seeds:
        model_seeds = [int(s) for s in args.seeds.split(",")]

    P_test, meta = build_rashomon_set(
        X=X,
        y=y,
        split=split,
        test_split=test_split,
        cv_splits=cv_splits,
        base_preprocessor=preprocessor,
        epsilon=args.epsilon,
        n_samples_per_model=args.n_models,
        seed=args.seed,
        selection_mode=selection_mode,
        family=family,
        model_seeds=model_seeds,
    )

    metrics = compute_multiplicity_metrics(P_test)

    # -------------------------------
    # Save artifacts
    # -------------------------------
    np.save(out_dir / "P_test.npy", P_test)
    np.savez(out_dir / "metrics.npz", **metrics)

    meta.to_csv(out_dir / "meta.csv", index=False)

    # Save split information
    if args.use_cv:
        np.savez(
            out_dir / "split.npz",
            test=test_split["test"],
            seed=test_split["seed"],
            cv_method=args.cv_method,
            n_folds=args.n_folds if args.cv_method == "kfold" else 0,
            n_repeats=args.n_repeats if args.cv_method == "repeated_holdout" else 0,
        )
        # Also save CV splits for reference
        cv_splits_dict = {f"fold_{i}": cv_split for i, cv_split in enumerate(cv_splits)}
        np.savez(out_dir / "cv_splits.npz", **cv_splits_dict)
    else:
        np.savez(
            out_dir / "split.npz",
            train=split["train"],
            val=split["val"],
            test=split["test"],
            seed=split["seed"],
        )

    X_test.to_csv(out_dir / "X_test.csv", index=False)
    np.save(out_dir / "y_test.npy", y_test.values)

    config = vars(args)
    config["n_models_rashomon"] = int(P_test.shape[0])
    config["use_cv"] = args.use_cv  # Ensure this is in config

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✔ Finished Rashomon run")
    print("  Dataset:", args.dataset)
    print("  Mode:", run_label)
    print("  Seed:", args.seed)
    print("  ε:", args.epsilon)
    if args.use_cv:
        print(f"  CV: {args.cv_method} ({args.n_folds if args.cv_method == 'kfold' else args.n_repeats} folds/repeats)")
    print("  Rashomon models:", P_test.shape[0])
    print("  Saved to:", out_dir)


if __name__ == "__main__":
    main()
