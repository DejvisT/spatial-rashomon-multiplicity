#!/usr/bin/env python
"""
Run training with a fixed test set across outer seeds.

This experiment is intended for observation-level cross-seed stability analyses
(e.g., HH mask overlap), where all runs must be evaluated on the same test
observations.

Usage:
  python run_training_pipeline_fixed_test.py --dataset compas
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data import load_dataset, make_preprocessor, make_split, make_split_with_fixed_test
from training_pipeline import (
    run_one_training_run,
    save_run_artifacts,
    TRAINING_MODEL_CONFIGS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Training pipeline with a fixed test split across outer seeds"
    )
    p.add_argument("--dataset", type=str, required=True, choices=["compas", "german", "breast_cancer"])
    p.add_argument("--out_dir", type=str, default="results_fixed_test", help="Base output directory")
    p.add_argument("--n_outer", type=int, default=10, help="Number of outer runs (seeds)")
    p.add_argument("--n_candidates", type=int, default=50, help="Candidates per model family per run")
    p.add_argument("--save_models", action="store_true", help="Save trained pipeline objects (pickles)")
    p.add_argument("--seed_start", type=int, default=0, help="First outer seed")
    p.add_argument("--fixed_test_seed", type=int, default=0, help="Seed used once to define the fixed test set")
    p.add_argument("--test_size", type=float, default=0.2, help="Fraction of dataset in fixed test set")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation fraction relative to full dataset")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    X, y, feature_info = load_dataset(args.dataset)
    preprocessor_factory = lambda fi: make_preprocessor(fi, scale_numeric=True)

    out_base = Path(args.out_dir) / args.dataset
    out_base.mkdir(parents=True, exist_ok=True)

    # Define fixed test indices once.
    base_split = make_split(
        n_samples=len(X),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.fixed_test_seed,
        stratify=y.values,
    )
    fixed_test_idx = np.asarray(base_split["test"], dtype=int)
    np.save(out_base / "fixed_test_idx.npy", fixed_test_idx)

    for i in range(args.n_outer):
        outer_seed = args.seed_start + i
        run_dir = out_base / f"seed={outer_seed}"
        if (run_dir / "P_test.npy").exists() and (run_dir / "meta.csv").exists():
            print(f"[Run {i+1}/{args.n_outer}] seed={outer_seed} already exists, skipping.")
            continue

        split = make_split_with_fixed_test(
            n_samples=len(X),
            fixed_test_idx=fixed_test_idx,
            val_size=args.val_size,
            seed=outer_seed,
            stratify=y.values,
        )

        print(f"\n[Run {i+1}/{args.n_outer}] seed={outer_seed} (fixed test set)")
        split_out, meta, P_val, P_test, pipelines = run_one_training_run(
            X=X,
            y=y,
            feature_info=feature_info,
            preprocessor_factory=preprocessor_factory,
            outer_seed=outer_seed,
            n_candidates_per_family=args.n_candidates,
            test_size=args.test_size,
            val_size=args.val_size,
            model_configs=TRAINING_MODEL_CONFIGS,
            families=list(TRAINING_MODEL_CONFIGS.keys()),
            save_pipelines=args.save_models,
            verbose=args.verbose,
            split=split,
        )

        save_run_artifacts(
            run_dir,
            split=split_out,
            meta=meta,
            P_val=P_val,
            P_test=P_test,
            pipelines=pipelines,
            config={
                "dataset": args.dataset,
                "outer_seed": outer_seed,
                "n_candidates_per_family": args.n_candidates,
                "n_outer_runs": args.n_outer,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "save_models": args.save_models,
                "fixed_test_seed": args.fixed_test_seed,
                "fixed_test_size": int(len(fixed_test_idx)),
            },
        )
        print(f"  Saved to {run_dir} (candidates={P_test.shape[0]}, n_test={P_test.shape[1]})")

    print("\nDone.")


if __name__ == "__main__":
    main()
