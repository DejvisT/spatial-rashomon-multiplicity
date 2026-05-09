#!/usr/bin/env python
"""
Run the main training pipeline: 10 outer runs, 60/20/20 stratified splits,
50 candidates per model family, deterministic seeds. Saves artifacts per run
(no Rashomon selection).

Usage:
  python run_training_pipeline.py --dataset compas [--out_dir results] [--n_outer 10] [--n_candidates 50]
  python run_training_pipeline.py --dataset compas --save_models  # also save trained pipelines
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data import load_dataset, make_preprocessor
from training_pipeline import (
    run_one_training_run,
    save_run_artifacts,
    TRAINING_MODEL_CONFIGS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Main training pipeline (10 outer runs, 50 candidates/family)")
    p.add_argument("--dataset", type=str, required=True, choices=["compas", "german", "adult"])
    p.add_argument("--out_dir", type=str, default="results", help="Base output directory")
    p.add_argument("--n_outer", type=int, default=10, help="Number of outer runs (seeds)")
    p.add_argument("--n_candidates", type=int, default=50, help="Candidates per model family per run")
    p.add_argument("--save_models", action="store_true", help="Save trained pipeline objects (pickles)")
    p.add_argument("--seed_start", type=int, default=0, help="First outer seed (seeds = seed_start, seed_start+1, ...)")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    X, y, feature_info = load_dataset(args.dataset)
    preprocessor_factory = lambda fi: make_preprocessor(fi, scale_numeric=True)

    out_base = Path(args.out_dir) / args.dataset
    out_base.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_outer):
        outer_seed = args.seed_start + i
        run_dir = out_base / f"seed={outer_seed}"
        if (run_dir / "P_test.npy").exists() and (run_dir / "meta.csv").exists():
            print(f"[Run {i+1}/{args.n_outer}] seed={outer_seed} already exists, skipping.")
            continue

        print(f"\n[Run {i+1}/{args.n_outer}] seed={outer_seed}")
        split, meta, P_val, P_test, pipelines = run_one_training_run(
            X=X,
            y=y,
            feature_info=feature_info,
            preprocessor_factory=preprocessor_factory,
            outer_seed=outer_seed,
            n_candidates_per_family=args.n_candidates,
            test_size=0.2,
            val_size=0.2,
            model_configs=TRAINING_MODEL_CONFIGS,
            families=list(TRAINING_MODEL_CONFIGS.keys()),
            save_pipelines=args.save_models,
            verbose=args.verbose,
        )

        save_run_artifacts(
            run_dir,
            split=split,
            meta=meta,
            P_val=P_val,
            P_test=P_test,
            pipelines=pipelines,
            config={
                "dataset": args.dataset,
                "outer_seed": outer_seed,
                "n_candidates_per_family": args.n_candidates,
                "n_outer_runs": args.n_outer,
                "test_size": 0.2,
                "val_size": 0.2,
                "save_models": args.save_models,
            },
        )
        print(f"  Saved to {run_dir} (candidates={P_test.shape[0]}, n_test={P_test.shape[1]})")

    print("\nDone.")


if __name__ == "__main__":
    main()
