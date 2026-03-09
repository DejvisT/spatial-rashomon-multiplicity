#!/usr/bin/env python
"""
Run experiments over training artifacts: Rashomon selection, multiplicity,
spatial (Moran's I + HH), and null (empirical p-value). Aggregates per dataset.

Usage:
  python run_experiments.py --results_dir results --dataset compas
  python run_experiments.py --results_dir results  # all datasets
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.experiment_runner import run_dataset_experiment, run_all_experiments


def main():
    parser = argparse.ArgumentParser(description="Run experiments over dataset run dirs")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Base results dir")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset (default: all)")
    parser.add_argument("--K", type=int, default=25, help="Rashomon top-K")
    parser.add_argument("--R_null", type=int, default=100, help="Null permutations")
    args = parser.parse_args()

    if args.dataset:
        dataset_dir = args.results_dir / args.dataset
        out = run_dataset_experiment(dataset_dir, K=args.K, R_null=args.R_null)
    else:
        out = run_all_experiments(args.results_dir, K=args.K, R_null=args.R_null)

    print(out.to_string())


if __name__ == "__main__":
    main()
