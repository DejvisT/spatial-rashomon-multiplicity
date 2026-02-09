#!/usr/bin/env python
"""
Run thesis experiments: Rashomon training + analysis notebooks logic.

Usage:
  python run_experiments.py [--skip-training] [--quick]

  --skip-training: Use existing results (default if results exist)
  --quick: Reduce null runs (10) and permutations (100) for faster execution
"""
import argparse
import sys
from pathlib import Path

# Ensure project root in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from analysis.spatial import build_knn_graph, moran_global, lisa_local, extract_hh_components
from analysis.nulls import run_null_experiment, run_null_experiments_with_hh
from analysis.rules import extract_component_rules, rules_summary_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training", action="store_true", help="Skip Rashomon training")
    ap.add_argument("--quick", action="store_true", help="Fewer null runs (5) and permutations (99)")
    ap.add_argument("--global-only", action="store_true", help="Run only Global (skip per-family)")
    args = ap.parse_args()

    RESULTS = ROOT / "results" / "compas"
    BASE_GLOBAL = RESULTS / "seed=42_eps=0.01"
    FAMILY_PATHS = {
        "Global": BASE_GLOBAL,
        "LogReg": RESULTS / "family=LogReg" / "seed=42_eps=0.01",
        "RF": RESULTS / "family=RF" / "seed=42_eps=0.01",
        "GBM": RESULTS / "family=GBM" / "seed=42_eps=0.01",
        "MLP": RESULTS / "family=MLP" / "seed=42_eps=0.01",
    }
    if args.global_only:
        FAMILY_PATHS = {"Global": BASE_GLOBAL}

    # Check results exist
    for name, p in FAMILY_PATHS.items():
        if not p.exists():
            print(f"ERROR: Missing results at {p} ({name})")
            print("Run Rashomon experiments first:")
            print("  $env:PYTHONPATH='.\\src'; python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42")
            print("  # Then per-family: add --family LogReg, --family RF, etc.")
            sys.exit(1)

    k = 10
    n_runs = 5 if args.quick else 50
    permutations = 99 if args.quick else 999

    # --- 1. Load real and compute Moran I / HH ---
    print("\n=== 1. Real spatial stats (Moran I, n_HH) ===")
    def load_real(path):
        P = np.load(path / "P_test.npy")
        metrics = np.load(path / "metrics.npz")
        X = pd.read_csv(path / "X_test.csv")
        v = metrics["variance"]
        X_knn = X.select_dtypes(include=[np.number])
        W = build_knn_graph(X_knn, k=k)
        moran = moran_global(v, W, permutations=permutations)
        lisa = lisa_local(v, W, permutations=permutations)
        n_hh = (lisa["cluster"] == "HH").sum()
        return {"P": P, "X_knn": X_knn, "moran": moran, "n_hh": int(n_hh), "lisa": lisa, "W": W}

    real = {n: load_real(p) for n, p in FAMILY_PATHS.items()}
    for n, r in real.items():
        print(f"  {n}: Moran I = {r['moran']['I']:.4f}, n_HH = {r['n_hh']}")

    # --- 2. Null experiments ---
    print(f"\n=== 2. Null experiments (n_runs={n_runs}, permutations={permutations}) ===")
    null = {}
    for n, r in real.items():
        print(f"  {n}...")
        null[n] = run_null_experiments_with_hh(
            r["P"], r["X_knn"], n_runs=n_runs, k=k, permutations=permutations
        )

    # --- 3. Comparison tables ---
    print("\n=== 3. Moran I: Real vs null ===")
    for n in FAMILY_PATHS:
        r, nr = real[n]["moran"]["I"], null[n]["I"]
        print(f"  {n}: real={r:.4f}  null_mean={nr.mean():.4f}  null_max={nr.max():.4f}")

    print("\n=== 4. n_HH: Real vs null ===")
    for n in FAMILY_PATHS:
        r, nr = real[n]["n_hh"], null[n]["n_hh"]
        print(f"  {n}: real={r}  null_mean={nr.mean():1f}  null_max={nr.max():0f}")

    # --- 5. Interpretable rules (global only) ---
    print("\n=== 5. Interpretable rules (Global) ===")
    comp_id, components = extract_hh_components(real["Global"]["lisa"], real["Global"]["W"], min_size=5)
    X_test = pd.read_csv(BASE_GLOBAL / "X_test.csv")
    if components:
        rules = extract_component_rules(X_test, components, max_depth=3, min_samples_leaf=5)
        for cid, r in rules.items():
            print(f"  Component {cid} (n={r['n_component']}): prec={r['precision_train']:.2f} rec={r['recall_train']:.2f}")
    else:
        print("  No HH components with min_size=5")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
