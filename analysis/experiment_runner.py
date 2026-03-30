"""
Experiment runner: iterate over runs in a dataset directory, compute
Rashomon selection, multiplicity, spatial, and null; aggregate to per-dataset summary.

Uses analysis.run_analysis and data loading only (no training). Preprocesses
test set per run (fit on train) to obtain X_test for spatial/null.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Use existing data loading (no modification)
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from analysis.run_analysis import (  # noqa: E402
    load_config,
    load_meta,
    load_P_test,
    load_split,
    run_multiplicity,
    run_spatial,
    run_null,
    select_rashomon_global,
    pointwise_conflict,
    hard_vote_variance,
    quadrant_analysis,
)
from analysis.preprocessing import get_transformed_test_features  # noqa: E402
from analysis.knn_defaults import default_k_nn  # noqa: E402

try:
    from sklearn.metrics import brier_score_loss  # noqa: E402
except ImportError:
    brier_score_loss = None

PathLike = Union[str, Path]


def _get_run_dirs(dataset_dir: PathLike) -> List[Path]:
    """List run directories (seed=*) under dataset_dir, sorted by seed."""
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


def _run_single(
    run_dir: Path,
    X_test: np.ndarray,
    *,
    K: int = 25,
    epsilon: float = 0.05,
    k_nn: int = 30,
    R_null: int = 100,
    seed: Optional[int] = 42,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For one run: multiplicity, spatial (Moran + HH count + n_ll, neighborhood_agreement, lcae),
    null (empirical p-value). If dataset_name is provided, also compute performance
    (acc_mean, brier_mean, acc_ensemble, brier_ensemble) over Rashomon and ensemble.
    """
    mult = run_multiplicity(run_dir, K=K, epsilon=epsilon)
    spatial = run_spatial(run_dir, X_test, K=K, k=k_nn, seed=seed)
    null = run_null(run_dir, X_test, K=K, R=R_null, k=k_nn, seed=seed)

    observed_moran = spatial["moran_i"]
    null_moran = null["null_moran_i"]
    null_n_hh = null["null_n_hh"]

    p_empirical = float(
        (1 + np.sum(null_moran >= observed_moran)) / (len(null_moran) + 1)
    )
    n_hh = int(np.sum(spatial["HH_mask"]))
    p_empirical_hh = float(
        (1 + np.sum(null_n_hh >= n_hh)) / (len(null_n_hh) + 1)
    )
    null_mean = float(np.mean(null_moran))
    null_std = float(np.std(null_moran, ddof=1))
    moran_p_sim = float(spatial["moran_p_sim"])
    n_ll = int(spatial["n_ll"])
    neighborhood_agreement = float(spatial["neighborhood_agreement"])
    lcae = float(spatial["lcae"])

    out = {
        "mean_variance": mult["mean_variance"],
        "ambiguity": mult["ambiguity"],
        "disagreement_rate": mult["disagreement_rate"],
        "discrepancy": mult["discrepancy"],
        "mean_conflict": mult["mean_conflict"],
        "frac_conflict_gt0": mult["frac_conflict_gt0"],
        "frac_conflict_ge025": mult["frac_conflict_ge025"],
        "moran_i": observed_moran,
        "moran_p_sim": moran_p_sim,
        "n_hh": n_hh,
        "n_ll": n_ll,
        "conflict_moran_i": float(spatial.get("conflict_moran_i", np.nan)),
        "conflict_n_hh": int(spatial.get("conflict_n_hh", 0)),
        "conflict_n_ll": int(spatial.get("conflict_n_ll", 0)),
        "hh_jaccard_var_conflict": float(spatial.get("hh_jaccard_var_conflict", np.nan)),
        "neighborhood_agreement": neighborhood_agreement,
        "lcae": lcae,
        "p_empirical": p_empirical,
        "significant_moran": p_empirical < 0.05,
        "p_empirical_hh": p_empirical_hh,
        "significant_hh": p_empirical_hh < 0.05,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_n_hh_mean": float(np.mean(null_n_hh)),
        "null_n_hh_std": float(np.std(null_n_hh, ddof=1)) if len(null_n_hh) > 1 else 0.0,
    }

    # Per-point arrays for optional saving
    out["_pointwise_variance"] = mult["pointwise_variance"]
    out["_pointwise_conflict"] = mult["pointwise_conflict"]
    out["_var_hard"] = mult["var_hard"]

    if dataset_name is not None and brier_score_loss is not None:
        from data import load_dataset  # noqa: E402
        _, y, _ = load_dataset(dataset_name)
        split = load_split(run_dir)
        y_test = np.asarray(y.iloc[split["test"]].values).flatten()
        P_test = load_P_test(run_dir)
        idx = select_rashomon_global(run_dir, K=K)
        P_sel = P_test[idx]
        acc_per_model = []
        brier_per_model = []
        for m in range(P_sel.shape[0]):
            p = P_sel[m]
            y_pred = (p >= 0.5).astype(int)
            acc_per_model.append(float(np.mean(y_pred == y_test)))
            brier_per_model.append(float(brier_score_loss(y_test, p)))
        P_ens = P_sel.mean(axis=0)
        acc_ensemble = float(np.mean((P_ens >= 0.5).astype(int) == y_test))
        brier_ensemble = float(brier_score_loss(y_test, P_ens))
        out["acc_mean"] = float(np.mean(acc_per_model))
        out["brier_mean"] = float(np.mean(brier_per_model))
        out["acc_ensemble"] = acc_ensemble
        out["brier_ensemble"] = brier_ensemble

        # Quadrant analysis (requires labels)
        qa = quadrant_analysis(
            mult["pointwise_variance"],
            mult["pointwise_conflict"],
            y_test=y_test,
            P_mean=P_ens,
        )
        out["_quadrant_summary"] = qa["summary"]
        out["quadrant_var_thresh"] = qa["var_thresh"]
        out["quadrant_conflict_thresh"] = qa["conflict_thresh"]
    else:
        qa = quadrant_analysis(
            mult["pointwise_variance"],
            mult["pointwise_conflict"],
        )
        out["_quadrant_summary"] = qa["summary"]
        out["quadrant_var_thresh"] = qa["var_thresh"]
        out["quadrant_conflict_thresh"] = qa["conflict_thresh"]

    return out


def run_dataset_experiment(
    dataset_dir: PathLike,
    dataset_name: Optional[str] = None,
    *,
    K: int = 25,
    epsilon: float = 0.05,
    k_nn: Optional[int] = None,
    R_null: int = 100,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For all runs in dataset_dir: select Rashomon (K=25), compute multiplicity,
    spatial (Moran's I + HH count), and null (empirical p-value). Saves
    results/{dataset}/summary_per_run.csv with per-run diagnostics. Aggregate
    to one row per dataset: mean ± std of mean_variance, mean ± std of Moran's I,
    average number of HH points, fraction of runs with significant Moran.

    Parameters
    ----------
    dataset_dir : path to results/{dataset}/ (e.g. results/compas)
    dataset_name : name for load_dataset (default: dataset_dir.name)
    K : Rashomon top-K
    epsilon : for disagreement_rate
    k_nn : k for kNN graph (None = dataset default from ``analysis.knn_defaults``)
    R_null : null permutations
    seed : for spatial LISA
    verbose : print progress

    Returns
    -------
    DataFrame with one row and columns:
      dataset, n_runs,
      mean_variance_mean, mean_variance_std,
      moran_i_mean, moran_i_std,
      n_hh_mean, frac_significant_moran,
      null_mean, null_std (mean of per-run null distribution stats)
    """
    dataset_dir = Path(dataset_dir)
    name = dataset_name if dataset_name is not None else dataset_dir.name
    k_resolved = k_nn if k_nn is not None else default_k_nn(name)

    run_dirs = _get_run_dirs(dataset_dir)
    if not run_dirs:
        return pd.DataFrame(
            columns=[
                "dataset", "n_runs",
                "mean_variance_mean", "mean_variance_std",
                "mean_conflict_mean", "mean_conflict_std",
                "frac_conflict_gt0_mean", "frac_conflict_ge025_mean",
                "moran_i_mean", "moran_i_std",
                "n_hh_mean", "n_ll_mean", "n_ll_std",
                "conflict_moran_i_mean", "conflict_n_hh_mean",
                "hh_jaccard_var_conflict_mean",
                "neighborhood_agreement_mean", "neighborhood_agreement_std",
                "lcae_mean", "lcae_std",
                "frac_significant_moran",
                "null_mean", "null_std",
                "acc_mean_mean", "acc_mean_std", "brier_mean_mean", "brier_mean_std",
                "acc_ensemble_mean", "brier_ensemble_mean",
            ]
        )

    records: List[Dict[str, Any]] = []
    for i, run_dir in enumerate(run_dirs):
        if verbose:
            print(f"  Run {i+1}/{len(run_dirs)}: {run_dir.name}")
        config = load_config(run_dir)
        outer_seed = config.get("outer_seed")
        if outer_seed is None and run_dir.name.startswith("seed="):
            try:
                outer_seed = int(run_dir.name.split("=")[1])
            except (IndexError, ValueError):
                outer_seed = i
        elif outer_seed is None:
            outer_seed = i
        n_candidates = len(load_meta(run_dir))
        K_actual = min(K, n_candidates)
        X_test = get_transformed_test_features(run_dir, name)
        rec = _run_single(
            run_dir,
            X_test,
            K=K_actual,
            epsilon=epsilon,
            k_nn=k_resolved,
            R_null=R_null,
            seed=seed,
            dataset_name=name,
        )
        rec["outer_seed"] = outer_seed
        records.append(rec)

    # Per-run diagnostics: results/{dataset}/summary_per_run.csv
    def _row(r: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "outer_seed": r["outer_seed"],
            "mean_variance": r["mean_variance"],
            "ambiguity": r["ambiguity"],
            "disagreement_rate": r["disagreement_rate"],
            "discrepancy": r["discrepancy"],
            "mean_conflict": r["mean_conflict"],
            "frac_conflict_gt0": r["frac_conflict_gt0"],
            "frac_conflict_ge025": r["frac_conflict_ge025"],
            "moran_i": r["moran_i"],
            "moran_p_sim": r["moran_p_sim"],
            "p_empirical": r["p_empirical"],
            "n_hh": r["n_hh"],
            "n_ll": r["n_ll"],
            "conflict_moran_i": r["conflict_moran_i"],
            "conflict_n_hh": r["conflict_n_hh"],
            "conflict_n_ll": r["conflict_n_ll"],
            "hh_jaccard_var_conflict": r["hh_jaccard_var_conflict"],
            "p_empirical_hh": r["p_empirical_hh"],
            "null_n_hh_mean": r["null_n_hh_mean"],
            "null_n_hh_std": r["null_n_hh_std"],
            "neighborhood_agreement": r["neighborhood_agreement"],
            "lcae": r["lcae"],
            "null_mean": r["null_mean"],
            "null_std": r["null_std"],
        }
        if "acc_mean" in r:
            row["acc_mean"] = r["acc_mean"]
            row["brier_mean"] = r["brier_mean"]
            row["acc_ensemble"] = r["acc_ensemble"]
            row["brier_ensemble"] = r["brier_ensemble"]
        return row

    per_run = pd.DataFrame([_row(r) for r in records])
    summary_path = dataset_dir / "summary_per_run.csv"
    per_run.to_csv(summary_path, index=False)

    # Save per-point arrays and quadrant summary for each run
    for i, (run_dir_i, rec) in enumerate(zip(run_dirs, records)):
        pp_dir = run_dir_i / "per_point"
        pp_dir.mkdir(exist_ok=True)
        np.save(pp_dir / "var_p.npy", rec["_pointwise_variance"])
        np.save(pp_dir / "conflict.npy", rec["_pointwise_conflict"])
        np.save(pp_dir / "var_hard.npy", rec["_var_hard"])
        if "_quadrant_summary" in rec:
            rec["_quadrant_summary"].to_csv(pp_dir / "quadrant_summary.csv", index=False)

    mean_var = np.array([r["mean_variance"] for r in records])
    ambiguity_arr = np.array([r["ambiguity"] for r in records])
    disagree_arr = np.array([r["disagreement_rate"] for r in records])
    discrep_arr = np.array([r["discrepancy"] for r in records])
    mean_conf = np.array([r["mean_conflict"] for r in records])
    frac_conf_gt0 = np.array([r["frac_conflict_gt0"] for r in records])
    frac_conf_ge025 = np.array([r["frac_conflict_ge025"] for r in records])
    moran = np.array([r["moran_i"] for r in records])
    n_hh = np.array([r["n_hh"] for r in records])
    n_ll = np.array([r["n_ll"] for r in records])
    conflict_moran = np.array([r["conflict_moran_i"] for r in records])
    conflict_n_hh = np.array([r["conflict_n_hh"] for r in records])
    hh_jaccard_vc = np.array([r["hh_jaccard_var_conflict"] for r in records])
    na = np.array([r["neighborhood_agreement"] for r in records])
    lcae = np.array([r["lcae"] for r in records])
    sig = np.array([r["significant_moran"] for r in records])
    sig_hh = np.array([r["significant_hh"] for r in records])
    null_mean_arr = np.array([r["null_mean"] for r in records])
    null_std_arr = np.array([r["null_std"] for r in records])
    null_n_hh_mean_arr = np.array([r["null_n_hh_mean"] for r in records])

    def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
        return (
            float(np.nanmean(arr)),
            float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        )

    row = {
        "dataset": name,
        "n_runs": len(records),
        "mean_variance_mean": _mean_std(mean_var)[0],
        "mean_variance_std": _mean_std(mean_var)[1],
        "ambiguity_mean": _mean_std(ambiguity_arr)[0],
        "ambiguity_std": _mean_std(ambiguity_arr)[1],
        "disagreement_rate_mean": _mean_std(disagree_arr)[0],
        "disagreement_rate_std": _mean_std(disagree_arr)[1],
        "discrepancy_mean": _mean_std(discrep_arr)[0],
        "discrepancy_std": _mean_std(discrep_arr)[1],
        "mean_conflict_mean": _mean_std(mean_conf)[0],
        "mean_conflict_std": _mean_std(mean_conf)[1],
        "frac_conflict_gt0_mean": _mean_std(frac_conf_gt0)[0],
        "frac_conflict_ge025_mean": _mean_std(frac_conf_ge025)[0],
        "moran_i_mean": _mean_std(moran)[0],
        "moran_i_std": _mean_std(moran)[1],
        "n_hh_mean": float(np.mean(n_hh)),
        "n_ll_mean": float(np.mean(n_ll)),
        "n_ll_std": _mean_std(n_ll)[1],
        "conflict_moran_i_mean": _mean_std(conflict_moran)[0],
        "conflict_n_hh_mean": float(np.nanmean(conflict_n_hh)),
        "hh_jaccard_var_conflict_mean": _mean_std(hh_jaccard_vc)[0],
        "neighborhood_agreement_mean": _mean_std(na)[0],
        "neighborhood_agreement_std": _mean_std(na)[1],
        "lcae_mean": _mean_std(lcae)[0],
        "lcae_std": _mean_std(lcae)[1],
        "frac_significant_moran": float(np.mean(sig)),
        "frac_significant_hh": float(np.mean(sig_hh)),
        "null_mean": float(np.mean(null_mean_arr)),
        "null_std": float(np.mean(null_std_arr)),
        "null_n_hh_mean": float(np.mean(null_n_hh_mean_arr)),
    }
    if "acc_mean" in records[0]:
        acc_mean = np.array([r["acc_mean"] for r in records])
        brier_mean = np.array([r["brier_mean"] for r in records])
        acc_ens = np.array([r["acc_ensemble"] for r in records])
        brier_ens = np.array([r["brier_ensemble"] for r in records])
        row["acc_mean_mean"], row["acc_mean_std"] = _mean_std(acc_mean)
        row["brier_mean_mean"], row["brier_mean_std"] = _mean_std(brier_mean)
        row["acc_ensemble_mean"] = float(np.mean(acc_ens))
        row["brier_ensemble_mean"] = float(np.mean(brier_ens))

    return pd.DataFrame([row])


def run_all_experiments(
    results_dir: PathLike,
    *,
    datasets: Optional[List[str]] = None,
    K: int = 25,
    epsilon: float = 0.05,
    k_nn: Optional[int] = None,
    R_null: int = 100,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run experiment for each dataset under results_dir; concatenate summaries.
    If datasets is None, use all subdirectories that contain at least one seed=* run.
    """
    results_dir = Path(results_dir)
    if datasets is not None:
        to_run = [results_dir / d for d in datasets]
    else:
        to_run = [p for p in results_dir.iterdir() if p.is_dir() and _get_run_dirs(p)]

    if not to_run:
        return pd.DataFrame()

    dfs = []
    for dataset_dir in to_run:
        if verbose:
            print(f"Dataset: {dataset_dir.name}")
        k_resolved = k_nn if k_nn is not None else default_k_nn(dataset_dir.name)
        df = run_dataset_experiment(
            dataset_dir,
            dataset_name=dataset_dir.name,
            K=K,
            epsilon=epsilon,
            k_nn=k_resolved,
            R_null=R_null,
            seed=seed,
            verbose=verbose,
        )
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiments over dataset run dirs")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Base results dir")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset (default: all)")
    parser.add_argument("--K", type=int, default=25)
    parser.add_argument("--R_null", type=int, default=100)
    args = parser.parse_args()
    if args.dataset:
        dataset_dir = args.results_dir / args.dataset
        out = run_dataset_experiment(dataset_dir, K=args.K, R_null=args.R_null)
    else:
        out = run_all_experiments(args.results_dir, K=args.K, R_null=args.R_null)
    print(out.to_string())
