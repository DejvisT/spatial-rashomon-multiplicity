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
    run_multiplicity,
    run_spatial,
    run_null,
    run_spatial_per_family,
    quadrant_analysis,
)
from analysis.preprocessing import get_transformed_test_features  # noqa: E402
from analysis.knn_defaults import default_k_nn  # noqa: E402

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


def _aggregate_quadrant_breakdown(
    records: List[Dict[str, Any]],
    *,
    dataset_name: str,
) -> Optional[pd.DataFrame]:
    """
    Mean ± std of quadrant counts/fractions and within-quadrant means across outer runs.
    Matches thesis-style Table: variance vs conflict quadrants (default 90th pct thresholds).
    """
    rows_long: List[Dict[str, Any]] = []
    for r in records:
        qdf = r.get("_quadrant_summary")
        if qdf is None or getattr(qdf, "empty", True):
            continue
        for _, row in qdf.iterrows():
            d = row.to_dict()
            d["outer_seed"] = r["outer_seed"]
            rows_long.append(d)
    if not rows_long:
        return None
    long = pd.DataFrame(rows_long)
    agg_rows: List[Dict[str, Any]] = []
    for grp in ["A", "B", "C", "D"]:
        sub = long[long["quadrant"] == grp]
        if sub.empty:
            continue

        def _ms(col: str) -> Tuple[float, float]:
            v = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return float("nan"), float("nan")
            m = float(np.nanmean(v))
            s = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
            return m, s

        cm, cs = _ms("count")
        fm, fs = _ms("fraction")
        vm, vs = _ms("mean_var_p")
        ccm, ccs = _ms("mean_conflict")
        out: Dict[str, Any] = {
            "dataset": dataset_name,
            "quadrant": grp,
            "n_runs": int(len(sub)),
            "count_mean": cm,
            "count_std": cs,
            "fraction_mean": fm,
            "fraction_std": fs,
            "mean_var_p_mean": vm,
            "mean_var_p_std": vs,
            "mean_conflict_mean": ccm,
            "mean_conflict_std": ccs,
        }
        if "error_rate" in sub.columns:
            em, es = _ms("error_rate")
            out["error_rate_mean"], out["error_rate_std"] = em, es
        if "brier" in sub.columns:
            bm, bs = _ms("brier")
            out["brier_mean"], out["brier_std"] = bm, bs
        agg_rows.append(out)
    if not agg_rows:
        return None
    return pd.DataFrame(agg_rows)


def _run_per_family_spatial_across_seeds(
    dataset_dir: Path,
    dataset_name: str,
    run_dirs: List[Path],
    *,
    K: int,
    k_nn: int,
    seed: Optional[int],
    verbose: bool,
) -> None:
    """
    For each outer seed, run spatial analysis on per-family Rashomon sets (K per family).
    Writes:
      - results/<dataset>/per_family_spatial_per_run.csv
      - results/<dataset>/per_family_spatial_aggregated.csv (mean ± std over seeds per family)
    """
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        config = load_config(run_dir)
        outer_seed = config.get("outer_seed")
        if outer_seed is None and run_dir.name.startswith("seed="):
            try:
                outer_seed = int(run_dir.name.split("=")[1])
            except (IndexError, ValueError):
                outer_seed = None
        if outer_seed is None:
            outer_seed = 0
        X_test = get_transformed_test_features(run_dir, dataset_name)
        n_cand = len(load_meta(run_dir))
        k_use = min(K, n_cand)
        fam_sp = run_spatial_per_family(
            run_dir,
            X_test,
            K=k_use,
            k=k_nn,
            seed=seed,
        )
        for family, res in fam_sp.items():
            rows.append(
                {
                    "outer_seed": outer_seed,
                    "family": family,
                    "mean_variance": res["mean_variance"],
                    "moran_i": float(res["moran_i"]),
                    "moran_p_sim": float(res["moran_p_sim"]),
                    "n_hh": int(res["n_hh"]),
                    "n_ll": int(res["n_ll"]),
                    "n_models": int(res["n_models"]),
                }
            )

    if not rows:
        return

    df_run = pd.DataFrame(rows)
    pr_path = dataset_dir / "per_family_spatial_per_run.csv"
    df_run.to_csv(pr_path, index=False)
    if verbose:
        print(f"  Wrote {pr_path}")

    def _ms(arr: np.ndarray) -> Tuple[float, float]:
        v = np.asarray(arr, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), float("nan")
        m = float(np.nanmean(v))
        s = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        return m, s

    agg_rows: List[Dict[str, Any]] = []
    for family in sorted(df_run["family"].unique()):
        grp = df_run[df_run["family"] == family]
        mv_m, mv_s = _ms(grp["mean_variance"].values)
        mi_m, mi_s = _ms(grp["moran_i"].values)
        hh_m, hh_s = _ms(grp["n_hh"].astype(float).values)
        frac_sig = float(np.mean(grp["moran_p_sim"].values < 0.05))
        agg_rows.append(
            {
                "dataset": dataset_name,
                "family": family,
                "n_runs": int(len(grp)),
                "mean_variance_mean": mv_m,
                "mean_variance_std": mv_s,
                "moran_i_mean": mi_m,
                "moran_i_std": mi_s,
                "n_hh_mean": hh_m,
                "n_hh_std": hh_s,
                "frac_significant_moran": frac_sig,
            }
        )
    df_agg = pd.DataFrame(agg_rows)
    ag_path = dataset_dir / "per_family_spatial_aggregated.csv"
    df_agg.to_csv(ag_path, index=False)
    if verbose:
        print(f"  Wrote {ag_path}")


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
    For one run: multiplicity, spatial metrics, conflict metrics, quadrant summaries, and null results.
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
    results/{dataset}/summary_per_run.csv with per-run diagnostics; per run also
    results/{dataset}/seed=*/per_point/quadrant_summary.csv (variance vs conflict
    quadrants, default 90th percentile thresholds). Aggregates quadrants to
    results/{dataset}/quadrant_breakdown_aggregated.csv and saves per-run
    thresholds to results/{dataset}/quadrant_thresholds_per_run.csv.
    Per-family Rashomon (K per family): results/{dataset}/per_family_spatial_per_run.csv
    and per_family_spatial_aggregated.csv (mean ± std over seeds). Aggregate
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
                "hh_jaccard_var_conflict_mean", "hh_jaccard_var_conflict_std",
                "frac_significant_moran",
                "null_mean", "null_std"
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
            "null_mean": r["null_mean"],
            "null_std": r["null_std"],
        }
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
        if "_quadrant_summary" in rec:
            rec["_quadrant_summary"].to_csv(pp_dir / "quadrant_summary.csv", index=False)

    thr_df = pd.DataFrame(
        [
            {
                "outer_seed": r["outer_seed"],
                "quadrant_var_thresh": r.get("quadrant_var_thresh"),
                "quadrant_conflict_thresh": r.get("quadrant_conflict_thresh"),
            }
            for r in records
        ]
    )
    thr_path = dataset_dir / "quadrant_thresholds_per_run.csv"
    thr_df.to_csv(thr_path, index=False)
    if verbose:
        print(f"  Wrote {thr_path}")

    q_agg = _aggregate_quadrant_breakdown(records, dataset_name=name)
    if q_agg is not None and not q_agg.empty:
        q_agg["var_quantile"] = 0.9
        q_agg["conflict_quantile"] = 0.9
        q_path = dataset_dir / "quadrant_breakdown_aggregated.csv"
        q_agg.to_csv(q_path, index=False)
        if verbose:
            print(f"  Wrote {q_path}")

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

    hh_jaccard_mean, hh_jaccard_std = _mean_std(hh_jaccard_vc)

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
        "hh_jaccard_var_conflict_mean": hh_jaccard_mean,
        "hh_jaccard_var_conflict_std": hh_jaccard_std,
        "frac_significant_moran": float(np.mean(sig)),
        "frac_significant_hh": float(np.mean(sig_hh)),
        "null_mean": float(np.mean(null_mean_arr)),
        "null_std": float(np.mean(null_std_arr)),
        "null_n_hh_mean": float(np.mean(null_n_hh_mean_arr)),
    }

    _run_per_family_spatial_across_seeds(
        dataset_dir,
        name,
        run_dirs,
        K=K,
        k_nn=k_resolved,
        seed=seed,
        verbose=verbose,
    )

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
