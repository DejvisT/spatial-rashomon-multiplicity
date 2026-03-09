"""
Export thesis assets from results: dataset summary table, null significance table,
and sensitivity figures (K and kNN). Writes into overleaf_bundle/presentation_assets/.
Run from repo root: python scripts/export_thesis_assets.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "tables"
OUT_DIR = ROOT / "overleaf_bundle" / "presentation_assets"
FIG_DIR = OUT_DIR / "fig"
TAB_DIR = OUT_DIR / "tab"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_DATASETS = ("compas", "german", "breast_cancer")


def build_dataset_summary():
    """Aggregate summary_per_run.csv per dataset and return DataFrame."""
    rows = []
    for name in SUPPORTED_DATASETS:
        path = RESULTS_DIR / name / "summary_per_run.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n = len(df)
        mv_mean = df["mean_variance"].mean()
        mv_std = df["mean_variance"].std(ddof=1) if n > 1 else 0.0
        mi_mean = df["moran_i"].mean()
        mi_std = df["moran_i"].std(ddof=1) if n > 1 else 0.0
        n_hh_mean = df["n_hh"].mean()
        sig = (df["p_empirical"] < 0.05).mean()
        rows.append({
            "dataset": name.replace("_", " ").title(),
            "n_runs": n,
            "mean_variance_mean": mv_mean,
            "mean_variance_std": mv_std,
            "moran_i_mean": mi_mean,
            "moran_i_std": mi_std,
            "n_hh_mean": n_hh_mean,
            "frac_significant": sig,
        })
    return pd.DataFrame(rows)


def write_dataset_summary_tex():
    df = build_dataset_summary()
    if df.empty:
        print("No dataset summary (missing summary_per_run.csv). Skipping dataset_summary.tex")
        return
    out = []
    out.append(r"\begin{tabular}{lccccc}")
    out.append(r"\hline")
    out.append(r"Dataset & $n$ runs & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count & Frac.\ sig. \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        mv = f"{r['mean_variance_mean']:.4f} $\\pm$ {r['mean_variance_std']:.4f}"
        mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
        out.append(f"{r['dataset']} & {int(r['n_runs'])} & {mv} & {mi} & {r['n_hh_mean']:.1f} & {r['frac_significant']:.2f} \\\\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "dataset_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "dataset_summary.tex")


def write_global_summary_tex():
    """Global Rashomon set: mean variance, Moran's I, HH count, frac sig per dataset."""
    df = build_dataset_summary()
    if df.empty:
        print("No dataset summary. Skipping global_summary.tex")
        return
    out = []
    out.append(r"% Global Rashomon set (top-K=25). Source: summary_per_run.csv per dataset.")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count & Frac.\ sig. \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        mv = f"{r['mean_variance_mean']:.4f} $\\pm$ {r['mean_variance_std']:.4f}"
        mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
        out.append(f"{r['dataset']} & {mv} & {mi} & {r['n_hh_mean']:.1f} & {r['frac_significant']:.2f} \\\\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "global_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "global_summary.tex")


def write_family_summary_tex():
    """Per-family (top-25 per family): from family_hv_hh_summary_*.csv if available (single run)."""
    path = TABLES_DIR / "family_hv_hh_summary_compas.csv"
    if not path.exists():
        print("Missing family_hv_hh_summary_compas.csv. Skipping family_summary.tex")
        return
    df = pd.read_csv(path)
    # One run: no std; report single values. Frac sig from moran_p < 0.05.
    df["frac_sig"] = (df["moran_p"] < 0.05).astype(int)
    out = []
    out.append(r"% Per-family Rashomon (top-K=25 per family). Compas, single run; fill mean$\pm$std when multi-run available.")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(r"Family & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count (mean $\pm$ std) & Frac.\ sig. \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        fam = r["family"]
        mv = f"{r['mean_var']:.6f}"
        mi = f"{r['moran_I']:.3f}"
        hh = str(int(r["hh_count"]))
        fs = "1.00" if r["frac_sig"] else "0.00"
        out.append(f"{fam} & {mv} & {mi} & {hh} & {fs} \\\\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "family_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "family_summary.tex")


def write_spatial_patterns_figure():
    """HH count and Moran's I per run (all datasets) -> spatial_patterns_per_run.pdf."""
    rows = []
    for name in SUPPORTED_DATASETS:
        path = RESULTS_DIR / name / "summary_per_run.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["dataset"] = name.replace("_", " ").title()
        rows.append(df)
    if not rows:
        print("No summary_per_run. Skipping spatial_patterns_per_run.pdf")
        return
    big = pd.concat(rows, ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"Compas": "C0", "German": "C1", "Breast Cancer": "C2"}
    x_offset = 0
    xticks, xlabels = [], []
    for ds in big["dataset"].unique():
        d = big[big["dataset"] == ds].sort_values("outer_seed")
        n = len(d)
        x = np.arange(x_offset, x_offset + n)
        ax1.bar(x, d["n_hh"].values, color=colors.get(ds, "gray"), edgecolor="white", label=ds)
        ax2.bar(x, d["moran_i"].values, color=colors.get(ds, "gray"), edgecolor="white")
        xticks.append(x_offset + (n - 1) / 2)
        xlabels.append(ds)
        x_offset += n + 1
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("HH count")
    ax1.set_title("HH count per run")
    ax1.legend()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I per run")
    plt.suptitle("Spatial patterns by dataset (notebook 03 / summary_per_run)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "spatial_patterns_per_run.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "spatial_patterns_per_run.pdf")


def write_hh_by_family_figure():
    """HH count by family (Compas per-family run) -> hh_by_family.pdf."""
    path = TABLES_DIR / "family_hv_hh_summary_compas.csv"
    if not path.exists():
        print("Missing family_hv_hh_summary_compas.csv. Skipping hh_by_family.pdf")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["family"], df["hh_count"], color="steelblue", edgecolor="white")
    ax.set_ylabel("HH count")
    ax.set_xlabel("Model family")
    ax.set_title("HH count by family (Compas, per-family top-25)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hh_by_family.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "hh_by_family.pdf")


def write_null_significance_tex():
    path = TABLES_DIR / "null_significance_summary.csv"
    if not path.exists():
        print("Missing tables/null_significance_summary.csv. Skipping null_significance.tex")
        return
    df = pd.read_csv(path)
    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & $n$ runs & Frac.\ significant ($p < 0.05$) & Moran's $I$ (mean $\pm$ std) \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        ds = r["dataset"].replace("_", " ").title()
        moran_fmt = str(r.get("mean_moran ± std", "")).replace("±", r"$\pm$")
        out.append(f"{ds} & {int(r['n_runs'])} & {r['frac_significant']:.2f} & {moran_fmt} \\\\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "null_significance.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "null_significance.tex")


def run_sensitivity_K_and_save_figure():
    from analysis.experiment_runner import _get_run_dirs
    from analysis.preprocessing import get_transformed_test_features
    from analysis.run_analysis import load_meta, run_multiplicity, run_spatial

    K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    k_nn = 30
    dataset_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name in SUPPORTED_DATASETS and _get_run_dirs(p)]
    if not dataset_dirs:
        print("No run dirs. Skipping sensitivity_K.pdf")
        return

    def run_one_k(run_dir, X_test, K):
        n_cand = len(load_meta(run_dir))
        if K > n_cand:
            return None
        K_actual = min(K, n_cand)
        mult = run_multiplicity(run_dir, K=K_actual)
        spatial = run_spatial(run_dir, X_test, K=K_actual, k=k_nn)
        return {"mean_variance": mult["mean_variance"], "moran_i": spatial["moran_i"], "n_hh": int(np.sum(spatial["HH_mask"]))}

    results_k = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        run_dirs = _get_run_dirs(dataset_dir)
        for K in K_LIST:
            for run_dir in run_dirs:
                try:
                    X_test = get_transformed_test_features(run_dir, dataset_name)
                except Exception:
                    continue
                res = run_one_k(run_dir, X_test, K)
                if res is not None:
                    results_k.append({"dataset": dataset_name, "K": K, **res})

    df_k = pd.DataFrame(results_k)
    if df_k.empty:
        print("No K sensitivity data. Skipping sensitivity_K.pdf")
        return

    agg_k = df_k.groupby(["dataset", "K"]).agg(
        mean_variance_mean=("mean_variance", "mean"), mean_variance_std=("mean_variance", "std"),
        moran_mean=("moran_i", "mean"), moran_std=("moran_i", "std"),
        n_hh_mean=("n_hh", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, agg_k["dataset"].nunique()))
    for (ds, grp), c in zip(agg_k.groupby("dataset"), colors):
        d = grp
        ax1.errorbar(d["K"], d["mean_variance_mean"], yerr=d["mean_variance_std"], marker="o", capsize=3, label=ds, color=c)
        ax2.errorbar(d["K"], d["moran_mean"], yerr=d["moran_std"], marker="o", capsize=3, label=ds, color=c)
    ax1.set_xlabel("K (Rashomon size)")
    ax1.set_ylabel("Mean variance")
    ax1.set_title("Mean variance vs K")
    ax1.legend()
    ax2.set_xlabel("K (Rashomon size)")
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I vs K")
    ax2.legend()
    plt.suptitle("K sensitivity (all datasets)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_K.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "sensitivity_K.pdf")


def run_sensitivity_kNN_and_save_figure():
    import warnings
    warnings.filterwarnings("ignore", message=".*not fully connected.*", category=UserWarning)

    from analysis.experiment_runner import _get_run_dirs
    from analysis.preprocessing import get_transformed_test_features
    from analysis.run_analysis import load_meta, run_multiplicity, run_spatial

    K = 25
    K_NN_LIST = [10, 20, 30, 50]
    dataset_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name in SUPPORTED_DATASETS and _get_run_dirs(p)]
    if not dataset_dirs:
        print("No run dirs. Skipping sensitivity_kNN.pdf")
        return

    def run_one_knn(run_dir, X_test, k_nn):
        n_cand = len(load_meta(run_dir))
        K_actual = min(K, n_cand)
        mult = run_multiplicity(run_dir, K=K_actual)
        spatial = run_spatial(run_dir, X_test, K=K_actual, k=k_nn)
        return {"mean_variance": mult["mean_variance"], "moran_i": spatial["moran_i"], "n_hh": int(np.sum(spatial["HH_mask"]))}

    results_knn = []
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        run_dirs = _get_run_dirs(dataset_dir)
        for k_nn in K_NN_LIST:
            for run_dir in run_dirs:
                try:
                    X_test = get_transformed_test_features(run_dir, dataset_name)
                except Exception:
                    continue
                res = run_one_knn(run_dir, X_test, k_nn)
                results_knn.append({"dataset": dataset_name, "k_nn": k_nn, **res})

    df_knn = pd.DataFrame(results_knn)
    if df_knn.empty:
        print("No kNN sensitivity data. Skipping sensitivity_kNN.pdf")
        return

    agg_knn = df_knn.groupby(["dataset", "k_nn"]).agg(
        mean_variance_mean=("mean_variance", "mean"), mean_variance_std=("mean_variance", "std"),
        moran_mean=("moran_i", "mean"), moran_std=("moran_i", "std"),
        n_hh_mean=("n_hh", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, agg_knn["dataset"].nunique()))
    for (ds, grp), c in zip(agg_knn.groupby("dataset"), colors):
        d = grp
        ax1.errorbar(d["k_nn"], d["mean_variance_mean"], yerr=d["mean_variance_std"], marker="s", capsize=3, label=ds, color=c)
        ax2.errorbar(d["k_nn"], d["moran_mean"], yerr=d["moran_std"], marker="s", capsize=3, label=ds, color=c)
    ax1.set_xlabel("k (kNN neighbors)")
    ax1.set_ylabel("Mean variance")
    ax1.set_title("Mean variance vs k")
    ax1.legend()
    ax2.set_xlabel("k (kNN neighbors)")
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I vs k")
    ax2.legend()
    plt.suptitle("kNN sensitivity (all datasets)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_kNN.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "sensitivity_kNN.pdf")


def main():
    write_dataset_summary_tex()
    write_global_summary_tex()
    write_family_summary_tex()
    write_null_significance_tex()
    run_sensitivity_K_and_save_figure()
    run_sensitivity_kNN_and_save_figure()
    write_spatial_patterns_figure()
    write_hh_by_family_figure()
    print("Done.")


if __name__ == "__main__":
    main()
