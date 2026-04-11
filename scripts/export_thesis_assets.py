"""
Export thesis assets from results: dataset summary table, null significance table,
and sensitivity figures (K and kNN). Writes into overleaf_bundle/presentation_assets/.

Notebook PDFs are copied into presentation_assets/fig/ only if they are referenced
by \\includegraphics in thesis.tex and overleaf_bundle/chapters/*.tex (use
--copy-all-figures for the old behaviour). Use --prune-presentation-figs to remove
PDFs in that folder that are not thesis- or export-generated.

Run from repo root: python scripts/export_thesis_assets.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from thesis_layout import (  # noqa: E402
    LEGACY_TABLES,
    THESIS_TABLES_ROOT,
    resolve_csv,
)

from thesis_presentation_figures import copy_notebook_figures  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results"
OVERLEAF_BUNDLE = ROOT / "overleaf_bundle"
OUT_DIR = OVERLEAF_BUNDLE / "presentation_assets"
FIG_DIR = OUT_DIR / "fig"
TAB_DIR = OUT_DIR / "tab"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_DATASETS = ("compas", "german", "adult")


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
        n_hh_std = df["n_hh"].std(ddof=1) if n > 1 else 0.0
        mc_mean = df["mean_conflict"].mean() if "mean_conflict" in df.columns else 0.0
        mc_std = df["mean_conflict"].std(ddof=1) if n > 1 and "mean_conflict" in df.columns else 0.0
        sig = (df["p_empirical"] < 0.05).mean()
        rows.append({
            "dataset": name.replace("_", " ").title(),
            "n_runs": n,
            "mean_variance_mean": mv_mean,
            "mean_variance_std": mv_std,
            "moran_i_mean": mi_mean,
            "moran_i_std": mi_std,
            "n_hh_mean": n_hh_mean,
            "n_hh_std": n_hh_std,
            "mean_conflict_mean": mc_mean,
            "mean_conflict_std": mc_std,
            "frac_significant": sig,
        })
    return pd.DataFrame(rows)


def write_dataset_summary_tex():
    df = build_dataset_summary()
    if df.empty:
        print("No dataset summary (missing summary_per_run.csv). Skipping dataset_summary.tex")
        return
    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & HH count (mean $\pm$ std) \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        mv = f"{r['mean_variance_mean']:.4f} $\\pm$ {r['mean_variance_std']:.4f}"
        mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
        hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"
        out.append(
            f"{r['dataset']} & {mv} & {mi} & {hh} \\\\"
        )
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
    """Per-family (top-25 per family): prefer aggregated CSV over seeds; else legacy single-run CSV."""
    agg_path = RESULTS_DIR / "compas" / "per_family_spatial_aggregated.csv"
    out = []
    out.append(r"% Per-family Rashomon (top-K=25 per family), COMPAS. Mean $\pm$ std over outer seeds from experiment runner.")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(r"Family & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count (mean $\pm$ std) & Frac.\ sig. \\")
    out.append(r"\hline")

    if agg_path.is_file():
        df = pd.read_csv(agg_path)
        for _, r in df.iterrows():
            fam = r["family"]
            mv = f"{r['mean_variance_mean']:.6f} $\\pm$ {r['mean_variance_std']:.6f}"
            mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
            hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"
            fs = f"{float(r['frac_significant_moran']):.2f}"
            out.append(f"{fam} & {mv} & {mi} & {hh} & {fs} \\\\")
    else:
        path = resolve_csv("family_hv_hh_summary_compas.csv", "nb06")
        if path is None:
            print("Missing per_family_spatial_aggregated.csv and family_hv_hh_summary_compas.csv. Skipping family_summary.tex")
            return
        df = pd.read_csv(path)
        df["frac_sig"] = (df["moran_p"] < 0.05).astype(int)
        out[0] = (
            r"% Per-family Rashomon (top-K=25 per family). Legacy single-run CSV; re-run notebook 01 to get mean$\pm$std."
        )
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


def write_dataset_comparison_bars_figure():
    """Four-panel bar chart: Moran's I, HH count, mean variance, mean conflict (dataset_comparison_bars.pdf)."""
    df = build_dataset_summary()
    if df.empty:
        print("No dataset summary. Skipping dataset_comparison_bars.pdf")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.arange(len(df))
    labels = df["dataset"].values

    axes[0, 0].bar(x, df["moran_i_mean"], yerr=df["moran_i_std"], capsize=4, color="seagreen", edgecolor="black")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=15)
    axes[0, 0].set_ylabel("Moran's I")

    axes[0, 1].bar(x, df["n_hh_mean"], yerr=df.get("n_hh_std", 0), capsize=4, color="steelblue", edgecolor="black")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=15)
    axes[0, 1].set_ylabel("HH count")

    axes[1, 0].bar(x, df["mean_variance_mean"], yerr=df["mean_variance_std"], capsize=4, color="steelblue", edgecolor="black")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=15)
    axes[1, 0].set_ylabel("Mean variance")

    axes[1, 1].bar(x, df["mean_conflict_mean"], yerr=df["mean_conflict_std"], capsize=4, color="coral", edgecolor="black")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=15)
    axes[1, 1].set_ylabel("Mean conflict")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "dataset_comparison_bars.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "dataset_comparison_bars.pdf")


def write_hh_moran_per_run_compas():
    """HH count and Moran's I per run for COMPAS only -> hh_moran_per_run_compas.pdf."""
    path = RESULTS_DIR / "compas" / "summary_per_run.csv"
    if not path.exists():
        print("Missing compas/summary_per_run.csv. Skipping hh_moran_per_run_compas.pdf")
        return
    df = pd.read_csv(path).sort_values("outer_seed")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(df))
    ax1.bar(x, df["n_hh"].values, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Run (outer seed)")
    ax1.set_ylabel("HH count")
    ax1.set_title("HH count per run (COMPAS)")
    ax2.bar(x, df["moran_i"].values, color="seagreen", edgecolor="white")
    ax2.set_xlabel("Run (outer seed)")
    ax2.set_ylabel("Moran's I")
    ax2.set_title("Moran's I per run (COMPAS)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hh_moran_per_run_compas.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "hh_moran_per_run_compas.pdf")


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
    colors = {"Compas": "C0", "German": "C1", "Adult": "C2"}
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
    """HH count by family (COMPAS, per-family top-K) -> hh_by_family.pdf."""
    agg_path = RESULTS_DIR / "compas" / "per_family_spatial_aggregated.csv"
    if agg_path.is_file():
        df = pd.read_csv(agg_path)
        yerr = df["n_hh_std"].values if "n_hh_std" in df.columns else None
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(df))
        ax.bar(x, df["n_hh_mean"], yerr=yerr, capsize=3, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["family"].values, rotation=45, ha="right")
    else:
        path = resolve_csv("family_hv_hh_summary_compas.csv", "nb06")
        if path is None:
            print("Missing per_family_spatial_aggregated.csv and family_hv_hh_summary_compas.csv. Skipping hh_by_family.pdf")
            return
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df["family"], df["hh_count"], color="steelblue", edgecolor="white")
        plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("HH count")
    ax.set_xlabel("Model family")
    ax.set_title("HH count by family (Compas, per-family top-25, mean ± std over runs)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hh_by_family.pdf", bbox_inches="tight")
    plt.close()
    print("Wrote", FIG_DIR / "hh_by_family.pdf")


def write_null_significance_tex():
    path = resolve_csv("null_significance_summary.csv", "nb02")
    if path is None:
        print("Missing null_significance_summary.csv (nb02 / legacy tables). Skipping null_significance.tex")
        return
    df = pd.read_csv(path)
    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & $n$ runs & Frac.\ significant ($p < 0.05$) & Moran's $I$ (mean $\pm$ std) \\")
    out.append(r"\hline")
    for _, r in df.iterrows():
        ds = r["dataset"].replace("_", " ").title()
        frac_sig = r.get("frac_sig_moran", r.get("frac_significant", 0))
        moran_fmt = f"{r['obs_mean_moran']:.3f} $\\pm$ {r['obs_std_moran']:.3f}" if "obs_mean_moran" in r else str(r.get("mean_moran ± std", ""))
        out.append(f"{ds} & {int(r['n_runs'])} & {frac_sig:.2f} & {moran_fmt} \\\\")
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

    # Combined figure (all datasets)
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

    # Per-dataset figures (3 panels: mean var, Moran, HH) for thesis
    for ds_name, grp in agg_k.groupby("dataset"):
        d = grp.sort_values("K")
        fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.errorbar(d["K"], d["mean_variance_mean"], yerr=d["mean_variance_std"], marker="o", capsize=3, color="steelblue")
        ax1.set_xlabel("K (Rashomon size)")
        ax1.set_ylabel("Mean variance")
        ax1.set_title("Mean variance vs K")
        ax2.errorbar(d["K"], d["moran_mean"], yerr=d["moran_std"], marker="o", capsize=3, color="seagreen")
        ax2.set_xlabel("K (Rashomon size)")
        ax2.set_ylabel("Moran's I")
        ax2.set_title("Moran's I vs K")
        ax3.plot(d["K"], d["n_hh_mean"], marker="o", color="coral")
        ax3.set_xlabel("K (Rashomon size)")
        ax3.set_ylabel("HH count")
        ax3.set_title("HH count vs K")
        plt.suptitle(f"K sensitivity ({ds_name})")
        plt.tight_layout()
        out_name = f"sensitivity_K_curves_{ds_name}.pdf"
        fig3.savefig(FIG_DIR / out_name, bbox_inches="tight")
        plt.close()
        print("Wrote", FIG_DIR / out_name)


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

    # Combined figure (all datasets)
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

    # Per-dataset figures for thesis
    for ds_name, grp in agg_knn.groupby("dataset"):
        d = grp.sort_values("k_nn")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.errorbar(d["k_nn"], d["mean_variance_mean"], yerr=d["mean_variance_std"], marker="s", capsize=3, color="steelblue")
        ax1.set_xlabel("k (kNN neighbors)")
        ax1.set_ylabel("Mean variance")
        ax1.set_title("Mean variance vs k")
        ax2.errorbar(d["k_nn"], d["moran_mean"], yerr=d["moran_std"], marker="s", capsize=3, color="seagreen")
        ax2.set_xlabel("k (kNN neighbors)")
        ax2.set_ylabel("Moran's I")
        ax2.set_title("Moran's I vs k")
        plt.suptitle(f"kNN sensitivity ({ds_name})")
        plt.tight_layout()
        out_name = f"sensitivity_kNN_curves_{ds_name}.pdf"
        fig2.savefig(FIG_DIR / out_name, bbox_inches="tight")
        plt.close()
        print("Wrote", FIG_DIR / out_name)


def write_hh_component_summary_tex():
    """
    HH connected-component summary (notebook 03, min component size 5).
    Reads thesis_outputs/tables/nb03/hh_component_summary_<dataset>.csv (fallback: tables/).
    """
    out = []
    out.append(
        r"% Auto-generated from notebook 03 CSVs: hh_component_summary_{compas,german,adult}.csv"
    )
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(
        r"Dataset & Mean $n$ components & Mean max comp.\ size & Median max comp.\ size & Max (over runs) \\"
    )
    out.append(r"\hline")
    for ds in ("compas", "german", "adult"):
        label = ds.replace("_", " ").title()
        name = f"hh_component_summary_{ds}.csv"
        path = THESIS_TABLES_ROOT / "nb03" / name
        if not path.is_file():
            path = LEGACY_TABLES / name
        if not path.is_file():
            print(f"Missing {name}; using --- row for {label}")
            out.append(f"{label} & --- & --- & --- & --- \\\\")
            continue
        df = pd.read_csv(path)
        if df.empty or "n_components" not in df.columns or "max_component_size" not in df.columns:
            out.append(f"{label} & --- & --- & --- & --- \\\\")
            continue
        mn_c = float(df["n_components"].mean())
        mn_mx = float(df["max_component_size"].mean())
        med_mx = float(df["max_component_size"].median())
        max_mx = int(df["max_component_size"].max())
        out.append(
            f"{label} & {mn_c:.2f} & {mn_mx:.2f} & {int(med_mx)} & {max_mx} \\\\"
        )
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "hh_component_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "hh_component_summary.tex")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export thesis assets")
    parser.add_argument("--quick", action="store_true", help="Skip slow sensitivity analysis (K, kNN)")
    parser.add_argument(
        "--copy-all-figures",
        action="store_true",
        help="Copy every PDF from thesis_outputs/figures and legacy figures/ (ignore thesis allowlist).",
    )
    parser.add_argument(
        "--prune-presentation-figs",
        action="store_true",
        help="Remove PDFs under presentation_assets/fig/ not referenced by the thesis or this export script.",
    )
    args = parser.parse_args()

    write_dataset_summary_tex()
    write_global_summary_tex()
    write_family_summary_tex()
    write_null_significance_tex()
    write_hh_component_summary_tex()
    write_dataset_comparison_bars_figure()
    write_hh_moran_per_run_compas()
    write_spatial_patterns_figure()
    write_hh_by_family_figure()
    if not args.quick:
        run_sensitivity_K_and_save_figure()
        run_sensitivity_kNN_and_save_figure()
    else:
        print("Skipping sensitivity figures (--quick). Run without --quick for sensitivity_K_curves_*.pdf and sensitivity_kNN_curves_*.pdf")
    copy_notebook_figures(
        FIG_DIR,
        copy_all=args.copy_all_figures,
        prune_orphans=args.prune_presentation_figs,
        overleaf_bundle=OVERLEAF_BUNDLE,
    )
    print("Done.")


if __name__ == "__main__":
    main()
