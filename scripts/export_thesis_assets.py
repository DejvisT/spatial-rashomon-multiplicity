"""
Export thesis assets from results: dataset summary, null significance, calibration
summary, HH component summary, and sensitivity figures (K and kNN). Writes into
overleaf_bundle/presentation_assets/.

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
    HH connected-component summary for COMPAS only
    (notebook 03, min component size 5).

    Reads thesis_outputs/tables/nb03/hh_component_summary_compas.csv
    with fallback to legacy tables/.
    """
    ds = "compas"
    label = "COMPAS"
    name = f"hh_component_summary_{ds}.csv"

    path = THESIS_TABLES_ROOT / "nb03" / name
    if not path.is_file():
        path = LEGACY_TABLES / name

    out = []
    out.append(r"% Auto-generated from notebook 03 CSV: hh_component_summary_compas.csv")
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(
        r"Dataset & Mean $n$ components & Mean max comp.\ size & Median max comp.\ size & Max (over runs) \\"
    )
    out.append(r"\hline")

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {name}. Expected at {THESIS_TABLES_ROOT / 'nb03' / name} "
            f"or {LEGACY_TABLES / name}."
        )

    df = pd.read_csv(path)

    required_cols = {"n_components", "max_component_size"}
    missing_cols = required_cols - set(df.columns)
    if df.empty:
        raise ValueError(f"{name} is empty; cannot write COMPAS component summary.")
    if missing_cols:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_cols)}")

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

def write_calibration_summary_tex():
    """
    Calibration robustness summary (mean ± std over runs), split by method.

    Reads ``thesis_outputs/tables/nb07/calibration_summary.csv`` (notebook 07),
    then legacy ``tables/calibration_summary.csv``. Expects long-format rows with
    ``dataset``, ``method``, ``metric``, ``mean``, and ``std`` columns.

    Writes two LaTeX tables:
      - calibration_summary_platt.tex
      - calibration_summary_isotonic.tex
    """
    path = resolve_csv("calibration_summary.csv", "nb07")

    metrics_order = [
        ("delta_mean_variance", 4, 4),
        ("delta_moran_i", 4, 4),
        ("delta_n_HH", 1, 1),
        ("jaccard_HH_before_after", 4, 4),
        ("brier_improvement", 4, 4),
    ]

    method_order = ["platt", "isotonic"]
    method_files = {
        "platt": TAB_DIR / "calibration_summary_platt.tex",
        "isotonic": TAB_DIR / "calibration_summary_isotonic.tex",
    }

    def write_placeholder():
        for method in method_order:
            out = [
                rf"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
                r"\begin{tabular}{lccccc}",
                r"\hline",
                r"Dataset & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & $\Delta$ $n_{\mathrm{HH}}$ "
                r"& Jaccard HH & $\Delta$ Brier \\",
                r"\hline",
            ]
            for name in SUPPORTED_DATASETS:
                label = name.replace("_", " ").title()
                out.append(f"{label} & --- & --- & --- & --- & --- \\\\")
            out.extend([r"\hline", r"\end{tabular}"])
            method_files[method].write_text("\n".join(out), encoding="utf-8")
            print("Wrote", method_files[method], "(placeholder)")

    if path is None or not path.is_file():
        print(
            "Missing calibration_summary.csv (nb07). "
            "Writing placeholder calibration summary tables."
        )
        write_placeholder()
        return

    df = pd.read_csv(path)
    required = {"dataset", "method", "metric", "mean", "std"}
    if df.empty or not required.issubset(df.columns):
        print("calibration_summary.csv empty or missing required columns. Skipping calibration summary tables.")
        return

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["method"] = df["method"].astype(str).str.lower()

    def cell_for(dataset_key: str, method_key: str, metric: str, dm: int, ds: int) -> str:
        sub = df[
            (df["dataset"] == dataset_key)
            & (df["method"] == method_key)
            & (df["metric"] == metric)
        ]
        if sub.empty:
            return "---"
        mn = float(sub["mean"].iloc[0])
        st = float(sub["std"].iloc[0])
        if pd.isna(st):
            st = 0.0
        return f"{mn:.{dm}f} $\\pm$ {st:.{ds}f}"

    for method in method_order:
        out = []
        out.append(
            rf"% Auto-generated from nb07/calibration_summary.csv ({method.title()} scaling)."
        )
        out.append(r"\begin{tabular}{lccccc}")
        out.append(r"\hline")
        out.append(
            r"Dataset & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & $\Delta$ $n_{\mathrm{HH}}$ "
            r"& Jaccard HH & $\Delta$ Brier \\"
        )
        out.append(r"\hline")

        for name in SUPPORTED_DATASETS:
            label = name.replace("_", " ").title()
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in metrics_order]
            out.append(f"{label} & " + " & ".join(cells) + r" \\")

        out.append(r"\hline")
        out.append(r"\end{tabular}")
        method_files[method].write_text("\n".join(out), encoding="utf-8")
        print("Wrote", method_files[method])
    """
    Calibration robustness summary (mean ± std over runs), per dataset and method.

    Reads ``thesis_outputs/tables/nb07/calibration_summary.csv`` (notebook 07),
    then legacy ``tables/calibration_summary.csv``. Expects long-format rows with
    ``dataset``, ``method``, ``metric``, ``mean``, and ``std`` columns.

    Writes two LaTeX tables:
      - calibration_summary_spatial.tex  (Δ mean var, Δ Moran's I, Jaccard HH)
      - calibration_summary_support.tex  (Δ n_HH, Δ Brier)
    """
    path = resolve_csv("calibration_summary.csv", "nb07")
    method_order = ["platt", "isotonic"]
    method_labels = {
        "platt": "Platt",
        "isotonic": "Isotonic",
    }

    spatial_metrics = [
        ("delta_mean_variance", 4, 4),
        ("delta_moran_i", 4, 4),
        ("jaccard_HH_before_after", 4, 4),
    ]
    support_metrics = [
        ("delta_n_HH", 1, 1),
        ("brier_improvement", 4, 4),
    ]

    spatial_file = TAB_DIR / "calibration_summary_spatial.tex"
    support_file = TAB_DIR / "calibration_summary_support.tex"

    def write_placeholder():
        spatial_out = [
            r"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
            r"\begin{tabular}{llccc}",
            r"\hline",
            r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & Jaccard HH \\",
            r"\hline",
        ]
        support_out = [
            r"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
            r"\begin{tabular}{llcc}",
            r"\hline",
            r"Dataset & Method & $\Delta$ $n_{\mathrm{HH}}$ & $\Delta$ Brier \\",
            r"\hline",
        ]

        for name in SUPPORTED_DATASETS:
            label = name.replace("_", " ").title()
            first_row = True
            for method in ["Platt", "Isotonic"]:
                if first_row:
                    spatial_out.append(f"{label} & {method} & --- & --- & --- \\\\")
                    support_out.append(f"{label} & {method} & --- & --- \\\\")
                    first_row = False
                else:
                    spatial_out.append(f" & {method} & --- & --- & --- \\\\")
                    support_out.append(f" & {method} & --- & --- \\\\")

        spatial_out.extend([r"\hline", r"\end{tabular}"])
        support_out.extend([r"\hline", r"\end{tabular}"])

        spatial_file.write_text("\n".join(spatial_out), encoding="utf-8")
        support_file.write_text("\n".join(support_out), encoding="utf-8")
        print("Wrote", spatial_file, "(placeholder)")
        print("Wrote", support_file, "(placeholder)")

    if path is None or not path.is_file():
        print(
            "Missing calibration_summary.csv (nb07). "
            "Writing placeholder calibration summary tables"
        )
        write_placeholder()
        return

    df = pd.read_csv(path)
    required = {"dataset", "method", "metric", "mean", "std"}
    if df.empty or not required.issubset(df.columns):
        print("calibration_summary.csv empty or missing required columns. Skipping calibration summary tables.")
        return

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["method"] = df["method"].astype(str).str.lower()

    def cell_for(dataset_key: str, method_key: str, metric: str, dm: int, ds: int) -> str:
        sub = df[
            (df["dataset"] == dataset_key)
            & (df["method"] == method_key)
            & (df["metric"] == metric)
        ]
        if sub.empty:
            return "---"
        mn = float(sub["mean"].iloc[0])
        st = float(sub["std"].iloc[0])
        if pd.isna(st):
            st = 0.0
        return f"{mn:.{dm}f} $\\pm$ {st:.{ds}f}"

    # Spatial table
    spatial_out = []
    spatial_out.append(
        r"% Auto-generated from nb07/calibration_summary.csv (spatial calibration summary)."
    )
    spatial_out.append(r"\begin{tabular}{llccc}")
    spatial_out.append(r"\hline")
    spatial_out.append(
        r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & Jaccard HH \\"
    )
    spatial_out.append(r"\hline")

    for name in SUPPORTED_DATASETS:
        label = name.replace("_", " ").title()
        first_row = True
        for method in method_order:
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in spatial_metrics]
            if first_row:
                spatial_out.append(f"{label} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
                first_row = False
            else:
                spatial_out.append(f" & {method_labels[method]} & " + " & ".join(cells) + r" \\")

    spatial_out.append(r"\hline")
    spatial_out.append(r"\end{tabular}")
    spatial_file.write_text("\n".join(spatial_out), encoding="utf-8")
    print("Wrote", spatial_file)

    # Support table
    support_out = []
    support_out.append(
        r"% Auto-generated from nb07/calibration_summary.csv (support calibration summary)."
    )
    support_out.append(r"\begin{tabular}{llcc}")
    support_out.append(r"\hline")
    support_out.append(
        r"Dataset & Method & $\Delta$ $n_{\mathrm{HH}}$ & $\Delta$ Brier \\"
    )
    support_out.append(r"\hline")

    for name in SUPPORTED_DATASETS:
        label = name.replace("_", " ").title()
        first_row = True
        for method in method_order:
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in support_metrics]
            if first_row:
                support_out.append(f"{label} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
                first_row = False
            else:
                support_out.append(f" & {method_labels[method]} & " + " & ".join(cells) + r" \\")

    support_out.append(r"\hline")
    support_out.append(r"\end{tabular}")
    support_file.write_text("\n".join(support_out), encoding="utf-8")
    print("Wrote", support_file)
    """
    Calibration robustness summary (mean ± std over runs), per dataset and method.

    Reads ``thesis_outputs/tables/nb07/calibration_summary.csv`` (notebook 07),
    then legacy ``tables/calibration_summary.csv``. Expects long-format rows with
    ``dataset``, ``method``, ``metric``, ``mean``, and ``std`` columns.

    Writes two LaTeX tables:
      - calibration_summary.tex        (main table: Δ mean var, Δ Moran's I, Jaccard HH)
      - calibration_summary_extra.tex  (secondary table: Δ n_HH, Δ Brier)
    """
    path = resolve_csv("calibration_summary.csv", "nb07")
    method_order = ["platt", "isotonic"]
    method_labels = {
        "platt": "Platt",
        "isotonic": "Isotonic",
    }

    main_metrics = [
        ("delta_mean_variance", 4, 4),
        ("delta_moran_i", 4, 4),
        ("jaccard_HH_before_after", 4, 4),
    ]
    extra_metrics = [
        ("delta_n_HH", 1, 1),
        ("brier_improvement", 4, 4),
    ]

    def write_placeholder():
        main_out = [
            r"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
            r"\begin{tabular}{llccc}",
            r"\hline",
            r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & Jaccard HH \\",
            r"\hline",
        ]
        extra_out = [
            r"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
            r"\begin{tabular}{llcc}",
            r"\hline",
            r"Dataset & Method & $\Delta$ $n_{\mathrm{HH}}$ & $\Delta$ Brier \\",
            r"\hline",
        ]

        for name in SUPPORTED_DATASETS:
            label = name.replace("_", " ").title()
            first_row = True
            for method in ["Platt", "Isotonic"]:
                if first_row:
                    main_out.append(f"{label} & {method} & --- & --- & --- \\\\")
                    extra_out.append(f"{label} & {method} & --- & --- \\\\")
                    first_row = False
                else:
                    main_out.append(f" & {method} & --- & --- & --- \\\\")
                    extra_out.append(f" & {method} & --- & --- \\\\")

        main_out.extend([r"\hline", r"\end{tabular}"])
        extra_out.extend([r"\hline", r"\end{tabular}"])

        (TAB_DIR / "calibration_summary.tex").write_text("\n".join(main_out), encoding="utf-8")
        (TAB_DIR / "calibration_summary_extra.tex").write_text("\n".join(extra_out), encoding="utf-8")
        print("Wrote", TAB_DIR / "calibration_summary.tex (placeholder)")
        print("Wrote", TAB_DIR / "calibration_summary_extra.tex (placeholder)")

    if path is None or not path.is_file():
        print(
            "Missing calibration_summary.csv (nb07). "
            "Writing placeholder calibration_summary.tex and calibration_summary_extra.tex"
        )
        write_placeholder()
        return

    df = pd.read_csv(path)
    required = {"dataset", "method", "metric", "mean", "std"}
    if df.empty or not required.issubset(df.columns):
        print("calibration_summary.csv empty or missing required columns. Skipping calibration summary tables.")
        return

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["method"] = df["method"].astype(str).str.lower()

    def cell_for(dataset_key: str, method_key: str, metric: str, dm: int, ds: int) -> str:
        sub = df[
            (df["dataset"] == dataset_key)
            & (df["method"] == method_key)
            & (df["metric"] == metric)
        ]
        if sub.empty:
            return "---"
        mn = float(sub["mean"].iloc[0])
        st = float(sub["std"].iloc[0])
        if pd.isna(st):
            st = 0.0
        return f"{mn:.{dm}f} $\\pm$ {st:.{ds}f}"

    # -------------------------
    # Main table
    # -------------------------
    main_out = []
    main_out.append(
        r"% Auto-generated from nb07/calibration_summary.csv (main calibration table)."
    )
    main_out.append(r"\begin{tabular}{llccc}")
    main_out.append(r"\hline")
    main_out.append(
        r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & Jaccard HH \\"
    )
    main_out.append(r"\hline")

    for name in SUPPORTED_DATASETS:
        label = name.replace("_", " ").title()
        first_row = True
        for method in method_order:
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in main_metrics]
            if first_row:
                main_out.append(f"{label} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
                first_row = False
            else:
                main_out.append(f" & {method_labels[method]} & " + " & ".join(cells) + r" \\")

    main_out.append(r"\hline")
    main_out.append(r"\end{tabular}")
    (TAB_DIR / "calibration_summary.tex").write_text("\n".join(main_out), encoding="utf-8")
    print("Wrote", TAB_DIR / "calibration_summary.tex")

    # -------------------------
    # Secondary / appendix table
    # -------------------------
    extra_out = []
    extra_out.append(
        r"% Auto-generated from nb07/calibration_summary.csv (secondary calibration table)."
    )
    extra_out.append(r"\begin{tabular}{llcc}")
    extra_out.append(r"\hline")
    extra_out.append(
        r"Dataset & Method & $\Delta$ $n_{\mathrm{HH}}$ & $\Delta$ Brier \\"
    )
    extra_out.append(r"\hline")

    for name in SUPPORTED_DATASETS:
        label = name.replace("_", " ").title()
        first_row = True
        for method in method_order:
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in extra_metrics]
            if first_row:
                extra_out.append(f"{label} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
                first_row = False
            else:
                extra_out.append(f" & {method_labels[method]} & " + " & ".join(cells) + r" \\")

    extra_out.append(r"\hline")
    extra_out.append(r"\end{tabular}")
    (TAB_DIR / "calibration_summary_extra.tex").write_text("\n".join(extra_out), encoding="utf-8")
    print("Wrote", TAB_DIR / "calibration_summary_extra.tex")

    """
    Calibration robustness summary (mean ± std over runs), per dataset and method.

    Reads ``thesis_outputs/tables/nb07/calibration_summary.csv`` (notebook 07),
    then legacy ``tables/calibration_summary.csv``. Expects long-format rows with
    ``dataset``, ``method``, ``metric``, ``mean``, and ``std`` columns.

    Writes a table with one row per (dataset, method), so both Platt and isotonic
    scaling are included when present.
    """
    path = resolve_csv("calibration_summary.csv", "nb07")
    if path is None or not path.is_file():
        print(
            "Missing calibration_summary.csv (nb07). "
            "Writing placeholder calibration_summary.tex"
        )
        out = [
            r"% Placeholder: run notebook 07 and export nb07/calibration_summary.csv, then re-run.",
            r"\begin{tabular}{llccccc}",
            r"\hline",
            r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & $\Delta$ $n_{\mathrm{HH}}$ "
            r"& Jaccard HH & $\Delta$ Brier \\",
            r"\hline",
        ]
        for name in SUPPORTED_DATASETS:
            label = name.replace("_", " ").title()
            for method in ["Platt", "Isotonic"]:
                out.append(
                    f"{label} & {method} & --- & --- & --- & --- & --- \\\\"
                )
        out.extend([r"\hline", r"\end{tabular}"])
        (TAB_DIR / "calibration_summary.tex").write_text("\n".join(out), encoding="utf-8")
        print("Wrote", TAB_DIR / "calibration_summary.tex (placeholder)")
        return

    df = pd.read_csv(path)
    required = {"dataset", "method", "metric", "mean", "std"}
    if df.empty or not required.issubset(df.columns):
        print("calibration_summary.csv empty or missing required columns. Skipping calibration_summary.tex")
        return

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["method"] = df["method"].astype(str).str.lower()

    metrics_order = [
        ("delta_mean_variance", 4, 4),
        ("delta_moran_i", 4, 4),
        ("delta_n_HH", 1, 1),
        ("jaccard_HH_before_after", 4, 4),
        ("brier_improvement", 4, 4),
    ]

    method_order = ["platt", "isotonic"]
    method_labels = {
        "platt": "Platt",
        "isotonic": "Isotonic",
    }

    def cell_for(dataset_key: str, method_key: str, metric: str, dm: int, ds: int) -> str:
        sub = df[
            (df["dataset"] == dataset_key)
            & (df["method"] == method_key)
            & (df["metric"] == metric)
        ]
        if sub.empty:
            return "---"
        mn = float(sub["mean"].iloc[0])
        st = float(sub["std"].iloc[0])
        if pd.isna(st):
            st = 0.0
        return f"{mn:.{dm}f} $\\pm$ {st:.{ds}f}"

    out = []
    out.append(
        r"% Auto-generated from nb07/calibration_summary.csv (all calibration methods)."
    )
    out.append(r"\begin{tabular}{llccccc}")
    out.append(r"\hline")
    out.append(
        r"Dataset & Method & $\Delta$ mean var.\ & $\Delta$ Moran's $I$ & $\Delta$ $n_{\mathrm{HH}}$ "
        r"& Jaccard HH & $\Delta$ Brier \\"
    )
    out.append(r"\hline")

    for name in SUPPORTED_DATASETS:
        label = name.replace("_", " ").title()
        first_row = True
        for method in method_order:
            cells = [cell_for(name, method, m, dm, ds) for m, dm, ds in metrics_order]
            if first_row:
                out.append(f"{label} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
                first_row = False
            else:
                out.append(f" & {method_labels[method]} & " + " & ".join(cells) + r" \\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")
    (TAB_DIR / "calibration_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "calibration_summary.tex")


def _tex_escape(s: str) -> str:
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("%", r"\%")
         .replace("&", r"\&")
         .replace("#", r"\#")
    )


def _tex_escape_rule_text(s: str) -> str:
    """
    Escape rule text for LaTeX while rendering comparison operators correctly.
    This avoids LaTeX text-mode rendering of < and > as inverted punctuation.
    """
    s = _tex_escape(str(s))
    return (
        s.replace(">=", r"$\geq$")
         .replace("<=", r"$\leq$")
         .replace(">", r"$>$")
         .replace("<", r"$<$")
    )

def _pretty_dataset(name: str) -> str:
    name = str(name).strip().lower()
    return {
        "compas": "COMPAS",
        "german": "German",
        "adult": "Adult",
    }.get(name, name.title())


def _pretty_family(name: str) -> str:
    name = str(name).strip()
    return {
        "LogReg": "LogReg",
        "kNN": "kNN",
        "RF": "RF",
        "GBM": "GBM",
        "MLP": "MLP",
    }.get(name, name)


def write_hp_top2_driver_summary_tex():
    path = resolve_csv("hp_top2_driver_summary.csv", "nb06")
    if path is None or not path.is_file():
        print("Missing hp_top2_driver_summary.csv (nb06). Skipping hp_top2_driver_summary.tex")
        return

    df = pd.read_csv(path)
    required = {"dataset", "family", "Top-1 driver", "Top-2 driver"}
    if df.empty or not required.issubset(df.columns):
        print("hp_top2_driver_summary.csv empty or missing required columns. Skipping hp_top2_driver_summary.tex")
        return

    dataset_order = {"compas": 0, "german": 1, "adult": 2}
    family_order = {"GBM": 0, "LogReg": 1, "MLP": 2, "RF": 3, "kNN": 4}

    df = df.copy()
    df["dataset_order"] = df["dataset"].astype(str).str.lower().map(dataset_order).fillna(999)
    df["family_order"] = df["family"].map(family_order).fillna(999)
    df = df.sort_values(["dataset_order", "family_order", "dataset", "family"])

    out = []
    out.append(r"% Auto-generated from nb06/hp_top2_driver_summary.csv")
    out.append(r"\begin{tabular}{llp{4.8cm}p{4.8cm}}")
    out.append(r"\hline")
    out.append(r"Dataset & Family & Top-1 driver & Top-2 driver \\")
    out.append(r"\hline")

    last_dataset = None
    for _, r in df.iterrows():
        ds = _pretty_dataset(r["dataset"])
        fam = _pretty_family(r["family"])
        top1 = _tex_escape(r["Top-1 driver"])
        top2 = _tex_escape(r["Top-2 driver"])

        if last_dataset != ds:
            out.append(f"{ds} & {fam} & {top1} & {top2} \\\\")
            last_dataset = ds
        else:
            out.append(f" & {fam} & {top1} & {top2} \\\\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "hp_top2_driver_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "hp_top2_driver_summary.tex")

import re


def _format_delta_cell(text: str) -> str:
    """
    Convert strings like:
        'subsample (mean_delta=0.0266)'
        'n_estimators (mean_delta=-0.0094)'
    into:
        'subsample (+0.027)'
        'n_estimators (-0.009)'
    """
    s = str(text).strip()

    m = re.match(r"^(.*?)\s*\(mean_delta\s*=\s*([+-]?\d*\.?\d+)\)\s*$", s)
    if not m:
        return _tex_escape(s)

    hp = m.group(1).strip()
    val = float(m.group(2))
    return _tex_escape(f"{hp} ({val:+.3f})")


def write_hp_hotspot_delta_compas_tex():
    path = resolve_csv("hp_hotspot_delta_compas_compact.csv", "nb06")
    if path is None or not path.is_file():
        print("Missing hp_hotspot_delta_compas_compact.csv (nb06). Skipping hp_hotspot_delta_compas_compact.tex")
        return

    df = pd.read_csv(path)
    required = {"Family", "Most increased in HH", "Most decreased in HH"}
    if df.empty or not required.issubset(df.columns):
        print("hp_hotspot_delta_compas_compact.csv empty or missing required columns. Skipping hp_hotspot_delta_compas_compact.tex")
        return

    family_order = {"GBM": 0, "LogReg": 1, "MLP": 2, "RF": 3, "kNN": 4}
    df = df.copy()
    df["family_order"] = df["Family"].map(family_order).fillna(999)
    df = df.sort_values(["family_order", "Family"])

    out = []
    out.append(r"% Auto-generated from nb06/hp_hotspot_delta_compas_compact.csv")
    out.append(r"\begin{tabular}{lp{5.2cm}p{5.2cm}}")
    out.append(r"\hline")
    out.append(r"Family & Most increased in HH & Most decreased in HH \\")
    out.append(r"\hline")

    for _, r in df.iterrows():
        fam = _pretty_family(r["Family"])
        inc = _format_delta_cell(r["Most increased in HH"])
        dec = _format_delta_cell(r["Most decreased in HH"])
        out.append(f"{fam} & {inc} & {dec} \\\\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "hp_hotspot_delta_compas_compact.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "hp_hotspot_delta_compas_compact.tex")

def write_interpretable_rules_top_compas_tex():
    """Top COMPAS HH rules from notebook 09."""
    path = resolve_csv("rules_summary_compas.csv", "nb09")
    if path is None or not path.is_file():
        print("Missing rules_summary_compas.csv")
        return

    df = pd.read_csv(path)
    df = df[df["label"].eq("HH")].copy()

    # Prefer rules that are not tiny but still pure
    df = df[df["support"] >= 20]
    df = df.sort_values(["purity", "lift", "support"], ascending=[False, False, False])
    df = df.head(5)

    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Seed & Rule & Support & Purity & Recall & Lift \\",
        r"\hline",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"{int(r['outer_seed'])} & "
            f"{_tex_escape_rule_text(str(r['rule_text']))} & "
            f"{int(r['support'])} & "
            f"{r['purity']:.3f} & "
            f"{r['recall']:.3f} & "
            f"{r['lift']:.2f} \\\\"
        )

    rows += [r"\hline", r"\end{tabular}"]

    out_path = TAB_DIR / "interpretable_rules_top_compas.tex"
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_interpretable_rule_features_compas_tex():
    """Most frequent features used by COMPAS HH rules."""
    path = resolve_csv("rule_feature_frequency_compas.csv", "nb09")
    if path is None or not path.is_file():
        print("Missing rule_feature_frequency_compas.csv")
        return

    df = pd.read_csv(path)
    df = df.sort_values(
        ["n_seeds_with_feature", "n_rules_with_feature", "mean_purity_when_used"],
        ascending=False,
    ).head(8)

    rows = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Feature & Rules & Seeds & Mean purity & Mean lift \\",
        r"\hline",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"{_tex_escape(str(r['feature']))} & "
            f"{int(r['n_rules_with_feature'])} & "
            f"{int(r['n_seeds_with_feature'])} & "
            f"{r['mean_purity_when_used']:.3f} & "
            f"{r.get('mean_lift_when_used', 0.0):.3f} \\\\"
        )

    rows += [r"\hline", r"\end{tabular}"]

    out_path = TAB_DIR / "interpretable_rule_features_compas.tex"
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_component_rules_compas_tex():
    """Best component-level COMPAS HH rules."""
    path = resolve_csv("component_rules_summary_compas.csv", "nb09")
    if path is None or not path.is_file():
        print("Missing component_rules_summary_compas.csv")
        return

    df = pd.read_csv(path)
    df = df[df["support"] >= 10].copy()
    df = df.sort_values(
        ["component_purity", "component_lift", "support"],
        ascending=[False, False, False],
    ).head(5)

    rows = [
        r"\begin{tabular}{lllrrrr}",
        r"\hline",
        r"Seed & Component & Rule & Support & Purity & Recall & Lift \\",
        r"\hline",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"{int(r['outer_seed'])} & "
            f"{int(r['component_id'])} & "
            f"{_tex_escape_rule_text(str(r['rule_text']))} & "
            f"{int(r['support'])} & "
            f"{r['component_purity']:.3f} & "
            f"{r['component_recall']:.3f} & "
            f"{r['component_lift']:.2f} \\\\"
        )

    rows += [r"\hline", r"\end{tabular}"]

    out_path = TAB_DIR / "component_rules_compas.tex"
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_component_rule_features_compas_tex():
    """Most frequent features used by component-level COMPAS HH rules."""
    path = resolve_csv("component_feature_frequency_compas.csv", "nb09")
    if path is None or not path.is_file():
        print("Missing component_feature_frequency_compas.csv")
        return

    df = pd.read_csv(path)

    required = {
        "feature",
        "n_rules_with_feature",
        "n_seeds_with_feature",
        "mean_purity_when_used",
        "mean_lift_when_used",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"component_feature_frequency_compas.csv missing columns: {missing}")
        return

    df = df.sort_values(
        ["n_rules_with_feature", "n_seeds_with_feature", "mean_lift_when_used"],
        ascending=[False, False, False],
    ).head(8)

    rows = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Feature & Rules & Seeds & Mean purity & Mean lift \\",
        r"\hline",
    ]

    for _, r in df.iterrows():
        rows.append(
            f"{_tex_escape(str(r['feature']))} & "
            f"{int(r['n_rules_with_feature'])} & "
            f"{int(r['n_seeds_with_feature'])} & "
            f"{r['mean_purity_when_used']:.3f} & "
            f"{r['mean_lift_when_used']:.2f} \\\\"
        )

    rows += [r"\hline", r"\end{tabular}"]

    out_path = TAB_DIR / "component_feature_frequency_compas.tex"
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out_path}")


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
    write_calibration_summary_tex()

    write_hp_top2_driver_summary_tex()
    write_hp_hotspot_delta_compas_tex()

    write_interpretable_rules_top_compas_tex()
    write_interpretable_rule_features_compas_tex()
    write_component_rules_compas_tex()
    write_component_rule_features_compas_tex()

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
