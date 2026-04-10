"""Generate only global_summary, family_summary, spatial_patterns_per_run.pdf, hh_by_family.pdf.
Optionally copy thesis-referenced notebook PDFs into presentation_assets/fig/ (same allowlist as full export).

Run from repo root: python scripts/export_thesis_assets_quick.py [--copy-thesis-figs] [--prune-presentation-figs]
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from thesis_layout import resolve_csv  # noqa: E402
from thesis_presentation_figures import copy_notebook_figures  # noqa: E402

RESULTS_DIR = ROOT / "results"
OVERLEAF_BUNDLE = ROOT / "overleaf_bundle"
FIG_DIR = OVERLEAF_BUNDLE / "presentation_assets" / "fig"
TAB_DIR = OVERLEAF_BUNDLE / "presentation_assets" / "tab"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_DATASETS = ("compas", "german", "adult")


def main():
    parser = argparse.ArgumentParser(description="Quick export of selected thesis tables/figures")
    parser.add_argument(
        "--copy-thesis-figs",
        action="store_true",
        help="Copy thesis-referenced PDFs from thesis_outputs/figures (and legacy figures/) into presentation_assets/fig/",
    )
    parser.add_argument(
        "--copy-all-figures",
        action="store_true",
        help="With --copy-thesis-figs: copy every derived PDF, not only thesis \\includegraphics references",
    )
    parser.add_argument(
        "--prune-presentation-figs",
        action="store_true",
        help="Remove orphan PDFs under presentation_assets/fig/ (thesis + export script allowlist only)",
    )
    args = parser.parse_args()

    path_fam = None
    # ----- global_summary.tex -----
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
            "mean_variance_mean": mv_mean, "mean_variance_std": mv_std,
            "moran_i_mean": mi_mean, "moran_i_std": mi_std,
            "n_hh_mean": n_hh_mean, "frac_significant": sig,
        })
    if rows:
        df = pd.DataFrame(rows)
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

    # ----- family_summary.tex -----
    agg_fam = RESULTS_DIR / "compas" / "per_family_spatial_aggregated.csv"
    path_fam = resolve_csv("family_hv_hh_summary_compas.csv", "nb06")
    if agg_fam.is_file():
        df = pd.read_csv(agg_fam)
        out = []
        out.append(r"% Per-family COMPAS; mean $\pm$ std over outer seeds (experiment runner).")
        out.append(r"\begin{tabular}{lcccc}")
        out.append(r"\hline")
        out.append(r"Family & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count (mean $\pm$ std) & Frac.\ sig. \\")
        out.append(r"\hline")
        for _, r in df.iterrows():
            mv = f"{r['mean_variance_mean']:.6f} $\\pm$ {r['mean_variance_std']:.6f}"
            mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
            hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"
            fs = f"{float(r['frac_significant_moran']):.2f}"
            out.append(f"{r['family']} & {mv} & {mi} & {hh} & {fs} \\\\")
        out.append(r"\hline")
        out.append(r"\end{tabular}")
        (TAB_DIR / "family_summary.tex").write_text("\n".join(out), encoding="utf-8")
        print("Wrote", TAB_DIR / "family_summary.tex")
    elif path_fam is not None:
        df = pd.read_csv(path_fam)
        df["frac_sig"] = (df["moran_p"] < 0.05).astype(int)
        out = []
        out.append(r"% Per-family (top-K=25 per family). Legacy single-run CSV.")
        out.append(r"\begin{tabular}{lcccc}")
        out.append(r"\hline")
        out.append(r"Family & Mean variance (mean $\pm$ std) & Moran's $I$ (mean $\pm$ std) & Mean HH count (mean $\pm$ std) & Frac.\ sig. \\")
        out.append(r"\hline")
        for _, r in df.iterrows():
            mv = f"{r['mean_var']:.6f}"
            mi = f"{r['moran_I']:.3f}"
            hh = str(int(r["hh_count"]))
            fs = "1.00" if r["frac_sig"] else "0.00"
            out.append(f"{r['family']} & {mv} & {mi} & {hh} & {fs} \\\\")
        out.append(r"\hline")
        out.append(r"\end{tabular}")
        (TAB_DIR / "family_summary.tex").write_text("\n".join(out), encoding="utf-8")
        print("Wrote", TAB_DIR / "family_summary.tex")

    # ----- spatial_patterns_per_run.pdf -----
    rows = []
    for name in SUPPORTED_DATASETS:
        p = RESULTS_DIR / name / "summary_per_run.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["dataset"] = name.replace("_", " ").title()
        rows.append(d)
    if rows:
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

    # ----- hh_by_family.pdf -----
    if agg_fam.is_file():
        df = pd.read_csv(agg_fam)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(df))
        ax.bar(x, df["n_hh_mean"], yerr=df["n_hh_std"], capsize=3, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(df["family"].values, rotation=45, ha="right")
        ax.set_ylabel("HH count")
        ax.set_xlabel("Model family")
        ax.set_title("HH count by family (Compas, per-family top-25, mean ± std)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "hh_by_family.pdf", bbox_inches="tight")
        plt.close()
        print("Wrote", FIG_DIR / "hh_by_family.pdf")
    elif path_fam is not None:
        df = pd.read_csv(path_fam)
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

    if args.copy_thesis_figs or args.prune_presentation_figs:
        copy_notebook_figures(
            FIG_DIR,
            copy_all=args.copy_all_figures,
            prune_orphans=args.prune_presentation_figs,
            overleaf_bundle=OVERLEAF_BUNDLE,
        )

    print("Done.")


if __name__ == "__main__":
    main()
