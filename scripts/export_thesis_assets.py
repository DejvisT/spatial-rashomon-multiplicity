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
import re

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
        hh_rate_mean = df["hh_rate"].mean()
        hh_rate_std = df["hh_rate"].std(ddof=1) if n > 1 else 0.0
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
            "hh_rate_mean": hh_rate_mean,
            "hh_rate_std": hh_rate_std,
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
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Mean variance & Moran's $I$ & HH count & HH rate \\")
    out.append(
        r" & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) \\"
    )
    out.append(r"\hline")

    for _, r in df.iterrows():
        mv = f"{r['mean_variance_mean']:.4f} $\\pm$ {r['mean_variance_std']:.4f}"
        mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
        hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"
        hh_rate = f"{100 * r['hh_rate_mean']:.1f}\\% $\\pm$ {100 * r['hh_rate_std']:.1f}\\%"

        out.append(
            f"{r['dataset']} & {mv} & {mi} & {hh} & {hh_rate} \\\\"
        )

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "dataset_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "dataset_summary.tex")


def write_global_summary_tex():
    df = build_dataset_summary()
    if df.empty:
        print("No global summary. Skipping global_summary.tex")
        return

    out = []
    out.append(r"\begin{tabular}{lccccc}")
    out.append(r"\hline")
    out.append(
        r"Dataset & Mean variance & Moran's $I$ & \shortstack{HH count} & \shortstack{HH rate} & \shortstack{Significant\\runs} \\"
    )
    out.append(
        r" & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) & (\%) \\"
    )
    out.append(r"\hline")

    for _, r in df.iterrows():
        mv = f"{r['mean_variance_mean']:.4f} $\\pm$ {r['mean_variance_std']:.4f}"
        mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
        hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"
        hh_rate = f"{100 * r['hh_rate_mean']:.1f}\\% $\\pm$ {100 * r['hh_rate_std']:.1f}\\%"
        sig = f"{100 * r['frac_significant']:.0f}\\%"

        out.append(
            f"{r['dataset']} & {mv} & {mi} & {hh} & {hh_rate} & {sig} \\\\"
        )

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "global_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "global_summary.tex")

def write_family_summary_tex():
    """Per-family (top-25 per family): prefer aggregated CSV over seeds; else legacy single-run CSV."""
    agg_path = RESULTS_DIR / "compas" / "per_family_spatial_aggregated.csv"

    # Used only if the per-family CSV does not yet contain hh_rate columns.
    compas_n_test = 1443

    out = []
    out.append(
        r"% Per-family Rashomon (top-K=25 per family), COMPAS. Mean $\pm$ std over outer seeds from experiment runner."
    )
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(
        r"Family & Mean variance & Moran's $I$ & HH count & HH rate \\"
    )
    out.append(
        r" & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) \\"
    )
    out.append(r"\hline")

    if agg_path.is_file():
        df = pd.read_csv(agg_path)

        for _, r in df.iterrows():
            fam = r["family"]
            mv = f"{r['mean_variance_mean']:.6f} $\\pm$ {r['mean_variance_std']:.6f}"
            mi = f"{r['moran_i_mean']:.3f} $\\pm$ {r['moran_i_std']:.3f}"
            hh = f"{r['n_hh_mean']:.1f} $\\pm$ {r['n_hh_std']:.1f}"

            if "hh_rate_mean" in df.columns and "hh_rate_std" in df.columns:
                hh_rate_mean = r["hh_rate_mean"]
                hh_rate_std = r["hh_rate_std"]
            else:
                hh_rate_mean = r["n_hh_mean"] / compas_n_test
                hh_rate_std = r["n_hh_std"] / compas_n_test

            hh_rate = (
                f"{100 * hh_rate_mean:.1f}\\% $\\pm$ "
                f"{100 * hh_rate_std:.1f}\\%"
            )

            out.append(f"{fam} & {mv} & {mi} & {hh} & {hh_rate} \\\\")

    else:
        path = resolve_csv("family_hv_hh_summary_compas.csv", "nb06")
        if path is None:
            print(
                "Missing per_family_spatial_aggregated.csv and "
                "family_hv_hh_summary_compas.csv. Skipping family_summary.tex"
            )
            return

        df = pd.read_csv(path)
        out[0] = (
            r"% Per-family Rashomon (top-K=25 per family). "
            r"Legacy single-run CSV; re-run notebook 01 to get mean$\pm$std."
        )

        for _, r in df.iterrows():
            fam = r["family"]
            mv = f"{r['mean_var']:.6f}"
            mi = f"{r['moran_I']:.3f}"
            hh_count = int(r["hh_count"])
            hh = str(hh_count)
            hh_rate = f"{100 * hh_count / compas_n_test:.1f}\\%"

            out.append(f"{fam} & {mv} & {mi} & {hh} & {hh_rate} \\\\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "family_summary.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "family_summary.tex")


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

    dataset_labels = {
        "compas": "COMPAS",
        "german": "German Credit",
        "adult": "Adult",
    }

    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Runs & Significant runs & Moran's $I$ \\")
    out.append(r"\hline")

    for _, r in df.iterrows():
        ds_key = str(r["dataset"]).lower()
        ds = dataset_labels.get(ds_key, str(r["dataset"]).replace("_", " ").title())

        frac_sig = r.get("frac_sig_moran", r.get("frac_significant", 0))
        sig_fmt = f"{100 * frac_sig:.0f}\\%"

        if "obs_mean_moran" in df.columns and "obs_std_moran" in df.columns:
            moran_fmt = f"{r['obs_mean_moran']:.3f} $\\pm$ {r['obs_std_moran']:.3f}"
        else:
            moran_fmt = str(r.get("mean_moran ± std", ""))

        out.append(f"{ds} & {int(r['n_runs'])} & {sig_fmt} & {moran_fmt} \\\\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "null_significance.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "null_significance.tex")


def write_conflict_null_significance_tex():
    path = resolve_csv("conflict_null_significance_summary.csv", "nb02")
    if path is None:
        print(
            "Missing conflict_null_significance_summary.csv (nb02 / legacy tables). "
            "Skipping conflict_null_significance.tex"
        )
        return

    df = pd.read_csv(path)

    dataset_labels = {
        "compas": "COMPAS",
        "german": "German Credit",
        "adult": "Adult",
    }

    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Runs & Significant runs & Conflict Moran's $I$ \\")
    out.append(r"\hline")

    for _, r in df.iterrows():
        ds_key = str(r["dataset"]).lower()
        ds = dataset_labels.get(ds_key, str(r["dataset"]).replace("_", " ").title())

        sig_fmt = f"{100 * r['frac_sig']:.0f}\\%"
        moran_fmt = f"{r['obs_mean']:.3f} $\\pm$ {r['obs_std']:.3f}"

        out.append(
            f"{ds} & {int(r['n_runs'])} & {sig_fmt} & {moran_fmt} \\\\"
        )

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "conflict_null_significance.tex").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("Wrote", TAB_DIR / "conflict_null_significance.tex")


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


def _pretty_family(name: str) -> str:
    name = str(name).strip()
    return {
        "LogReg": "LogReg",
        "kNN": "kNN",
        "RF": "RF",
        "GBM": "GBM",
        "MLP": "MLP",
    }.get(name, name)


def _format_delta_cell(value):
    """
    Format strings like:
        'learning_rate_init (mean_delta=0.0475, n_seeds=10)'
        'max_depth (mean_delta=-0.0399, n_seeds=10)'
        'none'

    as:
        learning\_rate\_init $(+0.048)$
        max\_depth $(-0.040)$
        none
    """
    if pd.isna(value):
        return "none"

    text = str(value).strip()
    if not text or text.lower() == "none":
        return "none"

    # Match current compact CSV format:
    # "hp_name (mean_delta=0.0266, n_seeds=10)"
    m = re.match(
        r"^(.*?)\s*\(.*?mean_delta\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?).*?\)\s*$",
        text,
    )
    if m:
        hp = m.group(1).strip().replace("_", r"\_")
        delta = float(m.group(2))
        return f"{hp} $({delta:+.3f})$"

    # Match already compact format:
    # "hp_name (+0.027)"
    m = re.match(
        r"^(.*?)\s*\(([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\)\s*$",
        text,
    )
    if m:
        hp = m.group(1).strip().replace("_", r"\_")
        delta = float(m.group(2))
        return f"{hp} $({delta:+.3f})$"

    # Fallback: escape underscores only.
    return text.replace("_", r"\_")


def write_hp_hotspot_delta_compas_tex():
    path = resolve_csv("hp_hotspot_delta_compas_compact.csv", "nb06")
    if path is None or not path.is_file():
        print(
            "Missing hp_hotspot_delta_compas_compact.csv (nb06). "
            "Skipping hp_hotspot_delta_compas_compact.tex"
        )
        return

    df = pd.read_csv(path)
    required = {"Family", "Most increased in HH", "Most decreased in HH"}
    if df.empty or not required.issubset(df.columns):
        print(
            "hp_hotspot_delta_compas_compact.csv empty or missing required columns. "
            "Skipping hp_hotspot_delta_compas_compact.tex"
        )
        return

    family_order = {"GBM": 0, "LogReg": 1, "MLP": 2, "RF": 3, "kNN": 4}
    df = df.copy()
    df["family_order"] = df["Family"].map(family_order).fillna(999)
    df = df.sort_values(["family_order", "Family"])

    out = []
    out.append(r"% Auto-generated from nb06/hp_hotspot_delta_compas_compact.csv")
    out.append(r"\begin{tabular}{lll}")
    out.append(r"\toprule")
    out.append(
        r"Family & Largest positive HH shift & Largest negative HH shift \\"
    )
    out.append(r"\midrule")

    for _, r in df.iterrows():
        fam = _pretty_family(r["Family"])
        inc = _format_delta_cell(r["Most increased in HH"])
        dec = _format_delta_cell(r["Most decreased in HH"])
        out.append(f"{fam} & {inc} & {dec} \\\\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")

    (TAB_DIR / "hp_hotspot_delta_compas_compact.tex").write_text(
        "\n".join(out),
        encoding="utf-8",
    )
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


def _pretty_dataset(name: str) -> str:
    name = str(name).strip().lower()
    return {
        "compas": "COMPAS",
        "german": "German",
        "adult": "Adult",
    }.get(name, name.title())


def write_alternative_knn_comparison_tex():
    """
    Alternative graph construction summary from Notebook 10.

    Reads:
        thesis_outputs/tables/nb10/alternative_knn_comparison.csv

    Writes:
        overleaf_bundle/presentation_assets/tab/alternative_knn_comparison.tex
    """
    path = resolve_csv("alternative_knn_comparison.csv", "nb10")
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Missing alternative_knn_comparison.csv from Notebook 10. "
            "Run notebook 10 before exporting thesis assets."
        )

    df = pd.read_csv(path)

    required = {
        "dataset",
        "method",
        "moran_mean",
        "moran_std",
        "hh_mean",
        "hh_std",
    }
    missing = required - set(df.columns)
    if df.empty:
        raise ValueError("alternative_knn_comparison.csv is empty.")
    if missing:
        raise ValueError(
            f"alternative_knn_comparison.csv is missing required columns: {sorted(missing)}"
        )

    dataset_order = {"compas": 0, "german": 1, "adult": 2}
    method_order = {"euclidean": 0, "pca_15": 1, "cosine": 2}
    method_labels = {
        "euclidean": "Euclidean (baseline)",
        "pca_15": "PCA (15 comp.)",
        "cosine": "Cosine",
    }

    df = df.copy()
    df["dataset_key"] = df["dataset"].astype(str).str.lower()
    df["method_key"] = df["method"].astype(str).str.lower()
    df["dataset_order"] = df["dataset_key"].map(dataset_order).fillna(999)
    df["method_order"] = df["method_key"].map(method_order).fillna(999)
    df = df.sort_values(["dataset_order", "method_order", "dataset_key", "method_key"])

    out = []
    out.append(r"% Auto-generated from nb10/alternative_knn_comparison.csv")
    out.append(r"\begin{tabular}{llcc}")
    out.append(r"\toprule")
    out.append(r"Dataset & Method & Moran's $I$ & HH count \\")
    out.append(r"\midrule")

    last_dataset = None
    for _, r in df.iterrows():
        ds_key = r["dataset_key"]
        method_key = r["method_key"]

        ds = _pretty_dataset(ds_key)
        method = method_labels.get(method_key, _tex_escape(method_key))

        moran = f"${float(r['moran_mean']):.3f} \\pm {float(r['moran_std']):.3f}$"
        hh = f"${float(r['hh_mean']):.1f} \\pm {float(r['hh_std']):.1f}$"

        if last_dataset is not None and ds_key != last_dataset:
            out.append(r"\midrule")

        ds_cell = ds if ds_key != last_dataset else ""
        out.append(f"{ds_cell} & {method} & {moran} & {hh} \\\\")

        last_dataset = ds_key

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")

    out_path = TAB_DIR / "alternative_knn_comparison.tex"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print("Wrote", out_path)


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


def write_conflict_summary_tex():
    """
    Conflict metrics by dataset.

    Reads:
        results/<dataset>/summary_per_run.csv

    Writes:
        overleaf_bundle/presentation_assets/tab/conflict_summary.tex
    """
    rows = []

    dataset_order = {"compas": 0, "german": 1, "adult": 2}

    for ds in SUPPORTED_DATASETS:
        path = RESULTS_DIR / ds / "summary_per_run.csv"
        if not path.is_file():
            print(f"Missing {path}. Skipping {ds} in conflict_summary.tex")
            continue

        df = pd.read_csv(path)

        required = {
            "mean_conflict",
            "frac_conflict_gt0",
            "frac_conflict_ge025",
            "conflict_moran_i",
            "conflict_n_hh",
        }
        missing = required - set(df.columns)
        if df.empty or missing:
            raise ValueError(
                f"{path} is empty or missing required columns: {sorted(missing)}"
            )

        n = len(df)
        ddof = 1 if n > 1 else 0

        rows.append({
            "dataset": ds,
            "dataset_order": dataset_order.get(ds, 999),
            "mean_conflict_mean": df["mean_conflict"].mean(),
            "mean_conflict_std": df["mean_conflict"].std(ddof=ddof),
            "frac_conflict_gt0_mean": 100.0 * df["frac_conflict_gt0"].mean(),
            "frac_conflict_gt0_std": 100.0 * df["frac_conflict_gt0"].std(ddof=ddof),
            "frac_conflict_ge025_mean": 100.0 * df["frac_conflict_ge025"].mean(),
            "frac_conflict_ge025_std": 100.0 * df["frac_conflict_ge025"].std(ddof=ddof),
            "conflict_moran_i_mean": df["conflict_moran_i"].mean(),
            "conflict_moran_i_std": df["conflict_moran_i"].std(ddof=ddof),
            "conflict_n_hh_mean": df["conflict_n_hh"].mean(),
            "conflict_n_hh_std": df["conflict_n_hh"].std(ddof=ddof),
        })

    if not rows:
        print("No conflict summary rows. Skipping conflict_summary.tex")
        return

    out_df = pd.DataFrame(rows).sort_values(["dataset_order", "dataset"])

    out = []
    out.append(r"% Auto-generated from results/<dataset>/summary_per_run.csv")
    out.append(r"\begin{tabular}{lccccc}")
    out.append(r"\toprule")
    out.append(
        r"Dataset & Mean $c_i$ & $c_i > 0$ (\%) & $c_i \geq 0.25$ (\%) & Conflict Moran's $I$ & Conflict HH \\"
    )
    out.append(r"\midrule")

    for _, r in out_df.iterrows():
        dataset = _pretty_dataset(r["dataset"])

        mean_ci = (
            f"${r['mean_conflict_mean']:.3f} "
            f"\\pm {r['mean_conflict_std']:.3f}$"
        )
        frac_gt0 = (
            f"${r['frac_conflict_gt0_mean']:.1f} "
            f"\\pm {r['frac_conflict_gt0_std']:.1f}$"
        )
        frac_ge025 = (
            f"${r['frac_conflict_ge025_mean']:.1f} "
            f"\\pm {r['frac_conflict_ge025_std']:.1f}$"
        )
        conflict_moran = (
            f"${r['conflict_moran_i_mean']:.2f} "
            f"\\pm {r['conflict_moran_i_std']:.2f}$"
        )
        conflict_hh = (
            f"${r['conflict_n_hh_mean']:.1f} "
            f"\\pm {r['conflict_n_hh_std']:.1f}$"
        )

        out.append(
            f"{dataset} & {mean_ci} & {frac_gt0} & {frac_ge025} & "
            f"{conflict_moran} & {conflict_hh} \\\\"
        )

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")

    out_path = TAB_DIR / "conflict_summary.tex"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print("Wrote", out_path)


def write_aggregate_multiplicity_summary_tex():
    path = THESIS_TABLES_ROOT / "nb01" / "dataset_summary.csv"
    if not path.exists():
        print("Missing nb01/dataset_summary.csv. Skipping aggregate_multiplicity_summary.tex")
        return

    df = pd.read_csv(path)

    out = []
    out.append(r"\begin{tabular}{lccc}")
    out.append(r"\hline")
    out.append(r"Dataset & Ambiguity & Disagreement rate & Discrepancy \\")
    out.append(r" & (mean $\pm$ std) & (mean $\pm$ std) & (mean $\pm$ std) \\")
    out.append(r"\hline")

    for _, r in df.iterrows():
        dataset = str(r["dataset"]).replace("_", " ").title()
        if dataset.lower() == "German":
            dataset = "German Credit"

        ambiguity = f"{r['ambiguity_mean']:.3f} $\\pm$ {r['ambiguity_std']:.3f}"
        disagreement = f"{r['disagreement_rate_mean']:.3f} $\\pm$ {r['disagreement_rate_std']:.3f}"
        discrepancy = f"{r['discrepancy_mean']:.3f} $\\pm$ {r['discrepancy_std']:.3f}"

        out.append(f"{dataset} & {ambiguity} & {disagreement} & {discrepancy} \\\\")

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "aggregate_multiplicity_summary.tex").write_text(
        "\n".join(out), encoding="utf-8"
    )
    print("Wrote", TAB_DIR / "aggregate_multiplicity_summary.tex")


def write_quadrant_compas_tex():
    """Write COMPAS variance/conflict quadrant table from aggregated CSV."""
    path = RESULTS_DIR / "compas" / "quadrant_breakdown_aggregated.csv"
    if not path.exists():
        print("Missing compas/quadrant_breakdown_aggregated.csv. Skipping quadrant_compas.tex")
        return

    df = pd.read_csv(path)

    labels = {
        "A": "A (high var, high conflict)",
        "B": "B (high var, low conflict)",
        "C": "C (low var, high conflict)",
        "D": "D (low var, low conflict)",
    }

    out = []
    out.append(r"\begin{tabular}{lcccc}")
    out.append(r"\hline")
    out.append(
        r"Quadrant & \shortstack{Number of test\\observations} & \shortstack{Share of test\\observations} & \shortstack{Predictive\\variance $v_i$} & \shortstack{Conflict\\ratio $c_i$} \\"
    )
    out.append(r"\hline")

    for q in ["A", "B", "C", "D"]:
        r = df[df["quadrant"] == q].iloc[0]

        count = f"{r['count_mean']:.1f} $\\pm$ {r['count_std']:.1f}"
        share = f"{100 * r['fraction_mean']:.1f}\\% $\\pm$ {100 * r['fraction_std']:.1f}\\%"
        mean_vi = f"{r['mean_var_p_mean']:.4f} $\\pm$ {r['mean_var_p_std']:.4f}"
        mean_ci = f"{r['mean_conflict_mean']:.3f} $\\pm$ {r['mean_conflict_std']:.3f}"

        out.append(
            f"{labels[q]} & {count} & {share} & {mean_vi} & {mean_ci} \\\\"
        )

    out.append(r"\hline")
    out.append(r"\end{tabular}")

    (TAB_DIR / "quadrant_compas.tex").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", TAB_DIR / "quadrant_compas.tex")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export thesis assets")
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
    write_conflict_null_significance_tex()
    write_hh_component_summary_tex()
    write_alternative_knn_comparison_tex()
    write_conflict_summary_tex()
    write_aggregate_multiplicity_summary_tex()
    write_hp_hotspot_delta_compas_tex()

    write_interpretable_rules_top_compas_tex()
    write_interpretable_rule_features_compas_tex()
    write_component_rules_compas_tex()
    write_component_rule_features_compas_tex()

    write_quadrant_compas_tex()
    write_hh_moran_per_run_compas()
    write_spatial_patterns_figure()
    write_hh_by_family_figure()
    copy_notebook_figures(
        FIG_DIR,
        copy_all=args.copy_all_figures,
        prune_orphans=args.prune_presentation_figs,
        overleaf_bundle=OVERLEAF_BUNDLE,
    )
    print("Done.")

if __name__ == "__main__":
    main()
