"""
Combined multi-panel figures for notebook 06 (Section 5.7).

Keeps thesis-facing single-panel PDFs where ``results.tex`` references them, while
providing overview grids so the notebook shows fewer (richer) figures.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.hp_analysis import POOL_TYPE_RASHOMON

FAMILY_ORDER_DEFAULT: Tuple[str, ...] = ("GBM", "LogReg", "MLP", "RF", "kNN")
DATASET_ORDER_DEFAULT: Tuple[str, ...] = ("compas", "german", "adult")


def _meta_importance_path(table_dir: Path, ds: str, fam: str) -> Optional[Path]:
    pt = POOL_TYPE_RASHOMON.replace(" ", "_")
    fp = table_dir / f"hp_meta_importance_{ds}_{fam}_{pt}.csv"
    if fp.is_file():
        return fp
    cands = sorted(glob.glob(str(table_dir / f"hp_meta_importance_{ds}_{fam}_*.csv")))
    return Path(cands[0]) if cands else None


def plot_meta_importance_and_stability_grids(
    *,
    table_dir: Path,
    fig_dir: Path,
    df_vm_agg: pd.DataFrame,
    top_hp: int,
    family_order: Sequence[str] = FAMILY_ORDER_DEFAULT,
    dataset_order: Sequence[str] = DATASET_ORDER_DEFAULT,
    save_individual_importance: bool = True,
) -> None:
    """
    Two overview figures:
    1) grouped meta-model importances (Rashomon, ``V_m``),
    2) secondary rank-stability (``V_m`` unique-value path, subset=all).

    Optionally re-saves per-(dataset, family) importance PDFs without ``plt.show`` noise.
    """
    meta_sum = table_dir / "hp_meta_model_summary.csv"
    if not meta_sum.is_file():
        print("plot_meta_importance_and_stability_grids: missing hp_meta_model_summary.csv")
        return

    ms = pd.read_csv(meta_sum)
    pool_ok = ms["pool_type"] == POOL_TYPE_RASHOMON if "pool_type" in ms.columns else True
    tgt_ok = ms["target_vm"] == "V_m" if "target_vm" in ms.columns else True
    groups = ms[pool_ok & tgt_ok][["dataset", "family"]].drop_duplicates()
    if groups.empty:
        print("plot_meta_importance_and_stability_grids: no Rashomon V_m rows")
        return

    n_r, n_c = len(family_order), len(dataset_order)

    # --- Grid 1: meta-model grouped importances ---
    fig1, axes1 = plt.subplots(n_r, n_c, figsize=(3.4 * n_c, 2.5 * n_r), squeeze=False)
    for i, fam in enumerate(family_order):
        for j, ds in enumerate(dataset_order):
            ax = axes1[i][j]
            fp = _meta_importance_path(table_dir, ds, fam)
            if fp is None:
                ax.set_axis_off()
                continue
            imp = pd.read_csv(fp).sort_values("importance", ascending=False).head(top_hp)
            if imp.empty:
                ax.set_axis_off()
                continue
            y = np.arange(len(imp))
            ax.barh(y, imp["importance"], color="steelblue", alpha=0.88)
            ax.set_yticks(y)
            ax.set_yticklabels(imp["feature_group"].astype(str), fontsize=7)
            ax.invert_yaxis()
            ax.tick_params(axis="x", labelsize=7)
            if i == 0:
                ax.set_title(ds, fontsize=9)
            if j == 0:
                ax.set_ylabel(fam, fontsize=9)
    fig1.suptitle("Meta-model: grouped importances (Rashomon, $V_m$)", fontsize=11)
    fig1.tight_layout()
    p1 = fig_dir / "hp_meta_importance_bar_rashomon_grid.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    plt.show()

    if save_individual_importance:
        for _, row in groups.iterrows():
            ds, fam = row["dataset"], row["family"]
            fp = _meta_importance_path(table_dir, ds, fam)
            if fp is None:
                continue
            imp = pd.read_csv(fp).sort_values("importance", ascending=False).head(top_hp)
            if imp.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(imp))))
            y = np.arange(len(imp))
            ax.barh(y, imp["importance"], color="steelblue", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(imp["feature_group"].astype(str))
            ax.invert_yaxis()
            ax.set_xlabel("Grouped meta-model importance (descriptive RF)")
            ax.set_title(f"{ds} — {fam} — {POOL_TYPE_RASHOMON}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"hp_importance_bar_rashomon_{ds}_{fam}.pdf", bbox_inches="tight")
            plt.close(fig)

    # --- Grid 2: rank stability (secondary V_m-by-HP-value aggregates) ---
    vm = df_vm_agg.copy()
    if not vm.empty and "pool_type" in vm.columns:
        vm = vm[vm["pool_type"] == POOL_TYPE_RASHOMON]
    if not vm.empty and "subset" in vm.columns:
        vm = vm[vm["subset"] == "all"].copy()
    if vm.empty:
        print("plot_meta_importance_and_stability_grids: no vm_agg for stability grid")
        return

    fig2, axes2 = plt.subplots(n_r, n_c, figsize=(3.4 * n_c, 2.5 * n_r), squeeze=False)
    for i, fam in enumerate(family_order):
        for j, ds in enumerate(dataset_order):
            ax = axes2[i][j]
            grp = vm[(vm["dataset"] == ds) & (vm["family"] == fam)]
            top = grp.nlargest(top_hp, "mean_importance").copy()
            if top.empty:
                ax.set_axis_off()
                continue
            y = np.arange(len(top))
            ax.barh(y, top["rank_freq_top3"], color="slategray")
            ax.set_yticks(y)
            ax.set_yticklabels(top["hp_name"], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05)
            ax.tick_params(axis="x", labelsize=7)
            if i == 0:
                ax.set_title(ds, fontsize=9)
            if j == 0:
                ax.set_ylabel(fam, fontsize=9)
    fig2.suptitle(
        "Secondary: top-3 rank frequency ($V_m$-by-HP-value, subset=all, Rashomon)",
        fontsize=11,
    )
    fig2.tight_layout()
    p2 = fig_dir / "hp_rank_stability_rashomon_grid.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    plt.show()

    for _, row in groups.iterrows():
        ds, fam = row["dataset"], row["family"]
        grp = vm[(vm["dataset"] == ds) & (vm["family"] == fam)]
        top = grp.nlargest(top_hp, "mean_importance").copy()
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(top))))
        y = np.arange(len(top))
        ax.barh(y, top["rank_freq_top3"], color="slategray")
        ax.set_yticks(y)
        ax.set_yticklabels(top["hp_name"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Frequency in top-3 within seed")
        ax.set_title(f"{ds} — {fam}: rank stability (subset=all)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"hp_rank_stability_{ds}_{fam}.pdf", bbox_inches="tight")
        plt.close(fig)

    print("Saved grids:", p1.name, p2.name)


def plot_vm_vs_brier_grids(
    perf_df: pd.DataFrame,
    *,
    perf_col: str,
    fig_dir: Path,
    pool_types: Optional[Sequence[str]] = None,
    family_order: Sequence[str] = FAMILY_ORDER_DEFAULT,
    dataset_order: Sequence[str] = DATASET_ORDER_DEFAULT,
) -> None:
    """One grid per pool type: scatter validation Brier vs $V_m$ (all families × datasets)."""
    pts = list(pool_types) if pool_types is not None else list(perf_df["pool_type"].dropna().unique())
    for pt in pts:
        sub = perf_df[perf_df["pool_type"] == pt]
        if sub.empty:
            continue
        n_r, n_c = len(family_order), len(dataset_order)
        fig, axes = plt.subplots(n_r, n_c, figsize=(2.8 * n_c, 2.4 * n_r), squeeze=False)
        for i, fam in enumerate(family_order):
            for j, ds in enumerate(dataset_order):
                ax = axes[i][j]
                g = sub[(sub["dataset"] == ds) & (sub["family"] == fam)]
                g = g[[perf_col, "V_m"]].dropna()
                if len(g) < 3:
                    ax.set_axis_off()
                    continue
                ax.scatter(g[perf_col], g["V_m"], alpha=0.65, s=12)
                ax.tick_params(axis="both", labelsize=6)
                if i == 0:
                    ax.set_title(ds, fontsize=8)
                if j == 0:
                    ax.set_ylabel(fam, fontsize=8)
        safe_pt = str(pt).replace(" ", "_")
        fig.suptitle(f"$V_m$ vs validation Brier ({pt})", fontsize=10)
        fig.tight_layout()
        outp = fig_dir / f"hp_vm_vs_performance_grid_{safe_pt}.pdf"
        fig.savefig(outp, bbox_inches="tight")
        plt.show()

        # Per-(ds,fam) PDFs for deep dives (no extra show)
        for i, fam in enumerate(family_order):
            for j, ds in enumerate(dataset_order):
                g = sub[(sub["dataset"] == ds) & (sub["family"] == fam)]
                g = g[[perf_col, "V_m"]].dropna()
                if len(g) < 3:
                    continue
                fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
                ax2.scatter(g[perf_col], g["V_m"], alpha=0.75)
                ax2.set_xlabel("validation Brier")
                ax2.set_ylabel("$V_m$")
                ax2.set_title(f"{ds} — {fam} — {pt}")
                fig2.tight_layout()
                fig2.savefig(
                    fig_dir / f"hp_vm_vs_performance_{ds}_{fam}_{safe_pt}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig2)

    print("Saved hp_vm_vs_performance_grid_*.pdf (+ per-panel PDFs).")


def plot_meta_pdp_tiles(
    *,
    table_dir: Path,
    fig_dir: Path,
    max_panels: int = 6,
) -> None:
    """Multi-panel PDP-style curves from exported ``hp_meta_effect_1d_*.csv`` (Rashomon, $V_m$)."""
    meta_summary_path = table_dir / "hp_meta_model_summary.csv"
    if not meta_summary_path.is_file():
        print("plot_meta_pdp_tiles: run meta-model stage first.")
        return
    ms = pd.read_csv(meta_summary_path)
    pool_ok = ms["pool_type"] == POOL_TYPE_RASHOMON if "pool_type" in ms.columns else True
    tgt_ok = ms["target_vm"] == "V_m" if "target_vm" in ms.columns else True
    ms_r = ms[pool_ok & tgt_ok]
    if ms_r.empty:
        return

    entries: List[Tuple[Path, str, str, str]] = []
    for _, row in ms_r.iterrows():
        ds, fam = row["dataset"], row["family"]
        pt = str(row["pool_type"]).replace(" ", "_")
        for p in sorted(glob.glob(str(table_dir / f"hp_meta_effect_1d_{ds}_{fam}_{pt}_*.csv"))):
            entries.append((Path(p), str(ds), str(fam), pt))
            if len(entries) >= max_panels:
                break
        if len(entries) >= max_panels:
            break

    if not entries:
        print("plot_meta_pdp_tiles: no effect CSVs found.")
        return

    n = len(entries)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)
    for idx, (fp, ds, fam, pt) in enumerate(entries):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        edf = pd.read_csv(fp)
        if edf.empty:
            ax.set_axis_off()
            continue
        feat = edf["feature"].iloc[0]
        if pd.api.types.is_numeric_dtype(edf["grid_value"]):
            ax.plot(edf["grid_value"], edf["predicted_V_m"], marker="o", ms=3)
        else:
            ax.plot(range(len(edf)), edf["predicted_V_m"], marker="o", ms=3)
            ax.set_xticks(range(len(edf)))
            ax.set_xticklabels(edf["grid_value"].astype(str), rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("mean pred. $V_m$", fontsize=8)
        ax.set_xlabel(str(feat), fontsize=8)
        ax.set_title(f"{ds} / {fam}", fontsize=8)
    for k in range(len(entries), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_axis_off()
    fig.suptitle("PDP-style summaries (first effect curves, Rashomon $V_m$)", fontsize=11)
    fig.tight_layout()
    out = fig_dir / "hp_meta_pdp_tiles_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.show()

    for fp, ds, fam, pt in entries:
        edf = pd.read_csv(fp)
        if edf.empty:
            continue
        feat = edf["feature"].iloc[0]
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
        if pd.api.types.is_numeric_dtype(edf["grid_value"]):
            ax2.plot(edf["grid_value"], edf["predicted_V_m"], marker="o")
        else:
            ax2.plot(range(len(edf)), edf["predicted_V_m"], marker="o")
            ax2.set_xticks(range(len(edf)))
            ax2.set_xticklabels(edf["grid_value"].astype(str), rotation=45, ha="right")
        ax2.set_ylabel("PDP-style mean pred. $V_m$")
        ax2.set_xlabel(str(feat))
        ax2.set_title(f"{ds} — {fam} — {POOL_TYPE_RASHOMON}")
        fig2.tight_layout()
        safe_feat = str(feat).replace("/", "_").replace(" ", "_")
        fig2.savefig(fig_dir / f"hp_meta_pdp_{ds}_{fam}_{pt}_{safe_feat}.pdf", bbox_inches="tight")
        plt.close(fig2)
    print("Saved", out.name, f"and {len(entries)} hp_meta_pdp_*.pdf panels.")


def plot_meta_interaction_tiles(
    interaction_df: pd.DataFrame,
    meta_summary: pd.DataFrame,
    *,
    fig_dir: Path,
    hp_cols: Sequence[str],
    resolve_feat,
    max_panels: int = 6,
) -> None:
    """Up to ``max_panels`` pairwise mean-$V_m$ heatmaps on one figure (meta top features)."""
    panels: List[Tuple[pd.DataFrame, str, str, str, str, str]] = []
    for _, row in meta_summary.iterrows():
        if len(panels) >= max_panels:
            break
        ds, fam, pt = row["dataset"], row["family"], row["pool_type"]
        if pt != POOL_TYPE_RASHOMON:
            continue
        f1 = resolve_feat(row.get("top_feature_1"))
        f2 = resolve_feat(row.get("top_feature_2"))
        if f1 is None or f2 is None or f1 == f2:
            continue
        g = interaction_df[
            (interaction_df["dataset"] == ds)
            & (interaction_df["family"] == fam)
            & (interaction_df["pool_type"] == pt)
        ][[f1, f2, "V_m"]].dropna()
        if len(g) < 20:
            continue
        panels.append((g.copy(), str(ds), str(fam), str(pt), f1, f2))

    if not panels:
        print("plot_meta_interaction_tiles: no panels.")
        return

    n = len(panels)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)
    for idx, (g, ds, fam, pt, f1, f2) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        if pd.api.types.is_numeric_dtype(g[f1]) and pd.api.types.is_numeric_dtype(g[f2]):
            try:
                g2 = g.copy()
                g2["_b1"] = pd.qcut(g2[f1], q=5, duplicates="drop")
                g2["_b2"] = pd.qcut(g2[f2], q=5, duplicates="drop")
                pivot = g2.groupby(["_b1", "_b2"], observed=False)["V_m"].mean().unstack()
                im = ax.imshow(pivot.values, aspect="auto")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception:
                ax.set_axis_off()
                continue
        else:
            gt = g.copy()
            gt[f1] = gt[f1].astype(str)
            gt[f2] = gt[f2].astype(str)
            pivot = gt.groupby([f1, f2], observed=False)["V_m"].mean().unstack()
            pivot = pivot.iloc[:8, :8]
            im = ax.imshow(pivot.values, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{ds}/{fam}", fontsize=8)
        ax.set_xlabel(str(f2), fontsize=7)
        ax.set_ylabel(str(f1), fontsize=7)
    for k in range(len(panels), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_axis_off()
    fig.suptitle("Pairwise mean $V_m$ (meta top-2 features, Rashomon)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "hp_meta_interaction_tiles_overview.pdf", bbox_inches="tight")
    plt.show()

    # Save first panel also as standalone (appendix-style)
    g, ds, fam, pt, f1, f2 = panels[0]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    if pd.api.types.is_numeric_dtype(g[f1]) and pd.api.types.is_numeric_dtype(g[f2]):
        try:
            g2 = g.copy()
            g2["_b1"] = pd.qcut(g2[f1], q=6, duplicates="drop")
            g2["_b2"] = pd.qcut(g2[f2], q=6, duplicates="drop")
            pivot = g2.groupby(["_b1", "_b2"], observed=False)["V_m"].mean().unstack()
            im = ax2.imshow(pivot.values, aspect="auto")
            fig2.colorbar(im, ax=ax2)
        except Exception:
            gt = g.copy()
            gt[f1] = gt[f1].astype(str)
            gt[f2] = gt[f2].astype(str)
            pivot = gt.groupby([f1, f2], observed=False)["V_m"].mean().unstack().iloc[:10, :10]
            im = ax2.imshow(pivot.values, aspect="auto")
            fig2.colorbar(im, ax=ax2)
    else:
        gt = g.copy()
        gt[f1] = gt[f1].astype(str)
        gt[f2] = gt[f2].astype(str)
        pivot = gt.groupby([f1, f2], observed=False)["V_m"].mean().unstack().iloc[:10, :10]
        im = ax2.imshow(pivot.values, aspect="auto")
        fig2.colorbar(im, ax=ax2)
    ax2.set_title(f"{ds} — {fam} — {pt}")
    ax2.set_xlabel(f2)
    ax2.set_ylabel(f1)
    fig2.tight_layout()
    safe_pt = str(pt).replace(" ", "_")
    fig2.savefig(
        fig_dir / f"hp_meta_interaction_{ds}_{fam}_{safe_pt}_{f1.replace('/', '_')}_{f2.replace('/', '_')}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig2)
    print("Saved hp_meta_interaction_tiles_overview.pdf (+ one standalone).")


def plot_pool_compare_grid(
    plot_df_builder,
    *,
    pairs: Sequence[Tuple[str, str]],
    fig_dir: Path,
    top_hp: int,
    family_order: Sequence[str] = FAMILY_ORDER_DEFAULT,
    dataset_order: Sequence[str] = DATASET_ORDER_DEFAULT,
) -> None:
    """
    ``plot_df_builder(ds, fam)`` returns a concat meta-importance long frame for both pools, or None.
    """
    n_r, n_c = len(family_order), len(dataset_order)
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.6 * n_c, 2.8 * n_r), squeeze=False)
    for i, fam in enumerate(family_order):
        for j, ds in enumerate(dataset_order):
            ax = axes[i][j]
            plot_df = plot_df_builder(ds, fam)
            if plot_df is None or plot_df.empty:
                ax.set_axis_off()
                continue
            hp_order = (
                plot_df.groupby("hp_name")["mean_importance"]
                .max()
                .sort_values(ascending=False)
                .head(top_hp)
                .index.tolist()
            )
            sub = plot_df[plot_df["hp_name"].isin(hp_order)].copy()
            pool_types = list(sub["pool_type"].dropna().unique())
            x = np.arange(len(hp_order))
            width = 0.8 / max(len(pool_types), 1)
            for k, pt in enumerate(pool_types):
                s = sub[sub["pool_type"] == pt].set_index("hp_name").reindex(hp_order).reset_index()
                ax.bar(
                    x + (k - (len(pool_types) - 1) / 2) * width,
                    s["mean_importance"],
                    width=width,
                    yerr=s["std_importance"],
                    capsize=1.5,
                    label=str(pt)[:10],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(hp_order, rotation=45, ha="right", fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            if i == 0:
                ax.set_title(ds, fontsize=9)
            if j == 0:
                ax.set_ylabel(fam, fontsize=9)
    fig.suptitle("Rashomon vs full pool (grouped meta-model importance)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "hp_importance_compare_grid.pdf", bbox_inches="tight")
    plt.show()

    for ds, fam in pairs:
        plot_df = plot_df_builder(ds, fam)
        if plot_df is None or plot_df.empty:
            continue
        hp_order = (
            plot_df.groupby("hp_name")["mean_importance"]
            .max()
            .sort_values(ascending=False)
            .head(top_hp)
            .index.tolist()
        )
        sub = plot_df[plot_df["hp_name"].isin(hp_order)].copy()
        pool_types = list(sub["pool_type"].dropna().unique())
        x = np.arange(len(hp_order))
        width = 0.8 / max(len(pool_types), 1)
        fig2, ax2 = plt.subplots(figsize=(7, max(3, 0.45 * len(hp_order))))
        for k, pt in enumerate(pool_types):
            s = sub[sub["pool_type"] == pt].set_index("hp_name").reindex(hp_order).reset_index()
            ax2.bar(
                x + (k - (len(pool_types) - 1) / 2) * width,
                s["mean_importance"],
                width=width,
                yerr=s["std_importance"],
                capsize=2,
                label=str(pt),
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels(hp_order, rotation=45, ha="right")
        ax2.set_ylabel("Grouped importance")
        ax2.set_title(f"{ds} — {fam}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(fig_dir / f"hp_importance_compare_{ds}_{fam}.pdf", bbox_inches="tight")
        plt.close(fig2)

    print("Saved hp_importance_compare_grid.pdf + per-(dataset,family) PDFs.")


__all__ = [
    "plot_meta_importance_and_stability_grids",
    "plot_meta_interaction_tiles",
    "plot_meta_pdp_tiles",
    "plot_pool_compare_grid",
    "plot_vm_vs_brier_grids",
]
