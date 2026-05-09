"""
Combined multi-panel figures for notebook 06 (Section 5.7).

Keeps thesis-facing single-panel PDFs where ``results.tex`` references them, while
providing overview grids so the notebook shows fewer (richer) figures.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple

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
    save_individual_importance: bool = False,
    save_individual_rank_stability: bool = False,
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

    if save_individual_rank_stability:
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
    "plot_pool_compare_grid"
]
