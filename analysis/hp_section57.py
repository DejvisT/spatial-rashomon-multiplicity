"""
Thesis Section 5.7 — hyperparameter / family multiplicity pipeline (notebook 06).

Stages (mirrors the notebook narrative):
1. **Canonical model-level table** — one tidy ``df_models`` (and per-seed wide tables).
2. **Family decomposition** — main structural analysis (``metrics_long`` → aggregates).
3. **Secondary HP decomposition** — unique HP values on predictions ``P`` (appendix-style).
4. **Meta-model** — descriptive conditional RF on ``V_m`` (``hp_meta_model``).
5. **Exports** — CSV splits and aggregates used by plots / thesis export scripts.

All downstream HP analyses should read from the **canonical** ``df_models`` built here
(unified ``validation_brier`` when possible).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.experiment_runner import _get_run_dirs
from analysis.hp_analysis import POOL_TYPE_FULL_POOL, POOL_TYPE_RASHOMON, aggregate_hp_importance
from analysis.hp_meta_model import resolve_outer_seed_column, unify_validation_brier
from analysis.hp_multiplicity_pipeline import (
    aggregate_decomposition_hp,
    aggregate_family_importance_long,
    aggregate_hotspot_delta,
    hotspot_delta_decomp,
    run_dataset_all_seeds,
)


@dataclass
class HPAnalysisConfig:
    """Single place for notebook-tunable settings (Section 5.7)."""

    results_dir: Path
    table_dir: Path
    fig_dir: Path
    datasets: Tuple[str, ...] = ("compas", "german", "adult")
    pool_types: Tuple[str, ...] = (POOL_TYPE_RASHOMON, POOL_TYPE_FULL_POOL)
    rashomon_k_each: int = 25
    min_hh_obs: int = 5
    top_hp: int = 10
    illustrate_seed: Optional[int] = None
    meta_stability_reps: int = 12
    meta_random_state: int = 0
    # Prefer training outer seed for LOSO / bootstrap when present on model rows.
    seed_column_preference: Tuple[str, ...] = ("outer_seed", "seed")


def run_all_pools_for_dataset(ds: str, config: HPAnalysisConfig) -> Dict[str, pd.DataFrame]:
    """One dataset, all ``pool_types``: concatenate per-seed pipeline outputs."""
    parts: List[Dict[str, pd.DataFrame]] = []
    for pt in config.pool_types:
        ds_dir = config.results_dir / ds
        if not ds_dir.is_dir() or not _get_run_dirs(ds_dir):
            continue
        out = run_dataset_all_seeds(
            ds_dir,
            ds,
            pool_type=pt,
            rashomon_k_each=config.rashomon_k_each,
            min_hh_obs=config.min_hh_obs,
        )
        parts.append(out)
    if not parts:
        return {
            "models": pd.DataFrame(),
            "metrics_long": pd.DataFrame(),
            "decomp_hp_wide": pd.DataFrame(),
            "vm_hp_wide": pd.DataFrame(),
        }
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}


def collect_multiseed_hp_tables(config: HPAnalysisConfig) -> Dict[str, pd.DataFrame]:
    """
    Stage 1 — build the canonical multi-seed tables used everywhere else.

    Returns keys: ``models``, ``metrics_long``, ``decomp_hp_wide``, ``vm_hp_wide``.
    """
    all_models: List[pd.DataFrame] = []
    all_long: List[pd.DataFrame] = []
    all_decomp: List[pd.DataFrame] = []
    all_vm: List[pd.DataFrame] = []

    for ds in config.datasets:
        out = run_all_pools_for_dataset(ds, config)
        if not out["models"].empty:
            all_models.append(out["models"])
        if not out["metrics_long"].empty:
            all_long.append(out["metrics_long"])
        if not out["decomp_hp_wide"].empty:
            all_decomp.append(out["decomp_hp_wide"])
        if not out["vm_hp_wide"].empty:
            all_vm.append(out["vm_hp_wide"])

    return {
        "models": pd.concat(all_models, ignore_index=True) if all_models else pd.DataFrame(),
        "metrics_long": pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame(),
        "decomp_hp_wide": pd.concat(all_decomp, ignore_index=True) if all_decomp else pd.DataFrame(),
        "vm_hp_wide": pd.concat(all_vm, ignore_index=True) if all_vm else pd.DataFrame(),
    }


def export_per_dataset_wide_tables(
    tables: Dict[str, pd.DataFrame],
    config: HPAnalysisConfig,
) -> None:
    """Write per-dataset slices of the raw multi-seed tables (debug / partial re-runs)."""
    td = Path(config.table_dir)
    td.mkdir(parents=True, exist_ok=True)
    df_models = tables["models"]
    df_metrics_long = tables["metrics_long"]
    df_decomp_hp = tables["decomp_hp_wide"]
    df_vm_hp = tables["vm_hp_wide"]

    for ds in config.datasets:
        for name, df in (
            ("models", df_models),
            ("metrics_long", df_metrics_long),
            ("decomp_hp_per_seed", df_decomp_hp),
            ("vm_hp_per_seed", df_vm_hp),
        ):
            sub = df[df["dataset"] == ds] if not df.empty and "dataset" in df.columns else df
            if sub is not None and not sub.empty:
                sub.to_csv(td / f"{name}_{ds}.csv", index=False)

        if not df_vm_hp.empty:
            legacy = df_vm_hp[(df_vm_hp["dataset"] == ds) & (df_vm_hp["subset"] == "all")]
            if not legacy.empty:
                legacy.to_csv(td / f"hp_importance_per_seed_{ds}.csv", index=False)


def ensure_canonical_model_table(df_models: pd.DataFrame) -> pd.DataFrame:
    """Return ``df_models`` with ``validation_brier`` unified for all HP / meta-model consumers."""
    if df_models.empty:
        return df_models
    return unify_validation_brier(df_models.copy())


def aggregate_hp_seed_tables(
    df_metrics_long: pd.DataFrame,
    df_decomp_hp: pd.DataFrame,
    df_vm_hp: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Stages 2–3 aggregates across seeds.

    - ``fam_agg``: family decomposition (main).
    - ``decomp_agg`` / ``vm_agg``: secondary robustness paths.
    """
    df_fam_agg = aggregate_family_importance_long(df_metrics_long)
    df_decomp_agg = aggregate_decomposition_hp(df_decomp_hp) if not df_decomp_hp.empty else pd.DataFrame()
    df_vm_agg = aggregate_hp_importance(df_vm_hp) if not df_vm_hp.empty else pd.DataFrame()
    delta_seed = hotspot_delta_decomp(df_decomp_hp) if not df_decomp_hp.empty else pd.DataFrame()
    df_delta_agg = aggregate_hotspot_delta(delta_seed) if not delta_seed.empty else pd.DataFrame()
    return {
        "fam_agg": df_fam_agg,
        "decomp_agg": df_decomp_agg,
        "vm_agg": df_vm_agg,
        "delta_seed": delta_seed,
        "delta_agg": df_delta_agg,
    }


def export_seed_aggregates_and_hotspot_delta(
    aggs: Dict[str, Any],
    config: HPAnalysisConfig,
) -> None:
    """Write compact family + secondary HP aggregates and pooled hotspot delta CSVs."""
    td = Path(config.table_dir)
    td.mkdir(parents=True, exist_ok=True)
    df_fam_agg = aggs["fam_agg"]
    df_decomp_agg = aggs["decomp_agg"]
    df_vm_agg = aggs["vm_agg"]
    df_delta_agg = aggs["delta_agg"]

    for ds in config.datasets:
        if not df_fam_agg.empty:
            sub_f = df_fam_agg[df_fam_agg["dataset"] == ds]
            if not sub_f.empty:
                sub_f.to_csv(td / f"family_importance_agg_{ds}.csv", index=False)

    if not df_decomp_agg.empty:
        df_decomp_agg.to_csv(
            td / "decomp_hp_unique_value_secondary_agg.csv",
            index=False,
        )

    if not df_vm_agg.empty:
        df_vm_agg.to_csv(
            td / "hp_vm_unique_value_secondary_agg.csv",
            index=False,
        )


    if not df_delta_agg.empty:
        df_delta_agg.to_csv(td / "decomp_hp_hotspot_delta_agg.csv", index=False)


def build_canonical_model_analysis_exports(df_models: pd.DataFrame, table_dir: Path) -> None:
    """
    Stage 1b — thesis-friendly canonical model-level CSVs (wide table + compact summary).

    Uses the same column detection logic in one place (replaces duplicated notebook cells).
    """
    td = Path(table_dir)
    td.mkdir(parents=True, exist_ok=True)
    if df_models.empty:
        print("No model-level rows for canonical export.")
        return

    df = ensure_canonical_model_table(df_models)

    base_cols = [
        "dataset",
        "seed",
        "outer_seed",
        "family",
        "pool_type",
        "V_m",
        "V_m_HH",
        "V_m_nonHH",
        "val_brier",
        "validation_brier",
        "brier_val",
        "brier_score",
        "K_actual",
    ]
    model_cols_base = [c for c in base_cols if c in df.columns]
    hp_cols = sorted([c for c in df.columns if c.startswith("hp_")])
    model_analysis = df[model_cols_base + hp_cols].copy()

    ordered_front = [
        c
        for c in [
            "dataset",
            "seed",
            "outer_seed",
            "family",
            "pool_type",
            "validation_brier",
            "V_m",
            "V_m_HH",
            "V_m_nonHH",
            "K_actual",
        ]
        if c in model_analysis.columns
    ]
    remaining = [c for c in model_analysis.columns if c not in ordered_front]
    model_analysis = model_analysis[ordered_front + remaining]

    model_analysis.to_csv(td / "hp_model_level_analysis_table.csv", index=False)

    summary_group_cols = [c for c in ["dataset", "family", "pool_type"] if c in model_analysis.columns]
    if summary_group_cols:
        agg_dict: Dict[str, Any] = {"V_m": ["mean", "std", "count"]}
        if "validation_brier" in model_analysis.columns:
            agg_dict["validation_brier"] = ["mean", "std"]
        model_summary = model_analysis.groupby(summary_group_cols).agg(agg_dict).reset_index()
        model_summary.columns = [
            "_".join([str(x) for x in c if str(x) != ""]).strip("_") if isinstance(c, tuple) else c
            for c in model_summary.columns
        ]
        model_summary.to_csv(td / "hp_model_level_analysis_summary.csv", index=False)

    print("Saved hp_model_level_analysis_table.csv and hp_model_level_analysis_summary.csv")


__all__ = [
    "HPAnalysisConfig",
    "aggregate_hp_seed_tables",
    "build_canonical_model_analysis_exports",
    "collect_multiseed_hp_tables",
    "ensure_canonical_model_table",
    "export_per_dataset_wide_tables",
    "export_seed_aggregates_and_hotspot_delta",
    "run_all_pools_for_dataset",
]
