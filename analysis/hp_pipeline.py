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


__all__ = [
    "HPAnalysisConfig",
    "aggregate_hp_seed_tables",
    "collect_multiseed_hp_tables",
    "export_seed_aggregates_and_hotspot_delta",
    "run_all_pools_for_dataset",
]
