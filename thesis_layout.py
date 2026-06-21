"""
Central layout for thesis-facing *derived* outputs (CSV tables and PDF figures).

Raw training artifacts (P_test, meta, splits) stay under ``results/<dataset>/seed=<n>/``
from ``run_training_pipeline.py``. The experiment runner still writes
``summary_per_run.csv`` and ``per_point/`` next to those runs.

Each analysis notebook writes exports under ``thesis_outputs/tables/<nbXX>/`` and
``thesis_outputs/figures/<nbXX>/`` so filenames map cleanly to a notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

_ROOT = Path(__file__).resolve().parent

RAW_RESULTS = _ROOT / "results"

THESIS_ROOT = _ROOT / "thesis_outputs"
THESIS_TABLES_ROOT = THESIS_ROOT / "tables"
THESIS_FIGURES_ROOT = THESIS_ROOT / "figures"

DATASET_DISPLAY_NAMES = {
    "compas": "COMPAS",
    "german": "German Credit",
    "adult": "Adult",
}

DATASET_PLOT_COLORS = {
    "compas": "C0",
    "german": "C1",
    "adult": "C2",
}


def _dataset_key(dataset: str) -> str:
    key = str(dataset).strip().lower().replace(" ", "_")
    if key in DATASET_DISPLAY_NAMES:
        return key
    if key.startswith("german"):
        return "german"
    if key.startswith("compas"):
        return "compas"
    return key


def display_dataset_name(dataset: str) -> str:
    """Map internal dataset keys to thesis-facing labels (tables/figures only)."""
    key = _dataset_key(dataset)
    return DATASET_DISPLAY_NAMES.get(key, str(dataset))


def dataset_plot_color(dataset: str) -> str:
    return DATASET_PLOT_COLORS.get(_dataset_key(dataset), "gray")


def thesis_table_dir(notebook_id: str) -> Path:
    """Directory for CSV exports from one notebook (e.g. ``\"nb03\"``)."""
    p = THESIS_TABLES_ROOT / notebook_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def thesis_figure_dir(notebook_id: str) -> Path:
    """Directory for PDF figures from one notebook."""
    p = THESIS_FIGURES_ROOT / notebook_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def thesis_output_dirs(notebook_id: str) -> tuple[Path, Path]:
    """Return ``(table_dir, figure_dir)`` for a notebook id."""
    return thesis_table_dir(notebook_id), thesis_figure_dir(notebook_id)


def resolve_csv(filename: str, *notebook_ids: str) -> Optional[Path]:
    """
    Find a CSV by basename under ``thesis_outputs/tables/<nb>/filename`` for each
    ``notebook_ids`` in order.
    """
    for nb in notebook_ids:
        p = THESIS_TABLES_ROOT / nb / filename
        if p.is_file():
            return p
    return None


def iter_derived_figure_pdfs() -> Iterator[Path]:
    """Yield PDF paths under ``thesis_outputs/figures/``."""
    if not THESIS_FIGURES_ROOT.is_dir():
        return
    for sub in sorted(THESIS_FIGURES_ROOT.iterdir()):
        if sub.is_dir():
            yield from sorted(sub.glob("*.pdf"))
