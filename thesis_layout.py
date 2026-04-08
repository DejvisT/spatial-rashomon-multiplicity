"""
Central layout for thesis-facing *derived* outputs (CSV tables and PDF figures).

Raw training artifacts (P_test, meta, splits) stay under ``results/<dataset>/seed=<n>/``
from ``run_training_pipeline.py``. The experiment runner still writes
``summary_per_run.csv`` and ``per_point/`` next to those runs.

Each analysis notebook writes exports under ``thesis_outputs/tables/<nbXX>/`` and
``thesis_outputs/figures/<nbXX>/`` so filenames map cleanly to a notebook.

Legacy flat ``tables/`` and ``figures/`` at the repo root are still checked when
resolving inputs (so old runs keep working until you re-execute notebooks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

_ROOT = Path(__file__).resolve().parent

RAW_RESULTS = _ROOT / "results"

THESIS_ROOT = _ROOT / "thesis_outputs"
THESIS_TABLES_ROOT = THESIS_ROOT / "tables"
THESIS_FIGURES_ROOT = THESIS_ROOT / "figures"

LEGACY_TABLES = _ROOT / "tables"
LEGACY_FIGURES = _ROOT / "figures"


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
    Find a CSV by basename: try ``thesis_outputs/tables/<nb>/filename`` for each
    ``notebook_ids`` in order, then ``tables/filename`` (legacy).
    """
    for nb in notebook_ids:
        p = THESIS_TABLES_ROOT / nb / filename
        if p.is_file():
            return p
    leg = LEGACY_TABLES / filename
    if leg.is_file():
        return leg
    return None


def iter_derived_figure_pdfs() -> Iterator[Path]:
    """
    Yield PDF paths for bundling. Legacy ``figures/`` paths come first; ``thesis_outputs/figures/``
    last so duplicate basenames prefer the new layout when copied in sequence.
    """
    if LEGACY_FIGURES.is_dir():
        for sub in sorted(LEGACY_FIGURES.glob("nb*")):
            if sub.is_dir():
                yield from sorted(sub.glob("*.pdf"))
        yield from sorted(LEGACY_FIGURES.glob("*.pdf"))
    if THESIS_FIGURES_ROOT.is_dir():
        for sub in sorted(THESIS_FIGURES_ROOT.iterdir()):
            if sub.is_dir():
                yield from sorted(sub.glob("*.pdf"))
