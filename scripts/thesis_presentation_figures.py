"""
Map thesis \\includegraphics{*.pdf} references to notebook outputs and copy into
overleaf_bundle/presentation_assets/fig/.

Sources (via thesis_layout.iter_derived_figure_pdfs): ``thesis_outputs/figures/``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Datasets used for sensitivity_*_curves_<ds>.pdf filenames in export_thesis_assets.py
_EXPORT_SENSITIVITY_DATASETS = ("compas", "german", "adult")

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])*\{([^}]+)\}")

_ALWAYS_EXPORT_FIGURES = {
    "decomp_hp_secondary_bar_rashomon_grid.pdf",
    "sensitivity_K.pdf",
    "sensitivity_kNN.pdf",
    "rules_support_purity_compas.pdf",
    "hh_stability_freq_compas.pdf"
}


def _iter_tex_sources(overleaf_bundle: Path) -> Iterator[Path]:
    thesis = overleaf_bundle / "thesis.tex"
    if thesis.is_file():
        yield thesis
    ch = overleaf_bundle / "chapters"
    if ch.is_dir():
        yield from sorted(ch.glob("*.tex"))


def collect_referenced_pdf_basenames(overleaf_bundle: Path | None = None) -> frozenset[str]:
    """Basenames of PDFs referenced via \\includegraphics in thesis.tex and chapters/*.tex."""
    bundle = overleaf_bundle if overleaf_bundle is not None else _REPO_ROOT / "overleaf_bundle"
    names: set[str] = set()
    for tex in _iter_tex_sources(bundle):
        text = tex.read_text(encoding="utf-8")
        for m in _INCLUDEGRAPHICS_RE.finditer(text):
            arg = m.group(1).strip()
            if not arg.lower().endswith(".pdf"):
                continue
            names.add(Path(arg.replace("\\", "/")).name)
    return frozenset(names)


def export_script_pdf_basenames() -> frozenset[str]:
    """PDFs written by export_thesis_assets.py into presentation_assets/fig/."""
    s = {
        "spatial_patterns_per_run.pdf",
        "hh_by_family.pdf",
        "hh_moran_per_run_compas.pdf"
    }
    for ds in _EXPORT_SENSITIVITY_DATASETS:
        s.add(f"sensitivity_K_variance_{ds}.pdf")
        s.add(f"sensitivity_K_conflict_{ds}.pdf")
        s.add(f"sensitivity_kNN_variance_{ds}.pdf")
        s.add(f"sensitivity_kNN_conflict_{ds}.pdf")
        s.add(f"family_vs_global_spatial_{ds}.pdf")
        s.add(f"calibration_delta_metrics_{ds}.pdf")
        s.add(f"hh_stability_freq_{ds}.pdf")
        s.add(f"rules_support_purity_{ds}.pdf")
    s.update(
        {
            "structural_exceptions_ground_truth.pdf",
            "structural_exceptions_pred_var_lisa.pdf",
            "structural_exceptions_fdr_sensitivity.pdf",
            "structural_exceptions_fdr_sensitivity_precision_recall.pdf",
            "structural_exceptions_variance_margin.pdf",
        }
    )
    return frozenset(s)


def build_figure_source_map() -> dict[str, Path]:
    """Map PDF basename -> source path under thesis_outputs/figures/."""
    from thesis_layout import iter_derived_figure_pdfs

    m: dict[str, Path] = {}
    for p in iter_derived_figure_pdfs():
        m[p.name] = p
    return m


def copy_notebook_figures(
    fig_dir: Path,
    *,
    copy_all: bool = False,
    prune_orphans: bool = False,
    overleaf_bundle: Path | None = None,
) -> None:
    """
    Copy PDFs from thesis_outputs/figures/ into fig_dir.

    By default only copies PDFs whose basenames appear in \\includegraphics in
    thesis.tex and overleaf_bundle/chapters/*.tex.

    :param copy_all: copy every derived PDF (previous exporter behaviour).
    :param prune_orphans: delete *.pdf in fig_dir not in (thesis refs ∪ export outputs).
    """
    import shutil

    fig_dir.mkdir(parents=True, exist_ok=True)
    by_name = build_figure_source_map()
    thesis_refs = collect_referenced_pdf_basenames(overleaf_bundle)
    export_pdfs = export_script_pdf_basenames()

    if copy_all:
        to_copy = sorted(by_name.items(), key=lambda x: x[0])
    else:
        # Do not copy over PDFs this export script generates into fig_dir (notebooks may have stale homonyms).
        from_trees = thesis_refs - export_pdfs
        allow_names = from_trees | _ALWAYS_EXPORT_FIGURES
        to_copy = [(n, by_name[n]) for n in sorted(allow_names) if n in by_name]
        missing_sources = allow_names - frozenset(by_name.keys())
        for n in sorted(missing_sources):
            print(
                f"Warning: thesis references {n} but no PDF found under "
                "thesis_outputs/figures/ (generate via notebooks or export script)."
            )

    copied = 0
    for name, src in to_copy:
        shutil.copy2(src, fig_dir / name)
        copied += 1
        print(f"  Copied {name}")
    mode = "all derived" if copy_all else "thesis-referenced"
    print(f"Copied {copied} {mode} notebook-sourced figures to {fig_dir}")

    if not copy_all:
        missing_export = [n for n in sorted(thesis_refs & export_pdfs) if not (fig_dir / n).is_file()]
        if missing_export:
            tail = ", ".join(missing_export[:6])
            extra = f", … (+{len(missing_export) - 6} more)" if len(missing_export) > 6 else ""
            print(
                f"Warning: {len(missing_export)} thesis-referenced PDF(s) are normally written by "
                f"export_thesis_assets.py but missing under {fig_dir}: {tail}{extra}. "
                "Run export_thesis_assets.py if needed."
            )

    if prune_orphans:
        keep = thesis_refs | export_pdfs | _ALWAYS_EXPORT_FIGURES
        removed = 0
        for f in sorted(fig_dir.glob("*.pdf")):
            if f.name not in keep:
                f.unlink()
                removed += 1
                print(f"  Pruned orphan {f.name}")
        if removed:
            print(f"Pruned {removed} PDF(s) not referenced by the thesis or export script.")
        else:
            print("Prune: no orphan PDFs to remove.")
