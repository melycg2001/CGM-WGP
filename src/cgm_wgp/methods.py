"""Method registry for the CGM-WGP framework.

Single source of truth for which prediction methods are surfaced in
cross-fold aggregation, paper tables, and figures. Add a new method by
adding one entry to ``METHODS`` (plus the corresponding fit / predict
code and CLI plumbing — see TECHNICAL.md § Adding a Method).

This registry intentionally covers only the display-side wiring (file
basename, label, color, order). The fitting code, CLI flags, config
blocks, and ``_OUTPUT_MAP`` entries still live with each method's
implementation. The goal here is to remove the 5-place duplication of
``[("Mechanistic", "raw_mechanistic"), ...]`` tuple lists that the
codebase had pre-refactor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MethodSpec:
    """Display metadata for a single prediction method.

    Attributes
    ----------
    key
        Internal identifier (lowercase snake_case). Used as the dict key
        in METHODS. Not displayed.
    pred_file
        Basename (no .csv) of the predictions file written under
        ``predictions/`` in each per-fold run directory. Combined into
        ``combined/{pred_file}.csv`` after the suite finishes.
    display
        Human-readable label shown in tables, plots, and legends.
    color
        Matplotlib color hex used for this method across all plots.
    in_summary
        If True, the method is included in ``combined/cv_summary.csv``
        (overall + per-planting RMSE / PA aggregation).
    in_paper
        If True, the method appears in paper tables and the
        publication-quality figures (head-to-head, bump, 8-panel).
    """
    key: str
    pred_file: str
    display: str
    color: str
    in_summary: bool = True
    in_paper: bool = True


# Insertion order is preserved by dict. The order here is the order
# methods appear in tables and plot legends.
METHODS: Dict[str, MethodSpec] = {
    "mechanistic": MethodSpec(
        key="mechanistic",
        pred_file="raw_mechanistic",
        display="Mechanistic",
        color="#457B9D",
    ),
    "gblup": MethodSpec(
        key="gblup",
        pred_file="gblup",
        display="GBLUP",
        color="#2A9D8F",
    ),
    "rn_gblup": MethodSpec(
        key="rn_gblup",
        pred_file="jarquin",
        display="RN-GBLUP",
        color="#E9C46A",
    ),
    "joint": MethodSpec(
        key="joint",
        pred_file="joint",
        display="CGM-WGP",
        color="#E63946",
    ),
}


def in_summary() -> List[MethodSpec]:
    """Methods to include in per-run cv_summary.csv (overall + per-planting)."""
    return [s for s in METHODS.values() if s.in_summary]


def in_paper() -> List[MethodSpec]:
    """Methods to include in paper tables and figures."""
    return [s for s in METHODS.values() if s.in_paper]


def display_order() -> List[str]:
    """Display names in canonical order (for METHOD_ORDER lists)."""
    return [s.display for s in METHODS.values() if s.in_paper]


def color_map() -> Dict[str, str]:
    """{display_name: color} for use in plotting."""
    return {s.display: s.color for s in METHODS.values() if s.in_paper}


def pred_file_to_display() -> Dict[str, str]:
    """{pred_file: display} for resolving combined/*.csv → label."""
    return {s.pred_file: s.display for s in METHODS.values()}
