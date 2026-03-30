"""Design Rule Checks (DRC) for the Quantum PDK.

This module provides a set of DRC rules and functions to validate
GDS layouts against the QPDK fabrication constraints. It wraps the
`gplugins.klayout.drc` checks and adds PDK-specific rules such as
TSV/Indium-bump overlap prevention.

Typical usage::

    import gdsfactory as gf
    from qpdk.drc import run_drc

    component = gf.get_component("transmon")
    results = run_drc(component)
    results.print_summary()

Rules can also be executed from the command line via *just*::

    just drc transmon
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import klayout.db as kdb
from gplugins.klayout.drc.check_exclusion import check_exclusion
from gplugins.klayout.drc.check_space import check_space
from gplugins.klayout.drc.check_width import check_width

from qpdk.logger import logger
from qpdk.tech import LAYER

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gdsfactory.typings import Layer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DRCViolation:
    """A single DRC rule violation."""

    rule_name: str
    description: str
    value: int
    """Non-zero value indicates a violation.
    Semantics depend on the check type (edge count, area, etc.)."""


@dataclass
class DRCResults:
    """Collection of DRC violations found during a check run."""

    component_name: str
    violations: list[DRCViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True if **no** violations were recorded."""
        return len(self.violations) == 0

    @property
    def num_violations(self) -> int:
        """Return the total number of violations."""
        return len(self.violations)

    def print_summary(self) -> None:
        """Log a human-readable DRC summary."""
        if self.passed:
            logger.info(
                "DRC PASSED for '{}' – no violations found.", self.component_name
            )
            return

        logger.warning(
            "DRC FAILED for '{}' – {} violation(s):",
            self.component_name,
            self.num_violations,
        )
        for v in self.violations:
            logger.warning(
                "  • {} – {} (value={})", v.rule_name, v.description, v.value
            )


# ---------------------------------------------------------------------------
# Design‑rule constants (all values in µm)
# ---------------------------------------------------------------------------

#: Minimum feature width for M1_ETCH (etch openings).
M1_ETCH_MIN_WIDTH: float = 1.0
#: Maximum feature width for M1_ETCH.
M1_ETCH_MAX_WIDTH: float = 5000.0
#: Minimum spacing between M1_ETCH features.
M1_ETCH_MIN_SPACE: float = 1.0

#: Minimum feature width for M2_ETCH.
M2_ETCH_MIN_WIDTH: float = 1.0
#: Maximum feature width for M2_ETCH.
M2_ETCH_MAX_WIDTH: float = 5000.0
#: Minimum spacing between M2_ETCH features.
M2_ETCH_MIN_SPACE: float = 1.0

#: Minimum exclusion distance between TSV and IND (must not overlap).
TSV_IND_MIN_EXCLUSION: float = 0.001

#: Minimum feature width for airbridge draw layer.
AB_DRAW_MIN_WIDTH: float = 2.0
#: Minimum feature width for airbridge via/pad layer.
AB_VIA_MIN_WIDTH: float = 2.0

#: Minimum feature size for Josephson junction area layer.
JJ_AREA_MIN_WIDTH: float = 0.05

#: KLayout database unit multiplier (1 nm per database unit → 1e3).
DRC_DBU: float = 1e3


# ---------------------------------------------------------------------------
# Low-level check helpers
# ---------------------------------------------------------------------------


def _check_min_width(
    component: Component,
    layer: Layer,
    min_width: float,
    *,
    dbu: float = DRC_DBU,
) -> int:
    """Return the number of edges violating *min_width* on *layer*."""
    return check_width(component, layer=layer, min_width=min_width, dbu=dbu)


def _check_min_space(
    component: Component,
    layer: Layer,
    min_space: float,
    *,
    dbu: float = DRC_DBU,
) -> int:
    """Return the area (in database units²) violating *min_space* on *layer*."""
    return check_space(component, layer=layer, min_space=min_space, dbu=dbu)


def _check_max_width(
    component: Component,
    layer: Layer,
    max_width: float,
    *,
    dbu: float = DRC_DBU,
) -> int:
    """Return the number of polygons whose width exceeds *max_width*.

    The check erodes (shrinks) every polygon by ``max_width / 2``.  Any
    polygon that still has area remaining is wider than *max_width* in at
    least one direction.
    """
    layout = component.kcl
    cell = component.kdb_cell
    layer_index = layout.layer(layer[0], layer[1])
    region = kdb.Region(cell.begin_shapes_rec(layer_index))
    if region.is_empty():
        return 0
    eroded = region.sized(-int(max_width * dbu / 2))
    return eroded.count()


def _check_no_overlap(
    component: Component,
    layer1: Layer,
    layer2: Layer,
) -> int:
    """Return the overlapping area (in database units²) between two layers.

    A non-zero value means shapes on *layer1* and *layer2* overlap, which
    is treated as a violation (e.g. TSV vs. Indium bump).
    """
    layout = component.kcl
    cell = component.kdb_cell
    idx1 = layout.layer(layer1[0], layer1[1])
    idx2 = layout.layer(layer2[0], layer2[1])
    region1 = kdb.Region(cell.begin_shapes_rec(idx1))
    region2 = kdb.Region(cell.begin_shapes_rec(idx2))
    if region1.is_empty() or region2.is_empty():
        return 0
    overlap = region1 & region2
    return overlap.area()


def _check_exclusion(
    component: Component,
    layer1: Layer,
    layer2: Layer,
    min_space: float,
    *,
    dbu: float = DRC_DBU,
) -> int:
    """Return the area violating min exclusion between two layers."""
    return check_exclusion(
        component,
        layer1=layer1,
        layer2=layer2,
        min_space=min_space,
        dbu=dbu,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_drc(
    component: Component | str | Path,
    *,
    m1_etch_min_width: float = M1_ETCH_MIN_WIDTH,
    m1_etch_max_width: float = M1_ETCH_MAX_WIDTH,
    m1_etch_min_space: float = M1_ETCH_MIN_SPACE,
    m2_etch_min_width: float = M2_ETCH_MIN_WIDTH,
    m2_etch_max_width: float = M2_ETCH_MAX_WIDTH,
    m2_etch_min_space: float = M2_ETCH_MIN_SPACE,
    tsv_ind_min_exclusion: float = TSV_IND_MIN_EXCLUSION,
    ab_draw_min_width: float = AB_DRAW_MIN_WIDTH,
    ab_via_min_width: float = AB_VIA_MIN_WIDTH,
    jj_area_min_width: float = JJ_AREA_MIN_WIDTH,
    dbu: float = DRC_DBU,
) -> DRCResults:
    """Run a full suite of QPDK design-rule checks on *component*.

    Args:
        component: A :class:`~gdsfactory.Component`, a path to a GDS file,
            or a component name registered in the active PDK.
        m1_etch_min_width: Minimum width for M1_ETCH features (µm).
        m1_etch_max_width: Maximum width for M1_ETCH features (µm).
        m1_etch_min_space: Minimum spacing between M1_ETCH features (µm).
        m2_etch_min_width: Minimum width for M2_ETCH features (µm).
        m2_etch_max_width: Maximum width for M2_ETCH features (µm).
        m2_etch_min_space: Minimum spacing between M2_ETCH features (µm).
        tsv_ind_min_exclusion: Minimum exclusion distance between TSV and
            IND layers (µm).  Use a tiny positive value (e.g. 0.001) to
            simply forbid overlap.
        ab_draw_min_width: Minimum width for AB_DRAW (µm).
        ab_via_min_width: Minimum width for AB_VIA (µm).
        jj_area_min_width: Minimum width for JJ_AREA (µm).
        dbu: Database-unit multiplier for KLayout.

    Returns:
        A :class:`DRCResults` object describing any violations found.
    """
    if isinstance(component, str | Path):
        path = Path(str(component))
        if path.suffix == ".gds":
            component = gf.import_gds(path)
        else:
            component = gf.get_component(str(component))

    name = getattr(component, "name", str(component))
    results = DRCResults(component_name=name)

    def _add_if_nonzero(value: int, rule: str, desc: str) -> None:
        if value:
            results.violations.append(
                DRCViolation(rule_name=rule, description=desc, value=value)
            )

    # ── M1_ETCH checks ────────────────────────────────────────────────
    _add_if_nonzero(
        _check_min_width(component, LAYER.M1_ETCH, m1_etch_min_width, dbu=dbu),
        "M1_ETCH.min_width",
        f"Feature(s) narrower than {m1_etch_min_width} µm",
    )
    _add_if_nonzero(
        _check_max_width(component, LAYER.M1_ETCH, m1_etch_max_width, dbu=dbu),
        "M1_ETCH.max_width",
        f"Feature(s) wider than {m1_etch_max_width} µm",
    )
    _add_if_nonzero(
        _check_min_space(component, LAYER.M1_ETCH, m1_etch_min_space, dbu=dbu),
        "M1_ETCH.min_space",
        f"Spacing less than {m1_etch_min_space} µm",
    )

    # ── M2_ETCH checks ────────────────────────────────────────────────
    _add_if_nonzero(
        _check_min_width(component, LAYER.M2_ETCH, m2_etch_min_width, dbu=dbu),
        "M2_ETCH.min_width",
        f"Feature(s) narrower than {m2_etch_min_width} µm",
    )
    _add_if_nonzero(
        _check_max_width(component, LAYER.M2_ETCH, m2_etch_max_width, dbu=dbu),
        "M2_ETCH.max_width",
        f"Feature(s) wider than {m2_etch_max_width} µm",
    )
    _add_if_nonzero(
        _check_min_space(component, LAYER.M2_ETCH, m2_etch_min_space, dbu=dbu),
        "M2_ETCH.min_space",
        f"Spacing less than {m2_etch_min_space} µm",
    )

    # ── TSV / Indium-bump overlap ─────────────────────────────────────
    _add_if_nonzero(
        _check_no_overlap(component, LAYER.TSV, LAYER.IND),
        "TSV_IND.no_overlap",
        "TSV and Indium bump shapes overlap",
    )
    _add_if_nonzero(
        _check_exclusion(
            component, LAYER.TSV, LAYER.IND, tsv_ind_min_exclusion, dbu=dbu
        ),
        "TSV_IND.min_exclusion",
        f"TSV–IND separation less than {tsv_ind_min_exclusion} µm",
    )

    # ── Airbridge checks ──────────────────────────────────────────────
    _add_if_nonzero(
        _check_min_width(component, LAYER.AB_DRAW, ab_draw_min_width, dbu=dbu),
        "AB_DRAW.min_width",
        f"Airbridge metal narrower than {ab_draw_min_width} µm",
    )
    _add_if_nonzero(
        _check_min_width(component, LAYER.AB_VIA, ab_via_min_width, dbu=dbu),
        "AB_VIA.min_width",
        f"Airbridge via/pad narrower than {ab_via_min_width} µm",
    )

    # ── Josephson junction checks ─────────────────────────────────────
    _add_if_nonzero(
        _check_min_width(component, LAYER.JJ_AREA, jj_area_min_width, dbu=dbu),
        "JJ_AREA.min_width",
        f"JJ area narrower than {jj_area_min_width} µm",
    )

    return results


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``just drc <component>`` or ``python -m qpdk.drc``."""
    import sys

    from qpdk import PDK

    PDK.activate()

    if len(sys.argv) < 2:
        logger.error("Usage: python -m qpdk.drc <component_name_or_gds_path>")
        sys.exit(1)

    target = sys.argv[1]
    logger.info("Running DRC on '{}'…", target)

    results = run_drc(target)
    results.print_summary()

    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
