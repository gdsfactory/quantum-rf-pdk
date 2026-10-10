"""Tests for the QPDK COMSOL wrappers' public surface.

The generic COMSOL code lives in :mod:`gplugins.comsol` and is tested there.
These tests cover what QPDK adds or keeps: the public names, the re-exports of
the gplugins helpers, and the QPDK defaults the builder wrappers pass through.
Nothing needs MPh or a COMSOL licence: the gplugins builders are replaced.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from gplugins.comsol import metal as gplugins_metal, sheet as gplugins_sheet

from qpdk import simulation
from qpdk.simulation import build_comsol_metal_model, build_comsol_sheet_model
from qpdk.simulation.comsol import (
    layout as comsol_layout,
    metal as comsol_metal,
    sheet as comsol_sheet,
)
from qpdk.simulation.comsol.layout import ComsolBoundingBox, ComsolLayout
from qpdk.tech import material_properties


def _empty_layout() -> ComsolLayout:
    """A layout with a bounding box but no metal.

    Returns:
        The empty layout.
    """
    return ComsolLayout(
        polygons=(),
        feed_ports=(),
        bbox=ComsolBoundingBox(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0),
    )


def test_historical_cpw_alias_is_gone():
    """The builder is named for what it does; the old CPW alias was removed."""
    assert not hasattr(simulation, "build_comsol_cpw_model")
    assert not hasattr(comsol_metal, "build_comsol_cpw_model")
    assert build_comsol_metal_model.__name__ == "build_comsol_metal_model"


def test_comsol_class_is_exposed_lazily():
    """The model class is public, and importing it stays the caller's choice."""
    assert "COMSOL" in simulation.__all__
    assert simulation._LAZY_IMPORTS["COMSOL"] == (
        "qpdk.simulation.comsol.model",
        "COMSOL",
    )


@pytest.mark.parametrize(
    "name",
    [
        "ComsolBoundingBox",
        "ComsolFeedPort",
        "ComsolLayout",
        "ComsolPolygon",
        "prepare_comsol_layout",
    ],
)
def test_comsol_layout_public_exports(name: str):
    """The lazy package exports resolve to the geometry module's objects."""
    assert getattr(simulation, name) is getattr(comsol_layout, name)


@pytest.mark.parametrize(
    "name", ["ComsolBoundingBox", "ComsolFeedPort", "ComsolLayout", "ComsolPolygon"]
)
def test_layout_types_are_the_gplugins_types(name: str):
    """QPDK layouts are gplugins layouts, so the gplugins builders accept them."""
    gplugins_layout = importlib.import_module("gplugins.comsol.layout")
    assert getattr(comsol_layout, name) is getattr(gplugins_layout, name)


@pytest.mark.parametrize(
    "module",
    ["rf", "capacitance", "mesh", "plotting", "results"],
)
def test_helper_modules_reexport_gplugins(module: str):
    """The study, mesh, plotting, and result modules re-export gplugins as-is."""
    qpdk_module = importlib.import_module(f"qpdk.simulation.comsol.{module}")
    gplugins_module = importlib.import_module(f"gplugins.comsol.{module}")
    assert qpdk_module.__all__
    for name in qpdk_module.__all__:
        assert getattr(qpdk_module, name) is getattr(gplugins_module, name), name


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("add_capacitance_study", "capacitance"),
        ("add_cpw_rf_study", "rf"),
        ("pin_absolute_edge_mesh_sizes", "mesh"),
        ("pin_absolute_mesh_sizes", "mesh"),
        ("refine_metal_plane_mesh", "mesh"),
    ],
)
def test_package_exports_the_gplugins_helpers(name: str, module: str):
    """The lazy package exports resolve to the gplugins helpers."""
    gplugins_module = importlib.import_module(f"gplugins.comsol.{module}")
    assert getattr(simulation, name) is getattr(gplugins_module, name)


def test_permittivities_come_from_the_tech():
    """The QPDK sheet defaults are the technology values, not the gplugins ones."""
    assert (
        material_properties["Si"]["relative_permittivity"]
        == comsol_sheet.SILICON_RELATIVE_PERMITTIVITY
    )
    assert (
        material_properties["vacuum"]["relative_permittivity"]
        == comsol_sheet.AIR_RELATIVE_PERMITTIVITY
    )


def test_sheet_builder_defaults_to_the_tech_permittivities(
    monkeypatch: pytest.MonkeyPatch,
):
    """The sheet wrapper passes the QPDK permittivities to the gplugins builder."""
    calls: list[tuple[Any, ...]] = []

    def fake_builder(client: Any, layout: Any, name: str, **kwargs: Any) -> str:
        calls.append((client, layout, name, kwargs))
        return "model"

    monkeypatch.setattr(gplugins_sheet, "build_comsol_sheet_model", fake_builder)
    client = MagicMock()
    layout = _empty_layout()

    assert build_comsol_sheet_model(client, layout, "sheet") == "model"
    assert calls == [
        (
            client,
            layout,
            "sheet",
            {
                "substrate_thickness_um": 200.0,
                "air_height_um": 200.0,
                "lateral_margin_um": 0.0,
                "silicon_relative_permittivity": (
                    comsol_sheet.SILICON_RELATIVE_PERMITTIVITY
                ),
                "air_relative_permittivity": comsol_sheet.AIR_RELATIVE_PERMITTIVITY,
            },
        )
    ]


def test_metal_builder_defaults_to_the_qpdk_name(monkeypatch: pytest.MonkeyPatch):
    """The metal wrapper names the COMSOL model after QPDK."""
    calls: list[tuple[Any, ...]] = []

    def fake_builder(client: Any, layout: Any, **kwargs: Any) -> str:
        calls.append((client, layout, kwargs))
        return "model"

    monkeypatch.setattr(gplugins_metal, "build_comsol_metal_model", fake_builder)
    client = MagicMock()
    layout = _empty_layout()

    assert build_comsol_metal_model(client, layout, metal_thickness_um=0.35) == "model"
    assert calls == [
        (client, layout, {"metal_thickness_um": 0.35, "name": "QPDK metal"})
    ]


def test_modules_import_without_gplugins_comsol():
    """Without the extra the modules import and using a helper names the fix."""
    code = """
import sys
sys.modules["gplugins.comsol"] = None
import qpdk.simulation.comsol
from qpdk.simulation.comsol import (
    capacitance, layout, mesh, metal, plotting, results, rf, sheet,
)
assert sheet.SILICON_RELATIVE_PERMITTIVITY > 1
try:
    results.result_file
except ImportError as error:
    assert "uv sync --extra comsol" in str(error), error
else:
    raise AssertionError("expected ImportError")
from qpdk.simulation.comsol.model import COMSOL
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode != 0, result.stderr
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith("ImportError: qpdk.simulation.comsol.model"), (
        result.stderr
    )
    assert "uv sync --extra comsol" in last_line
