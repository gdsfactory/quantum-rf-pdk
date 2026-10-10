"""Tests for the QPDK COMSOL model class.

The class subclasses :class:`gplugins.comsol.model.COMSOL`, whose study and mesh
methods are tested in gplugins. These tests cover only what QPDK changes: the
defaults its builders pass through, and that the class stays importable and
reachable through the QPDK packages. Nothing here needs a COMSOL licence or a
running client: the base class comes from MPh alone, the Java handle is a mock,
and the gplugins builders are replaced.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from qpdk.simulation.comsol.layout import ComsolBoundingBox, ComsolLayout, ComsolPolygon
from qpdk.simulation.comsol.sheet import (
    AIR_RELATIVE_PERMITTIVITY,
    SILICON_RELATIVE_PERMITTIVITY,
)


def test_module_imports_without_mph():
    """Pytest can collect the module when the optional COMSOL extra is absent."""
    code = (
        "import sys; sys.modules['mph'] = None; "
        "from qpdk.simulation.comsol.model import COMSOL; "
        "assert COMSOL.__base__.__base__ is object; "
        "COMSOL(None, None)"
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode != 0
    assert "ImportError: Install gplugins[comsol]" in result.stderr


def test_dependency_import_failure_is_not_reported_as_missing_mph():
    """An installed MPh with a broken dependency keeps its original error."""
    code = """
import importlib
original = importlib.import_module
def broken(name, package=None):
    if name == "mph":
        raise ModuleNotFoundError("JPype failed to load", name="jpype")
    return original(name, package=package)
importlib.import_module = broken
import qpdk.simulation.comsol.model
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError: JPype failed to load" in result.stderr


def _layout() -> ComsolLayout:
    """A one-polygon layout to attach to a model.

    Returns:
        The layout.
    """
    return ComsolLayout(
        polygons=(ComsolPolygon(outline=((0, 0), (10, 0), (10, 10), (0, 10))),),
        feed_ports=(),
        bbox=ComsolBoundingBox(xmin=0, ymin=0, xmax=10, ymax=10),
    )


@pytest.fixture
def comsol() -> tuple[Any, Any, Any]:
    """MPh, the gplugins model module, and the QPDK one, skipping without MPh.

    Returns:
        ``(mph, gplugins_model, comsol_model)``.
    """
    mph = pytest.importorskip("mph", reason="the comsol extra is not installed")
    return (
        mph,
        importlib.import_module("gplugins.comsol.model"),
        importlib.import_module("qpdk.simulation.comsol.model"),
    )


def test_subclasses_the_gplugins_model(comsol: tuple[Any, Any, Any]):
    """The QPDK class is the gplugins class with QPDK defaults, still an MPh model."""
    mph, gplugins_model, comsol_model = comsol
    java = MagicMock()
    layout = _layout()
    model = comsol_model.COMSOL(mph.Model(java), layout)

    assert issubclass(comsol_model.COMSOL, gplugins_model.COMSOL)
    assert isinstance(model, mph.Model)
    assert model.java is java
    assert model.layout is layout


def test_package_exposes_the_class(comsol: tuple[Any, Any, Any]):
    """The package and the builder module both offer the QPDK class."""
    _, _, comsol_model = comsol
    package = importlib.import_module("qpdk.simulation")
    builder_module = importlib.import_module("qpdk.simulation.comsol")
    assert package.COMSOL is comsol_model.COMSOL
    assert builder_module.COMSOL is comsol_model.COMSOL


def test_create_sheet_defaults_to_the_qpdk_permittivities(
    comsol: tuple[Any, Any, Any], monkeypatch: pytest.MonkeyPatch
):
    """create_sheet passes the QPDK technology permittivities to gplugins."""
    mph, gplugins_model, comsol_model = comsol
    java = MagicMock()
    calls: list[tuple[Any, ...]] = []

    def fake_builder(client: Any, layout: Any, name: str, **kwargs: Any) -> Any:
        calls.append((client, layout, name, kwargs))
        return mph.Model(java)

    monkeypatch.setattr(gplugins_model.sheet, "build_comsol_sheet_model", fake_builder)
    client = MagicMock()
    layout = _layout()
    model = comsol_model.COMSOL.create_sheet(
        client, layout, "sheet", substrate_thickness_um=250.0
    )

    assert isinstance(model, comsol_model.COMSOL)
    assert model.java is java
    assert model.layout is layout
    assert calls == [
        (
            client,
            layout,
            "sheet",
            {
                "substrate_thickness_um": 250.0,
                "air_height_um": 200.0,
                "lateral_margin_um": 0.0,
                "silicon_relative_permittivity": SILICON_RELATIVE_PERMITTIVITY,
                "air_relative_permittivity": AIR_RELATIVE_PERMITTIVITY,
            },
        )
    ]


def test_create_metal_defaults_to_the_qpdk_name(
    comsol: tuple[Any, Any, Any], monkeypatch: pytest.MonkeyPatch
):
    """create_metal names the model after QPDK unless told otherwise."""
    mph, gplugins_model, comsol_model = comsol
    java = MagicMock()
    calls: list[tuple[Any, ...]] = []

    def fake_builder(client: Any, layout: Any, **kwargs: Any) -> Any:
        calls.append((client, layout, kwargs))
        return mph.Model(java)

    monkeypatch.setattr(gplugins_model.metal, "build_comsol_metal_model", fake_builder)
    client = MagicMock()
    layout = _layout()
    model = comsol_model.COMSOL.create_metal(client, layout, metal_thickness_um=0.35)

    assert isinstance(model, comsol_model.COMSOL)
    assert model.layout is layout
    assert calls == [
        (client, layout, {"metal_thickness_um": 0.35, "name": "QPDK metal"})
    ]
