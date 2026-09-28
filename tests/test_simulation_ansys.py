"""Tests for the layout of the :mod:`qpdk.simulation.ansys` package."""

import importlib
import subprocess
import sys

import pytest

import qpdk.simulation
import qpdk.simulation.ansys

_ANSYS_NAMES = qpdk.simulation.ansys.__all__


@pytest.mark.parametrize("name", _ANSYS_NAMES)
def test_simulation_reexports_ansys_names(name: str):
    """Every AEDT name stays importable from ``qpdk.simulation``."""
    assert name in qpdk.simulation.__all__
    assert getattr(qpdk.simulation, name) is getattr(qpdk.simulation.ansys, name)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("qpdk.simulation.aedt_base", "qpdk.simulation.ansys.base"),
        ("qpdk.simulation.hfss", "qpdk.simulation.ansys.hfss"),
        ("qpdk.simulation.q3d", "qpdk.simulation.ansys.q3d"),
    ],
)
def test_old_module_paths_are_deprecated_aliases(
    monkeypatch: pytest.MonkeyPatch, old: str, new: str
):
    """The pre-``ansys`` module paths warn and forward to the moved modules."""
    monkeypatch.delitem(sys.modules, old, raising=False)
    with pytest.warns(DeprecationWarning, match=new):
        module = importlib.import_module(old)
    moved = importlib.import_module(new)
    public = [name for name in vars(moved) if not name.startswith("__")]
    assert public
    for name in public:
        assert getattr(module, name) is getattr(moved, name)


def test_ansys_import_defers_design_wrappers():
    """Importing the package leaves the HFSS and Q3D modules unloaded until used."""
    code = (
        "import sys, qpdk.simulation.ansys as a\n"
        "assert 'qpdk.simulation.ansys.hfss' not in sys.modules\n"
        "assert 'qpdk.simulation.ansys.q3d' not in sys.modules\n"
        "assert a.HFSS.__module__ == 'qpdk.simulation.ansys.hfss'\n"
        "assert a.Q3D.__module__ == 'qpdk.simulation.ansys.q3d'\n"
    )
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code], check=True
    )


def test_ansys_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = qpdk.simulation.ansys.not_a_name  # pyrefly: ignore[missing-attribute]
