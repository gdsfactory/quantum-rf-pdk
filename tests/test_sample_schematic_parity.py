"""Checks for layouts materialized from the sample schematics."""

from pathlib import Path

import pytest

from qpdk import PDK
from qpdk.read import from_gsch

SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
SAMPLE_STEMS = tuple(path.stem for path in sorted(SAMPLE_DIR.glob("*.gsch")))


@pytest.mark.filterwarnings("error:Routing failed:UserWarning")
@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_sample_materializes_from_schematic(stem: str) -> None:
    """Materialize every checked-in schematic without a YAML fallback."""
    PDK.activate()

    component = from_gsch(SAMPLE_DIR / f"{stem}.gsch")
    assert component.name == stem
    assert len(component.insts) > 0
    assert len(component.ports) > 0
