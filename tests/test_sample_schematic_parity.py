"""Checks for the license-free sample layout fallbacks."""

from pathlib import Path

import gdsfactory as gf
import pytest

from qpdk import PDK

SAMPLE_DIR = Path(__file__).parents[1] / "qpdk/samples"
SAMPLE_STEMS = tuple(path.stem for path in sorted(SAMPLE_DIR.glob("*.gsch")))


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_sample_resolves_from_pdk(stem: str) -> None:
    """Resolve every checked-in schematic sample by its short name."""
    PDK.activate()

    assert gf.get_component(stem).name == stem
