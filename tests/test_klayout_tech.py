"""Checks on the generated KLayout technology file."""

import re

import klayout.db as kdb

from qpdk.config import PATH
from qpdk.tech import LAYER_CONNECTIVITY


def test_tech_lyt_loads_in_klayout() -> None:
    tech = kdb.Technology()
    tech.load(str(PATH.lyt))
    assert tech.name == "qpdk"


def test_connectivity_is_conductor_via_conductor() -> None:
    # KLayout's net tracer only parses three-layer connections
    assert all(len(spec) == 3 for spec in LAYER_CONNECTIVITY)


def test_tech_lyt_connectivity_matches_layer_connectivity() -> None:
    text = PATH.lyt.read_text()
    written = {
        tuple(c.split(","))
        for c in re.findall(r"<connection>([^<]+)</connection>", text)
    }
    assert written == {tuple(spec) for spec in LAYER_CONNECTIVITY}
