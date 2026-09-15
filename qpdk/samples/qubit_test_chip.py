"""Qubit test chip sample, materialized from its ``.pic.yml`` netlist."""

from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

YAML_SAMPLE = Path(__file__).parent / "qubit_test_chip.pic.yml"


@gf.cell
def qubit_test_chip() -> gf.Component:
    """Layout of the qubit test chip, read from its ``.pic.yml`` netlist.

    The full chip layout is defined declaratively in the sample's netlist;
    this factory only reads that file, so editing the netlist edits the
    layout. The materialized name matches the ``.pic.yml``/GDS stem because
    elvis resolves the top cell from the GDS file name.

    Returns:
        The materialized sample component.
    """
    return from_yaml(YAML_SAMPLE, name="qubit_test_chip")
