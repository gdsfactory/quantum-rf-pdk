"""Flipmon test chip sample, materialized from its ``.gsch`` netlist."""

from pathlib import Path

import gdsfactory as gf

from qpdk.read import from_gsch

SCHEMATIC_SAMPLE = Path(__file__).parent / "flipmon_test_chip.gsch"


@gf.cell(tags=["samples", "qubits", "flip-chip"])
def flipmon_test_chip() -> gf.Component:
    """Layout of the flipmon test chip, read from its ``.gsch`` netlist.

    The full chip layout is defined declaratively in the sample's netlist;
    this factory only reads that file, so editing the netlist edits the
    layout. The materialized name matches the ``.gsch``/GDS stem because
    elvis resolves the top cell from the GDS file name.

    Returns:
        The materialized sample component.
    """
    return from_gsch(SCHEMATIC_SAMPLE)
