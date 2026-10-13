"""Qubit test chip sample, materialized from its ``.gsch`` netlist."""

from pathlib import Path

import gdsfactory as gf

from qpdk.read import from_gsch

SCHEMATIC_SAMPLE = Path(__file__).parent / "qubit_test_chip.gsch"


@gf.cell(tags=["samples", "qubits"])
def qubit_test_chip() -> gf.Component:
    """Layout of the qubit test chip, read from its ``.gsch`` netlist.

    Returns:
        The materialized sample component.
    """
    return from_gsch(SCHEMATIC_SAMPLE)
