"""Qubit test chip sample, materialized from its ``.pic.yml`` netlist."""

from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

YAML_SAMPLE = Path(__file__).parent / "qubit_test_chip.pic.yml"


@gf.cell(tags=["samples", "qubits"])
def qubit_test_chip() -> gf.Component:
    """Layout of the qubit test chip, read from its ``.pic.yml`` netlist.

    The matching ``.gsch`` remains the source for licensed schematic flows.

    Returns:
        The materialized sample component.
    """
    return from_yaml(YAML_SAMPLE, name="qubit_test_chip")
