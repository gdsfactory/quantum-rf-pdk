"""Qubit test chip example with fill."""

from pathlib import Path

import gdsfactory as gf
from gdsfactory.read import from_yaml

from qpdk import PDK, tech
from qpdk.cells.helpers import fill_magnetic_vortices


@gf.cell
def filled_qubit_test_chip():
    """Returns a qubit test chip filled with magnetic vortex trapping rectangles.

    Rouhly corresponds to the sample in :cite:`tuokkolaMethodsAchieveNearmillisecond2025`.
    """
    c = gf.Component()
    test_chip = from_yaml(
        Path(__file__).parent / "qubit_test_chip.pic.yml",
        routing_strategies=tech.routing_strategies,
    )
    c << fill_magnetic_vortices(
        component=test_chip,
        rectangle_size=(15.0, 15.0),
        gap=15.0,
        exclude_layers=[(tech.LAYER.M1_ETCH, 80)],
    )
    return c


if __name__ == "__main__":
    PDK.activate()

    filled_qubit_test_chip().show()
