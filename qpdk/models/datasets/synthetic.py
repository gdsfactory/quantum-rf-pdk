"""Synthetic placeholder datasets, shipped until real Palace extractions exist.

Each dataset is a :class:`~qpdk.models.datasets.metadata.DatasetMetadata`, a grid, and
a ``solve`` function passed to :func:`qpdk.models.datasets.generate.sweep`; a real FEM
dataset swaps only the ``solve`` function. Regenerate all of them from the
repository root with::

    uv run --extra models python -m qpdk.models.datasets.synthetic
"""

import numpy as np

from qpdk import logger
from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import ε_0
from qpdk.models.datasets.generate import sweep, write
from qpdk.models.datasets.metadata import Axis, DatasetMetadata, Quantity, QuantityKind
from qpdk.models.datasets.table import DATASETS_PATH
from qpdk.models.math import epsilon_eff

PLATE_CAPACITOR = DatasetMetadata(
    name="plate_capacitor_synthetic",
    description="Two-pad Maxwell capacitance of plate_capacitor over length, width, and gap (synthetic placeholder).",
    synthetic=True,
    axes=(
        Axis(name="length", unit="um"),
        Axis(name="width", unit="um"),
        Axis(name="gap", unit="um"),
    ),
    variants=("cross_section",),
    quantities=(
        Quantity(
            name="maxwell_capacitance", kind=QuantityKind.MAXWELL_CAPACITANCE, unit="F"
        ),
    ),
    terminals=("o1", "o2"),
    reference_ground="ground_plane",
    provenance={
        "cell": "plate_capacitor",
        "solver": "analytical placeholder: conformal-mapping mutual term, invented perimeter term to ground",
        "substrate": {"material": "Si", "relative_permittivity": 11.45},
    },
)
"""Metadata of ``plate_capacitor_synthetic``."""

PLATE_CAPACITOR_GRID = {
    "length": [20.0, 40.0, 80.0, 120.0, 160.0, 200.0, 250.0, 300.0],
    "width": [5.0, 10.0, 15.0, 20.0],
    "gap": [2.0, 4.0, 7.0, 10.0, 15.0, 20.0],
}
"""Sweep grid of ``plate_capacitor_synthetic``, in µm."""


def plate_capacitor_maxwell(
    length: float,
    width: float,
    gap: float,
    cross_section: str,  # ruff: ignore[unused-function-argument]
    ep_r: float = 11.45,
) -> dict[str, np.ndarray]:
    """Synthetic two-pad Maxwell capacitance matrix in F.

    The mutual term is
    :func:`~qpdk.models.capacitor.plate_capacitor_capacitance_analytical`; the
    capacitance of each pad to ground is an invented perimeter term, so the
    matrix has the structure of a two-terminal Maxwell matrix without being
    physics.

    Returns:
        The ``maxwell_capacitance`` quantity.
    """
    mutual = float(
        plate_capacitor_capacitance_analytical(
            length=length, width=width, gap=gap, ep_r=ep_r
        )
    )
    ground = float(ε_0 * epsilon_eff(ep_r) * (length + 2 * width) * 1e-6)
    return {
        "maxwell_capacitance": np.array([
            [ground + mutual, -mutual],
            [-mutual, ground + mutual],
        ])
    }


def main() -> None:
    """Regenerate every synthetic dataset in :data:`~qpdk.models.datasets.table.DATASETS_PATH`."""
    frame = sweep(
        PLATE_CAPACITOR,
        plate_capacitor_maxwell,
        PLATE_CAPACITOR_GRID,
        {"cross_section": ["cpw"]},
    )
    dataset = write(DATASETS_PATH / PLATE_CAPACITOR.name, PLATE_CAPACITOR, frame)
    logger.info(f"Wrote {dataset!r}: {frame.height} rows")


if __name__ == "__main__":
    main()
