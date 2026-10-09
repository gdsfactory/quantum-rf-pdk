"""Generic SAX models from stored N-port scattering or Maxwell matrices."""

from pathlib import Path

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.datasets.capacitance import check_maxwell
from qpdk.models.datasets.interpolation import GridInterpolator
from qpdk.models.datasets.metadata import QuantityKind
from qpdk.models.datasets.table import Dataset


def s_parameters_model(
    dataset: Dataset | Path | str,
    quantity: str = "s_parameters",
    *,
    frequency_axis: str = "frequency",
    **variants: str,
) -> sax.Model:
    """Load stored N-port S-parameters and return a jittable SAX lookup.

    ``f`` is in Hz and supplies ``frequency_axis``. Other continuous grid axes
    are passed by name; discrete variants are fixed when creating the model.
    Terminals become port names. Reference impedances and planes are those of
    the stored simulation, without renormalization. Out-of-domain values are
    NaN. Multilinear interpolation does not enforce losslessness between points.
    Unspecified geometry axes use their middle stored coordinates.

    Returns:
        SAX model with the dataset terminal names.

    Raises:
        ValueError: If the quantity or frequency axis has the wrong convention.
    """
    data = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
    grid = data.grid(quantity, allow_missing=False, **variants)
    if grid.quantity.kind != QuantityKind.S_PARAMETERS or not grid.quantity.complex:
        raise ValueError("Expected complex S-parameters")
    if not any(axis.name == frequency_axis and axis.unit == "Hz" for axis in grid.axes):
        raise ValueError("The frequency axis must be present and in Hz")
    lookup = GridInterpolator(grid)
    defaults = {
        name: float(values[len(values) // 2])
        for name, values in zip(grid.axis_names, grid.coords, strict=True)
        if name != frequency_axis
    }

    def model(f: ArrayLike = DEFAULT_FREQUENCY, **params: ArrayLike) -> sax.SDict:
        values = lookup(**{**defaults, frequency_axis: f, **params})
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(grid.terminals)
            for j, b in enumerate(grid.terminals)
        }

    sax.replace_kwargs(model, f=DEFAULT_FREQUENCY, **defaults)
    return jax.jit(model)


def capacitance_model(
    dataset: Dataset | Path | str,
    quantity: str = "maxwell_capacitance",
    **variants: str,
) -> sax.Model:
    r"""Return a lumped N-port SAX model from a Maxwell capacitance dataset.

    Continuous geometry axes are passed by name; discrete variants are fixed
    here. ``f`` is in Hz and ``z_ref`` is a common real port impedance in ohms.
    The admittance is :math:`Y = j 2 \pi f C`; all terminals reference the
    dataset ground. This model includes terminal-to-ground capacitances and
    mutual branches, with no distributed propagation or inductance.
    Unspecified geometry axes use their middle stored coordinates.

    Returns:
        SAX model with the dataset terminal names.

    Raises:
        ValueError: If the quantity is not a physical real Maxwell matrix.
    """
    data = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
    grid = data.grid(quantity, allow_missing=False, **variants)
    if grid.quantity.kind != QuantityKind.MAXWELL_CAPACITANCE or grid.quantity.complex:
        raise ValueError("Expected real Maxwell capacitances")
    check_maxwell(grid.values)
    lookup = GridInterpolator(grid)
    identity = jnp.eye(len(grid.terminals))
    defaults = {
        name: float(values[len(values) // 2])
        for name, values in zip(grid.axis_names, grid.coords, strict=True)
    }

    def model(
        f: ArrayLike = DEFAULT_FREQUENCY,
        z_ref: ArrayLike = 50.0,
        **params: ArrayLike,
    ) -> sax.SDict:
        capacitance = lookup(**{**defaults, **params})
        admittance = 2j * jnp.pi * jnp.asarray(f)[..., None, None] * capacitance
        normalized = jnp.asarray(z_ref)[..., None, None] * admittance
        values = jnp.linalg.solve(identity + normalized, identity - normalized)
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(grid.terminals)
            for j, b in enumerate(grid.terminals)
        }

    sax.replace_kwargs(model, f=DEFAULT_FREQUENCY, z_ref=50.0, **defaults)
    return jax.jit(model)
