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


def junction_model(
    dataset: Dataset | Path | str,
    *,
    junction: tuple[str, str],
    inductance: float,
    junction_capacitance: float = 0.0,
    terminal_groups: tuple[tuple[str, ...], ...] | None = None,
    **variants: str,
) -> sax.Model:
    r"""Build a small-signal circuit from electrode capacitances and one junction.

    The dataset supplies capacitances between electrodes and to ground. Between
    the two ``junction`` ports, this adds an inductor in parallel with an optional
    junction capacitor. The inductor approximates the Josephson junction for
    small oscillations around its operating point; it does not model nonlinear
    dynamics or qubit anharmonicity. See :cite:`voolIntroductionQuantumElectromagnetic2017`,
    sections 2.1.6 and 4.1, for this linearization and the Josephson element.

    The branch admittance is :math:`Y_\text{J}=1/(j\omega L_\text{J})+j\omega C_\text{J}`.
    Its inductance and capacitance are explicit inputs, since an electrostatic
    dataset cannot determine the Josephson inductance. Only add junction
    capacitance that is not already included in the dataset.

    ``terminal_groups`` optionally connects electrodes with ideal wires. For
    example, ``(("island", "top"), ("ring",))`` joins ``island`` and ``top`` and
    leaves ``ring`` separate. Each group's first name becomes its SAX port.
    Every dataset terminal must appear exactly once. This applies
    :math:`B^{\text{T}} C B`, with :math:`B` mapping group voltages to electrodes,
    and retains capacitance to ground. The wires add no field or inductance.
    All SAX ports reference dataset ground, and frequency is in Hz.

    Args:
        dataset: Dataset or path containing real Maxwell capacitance matrices in F.
        junction: Names of the two ports connected by the junction, after grouping.
        inductance: Positive small-signal Josephson inductance in H.
        junction_capacitance: Additional parallel junction capacitance in F.
        terminal_groups: Electrodes to connect together; defaults to separate ports.
        **variants: Discrete dataset selections.

    Returns:
        Jittable small-signal SAX model, with geometry in dataset axis units.

    Raises:
        ValueError: If terminals, grouping, matrix type or explicit inputs are invalid.
    """
    if inductance <= 0 or junction_capacitance < 0:
        raise ValueError(
            "Require positive inductance and nonnegative junction capacitance"
        )
    data = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
    grid = data.grid("maxwell_capacitance", allow_missing=False, **variants)
    if grid.quantity.kind != QuantityKind.MAXWELL_CAPACITANCE or grid.quantity.complex:
        raise ValueError("Expected real Maxwell capacitances")
    check_maxwell(grid.values)
    groups = (
        tuple((name,) for name in grid.terminals)
        if terminal_groups is None
        else terminal_groups
    )
    members = [name for group in groups for name in group]
    if any(not group for group in groups) or sorted(members) != sorted(grid.terminals):
        raise ValueError(
            "Terminal groups must partition every dataset terminal exactly once"
        )
    ports = tuple(group[0] for group in groups)
    if len(set(junction)) != 2 or not set(junction) <= set(ports):
        raise ValueError("The junction must name two distinct grouped ports")
    connection = jnp.asarray([
        [float(name in group) for group in groups] for name in grid.terminals
    ])
    d = jnp.asarray([
        float(name == junction[0]) - float(name == junction[1]) for name in ports
    ])
    branch = d[:, None] * d[None, :]
    lookup = GridInterpolator(grid)
    identity = jnp.eye(len(ports))
    defaults = {
        name: float(values[len(values) // 2])
        for name, values in zip(grid.axis_names, grid.coords, strict=True)
    }

    def model(
        f: ArrayLike = DEFAULT_FREQUENCY,
        z_ref: ArrayLike = 50.0,
        inductance: ArrayLike = inductance,
        junction_capacitance: ArrayLike = junction_capacitance,
        **params: ArrayLike,
    ) -> sax.SDict:
        """Evaluate a lossless linearized junction circuit at positive frequencies."""
        capacitance = connection.T @ lookup(**{**defaults, **params}) @ connection
        omega = 2 * jnp.pi * jnp.asarray(f)
        y_junction = 1j * omega * jnp.asarray(junction_capacitance) + 1 / (
            1j * omega * jnp.asarray(inductance)
        )
        admittance = (
            1j * omega[..., None, None] * capacitance
            + y_junction[..., None, None] * branch
        )
        normalized = jnp.asarray(z_ref)[..., None, None] * admittance
        values = jnp.linalg.solve(identity + normalized, identity - normalized)
        valid = (
            (jnp.asarray(f) > 0)
            & (jnp.asarray(inductance) > 0)
            & (jnp.asarray(junction_capacitance) >= 0)
        )
        values = jnp.where(valid[..., None, None], values, jnp.nan)
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(ports)
            for j, b in enumerate(ports)
        }

    sax.replace_kwargs(
        model,
        f=DEFAULT_FREQUENCY,
        z_ref=50.0,
        inductance=inductance,
        junction_capacitance=junction_capacitance,
        **defaults,
    )
    return jax.jit(model)
