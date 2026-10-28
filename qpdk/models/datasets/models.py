"""Generic SAX models from stored N-port scattering or Maxwell matrices."""

from pathlib import Path

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import LayerSpec
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.datasets.capacitance import check_maxwell
from qpdk.models.datasets.interpolation import GridInterpolator
from qpdk.models.datasets.metadata import QuantityKind
from qpdk.models.datasets.table import Dataset
from qpdk.models.jit import jit


def _matches_layer(
    value: LayerSpec | ArrayLike, index: int, pair: tuple[int, int], name: str
) -> jax.Array:
    """Distinguish static layer names, layout indices and GDS pairs."""
    if isinstance(value, str):
        return jnp.asarray(value == name)
    array = jnp.asarray(value)
    if array.ndim == 0:
        return array == index
    if array.shape == (2,):
        return jnp.all(array == jnp.asarray(pair))
    return jnp.asarray(False)


def s_parameters_model(
    dataset: Dataset | Path | str,
    quantity: str = "s_parameters",
    *,
    frequency_axis: str = "frequency",
    **variants: str,
) -> sax.SDictModel:
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
    return jit(model)


def capacitance_model(
    dataset: Dataset | Path | str,
    quantity: str = "maxwell_capacitance",
    **variants: str,
) -> sax.SDictModel:
    r"""Return a lumped N-port SAX model from a Maxwell capacitance dataset.

    Continuous geometry axes are passed by name; discrete variants are fixed
    here. ``f`` is in Hz and ``z_ref`` is a common finite positive port impedance
    in ohms. Invalid reference impedances return NaN.
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
        valid = jnp.isfinite(jnp.asarray(z_ref)) & (jnp.asarray(z_ref) > 0)
        values = jnp.where(valid[..., None, None], values, jnp.nan)
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(grid.terminals)
            for j, b in enumerate(grid.terminals)
        }

    sax.replace_kwargs(model, f=DEFAULT_FREQUENCY, z_ref=50.0, **defaults)
    return jit(model)


def junction_model(
    dataset: Dataset | Path | str,
    *,
    junction: tuple[str, str | None],
    inductance: float,
    junction_capacitance: float = 0.0,
    terminal_groups: tuple[tuple[str, ...], ...] | None = None,
    exposed_ports: tuple[str, ...] | None = None,
    **variants: str,
) -> sax.SDictModel:
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
    Reference impedance ``z_ref`` must be finite and positive, in ohms.

    Use ``None`` for the junction's second terminal to connect it to reference
    ground. ``exposed_ports`` selects the external circuit terminals. Other
    electrodes remain internal with zero injected current, rather than being
    terminated by SAX's implicit matched loads. This lets a local qubit and
    coupling paddle expose only its readout pin without damping the islands.
    The dataset must end at that pin; a distributed resonator remains outside
    this boundary.

    Args:
        dataset: Dataset or path containing real Maxwell capacitance matrices in F.
        junction: Names of the two ports connected by the junction, after grouping.
        inductance: Positive small-signal Josephson inductance in H.
        junction_capacitance: Additional parallel junction capacitance in F.
        terminal_groups: Electrodes to connect together; defaults to separate ports.
        exposed_ports: External ports in order; defaults to every grouped terminal.
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
    if (
        junction[0] not in ports
        or junction[1] == junction[0]
        or (junction[1] is not None and junction[1] not in ports)
    ):
        raise ValueError("The junction must name distinct grouped ports or ground")
    external = ports if exposed_ports is None else exposed_ports
    if (
        not external
        or len(set(external)) != len(external)
        or not set(external) <= set(ports)
    ):
        raise ValueError("Exposed ports must be a nonempty selection of grouped ports")
    indices = jnp.asarray([ports.index(name) for name in external])
    connection = jnp.asarray([
        [float(name in group) for group in groups] for name in grid.terminals
    ])
    d = jnp.asarray([
        float(name == junction[0]) - float(name == junction[1]) for name in ports
    ])
    branch = d[:, None] * d[None, :]
    lookup = GridInterpolator(grid)
    identity = jnp.eye(len(ports))
    loads = jnp.diag(jnp.asarray([float(name in external) for name in ports]))
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
        # Only external ports have reference loads; internal nodes obey KCL.
        response = jnp.linalg.solve(normalized + loads, identity[:, indices])
        values = 2 * response[..., indices, :] - jnp.eye(len(external))
        valid = (
            (jnp.asarray(f) > 0)
            & (jnp.asarray(inductance) > 0)
            & (jnp.asarray(junction_capacitance) >= 0)
            & jnp.isfinite(jnp.asarray(z_ref))
            & (jnp.asarray(z_ref) > 0)
        )
        values = jnp.where(valid[..., None, None], values, jnp.nan)
        return {
            (a, b): values[..., i, j]
            for i, a in enumerate(external)
            for j, b in enumerate(external)
        }

    sax.replace_kwargs(
        model,
        f=DEFAULT_FREQUENCY,
        z_ref=50.0,
        inductance=inductance,
        junction_capacitance=junction_capacitance,
        **defaults,
    )
    return jit(model)
