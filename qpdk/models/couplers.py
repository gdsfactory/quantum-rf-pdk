"""Coupler models."""

from dataclasses import replace
from functools import cache, partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from jax.typing import ArrayLike
from sax.models.rf import tee

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0, μ_0
from qpdk.models.cpw import (
    cpw_ep_r_from_cross_section,
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
    transmission_line_s_params,
)
from qpdk.models.datasets.capacitance import check_maxwell
from qpdk.models.datasets.interpolation import GridInterpolator
from qpdk.models.datasets.metadata import QuantityKind
from qpdk.models.datasets.table import Dataset, Grid
from qpdk.models.math import (
    capacitance_per_length_conformal,
    ellipk_ratio,
    epsilon_eff,
)
from qpdk.models.waveguides import straight


@partial(jax.jit, inline=True)
def cpw_cpw_coupling_capacitance_per_length_analytical(
    gap: float | ArrayLike,
    width: float | ArrayLike,
    cpw_gap: float | ArrayLike,
    ep_r: float | ArrayLike,
) -> float | jax.Array:
    r"""Analytical formula for ECCPW mutual capacitance per unit length.

    The model follows the edge-coupled coplanar waveguide (ECCPW) formula
    using conformal mapping for even and odd modes:

    .. math::

        \begin{aligned}
        x_1 &= s_\text{c} / 2 \\
        x_2 &= x_1 + W \\
        x_3 &= x_2 + G \\
        k_\text{e} &= \sqrt{\frac{x_2^2 - x_1^2}{x_3^2 - x_1^2}} \\
        k_\text{o} &= \frac{x_1}{x_2} \sqrt{\frac{x_3^2 - x_2^2}{x_3^2 - x_1^2}} \\
        C_{\text{even}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_\text{e})}{K(k_\text{e}')} \\
        C_{\text{odd}} &= 2 \epsilon_0 \epsilon_{\text{eff}} \frac{K(k_\text{o}')}{K(k_\text{o})} \\
        C_\text{m} &= \frac{C_{\text{odd}} - C_{\text{even}}}{2}
        \end{aligned}

    where :math:`s_\text{c}` is the separation (gap) between inner edges, :math:`W` is the
    center conductor width, and :math:`G` is the gap to the ground plane.

    The entire inner gap is etched. This expression excludes the ground strip
    left by ``coupler_straight`` when its gap exceeds twice the CPW slot width.

    See :cite:`simonsCoplanarWaveguideCircuits2001`.

    Args:
        gap: The gap (separation) between the two center conductors in µm.
        width: Center conductor width in µm.
        cpw_gap: Gap between center conductor and ground plane in µm.
        ep_r: Relative permittivity of the substrate.

    Returns:
        The mutual coupling capacitance per unit length in Farads/meter.
    """
    # Geometric parameters in m (convert from μm)
    s_c = gap * 1e-6
    w_m = width * 1e-6
    g_m = cpw_gap * 1e-6

    x1 = s_c / 2
    x2 = x1 + w_m
    x3 = x2 + g_m

    # Even-mode modulus squared
    ke_sq = (x2**2 - x1**2) / (x3**2 - x1**2)

    # Odd-mode modulus squared
    ko_sq = (x1**2 / x2**2) * ((x3**2 - x2**2) / (x3**2 - x1**2))

    # Capacitances per unit length
    # Factor is 2.0 since ECCPW formula uses 2 * ε_0 * ε_eff
    c_even_pul = 2.0 * capacitance_per_length_conformal(m=ke_sq, ep_r=ep_r)
    # c_odd uses K(1-m)/K(m) which is the inverse of ellipk_ratio(m)
    c_odd_pul = 2.0 * ε_0 * epsilon_eff(ep_r) / ellipk_ratio(ko_sq)

    # Mutual capacitance per unit length
    return (c_odd_pul - c_even_pul) / 2


class _GroundStripLookup:
    """Resolve shielding onset without interpolating across a geometry change."""

    def __init__(self, grid: Grid, unshielded: Grid) -> None:
        """Prepare the two independent geometry regimes outside JAX tracing."""
        if any(not bool(jnp.all(jnp.asarray(axis) > 0)) for axis in grid.coords):
            raise ValueError("Log interpolation requires positive geometry coordinates")
        self._etched = GridInterpolator(unshielded)
        self._domain = grid.domain
        values = jnp.asarray(grid.values)
        signs = jnp.array([[1, -1], [-1, 1]])
        if not bool(jnp.all(values * signs > 0)):
            raise ValueError(
                "Log interpolation requires positive self and negative mutual capacitances"
            )
        self._signs = signs
        self._interpolate = GridInterpolator(
            replace(
                grid,
                axes=tuple(
                    axis.model_copy(update={"validated": None}) for axis in grid.axes
                ),
                coords=tuple(jnp.log(jnp.asarray(axis)) for axis in grid.coords),
                values=jnp.log(jnp.abs(values)),
            )
        )

    def __call__(
        self, *, width: ArrayLike, cpw_gap: ArrayLike, gap: ArrayLike
    ) -> jax.Array:
        """Return the raw slice matrix; unsupported positive strips yield NaN."""
        epsilon = max(
            jnp.finfo(jnp.result_type(cpw_gap, 0.0)).eps,
            jnp.finfo(jnp.result_type(gap, 0.0)).eps,
        )
        width, cpw_gap, gap = jnp.broadcast_arrays(
            *(jnp.asarray(value, dtype=float) for value in (width, cpw_gap, gap))
        )
        strip = gap - 2 * cpw_gap
        lo_strip, hi_strip = self._domain["ground_strip_width"]
        # Subtracting the slot widths loses a few ulps at the thinnest strip.
        tolerance = 8 * epsilon * jnp.maximum(jnp.abs(gap), 1)
        points = (width, cpw_gap, jnp.clip(strip, lo_strip, hi_strip))
        grounded = (
            jnp.exp(
                self._interpolate(**{
                    name: jnp.log(value)
                    for name, value in zip(
                        self._interpolate.axis_names, points, strict=True
                    )
                })
            )
            * self._signs
        )
        inside = jnp.ones(width.shape, dtype=bool)
        for name, value in zip(
            ("width", "cpw_gap", "ground_strip_width"),
            (width, cpw_gap, strip),
            strict=True,
        ):
            lo, hi = self._domain[name]
            if name == "ground_strip_width":
                lo, hi = lo - tolerance, hi + tolerance
            inside &= (value >= lo) & (value <= hi)
        grounded = jnp.where(inside[..., None, None], grounded, jnp.nan)
        etched = self._etched(width=width, cpw_gap=cpw_gap, gap=gap)
        return jnp.where((strip <= 0)[..., None, None], etched, grounded)


def _cpw_matrix_lookup(
    data: Dataset, grid: Grid
) -> GridInterpolator | _GroundStripLookup:
    """Select interpolation for a continuous geometry regime."""
    if grid.axis_names == ("width", "cpw_gap", "ground_strip_width"):
        etched = Dataset("cpw_coupling_palace")
        for name in ("slice_length_um", "permittivity"):
            if (
                data.metadata.provenance["settings"][name]
                != etched.metadata.provenance["settings"][name]
            ):
                raise ValueError(
                    f"Grounded and fully etched datasets disagree on {name}"
                )
        return _GroundStripLookup(grid, etched.grid("maxwell_capacitance"))
    return GridInterpolator(grid)


def cpw_coupling_model(
    dataset: Dataset | Path | str = "cpw_coupling_ground_strip_palace",
) -> sax.Model:
    r"""Load a symmetric CPW dataset once and return a jittable four-port model.

    The dataset contains slice capacitances in F, swept over ``width``,
    ``cpw_gap`` and ``gap`` or ``ground_strip_width`` in µm. Grounded cases use
    logarithmic coordinates and capacitance magnitudes, with the fully etched
    dataset used where the slots touch or overlap. Positive ground strips below
    the sampled minimum return NaN. The generator's ``slice_length_um`` and
    substrate ``permittivity`` must be present in the provenance. Ports are
    ``o1`` lower-left, ``o2`` upper-left, ``o3`` upper-right, ``o4`` lower-right.
    The reference planes are the two ends of the uniform coupled section.
    The default dataset retains the ground strip between separated CPW slots,
    matching ``coupler_straight``. Select ``cpw_coupling_palace`` explicitly for
    a fully etched inner gap and comparison with the analytical ECCPW formula.

    For sheets between two dielectric half-spaces,
    :math:`C = (1 + \epsilon_\text{r}) C_0 / 2`. The quasi-TEM geometric
    inductance per length is :math:`L = \mu_0 \epsilon_0 C_0^{-1}`. Even and
    odd modes form a lossless distributed line, transformed back into the
    physical ports. Finite substrate thickness, kinetic inductance, conductor
    loss and dispersion are outside this model; see
    :cite:`simonsCoplanarWaveguideCircuits2001` for coupled-line theory. Check
    dielectric scaling with separate vacuum solves when changing the geometry
    or boundaries.

    Self-capacitances may differ by up to 1% from numerical meshing; their
    mean defines the symmetric model. Larger differences are rejected.
    File reads and grid validation happen here, outside JAX tracing. The
    returned model supports ``jit``, ``vmap`` and geometry derivatives; queries
    outside the dataset domain yield NaN.

    Returns:
        SAX model for the uniform four-port coupled section.

    Raises:
        ValueError: If the grid or slice settings do not describe symmetric CPWs.
    """
    data = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
    grid = data.grid("maxwell_capacitance")
    if grid.quantity.kind != QuantityKind.MAXWELL_CAPACITANCE or grid.quantity.complex:
        raise ValueError("Expected real Maxwell capacitances")
    if (
        grid.axis_names
        not in {
            ("width", "cpw_gap", "gap"),
            ("width", "cpw_gap", "ground_strip_width"),
        }
        or len(grid.terminals) != 2
    ):
        raise ValueError("Expected a symmetric two-conductor CPW geometry grid")
    if any(axis.unit != "um" for axis in grid.axes):
        raise ValueError("CPW geometry axes must be in um")
    check_maxwell(grid.values)
    if jnp.any(grid.values.sum(axis=-1) <= 0):
        raise ValueError("Both traces must have positive capacitance to ground")
    if not jnp.allclose(
        grid.values[..., 0, 0], grid.values[..., 1, 1], rtol=0.01, atol=0
    ):
        raise ValueError("The CPW model requires symmetric traces and ground rails")
    settings = data.metadata.provenance["settings"]
    slice_length = float(settings["slice_length_um"]) * 1e-6
    effective_permittivity = (1 + float(settings["permittivity"])) / 2
    if slice_length <= 0 or effective_permittivity <= 0:
        raise ValueError("Slice length and effective permittivity must be positive")
    lookup = _cpw_matrix_lookup(data, grid)

    @jax.jit
    def model(
        f: ArrayLike = DEFAULT_FREQUENCY,
        length: ArrayLike = 1000.0,
        width: ArrayLike = 10.0,
        cpw_gap: ArrayLike = 6.0,
        gap: ArrayLike = 8.0,
        z_ref: ArrayLike = 50.0,
    ) -> sax.SDict:
        """Uniform section; f in Hz, dimensions in µm, port impedance in ohms."""
        capacitance = lookup(width=width, cpw_gap=cpw_gap, gap=gap) / slice_length
        c_self = (capacitance[..., 0, 0] + capacitance[..., 1, 1]) / 2
        c_mutual = (capacitance[..., 0, 1] + capacitance[..., 1, 0]) / 2
        modal_c = jnp.stack((c_self + c_mutual, c_self - c_mutual), axis=-1)
        modal_l = μ_0 * ε_0 * effective_permittivity / modal_c
        gamma = 2j * jnp.pi * jnp.asarray(f)[..., None] * jnp.sqrt(modal_l * modal_c)
        impedance = jnp.sqrt(modal_l / modal_c)
        reflection, transmission = transmission_line_s_params(
            gamma,
            impedance,
            jnp.asarray(length)[..., None] * 1e-6,
            jnp.asarray(z_ref)[..., None],
        )
        r_same = (reflection[..., 0] + reflection[..., 1]) / 2
        r_other = (reflection[..., 0] - reflection[..., 1]) / 2
        t_same = (transmission[..., 0] + transmission[..., 1]) / 2
        t_other = (transmission[..., 0] - transmission[..., 1]) / 2
        return sax.reciprocal({
            ("o1", "o1"): r_same,
            ("o2", "o2"): r_same,
            ("o3", "o3"): r_same,
            ("o4", "o4"): r_same,
            ("o1", "o2"): r_other,
            ("o3", "o4"): r_other,
            ("o1", "o4"): t_same,
            ("o2", "o3"): t_same,
            ("o1", "o3"): t_other,
            ("o2", "o4"): t_other,
        })

    return model


@cache
def _cpw_lookup() -> tuple[GridInterpolator | _GroundStripLookup, float, float]:
    """Cache the grid, slice length and simulated substrate permittivity.

    Returns:
        Lookup and the dimensions needed to normalize its capacitances.
    """
    data = Dataset("cpw_coupling_ground_strip_palace")
    settings = data.metadata.provenance["settings"]
    return (
        _cpw_matrix_lookup(data, data.grid("maxwell_capacitance")),
        float(settings["slice_length_um"]) * 1e-6,
        float(settings["permittivity"]),
    )


def cpw_cpw_coupling_capacitance(
    f: sax.FloatArrayLike,  # ruff: ignore[unused-function-argument]
    length: float | ArrayLike,
    gap: float | ArrayLike,
    cross_section: CrossSectionSpec,
) -> float | jax.Array:
    r"""Look up Palace coupling capacitance between two parallel CPWs.

    The uniform sheet dataset supplies mutual capacitance per length. Values
    outside its width, outer-slot and inter-trace-gap domain return NaN.
    Its inner ground strip has width ``max(gap - 2 * cpw_gap, 0)``, matching
    the etch masks of ``coupler_straight``.
    Dielectric half-space scaling accounts for cross-section permittivity;
    kinetic inductance, thickness and fringing are omitted.

    Args:
        f: Frequency array in Hz.
        length: The coupling length in µm.
        gap: The gap between the two center conductors in µm.
        cross_section: The cross-section of the CPW.

    Returns:
        The total coupling capacitance in Farads.

    Note:
        Raises ``ValueError`` via :func:`~qpdk.models.cpw.get_cpw_dimensions`
        if the cross-section has no section with 'etch' in its name, so the
        CPW gap cannot be determined, or if the conductor width or etch gap
        is not positive.
    """
    ep_r = cpw_ep_r_from_cross_section(cross_section)

    width, cpw_gap = get_cpw_dimensions(cross_section)

    with jax.ensure_compile_time_eval():
        lookup, slice_length, simulated_permittivity = _cpw_lookup()
    capacitance = lookup(width=width, cpw_gap=cpw_gap, gap=gap)
    c_pul = -capacitance[..., 0, 1] / slice_length
    c_pul *= epsilon_eff(ep_r) / epsilon_eff(simulated_permittivity)
    return c_pul * length * 1e-6


def coupler_straight(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 10,
    gap: int | float = 16,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for two coupled coplanar waveguides, :func:`~qpdk.cells.waveguides.coupler_straight`.

    Args:
        f: Array of frequency points in Hz
        length: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SDict: S-parameters dictionary

    .. code::

        o2──────▲───────o3
                │gap
        o1──────▼───────o4
    """
    f = jnp.asarray(f)
    straight_settings = {"length": length / 2, "cross_section": cross_section}
    capacitance = cpw_cpw_coupling_capacitance(f, length, gap, cross_section)
    normalized_admittance = (
        2j * jnp.pi * f * capacitance * cpw_z0_from_cross_section(cross_section, f)
    )
    # Admittance avoids the capacitor's infinite impedance at DC.
    reflection = 1 / (1 + 2 * normalized_admittance)
    transmission = 2 * normalized_admittance * reflection
    capacitor = {
        ("o1", "o1"): reflection,
        ("o2", "o2"): reflection,
        ("o1", "o2"): transmission,
        ("o2", "o1"): transmission,
    }

    # Create straight instances with shared settings
    straight_instances = {
        f"straight_{i}_{j}": straight(f=f, **straight_settings)
        for i in [1, 2]
        for j in [1, 2]
    }
    tee_instances = {f"tee_{i}": tee(f=f) for i in [1, 2]}

    instances = {
        **straight_instances,
        **tee_instances,
        "capacitor": capacitor,
    }
    connections = {
        "straight_1_1,o1": "tee_1,o1",
        "straight_1_2,o1": "tee_1,o2",
        "straight_2_1,o1": "tee_2,o1",
        "straight_2_2,o1": "tee_2,o2",
        "tee_1,o3": "capacitor,o1",
        "tee_2,o3": "capacitor,o2",
    }
    ports = {
        "o2": "straight_1_1,o2",
        "o3": "straight_1_2,o2",
        "o1": "straight_2_1,o2",
        "o4": "straight_2_2,o2",
    }

    return sax.evaluate_circuit_fg((connections, ports), instances)


def coupler_ring(
    f: ArrayLike = DEFAULT_FREQUENCY,
    length_x: int | float = 20.0,
    gap: int | float = 16,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameter model for two coupled coplanar waveguides in a ring configuration.

    The implementation is the same as straight coupler for now.

    TODO: Fetch coupling capacitance from a curved simulation library.

    Args:
        f: Array of frequency points in Hz
        length_x: Physical length of coupling section in µm
        gap: Gap between the coupled waveguides in µm
        cross_section: The cross-section of the CPW.

    Returns:
        sax.SDict: S-parameters dictionary
    """
    return coupler_straight(f=f, length=length_x, gap=gap, cross_section=cross_section)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lengths = jnp.linspace(10, 1000, 10)
    gaps = jnp.geomspace(0.1, 5.0, 6)
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    plt.figure(figsize=(10, 6))

    # Calculate capacitance per unit length for all gaps simultaneously (shape: (6,))
    c_pul = cast(
        jax.Array,
        cpw_cpw_coupling_capacitance_per_length_analytical(
            gap=gaps, width=width, cpw_gap=cpw_gap, ep_r=ep_r
        ),
    )

    # Broadcast to compute total capacitance for all lengths and gaps (shape: (6, 1000))
    capacitances = c_pul[:, None] * lengths[None, :] * 1e-6 * 1e15  # Convert to fF

    for i, gap in enumerate(gaps):
        plt.plot(lengths, capacitances[i], label=f"gap = {gap:.1f} µm")

    plt.xlabel("Coupling Length (µm)")
    plt.ylabel("Mutual Capacitance (fF)")
    plt.title(
        rf"CPW-CPW Coupling Capacitance ($\mathtt{{width}} = {width}\,\text{{µm}}$, $\mathtt{{cpw\_gap}} = {cpw_gap}\,\text{{µm}}$, $\epsilon_\text{{r}} = {ep_r}$)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
