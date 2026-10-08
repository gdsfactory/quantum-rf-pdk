"""SAX models using capacitance extracted from uniform CPW slices."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sax
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY, ε_0, μ_0
from qpdk.models.cpw import transmission_line_s_params
from qpdk.models.datasets.capacitance import check_maxwell
from qpdk.models.datasets.interpolation import GridInterpolator
from qpdk.models.datasets.metadata import QuantityKind
from qpdk.models.datasets.table import Dataset


def cpw_coupling_model(
    dataset: Dataset | Path | str = "cpw_coupling_palace",
) -> sax.Model:
    r"""Load a symmetric CPW dataset once and return a jittable four-port model.

    The dataset contains slice capacitances in F, swept over ``width``,
    ``cpw_gap`` and ``gap`` in µm. The generator's ``slice_length_um`` and
    substrate ``permittivity`` must be present in the provenance. Ports are
    ``o1`` lower-left, ``o2`` upper-left, ``o3`` upper-right, ``o4`` lower-right.
    The reference planes are the two ends of the uniform coupled section.

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
    if grid.axis_names != ("width", "cpw_gap", "gap") or len(grid.terminals) != 2:
        raise ValueError("Expected a two-conductor grid over width, cpw_gap and gap")
    if any(axis.unit != "um" for axis in grid.axes):
        raise ValueError("CPW geometry axes must be in um")
    check_maxwell(grid.values)
    if np.any(grid.values.sum(axis=-1) <= 0):
        raise ValueError("Both traces must have positive capacitance to ground")
    if not np.allclose(
        grid.values[..., 0, 0], grid.values[..., 1, 1], rtol=0.01, atol=0
    ):
        raise ValueError("The CPW model requires symmetric traces and ground rails")
    settings = data.metadata.provenance["settings"]
    slice_length = float(settings["slice_length_um"]) * 1e-6
    effective_permittivity = (1 + float(settings["permittivity"])) / 2
    if slice_length <= 0 or effective_permittivity <= 0:
        raise ValueError("Slice length and effective permittivity must be positive")
    lookup = GridInterpolator(grid)

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
