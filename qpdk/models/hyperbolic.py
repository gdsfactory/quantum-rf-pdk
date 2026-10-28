r"""SAX models for hyperbolic resonator lattices.

Implements the edge-resonator scheme of Kollár et al.
:cite:`kollarHyperbolicLatticesCircuit2019`: identical CPW resonators sit
on the edges of a {p,q} tiling and meet at ``q``-pin capacitive couplers on
its vertices. Near the full-wave (second) mode of the resonators, whose
ends oscillate in phase, the lattice obeys

.. math::

    H = \omega_0 \sum_i a_i^\dagger a_i - t \sum_{\langle ij \rangle}
        \left(a_i^\dagger a_j + \text{h.c.}\right), \qquad t < 0,

on the line graph of the tiling, so the spectrum is
:math:`\omega_0 - t\,\lambda(A_\text{L})` with a flat band at
:math:`\omega_0 - 2|t|` (see :func:`tight_binding_hamiltonian`). This needs
every pair of resonators at a coupler to be coupled equally, which the
disk coupler does exactly only for :math:`q \le 3`.

:func:`hyperbolic_lattice` does not use the tight-binding approximation: it
solves the full circuit of transmission lines, couplers and boundary stubs.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec

from qpdk.hyperbolic import hyperbolic_tiling
from qpdk.models.capacitor import plate_capacitor_capacitance_analytical
from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.cpw import cpw_ep_r_from_cross_section, cpw_z0_from_cross_section
from qpdk.models.waveguides import straight, straight_shorted

_BATCH_SIZE = 32  # frequencies solved at once by the lattice model

__all__ = [
    "hyperbolic_cpw_path",
    "hyperbolic_edge_resonator",
    "hyperbolic_lattice",
    "hyperbolic_vertex_coupler",
    "hyperbolic_vertex_coupler_capacitances",
    "tight_binding_hamiltonian",
]


def tight_binding_hamiltonian(
    adjacency: jax.typing.ArrayLike, t: float, onsite: float = 0.0
) -> jax.Array:
    r"""Tight-binding Hamiltonian :math:`H = \omega_0 I - t A`.

    Follows the sign convention of Kollár et al.
    :cite:`kollarHyperbolicLatticesCircuit2019`, Eq. (1). Capacitively
    coupled CPW resonators have :math:`t < 0`, so on a line graph (whose
    adjacency spectrum is bounded below by :math:`-2`) the flat band sits at
    the bottom, :math:`\omega_0 + 2t = \omega_0 - 2|t|`.

    Args:
        adjacency: Symmetric (weighted) adjacency matrix, e.g.
            :attr:`qpdk.hyperbolic.HyperbolicTiling.line_graph_adjacency`.
        t: Hopping rate.
        onsite: On-site energy :math:`\omega_0`.

    Returns:
        The Hamiltonian matrix.
    """
    a = jnp.asarray(adjacency, dtype=float)
    return onsite * jnp.eye(a.shape[-1]) - t * a


def _cayley(x: jax.Array) -> jax.Array:
    r"""Cayley transform :math:`(I + X)^{-1}(I - X)` over the last two axes.

    It is its own inverse and converts between S-parameters and normalised
    admittance: :math:`Z_0 Y = \mathcal{C}(S)` and :math:`S = \mathcal{C}(Z_0 Y)`.

    Returns:
        The transformed matrices.
    """
    eye = jnp.eye(x.shape[-1])
    return jnp.linalg.solve(eye + x, eye - x)


def hyperbolic_vertex_coupler_capacitances(
    q: int = 3,
    pin_radius: float = 40.0,
    pin_gap: float = 4.0,
    ground_gap: float = 6.0,
    cross_section: CrossSectionSpec = "cpw",
) -> tuple[jax.Array, jax.Array]:
    r"""Pin-to-pin and pin-to-ground capacitance of a ``q``-pin vertex coupler.

    The coupler is a metal disk of radius ``pin_radius`` cut into ``q``
    sectors by radial gaps of width ``pin_gap`` and surrounded by a ground
    clearance of width ``ground_gap`` (see
    :func:`qpdk.cells.hyperbolic.hyperbolic_vertex_coupler`). Each gap is
    treated as a pair of coplanar pads of length ``pin_radius`` and width
    ``pin_radius / 2`` with :func:`~qpdk.models.capacitor.plate_capacitor_capacitance_analytical`,
    and each sector's arc as a pad of the same width facing the ground plane.
    Only neighbouring sectors are coupled; for :math:`q > 3` the weaker
    coupling across the disk is neglected, so the hopping between edges is
    not uniform and the line-graph picture holds only approximately.

    Returns:
        ``(C_c, C_g)`` in farads: coupling between neighbouring pins and
        capacitance of each pin to ground.
    """
    ep_r = cpw_ep_r_from_cross_section(cross_section)
    width = pin_radius / 2
    c_c = plate_capacitor_capacitance_analytical(
        length=pin_radius, width=width, gap=pin_gap, ep_r=ep_r
    )
    c_g = plate_capacitor_capacitance_analytical(
        length=2 * math.pi * pin_radius / q, width=width, gap=ground_gap, ep_r=ep_r
    )
    return c_c, c_g


def _coupler_capacitance_matrix(
    q: int, c_c: jax.Array | float, c_g: jax.Array | float
) -> jax.Array:
    """Maxwell capacitance matrix of ``q`` pins on a ring."""
    # For q = 2 both gaps lie between the same two pins, so they add up.
    neighbours = jnp.roll(jnp.eye(q), 1, axis=1) + jnp.roll(jnp.eye(q), -1, axis=1)
    return (c_g + c_c * neighbours.sum(axis=1)) * jnp.eye(q) - c_c * neighbours


def hyperbolic_vertex_coupler(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    q: int = 3,
    pin_radius: float = 40.0,
    pin_gap: float = 4.0,
    ground_gap: float = 6.0,
    lead_length: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""S-parameters of a ``q``-pin capacitive vertex coupler.

    The pins form a lumped capacitance network with Maxwell matrix
    :math:`C` (:func:`hyperbolic_vertex_coupler_capacitances`). Its nodal
    admittance :math:`Y = j\omega C` gives

    .. math::

        S = (I - Z_0 Y)(I + Z_0 Y)^{-1},

    referenced to the CPW impedance :math:`Z_0` of ``cross_section`` so the
    pins connect directly to :func:`~qpdk.models.waveguides.straight` lines.
    The matched leads then multiply :math:`S_{ij}` by their transmission
    :math:`t^2`, moving the reference planes to the layout ports.

    Args:
        f: Frequency in Hz.
        q: Number of pins.
        pin_radius: Radius of the metal disk in µm.
        pin_gap: Width of the radial gaps between pins in µm.
        ground_gap: Clearance between the disk and the ground plane in µm.
        lead_length: Length of the CPW lead on each pin in µm.
        cross_section: Cross-section of the leads.

    Returns:
        S-parameters with ports ``o1`` … ``oq``.
    """
    q = int(q)  # pyrefly: ignore[unnecessary-type-conversion]
    f_arr = jnp.asarray(f)
    c_c, c_g = hyperbolic_vertex_coupler_capacitances(
        q, pin_radius, pin_gap, ground_gap, cross_section
    )
    capacitance = _coupler_capacitance_matrix(q, c_c, c_g)
    z0 = cpw_z0_from_cross_section(cross_section, f_arr)[..., None, None]
    z0_y = 1j * 2 * jnp.pi * f_arr[..., None, None] * capacitance * z0
    t = straight(f=f_arr, length=lead_length, cross_section=cross_section)["o2", "o1"]
    s = _cayley(z0_y) * (t**2)[..., None, None]
    names = [f"o{k + 1}" for k in range(q)]
    return {(names[i], names[j]): s[..., i, j] for i in range(q) for j in range(q)}


def hyperbolic_cpw_path(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (100.0, 0.0)),
    radius: float = 100.0,
    shorted: bool = False,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""S-parameters of :func:`qpdk.cells.hyperbolic.hyperbolic_cpw_path`.

    A straight CPW of the centreline length: the polyline length minus
    :math:`2r\tan(\theta/2) - r\theta` at every corner turning by
    :math:`\theta`.

    Args:
        f: Frequency in Hz.
        points: Waypoints in µm.
        radius: Corner radius in µm.
        shorted: Short the far end, leaving only port ``o1``.
        cross_section: CPW cross-section.

    Returns:
        S-parameters with ports ``o1`` and, unless ``shorted``, ``o2``.
    """
    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turns = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)
    length = float(
        np.linalg.norm(seg, axis=1).sum()
        - (2 * radius * np.tan(turns / 2) - radius * turns).sum()
    )
    model = straight_shorted if shorted else straight
    return model(f=f, length=length, cross_section=cross_section)


def hyperbolic_edge_resonator(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: float = 4000.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    """S-parameters of a meandered hyperbolic-lattice edge resonator.

    The meander of :func:`qpdk.cells.hyperbolic.hyperbolic_edge_resonator`
    is modelled as a straight CPW of the same centreline length; bend
    corrections are neglected.

    Args:
        f: Frequency in Hz.
        length: Centreline length in µm.
        cross_section: CPW cross-section.

    Returns:
        S-parameters with ports ``o1`` and ``o2``.
    """
    return straight(f=f, length=length, cross_section=cross_section)


def hyperbolic_lattice(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    p: int = 7,
    q: int = 3,
    shells: int = 1,
    resonator_length: sax.FloatArrayLike = 7500.0,
    stub_length: float = 400.0,
    probe_pins: tuple[int, ...] = (11, 0),
    pin_radius: float = 40.0,
    pin_gap: float = 4.0,
    ground_gap: float = 6.0,
    lead_length: float = 20.0,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SDict:
    r"""Circuit model of a Kollár-type hyperbolic lattice of edge resonators.

    Builds the circuit of :func:`qpdk.cells.hyperbolic.hyperbolic_lattice`
    from the tiling rather than from the layout:

    - one :func:`~qpdk.models.waveguides.straight` of ``resonator_length``
      per tiling edge, open at both ends through the couplers;
    - one :func:`hyperbolic_vertex_coupler` per vertex, wired by
      :attr:`qpdk.hyperbolic.HyperbolicTiling.pins`;
    - a shorted CPW stub of ``stub_length`` on every free boundary pin
      (a :math:`\lambda/4` resonator far above the band) so boundary
      resonators see the same end capacitance as interior ones;
    - probe ports in place of the stubs on the free pins listed in
      ``probe_pins``. Their reference plane is the coupler pin, so the feed
      lines and launchers of the layout are not included.

    The circuit is solved exactly by nodal analysis rather than with
    :func:`sax.evaluate_circuit_fg`, whose JIT compilation does not scale to
    a lattice of this size. Each element's S-matrix is converted to an
    admittance matrix referenced to the CPW impedance :math:`Z_0`, the
    elements are stamped into the nodal admittance matrix :math:`Y` with one
    node per coupler pin, the nodes without a probe are eliminated by a
    Schur complement and the result is converted back to
    :math:`S = (I - Z_0 Y_\text{p})(I + Z_0 Y_\text{p})^{-1}`. This is the
    same S-matrix SAX produces for the netlist.

    Args:
        f: Frequency in Hz.
        p: Polygon side count of the tiling.
        q: Polygons per vertex, i.e. coupler pins.
        shells: Polygon rings around the central polygon.
        resonator_length: Length of every edge resonator in µm.
        stub_length: Length of the shorted boundary stubs in µm.
        probe_pins: Indices into
            :attr:`~qpdk.hyperbolic.HyperbolicTiling.free_pins` that become
            ports ``o1``, ``o2``, … in this order.
        pin_radius: Coupler disk radius in µm.
        pin_gap: Coupler radial gap in µm.
        ground_gap: Coupler ground clearance in µm.
        lead_length: Coupler lead length in µm. It is part of
            ``resonator_length`` and ``stub_length``, and moves the probe
            reference planes to the end of the probe pins' leads, like
            :func:`hyperbolic_vertex_coupler`.
        cross_section: CPW cross-section of resonators, stubs and probes.

    Returns:
        S-parameters between the probe ports.

    The structural arguments ``p``, ``q``, ``shells`` and ``probe_pins`` fix
    the circuit topology, so they are converted to Python integers and must
    be concrete; all other arguments may be traced by :func:`jax.jit`.

    Raises:
        ValueError: If ``probe_pins`` is empty, repeats a pin or is out of range.
    """
    # SAX passes settings as JAX arrays, which are unhashable.
    p, q, shells = int(p), int(q), int(shells)  # pyrefly: ignore[unnecessary-type-conversion]
    probe_pins = tuple(int(k) for k in np.ravel(probe_pins))
    tiling = hyperbolic_tiling(p, q, shells)
    free = tiling.free_pins
    if not probe_pins:
        raise ValueError("probe_pins must name at least one free pin.")
    if len(set(probe_pins)) != len(probe_pins) or not all(
        0 <= k < len(free) for k in probe_pins
    ):
        raise ValueError(
            f"probe_pins must be distinct indices below {len(free)}, got {probe_pins}."
        )

    f_in = jnp.asarray(f, dtype=float)
    f_arr = jnp.ravel(f_in)
    z0 = cpw_z0_from_cross_section(cross_section, f_arr)

    # Pin k of coupler v is node v * q + k, renumbered so the probes come first.
    n_nodes = tiling.num_vertices * q
    probes = [free[k][0] * q + free[k][1] for k in probe_pins]
    order = probes + [n for n in range(n_nodes) if n not in probes]
    position = np.empty(n_nodes, dtype=int)
    position[order] = np.arange(n_nodes)
    n_p = len(probes)

    c_c, c_g = hyperbolic_vertex_coupler_capacitances(
        q, pin_radius, pin_gap, ground_gap, cross_section
    )
    c_nodes = jnp.kron(
        jnp.eye(tiling.num_vertices), _coupler_capacitance_matrix(q, c_c, c_g)
    )[np.ix_(order, order)]
    ends = position[
        np.array([
            [i * q + tiling.pins[i][1].index(e), j * q + tiling.pins[j][1].index(e)]
            for e, (i, j) in enumerate(tiling.edges.tolist())
        ])
    ]
    stubs = position[[v * q + k for v, k in free if v * q + k not in probes]]

    # Element admittances from their S-matrices, Z0 Y = C(S).
    s_line, line_ports = sax.sdense(
        straight(f=f_arr, length=resonator_length, cross_section=cross_section)
    )
    order_line = [line_ports["o1"], line_ports["o2"]]
    line = _cayley(s_line[..., order_line, :][..., order_line]) / z0[..., None, None]
    s_stub = straight_shorted(f=f_arr, length=stub_length, cross_section=cross_section)
    y_stub = _cayley(s_stub["o1", "o1"][..., None, None])[..., 0, 0] / z0

    def solve_one(args: tuple[jax.Array, ...]) -> jax.Array:
        """Probe S-matrix at one frequency; mapped so memory stays O(N²)."""
        omega_k, line_k, y_stub_k, z0_k = args
        y = 1j * omega_k * c_nodes
        for a in range(2):
            for b in range(2):
                y = y.at[ends[:, a], ends[:, b]].add(line_k[a, b])
        y = y.at[stubs, stubs].add(y_stub_k)
        y_ports = y[:n_p, :n_p] - y[:n_p, n_p:] @ jnp.linalg.solve(
            y[n_p:, n_p:], y[n_p:, :n_p]
        )
        return _cayley(z0_k * y_ports)

    s = jax.lax.map(
        solve_one, (2 * jnp.pi * f_arr, line, y_stub, z0), batch_size=_BATCH_SIZE
    )
    t = straight(f=f_arr, length=lead_length, cross_section=cross_section)["o2", "o1"]
    s = s * (t**2)[:, None, None]
    s = s.reshape(*f_in.shape, len(probes), len(probes))
    names = [f"o{n + 1}" for n in range(len(probes))]
    return {
        (names[i], names[j]): s[..., i, j]
        for i in range(len(probes))
        for j in range(len(probes))
    }
