r"""Su-Schrieffer-Heeger (SSH) coupled-resonator chain models.

A chain of :math:`N` unit cells, each holding two lumped LC resonators (sites
A and B) with alternating intra-cell and inter-cell coupling capacitances,
realises the photonic SSH model of :cite:`kimQuantumElectrodynamicsTopological2021`.

The circuit maps to the SSH Hamiltonian in the weak-coupling limit
(:cite:`kimQuantumElectrodynamicsTopological2021`, App. B, Eq. (B1)):

.. math::

    C_\Sigma &= C_0 + C_v + C_w, \qquad
    \omega_0 = \frac{1}{\sqrt{L_0 C_\Sigma}} \\
    J &= \frac{\omega_0}{4}\left(\frac{C_v + C_w}{C_\Sigma}
        - \frac{M_v + M_w}{L_0}\right) \\
    \delta &= \frac{L_0 (C_v - C_w) - C_\Sigma (M_v - M_w)}
        {L_0 (C_v + C_w) - C_\Sigma (M_v + M_w)}

with intra-cell coupling :math:`J(1+\delta)` and inter-cell coupling
:math:`J(1-\delta)`. :math:`\delta > 0` is the trivial phase and
:math:`\delta < 0` the topological phase, which hosts two edge modes at
:math:`\omega_0` in an open chain.

Assumptions of the ideal SSH mapping, as stated in
:cite:`kimQuantumElectrodynamicsTopological2021` (App. B):

- Rotating-wave approximation.
- Weak coupling, :math:`C_v, C_w \ll C_\Sigma` and
  :math:`|M_v|, |M_w| \ll L_0`. This drops the :math:`k`-dependence of the
  on-site frequency :math:`\omega_0(k)`. That dependence breaks chiral
  symmetry and shifts the circuit bands from the ideal ones by a relative
  amount of order :math:`(C_v + C_w)^2 / C_\Sigma^2`.
- Nearest-neighbour coupling only.

The SAX circuit model (:func:`ssh_resonator_chain`) makes no such
approximation for the capacitive network, but it neglects mutual inductance
between neighbouring resonators.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sax
from jax.typing import ArrayLike

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.generic import capacitor, lc_resonator, short, tee


def ssh_parameters(
    inductance: ArrayLike,
    capacitance: ArrayLike,
    intra_cell_capacitance: ArrayLike,
    inter_cell_capacitance: ArrayLike,
    intra_cell_mutual_inductance: ArrayLike = 0.0,
    inter_cell_mutual_inductance: ArrayLike = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Map lumped SSH-chain circuit values to the SSH model parameters.

    Implements Eq. (B1) of :cite:`kimQuantumElectrodynamicsTopological2021`
    (see the module docstring), with :math:`L_0` = ``inductance``,
    :math:`C_0` = ``capacitance``, :math:`C_v` / :math:`C_w` the intra- and
    inter-cell coupling capacitances, and :math:`M_v` / :math:`M_w` the intra-
    and inter-cell mutual inductances.

    Args:
        inductance: Site inductance :math:`L_0` in H.
        capacitance: Site capacitance to ground :math:`C_0` in F.
        intra_cell_capacitance: Intra-cell coupling capacitance :math:`C_v` in F.
        inter_cell_capacitance: Inter-cell coupling capacitance :math:`C_w` in F.
        intra_cell_mutual_inductance: Intra-cell mutual inductance :math:`M_v` in H.
        inter_cell_mutual_inductance: Inter-cell mutual inductance :math:`M_w` in H.

    Returns:
        Tuple of the bare site angular frequency :math:`\omega_0` in rad/s,
        the mean coupling :math:`J` in rad/s and the dimerisation
        :math:`\delta` (dimensionless).
    """
    l0 = jnp.asarray(inductance)
    c_v = jnp.asarray(intra_cell_capacitance)
    c_w = jnp.asarray(inter_cell_capacitance)
    m_v = jnp.asarray(intra_cell_mutual_inductance)
    m_w = jnp.asarray(inter_cell_mutual_inductance)
    c_sigma = jnp.asarray(capacitance) + c_v + c_w
    omega_0 = 1 / jnp.sqrt(l0 * c_sigma)
    coupling = omega_0 / 4 * ((c_v + c_w) / c_sigma - (m_v + m_w) / l0)
    dimerisation = (l0 * (c_v - c_w) - c_sigma * (m_v - m_w)) / (
        l0 * (c_v + c_w) - c_sigma * (m_v + m_w)
    )
    return omega_0, coupling, dimerisation


def ssh_bands(
    k: ArrayLike, omega_0: ArrayLike, coupling: ArrayLike, dimerisation: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    r"""Ideal SSH band structure of an infinite chain.

    .. math::

        \omega_\pm(k) = \omega_0 \pm J \sqrt{2(1+\delta^2) + 2(1-\delta^2)\cos(k d)}

    See :cite:`kimQuantumElectrodynamicsTopological2021`, Eq. (2).

    Args:
        k: Dimensionless Bloch wavenumber :math:`k d` in rad, with :math:`d`
            the lattice constant.
        omega_0: Bare site angular frequency :math:`\omega_0`.
        coupling: Mean coupling :math:`J`, in the same units as ``omega_0``.
        dimerisation: Dimerisation :math:`\delta`.

    Returns:
        Tuple of the lower and upper band frequencies
        :math:`(\omega_-(k), \omega_+(k))`.
    """
    delta = jnp.asarray(dimerisation)
    half_width = jnp.asarray(coupling) * jnp.sqrt(
        2 * (1 + delta**2) + 2 * (1 - delta**2) * jnp.cos(jnp.asarray(k))
    )
    omega_0 = jnp.asarray(omega_0)
    return omega_0 - half_width, omega_0 + half_width


def ssh_band_edges(
    omega_0: ArrayLike, coupling: ArrayLike, dimerisation: ArrayLike
) -> jax.Array:
    r"""Ideal SSH band edges of an infinite chain.

    The two bands span :math:`\omega_0 \pm [2J|\delta|, 2J]`, so the middle
    gap is :math:`4J|\delta|` wide and centred on :math:`\omega_0`.

    Args:
        omega_0: Bare site angular frequency :math:`\omega_0`.
        coupling: Mean coupling :math:`J`, in the same units as ``omega_0``.
        dimerisation: Dimerisation :math:`\delta`.

    Returns:
        Array ``[lower band bottom, lower band top, upper band bottom, upper
        band top]``.
    """
    omega_0 = jnp.asarray(omega_0)
    outer = 2 * jnp.asarray(coupling)
    inner = outer * jnp.abs(jnp.asarray(dimerisation))
    return jnp.stack([
        omega_0 - outer,
        omega_0 - inner,
        omega_0 + inner,
        omega_0 + outer,
    ])


def ssh_circuit_band_edges(
    inductance: ArrayLike,
    capacitance: ArrayLike,
    intra_cell_capacitance: ArrayLike,
    inter_cell_capacitance: ArrayLike,
) -> jax.Array:
    r"""Exact band edges of an infinite capacitively coupled LC chain.

    Without mutual inductance, the circuit bands follow from Eq. (A16) of
    :cite:`kimQuantumElectrodynamicsTopological2021` beyond the
    weak-coupling limit:

    .. math::

        \omega_{\text{min}-} &= 1/\sqrt{L_0 (C_\Sigma + C_v + C_w)}, &
        \omega_{\text{max}-} &= 1/\sqrt{L_0 (C_\Sigma + |C_v - C_w|)}, \\
        \omega_{\text{min}+} &= 1/\sqrt{L_0 (C_\Sigma - |C_v - C_w|)}, &
        \omega_{\text{max}+} &= 1/\sqrt{L_0 C_0}.

    Args:
        inductance: Site inductance :math:`L_0` in H.
        capacitance: Site capacitance to ground :math:`C_0` in F.
        intra_cell_capacitance: Intra-cell coupling capacitance :math:`C_v` in F.
        inter_cell_capacitance: Inter-cell coupling capacitance :math:`C_w` in F.

    Returns:
        Array of angular frequencies in rad/s, ordered like
        :func:`ssh_band_edges`.
    """
    l0 = jnp.asarray(inductance)
    c_v = jnp.asarray(intra_cell_capacitance)
    c_w = jnp.asarray(inter_cell_capacitance)
    c_sigma = jnp.asarray(capacitance) + c_v + c_w
    effective_capacitance = jnp.stack([
        c_sigma + c_v + c_w,
        c_sigma + jnp.abs(c_v - c_w),
        c_sigma - jnp.abs(c_v - c_w),
        c_sigma - c_v - c_w,
    ])
    return 1 / jnp.sqrt(l0 * effective_capacitance)


def ssh_chain_spectrum(
    n_cells: int, omega_0: ArrayLike, coupling: ArrayLike, dimerisation: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    r"""Normal modes of a finite open SSH chain in the tight-binding model.

    Diagonalises the :math:`2N \times 2N` single-particle Hamiltonian

    .. math::

        H = \omega_0 \mathbb{1} + \sum_{n} \left[ J(1+\delta)
            |n, \text{A}\rangle\langle n, \text{B}| + J(1-\delta)
            |n, \text{B}\rangle\langle n+1, \text{A}| + \text{h.c.} \right]

    of :cite:`kimQuantumElectrodynamicsTopological2021`, Eq. (1). Sites are
    ordered :math:`(1,\text{A}), (1,\text{B}), (2,\text{A}), \ldots`.

    For :math:`\delta < 0` and :math:`N` large enough that
    :math:`((1+\delta)/(1-\delta))^N \ll 1`, two eigenvalues lie inside the
    middle gap at :math:`\omega_0`, with eigenvectors localised on the end
    sites. For :math:`\delta > 0` the middle gap is empty.

    Args:
        n_cells: Number of unit cells :math:`N`.
        omega_0: Bare site angular frequency :math:`\omega_0`.
        coupling: Mean coupling :math:`J`, in the same units as ``omega_0``.
        dimerisation: Dimerisation :math:`\delta`.

    Returns:
        Tuple of ascending eigenfrequencies (shape ``(2N,)``) and the
        corresponding eigenvectors as columns (shape ``(2N, 2N)``).
    """
    hoppings = jnp.asarray(
        jnp.where(
            jnp.arange(2 * n_cells - 1) % 2 == 0,
            coupling * (1 + dimerisation),
            coupling * (1 - dimerisation),
        )
    )
    hamiltonian = (
        omega_0 * jnp.eye(2 * n_cells)
        + jnp.diag(hoppings, k=1)
        + jnp.diag(hoppings, k=-1)
    )
    return jnp.linalg.eigh(hamiltonian)


def _ssh_netlist(n_cells: int, tap_sites: tuple[int, ...]) -> sax.Netlist:
    """Build the SAX netlist of an SSH chain with optional site taps.

    Args:
        n_cells: Number of unit cells.
        tap_sites: Site indices with a coupling tap, in port order.

    Returns:
        SAX netlist with ports ``o1``, ``o2`` and one more per tap.
    """
    n_sites = 2 * n_cells
    instances: dict[str, str] = {
        "port_in": "port_capacitor",
        "port_out": "port_capacitor",
    }
    connections = {"port_in,o2": "node0,o1", f"node{n_sites - 1},o2": "port_out,o1"}
    ports = {"o1": "port_in,o1", "o2": "port_out,o2"}
    for site in range(n_sites):
        instances[f"node{site}"] = "tee"
        instances[f"site{site}"] = "site"
        instances[f"ground{site}"] = "short"
        connections[f"site{site},o2"] = f"ground{site},o1"
        if site in tap_sites:
            # A second tee splits the site branch into the resonator and the tap.
            instances[f"tap_node{site}"] = "tee"
            instances[f"tap{site}"] = "tap_capacitor"
            connections[f"node{site},o3"] = f"tap_node{site},o1"
            connections[f"tap_node{site},o2"] = f"site{site},o1"
            connections[f"tap_node{site},o3"] = f"tap{site},o1"
            ports[f"o{3 + tap_sites.index(site)}"] = f"tap{site},o2"
        else:
            connections[f"node{site},o3"] = f"site{site},o1"
        if site < n_sites - 1:
            coupler = "intra" if site % 2 == 0 else "inter"
            instances[f"{coupler}{site}"] = f"{coupler}_capacitor"
            connections[f"node{site},o2"] = f"{coupler}{site},o1"
            connections[f"{coupler}{site},o2"] = f"node{site + 1},o1"
    return {
        "instances": {
            name: {"component": component} for name, component in instances.items()
        },
        "connections": connections,
        "ports": ports,
    }


def ssh_resonator_chain(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    n_cells: int = 4,
    inductance: sax.FloatArrayLike = 1.9e-9,
    capacitance: sax.FloatArrayLike = 253e-15,
    intra_cell_capacitance: sax.FloatArrayLike = 33e-15,
    inter_cell_capacitance: sax.FloatArrayLike = 17e-15,
    port_capacitance: sax.FloatArrayLike = 17e-15,
    tap_sites: tuple[int, ...] = (),
    tap_capacitance: sax.FloatArrayLike = 1e-15,
) -> sax.SDict:
    r"""SAX model of an SSH chain of capacitively coupled lumped LC resonators.

    Each of the :math:`2N` sites is a parallel LC resonator
    (:func:`~qpdk.models.generic.lc_resonator`, :math:`L_0 \parallel C_0`) to
    ground. Neighbouring sites are coupled by series capacitors
    (:func:`~qpdk.models.generic.capacitor`), alternating :math:`C_v`
    (intra-cell, A→B) and :math:`C_w` (inter-cell, B→A). The end sites couple
    to ports ``o1`` and ``o2`` through ``port_capacitance``.

    .. svgbob::

              Cp      Cv      Cw      Cv            Cp
        o1 ───||──┬───||──┬───||──┬───||──┬ ··· ┬───||─── o2
                  │       │       │       │     │
                 LC      LC      LC      LC    LC
                  │       │       │       │     │
                 GND     GND     GND     GND   GND
                 1,A     1,B     2,A     2,B   N,B

    Optional taps couple chosen sites to extra ports ``o3``, ``o4``, …
    through ``tap_capacitance``, e.g. for a qubit coupled to that site. Taps
    are numbered in the order given in ``tap_sites``.

    The defaults are the test-structure values of
    :cite:`kimQuantumElectrodynamicsTopological2021` (App. C, trivial phase),
    except for ``n_cells``, which matches the layout cell (the paper uses 8). Swap ``intra_cell_capacitance`` and
    ``inter_cell_capacitance`` for the topological phase.

    An end site lacks one coupling capacitor of the bulk sites. Setting
    ``port_capacitance`` equal to that missing capacitor (``inter_cell_capacitance``,
    the default) gives the end sites the same total capacitance
    :math:`C_\Sigma` as the bulk, so the edge modes stay at :math:`\omega_0`.
    Taps load their sites by ``tap_capacitance`` and are not compensated.

    The model neglects mutual inductance between sites. Compare with
    :func:`ssh_chain_spectrum` and :func:`ssh_band_edges` via
    :func:`ssh_parameters` for the ideal SSH picture, or with
    :func:`ssh_circuit_band_edges` for the exact infinite-chain circuit bands.

    Args:
        f: Array of frequency points in Hz.
        n_cells: Number of unit cells :math:`N`. Sets the netlist structure,
            so it must be a concrete integer.
        inductance: Site inductance :math:`L_0` in H.
        capacitance: Site capacitance to ground :math:`C_0` in F.
        intra_cell_capacitance: Intra-cell coupling capacitance :math:`C_v` in F.
        inter_cell_capacitance: Inter-cell coupling capacitance :math:`C_w` in F.
        port_capacitance: Coupling capacitance from each end site to its port in F.
        tap_sites: Indices of the sites (0 to :math:`2N-1`, ordered
            :math:`(1,\text{A}), (1,\text{B}), \ldots`) with a tap port.
            Must be concrete and unique.
        tap_capacitance: Coupling capacitance of each tap in F.

    Returns:
        sax.SDict: S-parameters with ports ``o1``, ``o2`` and ``o3``… for
        the taps.

    Raises:
        ValueError: If ``n_cells`` < 1 or ``tap_sites`` holds non-integer,
            duplicate or out-of-range site indices.
    """
    # Static topology, so plain numpy: jnp would trace it under an outer jit.
    sites = np.atleast_1d(np.asarray(tap_sites)).tolist()
    if not all(float(site).is_integer() for site in sites):
        msg = f"tap_sites must be integer site indices, got {tap_sites}."
        raise ValueError(msg)
    tap_sites = tuple(int(site) for site in sites)
    if n_cells < 1:
        msg = f"n_cells must be at least 1, got {n_cells}."
        raise ValueError(msg)
    if len(set(tap_sites)) != len(tap_sites) or any(
        not 0 <= site < 2 * n_cells for site in tap_sites
    ):
        msg = f"tap_sites must be unique site indices in [0, {2 * n_cells}), got {tap_sites}."
        raise ValueError(msg)

    def site(f: sax.FloatArrayLike = DEFAULT_FREQUENCY) -> sax.SDict:
        # A closure instead of a partial: sax would pass the static
        # ``grounded`` default of ``lc_resonator`` as an unhashable array.
        return lc_resonator(f=f, capacitance=capacitance, inductance=inductance)

    models: dict[str, sax.Model] = {
        "tee": tee,
        "short": short,
        "site": site,
        "intra_capacitor": partial(capacitor, capacitance=intra_cell_capacitance),
        "inter_capacitor": partial(capacitor, capacitance=inter_cell_capacitance),
        "port_capacitor": partial(capacitor, capacitance=port_capacitance),
        "tap_capacitor": partial(capacitor, capacitance=tap_capacitance),
    }
    circuit, _ = sax.circuit(
        _ssh_netlist(n_cells, tap_sites), models, backend="filipsson_gunnar"
    )
    return circuit(f=jnp.asarray(f))
