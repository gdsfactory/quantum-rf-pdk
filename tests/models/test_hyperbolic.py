r"""Tests for qpdk.models.hyperbolic.

The lattice tests check the full-wave circuit against the tight-binding model
of :cite:`kollarHyperbolicLatticesCircuit2019`: a closed-circuit nodal
analysis of the default {7,3} flake gives one resonance per edge resonator
near the full-wave (second-harmonic) mode, with the line-graph spectrum,
hopping :math:`t < 0` and a flat band at :math:`\omega_0 - 2|t|`. The
lattice model is checked against a SAX netlist of the same circuit and
against those eigenfrequencies through a probe.
"""

from typing import ClassVar, final

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
from scipy.signal import find_peaks

from qpdk.cells.hyperbolic import hyperbolic_cpw_path as cpw_path_cell
from qpdk.hyperbolic import hyperbolic_tiling
from qpdk.models.constants import c_0
from qpdk.models.cpw import (
    cpw_parameters,
    cpw_z0_from_cross_section,
    get_cpw_dimensions,
)
from qpdk.models.generic import electrical_short
from qpdk.models.hyperbolic import (
    _cayley,
    hyperbolic_cpw_path,
    hyperbolic_edge_resonator,
    hyperbolic_lattice,
    hyperbolic_vertex_coupler,
    hyperbolic_vertex_coupler_capacitances,
    tight_binding_hamiltonian,
)
from qpdk.models.waveguides import straight

from .base import BaseModelTestSuite

RESONATOR_LENGTH = 7500.0
STUB_LENGTH = 400.0
LEAD_LENGTH = 20.0
# Window around the full-wave mode of the default resonators (bare 16.24 GHz,
# pulled down by the coupler capacitance), clear of the open-line poles.
F_MIN, F_MAX = 15.5e9, 16.0e9


@final
class TestHyperbolicVertexCoupler(BaseModelTestSuite):
    """Generic S-matrix checks for the three-pin coupler."""

    model_function = staticmethod(hyperbolic_vertex_coupler)
    expected_ports: ClassVar[set[str]] = {"o1", "o2", "o3"}


@final
class TestHyperbolicEdgeResonator(BaseModelTestSuite):
    """Generic S-matrix checks for the edge resonator."""

    model_function = staticmethod(hyperbolic_edge_resonator)
    expected_ports: ClassVar[set[str]] = {"o1", "o2"}


@final
class TestHyperbolicLatticeSParameters(BaseModelTestSuite):
    """Generic S-matrix checks for the default two-probe lattice."""

    model_function = staticmethod(hyperbolic_lattice)
    expected_ports: ClassVar[set[str]] = {"o1", "o2"}


def test_coupler_capacitances_positive() -> None:
    """Coupling and ground capacitances are positive and scale with pin size."""
    c_c, c_g = hyperbolic_vertex_coupler_capacitances()
    assert c_c > 0
    assert c_g > 0
    c_c_big, _ = hyperbolic_vertex_coupler_capacitances(pin_radius=80.0)
    assert c_c_big > c_c


def test_two_pin_coupler_counts_both_gaps() -> None:
    """With ``q=2`` both radial gaps lie between the same two pins."""
    c_c, c_g = hyperbolic_vertex_coupler_capacitances(q=2)
    s = hyperbolic_vertex_coupler(f=5e9, q=2, lead_length=0.0)
    z0_y = 1j * 2 * np.pi * 5e9 * cpw_z0_from_cross_section("cpw", 5e9)
    expected = _cayley(
        z0_y * jnp.array([[c_g + 2 * c_c, -2 * c_c], [-2 * c_c, c_g + 2 * c_c]])
    )
    np.testing.assert_allclose(s["o2", "o1"], expected[1, 0], rtol=1e-9)


def test_coupler_leads_shift_reference_planes() -> None:
    """The leads multiply every S-parameter by the lead transmission squared."""
    f = jnp.array([5e9, 16e9])
    bare = hyperbolic_vertex_coupler(f=f, lead_length=0.0)
    t = straight(f=f, length=20.0)["o2", "o1"]
    with_leads = hyperbolic_vertex_coupler(f=f, lead_length=20.0)
    np.testing.assert_allclose(with_leads["o2", "o1"], bare["o2", "o1"] * t**2)


def test_cpw_path_model_matches_layout_length() -> None:
    """The path model is a straight of the layout's rounded centreline length."""
    points = ((0.0, 0.0), (300.0, 0.0), (500.0, 400.0), (500.0, 900.0))
    length = cpw_path_cell(points=points, radius=100.0).info["length"]
    f = jnp.array([5e9, 16e9])
    s = hyperbolic_cpw_path(f=f, points=points, radius=100.0)
    np.testing.assert_allclose(
        s["o2", "o1"], straight(f=f, length=length)["o2", "o1"], rtol=1e-4
    )
    shorted = hyperbolic_cpw_path(f=f, points=points[:2], shorted=True)
    assert set(shorted) == {("o1", "o1")}


def test_lattice_keeps_frequency_shape() -> None:
    """A 2D frequency grid returns S-parameters of the same shape."""
    f = jnp.linspace(F_MIN, F_MAX, 6).reshape(2, 3)
    s = hyperbolic_lattice(f=f)
    assert s["o2", "o1"].shape == (2, 3)
    np.testing.assert_allclose(
        s["o2", "o1"].ravel(), hyperbolic_lattice(f=f.ravel())["o2", "o1"]
    )


@pytest.mark.parametrize("q", [3, 4, 5])
def test_coupler_is_lossless_and_reciprocal(q: int) -> None:
    """A lumped capacitance network is lossless and reciprocal."""
    f = jnp.array([5e9, 16e9])
    s = hyperbolic_vertex_coupler(f=f, q=q)
    pins = range(1, q + 1)
    matrix = jnp.stack(
        [jnp.stack([s[f"o{i}", f"o{j}"] for j in pins], -1) for i in pins], -2
    )
    np.testing.assert_allclose(matrix, jnp.swapaxes(matrix, -1, -2), atol=1e-12)
    product = jnp.einsum("fki,fkj->fij", matrix.conj(), matrix)
    np.testing.assert_allclose(
        product, np.broadcast_to(np.eye(q), product.shape), atol=1e-9
    )


@pytest.mark.parametrize("probe_pins", [(), (0, 0), (21,)])
def test_lattice_rejects_bad_probe_pins(probe_pins: tuple[int, ...]) -> None:
    """Probe pins must be distinct, in range and non-empty."""
    with pytest.raises(ValueError, match="probe_pins"):
        hyperbolic_lattice(f=jnp.array([16e9]), probe_pins=probe_pins)


def test_tight_binding_flat_band_is_lowest() -> None:
    r"""With :math:`t < 0` the flat band sits at :math:`\omega_0 - 2|t|`, below the rest."""
    tiling = hyperbolic_tiling(7, 3, 1)
    t, onsite = -0.1, 8.0
    energies = jnp.linalg.eigvalsh(
        tight_binding_hamiltonian(tiling.line_graph_adjacency, t, onsite)
    )
    n_flat = tiling.num_edges - tiling.num_vertices
    assert energies[0] == pytest.approx(onsite - 2 * abs(t))
    assert energies[n_flat - 1] == pytest.approx(energies[0])
    assert energies[n_flat] > energies[0] + 1e-6


def test_lattice_jits_over_frequency_and_lengths() -> None:
    """Continuous arguments may be traced; the result matches eager evaluation."""
    f = jnp.linspace(F_MIN, F_MAX, 5)

    @jax.jit
    def s21(f: jax.Array, length: jax.Array) -> jax.Array:
        return hyperbolic_lattice(f=f, resonator_length=length)["o2", "o1"]

    np.testing.assert_allclose(
        s21(f, RESONATOR_LENGTH),
        hyperbolic_lattice(f=f, resonator_length=RESONATOR_LENGTH)["o2", "o1"],
        atol=1e-9,
    )


def test_lattice_in_sax_circuit_with_array_settings() -> None:
    """SAX passes instance settings as arrays; structural ones must still work."""
    circuit, _ = sax.circuit(
        netlist={
            "instances": {
                "lattice": {
                    "component": "hyperbolic_lattice",
                    "settings": {"p": 7, "q": 3, "shells": 1, "probe_pins": [11, 0]},
                }
            },
            "ports": {"o1": "lattice,o1", "o2": "lattice,o2"},
        },
        models={"hyperbolic_lattice": hyperbolic_lattice},
    )
    f = jnp.linspace(F_MIN, F_MAX, 3)
    s = circuit(
        f=f,
        lattice={
            "p": jnp.array(7),
            "q": jnp.array(3),
            "shells": jnp.array(1),
            "probe_pins": jnp.array([11, 0]),
        },
    )
    np.testing.assert_allclose(
        s["o2", "o1"], hyperbolic_lattice(f=f)["o2", "o1"], atol=1e-9
    )


def _nodal_susceptance(f: float) -> np.ndarray:
    """Nodal susceptance matrix of the closed default lattice (no probes).

    One node per coupler pin. Lines contribute the exact transmission-line
    admittance matrix, couplers ``ω C`` and boundary pins a shorted stub.
    The circuit resonates where an eigenvalue crosses zero.

    Returns:
        Real symmetric susceptance matrix in siemens.
    """
    tiling = hyperbolic_tiling(7, 3, 1)
    q = tiling.q
    width, gap = get_cpw_dimensions("cpw")
    ep_eff, z0 = (float(np.real(x)) for x in cpw_parameters(width, gap))
    c_c, c_g = hyperbolic_vertex_coupler_capacitances()
    omega = 2 * np.pi * f
    beta = omega * np.sqrt(ep_eff) / c_0
    theta = beta * RESONATOR_LENGTH * 1e-6
    theta_stub = beta * STUB_LENGTH * 1e-6

    size = tiling.num_vertices * q
    b = np.zeros((size, size))
    for v in range(tiling.num_vertices):
        for i in range(q):
            b[v * q + i, v * q + i] += omega * (c_g + 2 * c_c)
            for j in ((i + 1) % q, (i - 1) % q):
                b[v * q + i, v * q + j] -= omega * c_c
    for e, (i, j) in enumerate(tiling.edges.tolist()):
        n_i = i * q + tiling.pins[i][1].index(e)
        n_j = j * q + tiling.pins[j][1].index(e)
        b[n_i, n_i] -= 1 / (np.tan(theta) * z0)
        b[n_j, n_j] -= 1 / (np.tan(theta) * z0)
        b[n_i, n_j] += 1 / (np.sin(theta) * z0)
        b[n_j, n_i] += 1 / (np.sin(theta) * z0)
    for v, k in tiling.free_pins:
        b[v * q + k, v * q + k] -= 1 / (np.tan(theta_stub) * z0)
    return b


@pytest.fixture(scope="module")
def nodal_modes() -> np.ndarray:
    """Eigenfrequencies of the closed lattice, repeated by degeneracy."""
    fs = np.linspace(F_MIN, F_MAX, 1001)
    negative = np.array([
        np.sum(np.linalg.eigvalsh(_nodal_susceptance(f)) < 0) for f in fs
    ])
    steps = np.nonzero(np.diff(negative))[0]
    return np.sort(
        np.concatenate([
            np.full(abs(negative[i + 1] - negative[i]), (fs[i] + fs[i + 1]) / 2)
            for i in steps
        ])
    )


def test_full_wave_spectrum_matches_line_graph(nodal_modes: np.ndarray) -> None:
    """One mode per resonator, ordered like the line-graph TB with ``t < 0``."""
    tiling = hyperbolic_tiling(7, 3, 1)
    assert len(nodal_modes) == tiling.num_edges

    lam = np.sort(np.linalg.eigvalsh(tiling.line_graph_adjacency))
    # H = ω0 - t A with t < 0 maps eigenvalue λ of A to ω0 + |t| λ.
    abs_t, f0 = np.polyfit(lam, nodal_modes, 1)
    assert abs_t > 0, "hopping must be negative"
    residual = nodal_modes - (abs_t * lam + f0)
    assert np.max(np.abs(residual)) < 0.1 * abs_t


def test_flat_band_at_minus_two_t(nodal_modes: np.ndarray) -> None:
    """The lowest E - V modes are degenerate at ω0 - 2|t|."""
    tiling = hyperbolic_tiling(7, 3, 1)
    n_flat = tiling.num_edges - tiling.num_vertices
    lam = np.sort(np.linalg.eigvalsh(tiling.line_graph_adjacency))
    abs_t, f0 = np.polyfit(lam, nodal_modes, 1)

    flat = nodal_modes[:n_flat]
    assert np.ptp(flat) == 0, "flat band must be exactly degenerate"
    assert nodal_modes[n_flat] - flat[0] > 0.1 * abs_t
    assert abs(flat[0] - (f0 - 2 * abs_t)) < 0.1 * abs_t


def test_sax_probe_resonances_match_nodal_modes(nodal_modes: np.ndarray) -> None:
    """Group-delay peaks of the SAX model sit on the dispersive eigenmodes.

    Flat-band states have zero summed amplitude at every coupler, so a probe on
    a free pin cannot excite them; no peak may appear there.
    """
    tiling = hyperbolic_tiling(7, 3, 1)
    n_flat = tiling.num_edges - tiling.num_vertices
    f = np.linspace(F_MIN, F_MAX, 2001)
    s = hyperbolic_lattice(
        f=jnp.asarray(f),
        resonator_length=RESONATOR_LENGTH,
        stub_length=STUB_LENGTH,
        probe_pins=(0,),
    )
    phase = np.unwrap(np.angle(np.asarray(s["o1", "o1"])))
    group_delay = -np.gradient(phase, f)
    peaks, _ = find_peaks(group_delay, prominence=1e-3 * group_delay.max())
    peak_f = f[peaks]

    assert len(peak_f) >= 10
    dispersive = nodal_modes[n_flat:]
    distance = np.min(np.abs(peak_f[:, None] - dispersive[None, :]), axis=1)
    np.testing.assert_array_less(distance, 1e6)
    assert np.min(np.abs(peak_f - nodal_modes[0])) > 5e6


def _sax_netlist_lattice(f: jnp.ndarray, probe_pins: tuple[int, ...]) -> sax.SDict:
    """The default lattice wired as a SAX netlist of its component models.

    Returns:
        S-parameters between the probe ports.
    """
    tiling = hyperbolic_tiling(7, 3, 1)
    free = tiling.free_pins
    # The coupler models include their leads, as in the layout.
    line = straight(f=f, length=RESONATOR_LENGTH - 2 * LEAD_LENGTH)
    stub = straight(f=f, length=STUB_LENGTH - LEAD_LENGTH)
    instances: dict[str, sax.SType] = {
        f"c{v}": hyperbolic_vertex_coupler(f=f) for v in range(tiling.num_vertices)
    }
    connections: dict[str, str] = {}
    ports: dict[str, str] = {}
    for e, (i, j) in enumerate(tiling.edges.tolist()):
        instances[f"r{e}"] = line
        for end, v in (("o1", i), ("o2", j)):
            connections[f"r{e},{end}"] = f"c{v},o{tiling.pins[v][1].index(e) + 1}"
    probe_index = {free[k]: n for n, k in enumerate(probe_pins)}
    for v, k in free:
        pin = f"c{v},o{k + 1}"
        if (v, k) in probe_index:
            ports[f"o{probe_index[v, k] + 1}"] = pin
            continue
        instances[f"s{v}_{k}"] = stub
        instances[f"s{v}_{k}_short"] = electrical_short(f=f)
        connections[pin] = f"s{v}_{k},o1"
        connections[f"s{v}_{k},o2"] = f"s{v}_{k}_short,o1"
    return sax.evaluate_circuit_fg((connections, ports), instances)


def test_nodal_model_matches_sax_netlist() -> None:
    """The nodal solution equals SAX evaluating the netlist of component models."""
    f = jnp.linspace(F_MIN, F_MAX, 7)
    probe_pins = (11, 0)
    nodal = hyperbolic_lattice(f=f, probe_pins=probe_pins)
    netlist = _sax_netlist_lattice(f, probe_pins)
    assert nodal.keys() == netlist.keys()
    for key, value in nodal.items():
        np.testing.assert_allclose(value, netlist[key], atol=1e-9)
