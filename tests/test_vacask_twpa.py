"""Independent cold-network checks for the resonantly phase-matched TWPA."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from qpdk.simulation.vacask import RawPlot, Vacask
from qpdk.simulation.vacask_twpa import rpm_twpa_netlist, twpa_pump_residual


def _cold_scattering(f: jnp.ndarray, cells: int, phase_matching: bool) -> jnp.ndarray:
    omega = 2 * jnp.pi * f
    # OpenVAF defaults to the NIST 1998 constants in constants.vams.
    phi0 = 6.62606876e-34 / (2 * 1.602176462e-19)
    lj = phi0 / (2 * jnp.pi * 3.4e-6)
    z_j = 1 / (1 / (1j * omega * lj) + 1j * omega * 55e-15 + 1e-9)
    matrix = jnp.broadcast_to(jnp.eye(2, dtype=complex), (len(f), 2, 2))
    one, zero = jnp.ones_like(f), jnp.zeros_like(f)

    def shunt(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack((jnp.stack((one, zero), -1), jnp.stack((y, one), -1)), -2)

    matrix = matrix @ shunt(1j * omega * 22.5e-15)
    series = jnp.stack((jnp.stack((one, z_j), -1), jnp.stack((zero, one), -1)), -2)
    for i in range(1, cells + 1):
        cap = 22.5e-15 if i == cells else 45e-15
        y = 1j * omega * cap
        if phase_matching and i < cells and i % 4 == 1:
            z_lc = 1 / (1j * omega * 2.8153e-12 + 1 / (1j * omega * 170e-12))
            y = 1j * omega * 15e-15 + 1 / (1 / (1j * omega * 30e-15) + z_lc)
        matrix = matrix @ series @ shunt(y)
    a, b, c, d = (
        matrix[:, 0, 0],
        matrix[:, 0, 1] / 50,
        matrix[:, 1, 0] * 50,
        matrix[:, 1, 1],
    )
    denominator = a + b + c + d
    return jnp.stack(
        [
            jnp.stack([(a + b - c - d) / denominator, 2 / denominator], axis=-1),
            jnp.stack([2 / denominator, (-a + b - c + d) / denominator], axis=-1),
        ],
        axis=-2,
    )


@pytest.mark.vacask
@pytest.mark.parametrize("phase_matching", [True, False])
@pytest.mark.parametrize("probe_port", ["input", "output"])
def test_twpa_cold_matches_abcd(
    phase_matching: bool, probe_port: Literal["input", "output"]
) -> None:
    cells = 20
    deck = rpm_twpa_netlist(
        cells,
        "options reltol=1e-9 tolscale=1e-6\nanalysis cold ac values=[3G,5G,7.12G,9G,11G]",
        pump_current=0,
        phase_matching=phase_matching,
        probe_port=probe_port,
    )
    result = Vacask.find().run(deck)["cold"]
    f = jnp.asarray(result["frequency"].real)
    column = jnp.stack(
        [result[node] * 2 / (1e-9 * 50) for node in ("n0", f"n{cells}")], axis=-1
    )
    incident = 0 if probe_port == "input" else 1
    column -= jnp.eye(2)[incident]
    np.testing.assert_allclose(
        column,
        _cold_scattering(f, cells, phase_matching)[:, :, incident],
        rtol=2e-7,
        atol=2e-9,
    )


@pytest.mark.parametrize("cells", [0, 1, True, 2.5])
def test_twpa_invalid_length(cells: int) -> None:
    with pytest.raises(ValueError, match="cells"):
        rpm_twpa_netlist(cells, "")


@pytest.mark.vacask
@pytest.mark.parametrize("phase_matching", [True, False])
@pytest.mark.parametrize("cells", [4, 20])
def test_twpa_cold_native_scattering_matches_abcd(
    phase_matching: bool, cells: int
) -> None:
    deck = rpm_twpa_netlist(
        cells,
        "options reltol=1e-9 tolscale=1e-6\n"
        'analysis cold acsp ports=["vp1","rs","vp2","rl"] values=[3G,5G,7.12G,7.24G,7.3G,9G,11G]',
        pump_current=0,
        phase_matching=phase_matching,
    )
    result = Vacask.find().run(deck)["cold"]
    expected = _cold_scattering(
        jnp.asarray(result["frequency"].real), cells, phase_matching
    )
    for output in range(2):
        for incident in range(2):
            np.testing.assert_allclose(
                result[f"s({output + 1},{incident + 1})"],
                expected[:, output, incident],
                rtol=2e-7,
                atol=2e-9,
            )


def test_twpa_invalid_probe_port() -> None:
    with pytest.raises(ValueError, match="probe_port"):
        rpm_twpa_netlist(20, "", probe_port="missing")  # pyrefly: ignore[bad-argument-type]


@pytest.fixture
def idle_pump() -> RawPlot:
    """A zero-drive equilibrium, including one loaded cell."""
    names = ("frequency", "n0", "n1", "n2", "j0:ph", "j1:ph", "r1", "lr1:flow(br)")
    data = np.zeros((2, len(names)), dtype=complex)
    data[1, 0] = 7.12e9
    return RawPlot("Idle pump", "HB Analysis", names, data)


def test_pump_residual_at_equilibrium(idle_pump: RawPlot) -> None:
    residual = twpa_pump_residual(idle_pump, 2, pump_current=0)
    assert residual.mainline_kcl_over_ic == 0
    assert residual.resonator_kcl_over_ic == 0
    assert residual.inductor_voltage_relative == 0
    assert residual.phase_voltage_relative == 0


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("j0:ph", 0.5, "mainline_kcl_over_ic", float(jnp.sin(0.5))),
        ("lr1:flow(br)", 1e-6, "resonator_kcl_over_ic", 1 / 3.4),
        ("r1", 1e-6, "inductor_voltage_relative", 1.0),
        ("n2", 1e-6, "phase_voltage_relative", 1.0),
    ],
)
def test_pump_residual_detects_inconsistent_states(
    idle_pump: RawPlot, variable: str, value: float, field: str, expected: float
) -> None:
    idle_pump.data[0, idle_pump.names.index(variable)] = value
    residual = twpa_pump_residual(idle_pump, 2, pump_current=0)
    assert getattr(residual, field) == pytest.approx(expected)


def test_pump_residual_detects_missing_drive_response(idle_pump: RawPlot) -> None:
    residual = twpa_pump_residual(idle_pump, 2)
    assert residual.mainline_kcl_over_ic == pytest.approx(3.7 / 3.4, rel=1e-6)


def test_pump_residual_rejects_undersampled_grid(idle_pump: RawPlot) -> None:
    with pytest.raises(ValueError, match="samples"):
        twpa_pump_residual(idle_pump, 2, samples=2)


def test_pump_residual_rejects_sideband_plot(idle_pump: RawPlot) -> None:
    idle_pump.data[:, 0] += 5e9
    with pytest.raises(ValueError, match="HB pump"):
        twpa_pump_residual(idle_pump, 2)


def test_pump_residual_rejects_empty_plot(idle_pump: RawPlot) -> None:
    empty = RawPlot("Empty", "HB Analysis", idle_pump.names, idle_pump.data[:0])
    with pytest.raises(ValueError, match="HB pump"):
        twpa_pump_residual(empty, 2)


def test_pump_residual_rejects_nonfinite_values(idle_pump: RawPlot) -> None:
    idle_pump.data[0, 1] = jnp.nan
    with pytest.raises(ValueError, match="nonfinite"):
        twpa_pump_residual(idle_pump, 2)


@pytest.mark.vacask
@pytest.mark.parametrize("phase_matching", [True, False])
@pytest.mark.parametrize("probe_port", ["input", "output"])
def test_twpa_pump_satisfies_original_equations(
    phase_matching: bool, probe_port: Literal["input", "output"]
) -> None:
    cells = 20
    deck = rpm_twpa_netlist(
        cells,
        "options reltol=1e-9 tolscale=1e-6\nanalysis pump hb freq=[7.12G] nharm=19",
        phase_matching=phase_matching,
        probe_port=probe_port,
    )
    pump = Vacask.find().run(deck)["pump"]
    residual = twpa_pump_residual(pump, cells, phase_matching=phase_matching)
    assert residual.mainline_kcl_over_ic < 1e-5
    assert residual.resonator_kcl_over_ic < 1e-8
    assert residual.inductor_voltage_relative < 1e-8
    assert residual.phase_voltage_relative < 1e-8
