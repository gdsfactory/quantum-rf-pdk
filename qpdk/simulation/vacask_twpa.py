"""A resonantly phase-matched Josephson TWPA for VACASK.

The circuit follows the JosephsonCircuits JTWPA example based on
:cite:p:`macklinNearquantumlimitedJosephsonTravelingwave2015` and the resonant
phase-matching scheme in :cite:p:`obrienResonantPhaseMatching2014`.
The 2048-node example has 2047 junctions and 512 coupled LC resonators.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qpdk.simulation.vacask.netlist import Netlist, _templates
from qpdk.simulation.vacask.raw import RawPlot


def rpm_twpa_netlist(
    cells: int,
    control: str,
    *,
    pump_current: float = 3.7e-6,
    phase_matching: bool = True,
    probe_port: Literal["input", "output"] = "input",
) -> Netlist:
    """Build a JJ ladder with a phase-matching resonator every fourth cell.

    The loaded ladder follows :cite:p:`obrienResonantPhaseMatching2014` and
    :cite:p:`macklinNearquantumlimitedJosephsonTravelingwave2015`.

    Each junction has 3.4 uA critical current, 55 fF capacitance and a 1 Gohm
    subgap resistance. Ground capacitance is 45 fF per cell, split at the two
    ends. Loaded cells replace 30 fF of it with a coupling capacitor to a
    2.8153 pF / 170 pH parallel resonator. The 50 ohm terminations at ``n0``
    and ``n{cells}`` include zero-voltage sources for ``acsp`` with
    ``ports=["vp1","rs","vp2","rl"]``. The weak current probe is 1 nA
    on spur zero; native scattering analyses drive the ports themselves.
    The pump always enters at ``n0``, independently of the probe port.

    Args:
        cells: Physical junction count, at least two.
        control: Raw VACASK options and analysis statements. Development
            analyses need a compatible local VACASK build.
        pump_current: Peak Norton source current at 7.12 GHz, twice the
            positive-frequency Fourier coefficient in JosephsonCircuits.
            The incident traveling current is half this value at a matched port.
        phase_matching: Attach the LC branches. False gives a uniform ladder
            with the same low-frequency shunt capacitance for comparison.
        probe_port: Apply the weak probe at the input or output port. Separate
            invocations give the two scattering columns about the same pump.

    Returns:
        A netlist using qpdk's RCSJ model, with explicit junction parasitics.

    Raises:
        ValueError: If the cell count, pump current or probe port is invalid.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError("cells must be an integer of at least two")
    if not math.isfinite(pump_current) or pump_current < 0:
        raise ValueError("pump_current must be finite and nonnegative")
    if probe_port not in {"input", "output"}:
        raise ValueError("probe_port must be 'input' or 'output'")
    circuit = (
        _templates()
        .get_template("rpm_twpa.sim.j2")
        .render(
            cells=cells,
            pump_current=pump_current,
            phase_matching=phase_matching,
            probe_port=probe_port,
        )
    )
    return Netlist(
        f"Resonantly phase-matched TWPA, {cells} junctions",
        loads=(
            "josephson_junction.va",
            "capacitor.osdi",
            "resistor.osdi",
            "inductor.osdi",
        ),
        circuit=circuit,
        control=control,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TwpaPumpResidual:
    """Original-equation residuals on a shifted time grid.

    KCL errors are normalized by the 3.4 uA critical current. Voltage errors
    are relative to each branch's peak voltage, with a 1 pV floor at zero drive.
    These residuals test the pump waveform, not its temporal stability or the
    small-signal sideband cutoff.
    """

    mainline_kcl_over_ic: float
    resonator_kcl_over_ic: float
    inductor_voltage_relative: float
    phase_voltage_relative: float


def twpa_pump_residual(
    pump: RawPlot,
    cells: int,
    *,
    pump_current: float = 3.7e-6,
    phase_matching: bool = True,
    samples: int = 4096,
) -> TwpaPumpResidual:
    """Check an RPM TWPA HB orbit against its original circuit equations.

    The RCSJ constitutive law follows :cite:p:`maasNonlinearMicrowaveRF2003`;
    the ladder and LC loading follow :cite:p:`obrienResonantPhaseMatching2014`.

    Reconstruct all saved node voltages, junction phases and inductor currents
    on a time grid shifted from the HB collocation grid. Evaluate the RCSJ
    currents and Kirchhoff equations directly, without using the HB Jacobian.
    The circuit parameters and cosine pump must match :func:`rpm_twpa_netlist`.

    Args:
        pump: Ordinary HB plot including every node, ``j{i}:ph`` and the
            loaded-cell ``lr{i}:flow(br)`` currents. HBAC plots are not pump orbits.
            Missing circuit variables raise ``KeyError``.
        cells: Physical junction count.
        pump_current: Peak current of the input cosine pump, in amperes.
        phase_matching: Whether the circuit has the coupled LC branches.
        samples: Number of holdout time samples, larger than twice the pump cutoff.

    Returns:
        Maximum normalized residuals over every branch and sampled time.

    Raises:
        ValueError: If the grid is invalid or a saved value is nonfinite.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError("cells must be an integer of at least two")
    if not math.isfinite(pump_current) or pump_current < 0:
        raise ValueError("pump_current must be finite and nonnegative")
    frequency = pump["frequency"].real
    harmonics = np.arange(len(frequency))
    if not len(frequency) or not np.allclose(
        frequency, harmonics * 7.12e9, rtol=1e-12, atol=1e-3
    ):
        raise ValueError("Expected a single-tone 7.12 GHz HB pump plot")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 2 * harmonics[-1]
    ):
        raise ValueError("samples must be an integer larger than twice the pump cutoff")
    if not np.all(np.isfinite(pump.data)):
        raise ValueError("Pump plot contains nonfinite values")

    theta = 2 * np.pi * (np.arange(samples) + 0.371) / samples
    evaluation = np.exp(1j * np.outer(theta, harmonics))

    def waveform(names: list[str], *, derivative: bool = False) -> np.ndarray:
        coefficients = np.stack([pump[name] for name in names], axis=1)
        if derivative:
            coefficients = coefficients * (2j * np.pi * frequency[:, None])
        return (evaluation @ coefficients).real

    nodes = [f"n{i}" for i in range(cells + 1)]
    voltage, derivative = waveform(nodes), waveform(nodes, derivative=True)
    phases = [f"j{i}:ph" for i in range(cells)]
    phase, dphase = waveform(phases), waveform(phases, derivative=True)
    branch_voltage = voltage[:, :-1] - voltage[:, 1:]
    dbranch_voltage = derivative[:, :-1] - derivative[:, 1:]
    current = 3.4e-6 * np.sin(phase) + branch_voltage / 1e9 + 55e-15 * dbranch_voltage
    residual = 45e-15 * derivative
    residual[:, 0] = (
        22.5e-15 * derivative[:, 0] + voltage[:, 0] / 50 - pump_current * np.cos(theta)
    )
    residual[:, -1] = 22.5e-15 * derivative[:, -1] + voltage[:, -1] / 50
    residual[:, :-1] += current
    residual[:, 1:] -= current

    lc_error = inductor_error = 0.0
    if phase_matching:
        loaded = np.arange(1, cells, 4)
        vr = waveform([f"r{i}" for i in loaded])
        dvr = waveform([f"r{i}" for i in loaded], derivative=True)
        il = waveform([f"lr{i}:flow(br)" for i in loaded])
        dil = waveform([f"lr{i}:flow(br)" for i in loaded], derivative=True)
        residual[:, loaded] -= 30e-15 * dvr
        lc_error = float(
            np.max(
                np.abs(2.8153e-12 * dvr + il + 30e-15 * (dvr - derivative[:, loaded]))
            )
            / 3.4e-6
        )
        inductor_error = float(
            np.max(
                np.abs(vr - 170e-12 * dil)
                / np.maximum(np.max(np.abs(vr), axis=0), 1e-12)
            )
        )

    # The compiled Verilog-A model uses the NIST 1998 constants, not modern SI values.
    phi0 = 6.62606876e-34 / (2 * 1.602176462e-19)
    phase_error = np.abs(branch_voltage - phi0 / (2 * np.pi) * dphase)
    return TwpaPumpResidual(
        mainline_kcl_over_ic=float(np.max(np.abs(residual)) / 3.4e-6),
        resonator_kcl_over_ic=lc_error,
        inductor_voltage_relative=inductor_error,
        phase_voltage_relative=float(
            np.max(
                phase_error / np.maximum(np.max(np.abs(branch_voltage), axis=0), 1e-12)
            )
        ),
    )
