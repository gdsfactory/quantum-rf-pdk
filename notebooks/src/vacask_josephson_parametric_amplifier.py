# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [raw] tags=["remove-cell"]
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///

# %% [markdown]
# # Josephson traveling-wave parametric amplifiers: phase matching
#
# ::::{admonition} Requirements
# :class: tip
#
# From a source checkout, run `uv sync --extra models`. The full-size gain plot reads JosephsonCircuits results bundled with the notebook data. Julia is needed only to recompute them, as described in `notebooks/data/vacask_twpa/README.md`. See {ref}`notebook-extras`.
# ::::
#
# A traveling-wave parametric amplifier (TWPA) uses a chain of Josephson junctions to amplify a weak signal. A strong pump changes the junction inductance. The signal at $f_\text{s}$ mixes with the pump at $f_\text{p}$ and creates an idler at $2f_\text{p}-f_\text{s}$ {cite:p}`macklinNearquantumlimitedJosephsonTravelingwave2015`.
#
# Can resonators along the chain improve gain at the same incident pump power? We compare a uniform chain with a chain loaded by a resonator every fourth cell {cite:p}`obrienResonantPhaseMatching2014`. Blue means uniform and orange means loaded in every plot. Each circuit has its own pump solution.
#
# Each junction follows the RCSJ equation {cite:p}`McCumber1968,tinkhamIntroductionSuperconductivity2015`:
#
# $$I=I_\text{c}\sin\varphi+\frac{\Phi_0}{2\pi R}\dot\varphi+\frac{\Phi_0 C_\text{J}}{2\pi}\ddot\varphi.$$
#
# The circuit follows the resonantly phase-matched JTWPA example distributed with JosephsonCircuits.jl {cite:p}`pengFloquetModeTravelingWaveParametric2022`. It has 2047 junctions. Each has $I_\text{c}=3.4\,\text{µA}$ and $C_\text{J}=55\,\text{fF}$. Ground capacitance is 45 fF per cell, halved at each end. A loaded cell replaces 30 fF of that capacitance with a 30 fF coupling capacitor to a 2.8153 pF / 170 pH parallel resonator at 7.275 GHz. The pump is 3.7 µA peak Norton current at 7.12 GHz, just below that resonance. Half that current is incident at the matched port. Its available power is $Z_0 I_\text{N}^2/8=85.56\,\text{pW}$, or -70.68 dBm {cite:p}`m.pozarMicrowaveEngineering2012`.
#
# ![Josephson ladder with ground capacitors and optional coupled LC resonators](figures/vacask-twpa-ladder.svg)
#
# The {doc}`lumped JPA <vacask_lumped_parametric_amplifier>` and {doc}`CPW JPA <vacask_cpw_parametric_amplifier>` notebooks cover resonator amplifiers.

# %% [markdown]
# (vacask-analysis-guide)=
# ## Choose a VACASK analysis
#
# An analysis chooses the circuit problem to solve. VACASK then uses numerical methods such as Newton iteration or time stepping to solve it. The manual groups the analyses by the state they use {cite:p}`vacaskCircuitAnalyses`.
#
# | Question | Analysis | State and result |
# | --- | --- | --- |
# | What is the DC state? | `op` | Solve the nonlinear DC equations. |
# | What is the unpumped response? | `ac`, `acsp` | Linearize at a DC state. `ac` returns voltages and currents; `acsp` returns port S-parameters. |
# | What happens in time? | `tran` | Integrate the nonlinear equations, including startup and transients. |
# | What periodic state repeats in time? | `pss` | Use shooting to match the state after one period to its initial state. |
# | What weak signal responds to that state? | `pac` | Linearize around a shooting-based periodic state. |
# | What is the periodic pump spectrum? | `hb` | Solve for Fourier coefficients using harmonic balance. |
# | What weak signal and idlers respond to the pump? | `hbac` | Linearize around an HB state and couple its frequency sidebands. |
#
# The manual also covers transfer functions, feedback-loop analysis and noise. `noise` uses a DC state and `hbnoise` uses an HB state. Those are classical device-noise calculations. They do not by themselves give quantum added noise. The current manual includes features that an installed release may lack.
#
# For a TWPA, use `hb` for the pump, then `hbac` for the weak probe {cite:p}`kundertSimulationNonlinearCircuits1986,maasNonlinearMicrowaveRF2003`. `hbac` labels the response at $f_\text{s}+k f_\text{p}$. Our idler uses $k=-2$, the negative-frequency partner of $2f_\text{p}-f_\text{s}$.


# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing dependencies...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input"]
import html
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if "ipykernel" in sys.modules:
    from IPython.display import display
from matplotlib import font_manager, pyplot as plt

import qpdk.colormaps  # ruff: ignore[unused-import] - registers the style colormap
from qpdk import PDK, logger
from qpdk.config import PATH

TWPA_DATA = PATH.repo / "notebooks/data/vacask_twpa"


# %% tags=["hide-input"]
if "ipykernel" in sys.modules:
    from IPython import get_ipython
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    if (ipython := get_ipython()) is not None:
        ipython.run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("png", "svg")

jax.config.update("jax_enable_x64", True)
PDK.activate()

# The checkout stylesheet is absent from installed wheels.
for _style in (PATH.docs / "qpdk.mplstyle", "qpdk"):
    try:
        plt.style.use(_style)
    except OSError:
        continue
    break

# The stylesheet names fonts that only the docs build ships; register them so
# they resolve without docs/ipython_config.py.
_font_dir = os.environ.get("QPDK_DOCS_FONTS_DIR")
for _fonts in (Path(_font_dir) if _font_dir else None, PATH.repo / "build/docs-fonts"):
    if _fonts is not None and _fonts.is_dir():
        for _font in sorted(_fonts.iterdir()):
            if _font.suffix in {".otf", ".ttf"}:
                font_manager.fontManager.addfont(str(_font))
        break

# %% tags=["hide-input"]
F_PUMP = 7.12e9
COLORS = {False: "C0", True: "C1"}
LABELS = {False: "Uniform", True: "Phase matched"}


def db(x: ArrayLike) -> jax.Array:
    """Convert a voltage-wave ratio to dB; :cite:p:`m.pozarMicrowaveEngineering2012`."""
    return 20 * jnp.log10(jnp.abs(x))


def source_details(title: str, source: str, language: str = "text") -> str:
    """Collapsible source listing for notebook and documentation output."""
    escaped = html.escape(source.strip())
    return (
        f"<details><summary>{title}</summary>\n"
        f'<pre><code class="language-{language}">{escaped}</code></pre>\n'
        "</details>"
    )


def show_source(title: str, source: str, language: str = "text") -> None:
    """Display a source listing collapsed, or print it outside a notebook."""
    if "ipykernel" in sys.modules:
        display(
            {
                "text/html": source_details(title, source, language),
                "text/plain": f"{title}: see the collapsible listing in the notebook or HTML documentation.",
            },
            raw=True,
        )
    else:
        print(source)


def show_plot() -> None:
    """Show the current figure with Outfit titles, as the docs config does."""
    fig = plt.gcf()
    for text in [ax.title for ax in fig.axes] + list(fig.texts):
        if text.get_text():
            text.set_fontfamily("Outfit")
            text.set_fontweight("bold")
    plt.show()


# %% [markdown]
# ## Cold dispersion
#
# First set the pump to zero. The helper below combines the ABCD matrices of four cells. It uses the same junction and resonator values as VACASK. The series junction impedance includes its linear inductance, 55 fF capacitance and 1 GΩ resistance. The constants match OpenVAF's NIST 1998 convention. Native VACASK `acsp` tests check the result.
#
# For 50 Ω reference ports, the Bloch phase follows from the transfer-matrix trace {cite:p}`m.pozarMicrowaveEngineering2012`:
#
# $$\cosh\gamma_4=\frac{A+D}{2}=\frac{1-S_{11}S_{22}+S_{12}S_{21}}{2S_{21}}.$$
#
# Here $\operatorname{Re}\gamma_4$ is attenuation over four cells and $q=\operatorname{Im}\gamma_4/4$ is phase per junction. The plot hides the stopband where attenuation exceeds $10^{-3}$. The coupled resonators raise the pump wavenumber, which sits just below their resonance, and barely change the signal. Above the resonance, in the idler band, they lower it slightly. Both make the cold mismatch $2q_\text{p}-q_\text{s}-q_\text{i}$ positive.
#
# The pump also shifts the wavenumbers through the junction nonlinearity. Expanding $I_\text{c}\sin\varphi$ to third order, a pump current of amplitude $I_\text{p}$ raises its own inductance by the factor $1+x/8$, with $x=(I_\text{p}/I_\text{c})^2$. It raises the inductance seen by the weak signal and idler by $1+x/4$. These are self- and cross-phase modulation. Since $q\propto\sqrt{L}$ for $q\ll1$, and $q_\text{s}+q_\text{i}\approx2q_\text{p}$, the total mismatch per junction is
#
# ```{math}
# :label: eq:vacask-twpa-mismatch
# \Delta q \simeq \left(2q_\text{p}-q_\text{s}-q_\text{i}\right)_\text{cold} - \frac{x}{8}\,q_\text{p}.
# ```
#
# The Kerr term is negative, so a uniform chain is never phase matched under pump. The resonators supply a positive cold mismatch to cancel it {cite:p}`obrienResonantPhaseMatching2014`. The dashed lines show the cold mismatch that would give $\Delta q=0$. They assume the port is matched to the chain, so $I_\text{p}\approx I_\text{N}/2$. The estimate ignores pump depletion and the chain's 46 Ω impedance, so it explains trends rather than predicting gain.


# %% tags=["hide-input"]
def cold_scattering(
    frequency: ArrayLike, cells: int, *, phase_matching: bool
) -> jax.Array:
    """Cold ladder S-parameters from a cascade of series and shunt elements.

    Uses the RCSJ linearization and the loaded circuit in
    :cite:p:`McCumber1968,obrienResonantPhaseMatching2014`, with ABCD-to-S
    conversion from :cite:p:`m.pozarMicrowaveEngineering2012`.

    Args:
        frequency: Frequencies in Hz.
        cells: Number of junctions.
        phase_matching: Include the coupled resonators.

    Returns:
        Complex S-parameters with axes (frequency, output port, input port).
    """
    f = jnp.asarray(frequency)
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


# %% tags=["hide-input", "twpa-dispersion"]
F_RESONATOR = 1 / (2 * jnp.pi * jnp.sqrt(170e-12 * 2.8153e-12))
# The port is nearly matched to the chain, so about half the Norton current flows in.
X_PUMP = (3.7e-6 / 2 / 3.4e-6) ** 2
PUMP_SHARES = (0.5, 1.0, 1.1)
print(f"Loading resonator at {F_RESONATOR / 1e9:.3f} GHz, (I_p/I_c)² = {X_PUMP:.3f}")

frequency = jnp.linspace(3e9, 12e9, 901)
signal = jnp.linspace(4e9, 6.5e9, 51)
fig, (ax_bloch, ax_mismatch) = plt.subplots(1, 2, figsize=(11, 4))
for matched in (False, True):
    cold = jnp.asarray(cold_scattering(frequency, 4, phase_matching=matched))
    s11, s12, s21, s22 = (
        cold[:, output, incident]
        for output, incident in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    gamma = jnp.acosh((1 - s11 * s22 + s12 * s21) / (2 * s21))
    q = gamma.imag / 4
    ax_bloch.plot(
        frequency / 1e9,
        jnp.where(gamma.real < 1e-3, q, jnp.nan),
        color=COLORS[matched],
        label=LABELS[matched],
    )
    q_p = jnp.interp(F_PUMP, frequency, q)
    mismatch = (
        2 * q_p
        - jnp.interp(signal, frequency, q)
        - jnp.interp(2 * F_PUMP - signal, frequency, q)
    )
    ax_mismatch.plot(
        signal / 1e9, mismatch, color=COLORS[matched], label=LABELS[matched]
    )
    cold_5ghz = jnp.interp(5e9, signal, mismatch)
    totals = ", ".join(
        f"{share:.0%}: {cold_5ghz - q_p * share**2 * X_PUMP / 8:+.4f}"
        for share in PUMP_SHARES
    )
    print(
        f"{LABELS[matched]}: q_p = {q_p:.4f} rad, cold mismatch at 5 GHz = "
        f"{cold_5ghz:+.4f} rad; with Kerr shift at pump current {totals} rad"
    )
# Targets use the loaded chain's q_p; the uniform one is 3 % lower.
for share in PUMP_SHARES:
    target = q_p * share**2 * X_PUMP / 8
    ax_mismatch.axhline(target, color="gray", ls="--", lw=1)
    ax_mismatch.annotate(
        f"Kerr target, {share:.0%} pump",
        (signal[0] / 1e9, target),
        xytext=(2, 2),
        textcoords="offset points",
        fontsize="small",
        color="gray",
    )
ax_bloch.axvline(F_PUMP / 1e9, color="gray", ls=":", label="Pump")
ax_bloch.axvline(F_RESONATOR / 1e9, color="C1", ls=":", label="Loading resonator")
ax_bloch.set(xlabel="Frequency (GHz)", ylabel="Cold Bloch phase (rad / junction)")
ax_mismatch.axhline(0, color="gray", ls=":")
ax_mismatch.set(
    xlabel="Signal frequency (GHz)",
    ylabel=r"Cold $2q_\text{p}-q_\text{s}-q_\text{i}$ (rad / junction)",
)
ax_bloch.legend()
ax_mismatch.legend()
fig.tight_layout()
show_plot()

# %% [markdown]
# At 5 GHz and full pump, Eq. {eq}`eq:vacask-twpa-mismatch` leaves the loaded chain at $+0.0013$ rad per junction and the uniform chain at $-0.0038$ rad. Over 2047 junctions the uniform chain accumulates about 8 rad of phase error, the loaded chain under 3 rad.
#
# The Kerr term grows with pump power while the cold mismatch does not, so the loading is tuned to one pump power {cite:p}`obrienResonantPhaseMatching2014`. At half the current the estimate reverses the order: the uniform chain is then closer to phase matching ($-0.0012$ against $+0.0040$ rad per junction).
#
# ## Full-size gain from JosephsonCircuits
#
# JosephsonCircuits.jl solves the same 2047-junction circuits with harmonic balance {cite:p}`pengFloquetModeTravelingWaveParametric2022`. The pump keeps odd harmonics through 35 and the probe keeps sideband offsets through 40, with 1609 time samples and five pump-current continuation stages. The saved data contain the complex signal and idler response at both ports, at 51 frequencies from 4 to 6.5 GHz. They come from one harmonic basis and carry no basis-refinement check.
#
# The data store voltage-wave ratios. The right panel converts the idler to photon conversion gain $n_\text{i}=|S_\text{is}|^2 f_\text{s}/|f_\text{i}|$ {cite:p}`royIntroductionParametricAmplification2016`. If the pump coupled the signal only to this idler, Manley-Rowe would make the signal photons minus the idler photons, summed over both ports, equal 1 {cite:p}`manleyGeneralPropertiesNonlinear1956`. The printed two-mode balance tests that.

# %% tags=["hide-input", "twpa-reference"]
reference_path = TWPA_DATA / "julia-reference.json"
try:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    logger.info(
        "The recorded full-size TWPA figure needs the notebook data in a source checkout."
    )
else:
    f_signal = jnp.asarray(reference["signal_hz"])
    f_idler = jnp.abs(f_signal - 2 * F_PUMP)
    i5 = int(jnp.argmin(jnp.abs(f_signal - 5e9)))
    fig, (ax_gain, ax_idler) = plt.subplots(1, 2, figsize=(11, 4))
    for chain in reference["chains"]:
        matched = chain["phase_matching"]
        s = jnp.asarray(chain["response_re"]) + 1j * jnp.asarray(chain["response_im"])
        idler_photons = jnp.abs(s[0, 1, 1]) ** 2 * f_signal / f_idler
        # Both output ports, so reflected signal and backward idler count too.
        balance = jnp.sum(
            jnp.abs(s[0, :, 0]) ** 2 - jnp.abs(s[0, :, 1]) ** 2 * f_signal / f_idler,
            axis=0,
        )
        print(
            f"{LABELS[matched]}: G = {db(s[0, 1, 0, i5]):.2f} dB at 5 GHz, "
            f"two-mode balance {balance.min():.2f} to {balance.max():.2f}"
        )
        ax_gain.plot(
            f_signal / 1e9, db(s[0, 1, 0]), color=COLORS[matched], label=LABELS[matched]
        )
        ax_idler.plot(
            f_signal / 1e9,
            10 * jnp.log10(idler_photons),
            color=COLORS[matched],
            label=LABELS[matched],
        )
    ax_gain.set(
        xlabel="Signal frequency (GHz)",
        ylabel="Signal power gain (dB)",
        title="2047 junctions, JosephsonCircuits",
    )
    ax_idler.set(
        xlabel="Signal frequency (GHz)",
        ylabel=r"Idler photon gain $n_\text{i}$ (dB)",
        title=r"Idler at $2f_\text{p}-f_\text{s}$",
    )
    ax_gain.legend()
    fig.tight_layout()
    show_plot()

# %% [markdown]
# At 5 GHz the loaded chain gives 25.00 dB and the uniform chain 5.25 dB, as the phase errors above suggest. The uniform chain keeps the two-mode balance within 15 % of 1 up to 6.25 GHz, rising toward the edge of the band. The loaded chain does not: the balance reaches tens to hundreds. Net photons leave through sidebands outside this signal/idler submatrix, for example those mixed through the pump harmonics. {cite:t}`pengFloquetModeTravelingWaveParametric2022` show that such sideband leakage limits the quantum efficiency of resonantly phase-matched TWPAs. A design for low added noise therefore needs the full sideband response, not only the signal gain.
#
# The Julia circuit and solver are listed below.

# %% tags=["hide-input"]
JULIA_REFERENCE = TWPA_DATA / "julia/reference.jl"
if JULIA_REFERENCE.is_file():
    show_source(
        "Julia circuit and solver", JULIA_REFERENCE.read_text(encoding="utf-8"), "julia"
    )
else:
    logger.info(
        "The Julia source listing needs a source checkout at {}.", JULIA_REFERENCE
    )

# %% [markdown]
# ## Connect the circuit to layout
#
# These TWPA values are circuit targets from the reference design. The {doc}`CPW notebook <vacask_cpw_parametric_amplifier>` shows a direct link: one QPDK cross-section and the measured layout length feed both SAX and VACASK.
#
# For a TWPA layout, use the following starting points and replace target values with measured or extracted ones.
#
# | Circuit part | QPDK layout starting point | Values still needed |
# | --- | --- | --- |
# | Junction | {func}`~qpdk.cells.junction.josephson_junction` | Process-calibrated $I_\text{c}$, $C_\text{J}$ and shunt resistance. |
# | Ground and coupling capacitors | {func}`~qpdk.cells.capacitor.interdigital_capacitor` | A capacitance extraction including nearby ground and pads. |
# | Loading resonator | {func}`~qpdk.cells.inductor.meander_inductor` and capacitor cells | Inductance including kinetic inductance, capacitance and parasitic coupling. |
#
# The {doc}`capacitor extraction <elmer_capacitance_interdigital>` and {doc}`CPW field simulation <comsol_cpw_resonator>` notebooks show extraction routes. This benchmark does not yet contain an extracted TWPA mask.

# %% [markdown]
# ## Limits of these results
#
# These are classical small-signal calculations. They do not establish pump stability, quantum added noise or squeezing {cite:p}`clerkIntroductionQuantumNoise2010,royIntroductionParametricAmplification2016`. Our junction shunt has no noise source. The RCSJ model also omits higher Josephson harmonics seen in real junctions {cite:p}`willschObservationJosephsonHarmonics2024`. The full-size gain comes from one solver and one harmonic basis, and the phase-mismatch estimate ignores pump depletion. Neither establishes agreement with a fabricated device.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
