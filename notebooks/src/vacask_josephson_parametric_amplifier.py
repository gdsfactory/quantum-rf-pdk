# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [raw] tags=["remove-cell"]
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///

# %% [markdown]
# # Josephson Parametric Amplifiers with VACASK Harmonic Balance
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra and the external
# [VACASK](https://codeberg.org/arpadbuermen/VACASK) circuit simulator
# (a separate program, not a Python package):
#
# ```bash
# uv add "qpdk[models]"
# # or, from a checkout of this repository:
# uv sync --extra models
# # or with pip:
# pip install "qpdk[models]"
# ```
#
# Install VACASK from its
# [release page](https://codeberg.org/arpadbuermen/VACASK/releases) and put
# `vacask` on your `PATH`, or point the `VACASK` environment variable at the
# binary. VACASK compiles the Verilog-A junction models with OpenVAF, which
# ships with VACASK releases as `openvaf-r`.
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# Josephson parametric amplifiers (JPAs) and travelling-wave parametric
# amplifiers (TWPAs) amplify a weak signal by mixing it with a strong pump in
# a nonlinear inductance
# {cite:p}`clerkIntroductionQuantumNoise2010,royIntroductionParametricAmplification2016`.
# Simulating them classically needs two things that a plain AC analysis does
# not provide:
#
# 1. the **periodic steady state** of the circuit under the pump, found with
#    harmonic balance (HB) {cite:p}`kundertSimulationNonlinearCircuits1986`, and
# 2. the **small-signal response around that pumped state**, in which a signal
#    at $f_\text{s}$ is converted to idlers at $f_\text{s} + k f_\text{p}$.
#    This is the conversion-matrix analysis of mixer design
#    {cite:p}`maasNonlinearMicrowaveRF2003`; VACASK calls it `hbac`.
#
# [Circulax](https://gdsfactory.github.io/circulax/) (see
# {doc}`circulax_transmon_optimization`) offers HB with a single fundamental
# and AC analysis only around the DC operating point, so it cannot compute
# parametric gain. SPICE-class tools with Josephson elements have been used
# for TWPAs {cite:p}`dixonCapturingComplexBehavior2020`. This notebook uses
# VACASK, an open-source SPICE-class simulator with `hb`, `hbac`, `tran` and
# `acsp` (S-parameter) analyses and Verilog-A device models compiled with
# OpenVAF {cite:p}`burmenFreeSoftwareSupport2024`.
#
# ```{note}
# VACASK is licensed under the AGPL-3.0. qpdk does not include or link any
# VACASK code: {mod}`qpdk.simulation.vacask` writes a netlist, runs the
# `vacask` binary as a separate process, and reads the `.raw` files it
# writes. The Josephson junction and SQUID Verilog-A models are part of qpdk
# (MIT); VACASK's own resistor, capacitor and transmission-line models are
# loaded from the VACASK installation at run time.
# ```
#
# The notebook covers:
#
# 1. a Verilog-A Josephson junction, checked against the analytic plasma
#    frequency and against the qpdk SAX model;
# 2. a current-pumped Kerr JPA (four-wave mixing);
# 3. a flux-pumped JPA (three-wave mixing), first lumped, then as a
#    quarter-wave coplanar-waveguide (CPW) resonator terminated by a SQUID
#    {cite:p}`yamamotoFluxdrivenJosephsonParametric2008`;
# 4. a Josephson-junction ladder TWPA, as a scaling test.
#
# ![A current-pumped Kerr JPA (junction and shunt capacitor coupled to a 50 Ω port) and a flux-pumped JPA (quarter-wave CPW terminated in a SQUID whose flux is modulated)](figures/vacask-jpa-circuits.svg)
#
# All numbers quoted in the text are the ones printed by the cells above them.

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
import time

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from sax.models import rf

from qpdk import PDK
from qpdk.config import PATH
from qpdk.models.constants import Φ_0
from qpdk.models.cpw import get_cpw_dimensions, get_cpw_substrate_params
from qpdk.models.junction import josephson_junction, squid_junction
from qpdk.simulation.vacask import (
    VACASK_MODELS_DIR,
    Netlist,
    Vacask,
    VacaskError,
    cpw_tline_params,
    format_value,
    vector_fit,
    vector_fit_subckt,
)

jax.config.update("jax_enable_x64", True)
PDK.activate()

# The checkout stylesheet is absent from installed wheels.
for _style in (PATH.docs / "qpdk.mplstyle", "qpdk"):
    try:
        plt.style.use(_style)
    except OSError:
        continue
    break

# %% [markdown]
# ## Finding VACASK
#
# {meth}`Vacask.find() <qpdk.simulation.vacask.Vacask.find>` looks at the
# `VACASK` and `QPDK_VACASK` environment variables, then at `PATH`. The rest
# of the notebook needs the binary, so it stops here if VACASK is not
# installed.

# %%
try:
    vacask = Vacask.find()
except VacaskError as error:
    print(error)
    raise SystemExit(
        "VACASK is not installed; skipping the rest of this notebook."
    ) from None
print(f"Using VACASK at {vacask.binary}")

# %% [markdown]
# Circuits and analyses are written directly in VACASK's netlist language.
# A {class}`~qpdk.simulation.vacask.Netlist` takes that text in blocks:
# `circuit` (models and instances), `control` (options, sweeps and analyses)
# and optional `subckts`, plus the device modules to `load`. It fills them into
# a Jinja2 template and copies the qpdk Verilog-A models next to the netlist.
# {meth}`Vacask.run() <qpdk.simulation.vacask.Vacask.run>` returns one result
# per `analysis` line. Python values are written into the text with f-strings;
# {func}`~qpdk.simulation.vacask.format_value` formats long vectors.
#
# Every netlist below sets `tolscale=1e-6 reltol=1e-6`: with VACASK's default
# tolerances the small-signal gain of the amplifiers below changes by more
# than the differences we want to resolve.
#
# The source conventions used throughout:
#
# - A port is a Norton source $I_\text{s}$ in parallel with $R = 50\,\Omega$.
#   The incident wave is $V_\text{inc} = I_\text{s} R / 2$ and the reflection
#   coefficient is $\Gamma = 2V/(I_\text{s} R) - 1$, with $V$ the port node
#   voltage.
# - VACASK reports a `sine` source with amplitude $A$ as the phasor
#   $-\text{j}A$, so in harmonic balance the incident phasor is
#   $-\text{j} I_\text{s} R / 2$.
# - `hbac` writes the response at $f_\text{s} + k f_\text{p}$ as `node;k`.

# %%
R_PORT = 50.0
I_SIG = 1e-9
OPTIONS = "options tolscale=1e-6 reltol=1e-6"


def db(x: jax.Array) -> jax.Array:
    """Return the power gain in dB of an amplitude ratio."""
    return 20 * jnp.log10(jnp.abs(x))


def reflection(v: jax.Array, source_current: float) -> jax.Array:
    """Reflection coefficient of a Norton port from its node voltage."""
    return 2 * jnp.asarray(v) / (source_current * R_PORT) - 1


def conversion(v: jax.Array, source_current: float) -> jax.Array:
    """Conversion gain into a sideband other than the one driven."""
    return 2 * jnp.asarray(v) / (source_current * R_PORT)


# %% [markdown]
# ## 1. A Josephson junction in Verilog-A
#
# The junction is the resistively and capacitively shunted junction (RCSJ)
# {cite:p}`McCumber1968`:
#
# ```{math}
# :label: eq:vacask-rcsj
# I = I_\text{c} \sin\varphi + \frac{V}{R} + C \frac{\text{d}V}{\text{d}t},
# \qquad
# V = \frac{\Phi_0}{2\pi} \frac{\text{d}\varphi}{\text{d}t}.
# ```
#
# The model below stores the phase $\varphi$ as the voltage of an internal
# node, so that the supercurrent is an algebraic function of a state variable.

# %%
print((VACASK_MODELS_DIR / "josephson_junction.va").read_text())

# %% [markdown]
# ### 1.1 Plasma frequency versus bias
#
# A junction in parallel with a capacitor $C$ resonates at
# $f = 1/(2\pi\sqrt{L_\text{J} C})$, where the bias-dependent Josephson
# inductance is
#
# ```{math}
# :label: eq:vacask-lj
# L_\text{J} = \frac{\Phi_0}{2\pi I_\text{c} \cos\varphi_0},
# \qquad \sin\varphi_0 = I_\text{b}/I_\text{c}.
# ```
#
# We sweep the DC bias and locate the peak of a 90 000-point AC sweep. This is
# the full netlist that VACASK runs:

# %%
IC, C_SHUNT = 1e-6, 2e-12
BIASES = jnp.array([0.0, 0.3e-6, 0.5e-6, 0.8e-6])

circuit = Netlist(
    "JJ LC resonance vs DC bias",
    loads=["josephson_junction.va", "capacitor.osdi"],
    circuit=f"""
        model isource isource
        model capacitor capacitor
        model jj jj

        iin (0 1) isource dc=0 mag=1e-9
        j1 (1 0) jj ic={IC} r=1e6
        c1 (1 0) capacitor c={C_SHUNT}
    """,
    control=f"""
        {OPTIONS}
        sweep bias instance="iin" parameter="dc" values={format_value(BIASES)}
        analysis ac1 ac from=1e9 to=10e9 mode="lin" points=90000
    """,
)
print(circuit.render())

# %%
ac = vacask.run(circuit)["ac1"]


def plasma_frequency(bias: jax.Array) -> jax.Array:
    """Small-signal resonance of the biased junction and shunt capacitor."""
    lj = Φ_0 / (2 * jnp.pi * IC * jnp.sqrt(1 - (bias / IC) ** 2))
    return 1 / (2 * jnp.pi * jnp.sqrt(lj * C_SHUNT))


f_analytic = plasma_frequency(BIASES)
f_peak = jnp.array([
    g["frequency"].real[jnp.argmax(jnp.abs(g["1"]))] for g in ac.split("bias")
])
for ib, fp_, fa in zip(BIASES, f_peak, f_analytic, strict=True):
    print(
        f"I_b = {ib * 1e6:.1f} µA: VACASK {fp_ / 1e9:.5f} GHz, "
        f"analytic {fa / 1e9:.5f} GHz, rel. error {abs(fp_ / fa - 1):.1e}"
    )

bias_fine = jnp.linspace(0, 0.95 * IC, 200)
fig, (ax, ax_ac) = plt.subplots(1, 2, figsize=(10, 4))
ax.plot(bias_fine * 1e6, plasma_frequency(bias_fine) / 1e9, label="Analytic")
ax.plot(BIASES * 1e6, f_peak / 1e9, "o", label="VACASK")
ax.set_xlabel(r"DC bias $I_\text{b}$ (µA)")
ax.set_ylabel("Resonance (GHz)")
ax.legend()
for group, ib in zip(ac.split("bias"), BIASES, strict=True):
    ax_ac.semilogy(
        group["frequency"].real / 1e9,
        jnp.abs(jnp.asarray(group["1"])),
        label=f"{ib * 1e6:.1f} µA",
    )
ax_ac.set_xlabel("Frequency (GHz)")
ax_ac.set_ylabel(r"$|V_1|$ for 1 nA drive (V)")
ax_ac.legend(title="Bias")
fig.tight_layout()
plt.show()

# %% [markdown]
# The peaks agree with Eq. {eq}`eq:vacask-lj` to within the 0.1 MHz frequency
# grid.
#
# ### 1.2 S-parameters against the SAX model
#
# The same junction, with a 2 pF shunt and a 0.5 µA bias, is put across a
# 50 Ω port of an `acsp` analysis. We convert the simulated $S_{11}$ to the
# load admittance and compare it with
# {func}`~qpdk.models.junction.josephson_junction`. That model is a
# series two-port with $S_{11} = 1/(1 + Y)$, so its admittance is
# $Y = 1/S_{11} - 1$.

# %%
f_sp = jnp.linspace(3e9, 9e9, 601)
circuit = Netlist(
    "JJ + C S-parameters",
    loads=["josephson_junction.va", "capacitor.osdi", "resistor.osdi"],
    circuit=f"""
        model isource isource
        model vsource vsource
        model resistor resistor
        model capacitor capacitor
        model jj jj

        ib (0 1) isource dc=0.5e-6
        vp (a 0) vsource dc=0
        rp (a 1) resistor r={R_PORT}
        j1 (1 0) jj ic={IC} r=10e3
        c1 (1 0) capacitor c={C_SHUNT}
    """,
    control=f"""
        {OPTIONS}
        analysis sp acsp ports=["vp", "rp"] values={format_value(f_sp)}
    """,
)
s11 = jnp.asarray(vacask.run(circuit)["sp"]["s(1,1)"])
y_vacask = (1 - s11) / (R_PORT * (1 + s11))

s_sax = josephson_junction(
    f=f_sp, ic=IC, capacitance=C_SHUNT, resistance=10e3, ib=0.5e-6
)
y_sax = 1 / s_sax["o1", "o1"] - 1
print(f"max |Y_VACASK / Y_SAX - 1| = {jnp.max(jnp.abs(y_vacask / y_sax - 1)):.1e}")

fig, ax = plt.subplots()
ax.plot(f_sp / 1e9, y_sax.imag * 1e3, label="SAX")
ax.plot(f_sp / 1e9, y_vacask.imag * 1e3, "--", label="VACASK acsp")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$\text{Im}\,Y$ (mS)")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. Current-pumped Kerr JPA
#
# A junction ($I_\text{c} = 1$ µA) in parallel with 2 pF is coupled to the
# 50 Ω port through 0.185 pF (left of the figure above). Its resonance sits
# slightly above the 5.84 GHz pump. A strong pump at $f_\text{p}$ modulates
# $L_\text{J}$ at $2 f_\text{p}$, so a signal at $f_\text{p} + \delta$ is
# amplified and an idler appears at $f_\text{p} - \delta$ (four-wave mixing,
# $2 f_\text{p} = f_\text{s} + f_\text{i}$)
# {cite:p}`royIntroductionParametricAmplification2016`.
#
# The pump source also injects the small signal on its first-harmonic
# sideband (`spur={[1]}`), so `hbac` offsets are relative to the pump:
# `in;1` is the reflected signal and `in;-1` is the idler.

# %%
F_PUMP_KERR = 5.84e9
KERR_LOADS = ["josephson_junction.va", "capacitor.osdi", "resistor.osdi"]


def kerr_circuit(pump_source: str) -> str:
    """Kerr JPA circuit with the given pump source parameters."""
    return f"""
        model isource isource
        model resistor resistor
        model capacitor capacitor
        model jj jj

        ipump (0 in) isource dc=0 {pump_source}
        rpump (in 0) resistor r={R_PORT}
        cc (in 1) capacitor c=0.185e-12
        j1 (1 0) jj ic={IC}
        c1 (1 0) capacitor c={C_SHUNT}
    """


KERR_PUMP = (
    f'type="sine" sinedc=0 ampl=40e-9 freq={F_PUMP_KERR} spur={{[1]}} smag=[{I_SIG}]'
)
pumps = [38e-9, 40e-9, 41e-9, 42e-9, 100e-9]
circuit = Netlist(
    "Kerr JPA",
    loads=KERR_LOADS,
    circuit=kerr_circuit(KERR_PUMP),
    control=f"""
        {OPTIONS}
        sweep pump instance="ipump" parameter="ampl" values={format_value(pumps)}
        analysis hbac1 hbac freq=[{F_PUMP_KERR}] nharm=7 outspur={{[1],[-1]}} \\
            from=0.5e6 to=40e6 mode="lin" points=79
    """,
)
start = time.perf_counter()
plot = vacask.run(circuit)["hbac1"]
print(f"5 pump amplitudes × 80 offsets in {time.perf_counter() - start:.2f} s")

fig, (ax, ax_idler) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for group in plot.split("pump"):
    offset = jnp.asarray(group["frequency"].real)
    gain = db(reflection(group["in;1"], I_SIG))
    idler = db(conversion(group["in;-1"], I_SIG))
    manley_rowe = 10 ** (gain[0] / 10) - 10 ** (idler[0] / 10)
    i5 = jnp.argmin(jnp.abs(offset - 5e6))
    label = f"{group['pump'][0].real * 1e9:.0f} nA"
    print(
        f"I_p = {group['pump'][0].real * 1e9:5.1f} nA: "
        f"G = {gain[0]:6.2f} dB, idler {idler[0]:6.2f} dB at 0.5 MHz; "
        f"G = {gain[i5]:6.2f} dB at 5 MHz; "
        f"|S_ss|² - |S_is|² = {manley_rowe:.5f}"
    )
    ax.plot(offset / 1e6, gain, label=label)
    ax_idler.plot(offset / 1e6, idler, label=label)
ax.set_xlabel(r"Signal offset $f_\text{s} - f_\text{p}$ (MHz)")
ax.set_ylabel("Gain (dB)")
ax.set_title("Signal")
ax_idler.set_xlabel(r"Signal offset $f_\text{s} - f_\text{p}$ (MHz)")
ax_idler.set_title("Idler")
ax_idler.legend(title="Pump")
fig.tight_layout()
plt.show()

# %% [markdown]
# For a lossless amplifier the Manley–Rowe relations
# {cite:p}`manleyGeneralPropertiesNonlinear1956` require equal signal and
# idler photon fluxes, so $|S_\text{ss}|^2 - |S_\text{is}|^2 = 1$
# {cite:p}`clerkIntroductionQuantumNoise2010`. The simulation satisfies it to
# 1.4e-3 at 40 nA (11.8 dB of gain). At 41 nA (23.1 dB) the difference is 1.030;
# we have not established where the 3 % comes from.
#
# The gain is not monotonic in the pump: the pump also shifts the resonance
# through the Kerr effect, so the operating point moves through and past the
# resonance between 41 nA and 42 nA. At 100 nA the gain has fallen to about
# 1 dB.
#
# ### 2.1 Transient check at low gain
#
# A transient simulation with pump and signal as two sine sources should give
# the same gain as `hbac`. We only make this comparison at 100 nA, where the
# gain is about 1 dB: near the gain peak the transient settles slowly and its
# answer depends on the simulated time, so it does not test anything there.
# The signal is offset by 10 MHz and the phasors are extracted from the last
# 400 ns of a 600 ns run.

# %%
F_OFFSET = 10e6
I_SIG_TRAN = 0.5e-9
circuit = Netlist(
    "Kerr JPA transient",
    loads=KERR_LOADS,
    circuit=kerr_circuit(f'type="sine" sinedc=0 ampl=100e-9 freq={F_PUMP_KERR}')
    + f"""
        isig (0 in) isource type="sine" sinedc=0 ampl={I_SIG_TRAN} \\
            freq={F_PUMP_KERR + F_OFFSET}
    """,
    control=f"""
        {OPTIONS}
        analysis tr tran stop=600e-9 step=2e-12 maxstep=2e-12
    """,
)
start = time.perf_counter()
tran = vacask.run(circuit)["tr"]
print(
    f"Transient: {len(tran['time'])} time points in {time.perf_counter() - start:.1f} s"
)

t_uniform = jnp.arange(200e-9, 600e-9, 2e-12)
v_uniform = jnp.interp(t_uniform, jnp.asarray(tran["time"]), jnp.asarray(tran["in"]))


def phasor(freq: float) -> jax.Array:
    """Cosine-referenced phasor of ``v_uniform`` at ``freq``."""
    return 2 * jnp.mean(v_uniform * jnp.exp(-2j * jnp.pi * freq * t_uniform))


incident = -1j * I_SIG_TRAN * R_PORT / 2
gain_tran = db(phasor(F_PUMP_KERR + F_OFFSET) / incident - 1)
idler_tran = db(phasor(F_PUMP_KERR - F_OFFSET) / incident)

circuit = Netlist(
    "Kerr JPA, 100 nA pump",
    loads=KERR_LOADS,
    circuit=kerr_circuit(KERR_PUMP.replace("ampl=40e-9", "ampl=100e-9")),
    control=f"""
        {OPTIONS}
        analysis hbac1 hbac freq=[{F_PUMP_KERR}] nharm=7 outspur={{[1],[-1]}} \\
            values=[{F_OFFSET}]
    """,
)
plot = vacask.run(circuit)["hbac1"]
gain_hb = db(reflection(plot["in;1"], I_SIG))[0]
idler_hb = db(conversion(plot["in;-1"], I_SIG))[0]
print(f"transient: G = {gain_tran:.3f} dB, idler = {idler_tran:.3f} dB")
print(f"hbac:      G = {gain_hb:.3f} dB, idler = {idler_hb:.3f} dB")

# Spectrum of the steady-state part of the transient around the pump
window = jnp.hanning(len(t_uniform))
spectrum = jnp.fft.rfft(v_uniform * window) * 2 / window.sum()
f_fft = jnp.fft.rfftfreq(len(t_uniform), 2e-12)
near = jnp.abs(f_fft - F_PUMP_KERR) < 40e6
fig, ax = plt.subplots()
ax.plot((f_fft[near] - F_PUMP_KERR) / 1e6, db(spectrum[near]))
for x, name in ((0, "pump"), (F_OFFSET, "signal"), (-F_OFFSET, "idler")):
    ax.axvline(x / 1e6, color="gray", ls=":")
    ax.annotate(name, (x / 1e6, 0.97), xycoords=("data", "axes fraction"), ha="center")
ax.set_xlabel(r"$f - f_\text{p}$ (MHz)")
ax.set_ylabel(r"$|V_\text{in}|$ (dBV)")
plt.show()

# %% [markdown]
# ## 3. Flux-pumped JPA
#
# In a flux-pumped JPA a SQUID terminates a resonator and the pump modulates
# the flux through the SQUID loop at $f_\text{p} \approx 2 f_0$
# {cite:p}`yamamotoFluxdrivenJosephsonParametric2008`. A signal at
# $f_\text{s}$ is amplified and an idler appears at $f_\text{p} - f_\text{s}$
# (three-wave mixing).
#
# Our SQUID model treats the symmetric SQUID as a single junction with
# critical current $I_\text{c,tot} \cos(\pi\Phi/\Phi_0)$ and neglects the loop
# inductance. That is valid for a screening parameter
# $\beta_L = 2 L_\text{loop} I_\text{c} / \Phi_0 \ll 1$
# {cite:p}`tescheDcSQUIDNoise1977`. The flux is the voltage of the third
# terminal, in units of $\Phi_0$.

# %%
print((VACASK_MODELS_DIR / "squid.va").read_text())

# %% [markdown]
# Here the signal is injected on the zeroth sideband (`spur={[0]}`), so the
# `hbac` sweep variable is the absolute signal frequency. The idler is
# `in;-1`, at $f_\text{s} - f_\text{p} = -f_\text{i}$.
#
# ### 3.1 Lumped flux-pumped JPA
#
# We start with the Kerr JPA circuit, with the junction replaced by a SQUID
# ($I_\text{c,tot} = 2$ µA) biased at $0.3\,\Phi_0$. The pump is at 12.9358 GHz,
# twice the resonance.

# %%
F_PUMP_LUMPED = 12.9358e9
SQUID_LOADS = ["squid.va", "capacitor.osdi", "resistor.osdi"]
SMALL_SIGNAL = f"mag=1e-9 spur={{[0]}} smag=[{I_SIG}]"


def lumped_circuit(source: str = SMALL_SIGNAL, pump: float = 0.01) -> str:
    """Lumped flux-pumped JPA circuit."""
    return f"""
        model isource isource
        model vsource vsource
        model resistor resistor
        model capacitor capacitor
        model squid squid

        iin (0 in) isource dc=0 {source}
        rport (in 0) resistor r={R_PORT}
        cc (in 1) capacitor c=0.185e-12
        s1 (1 0 fl) squid ic_tot=2e-6
        c1 (1 0) capacitor c={C_SHUNT}
        vfl (fl 0) vsource type="sine" sinedc=0.3 ampl={pump} freq={F_PUMP_LUMPED}
    """


pumps_flux = [0.002, 0.005, 0.008, 0.010, 0.011, 0.012]
circuit = Netlist(
    "Flux-pumped JPA",
    loads=SQUID_LOADS,
    circuit=lumped_circuit(),
    control=f"""
        {OPTIONS}
        sweep pump instance="vfl" parameter="ampl" values={format_value(pumps_flux)}
        analysis hbac1 hbac freq=[{F_PUMP_LUMPED}] nharm=5 outspur={{[0],[-1]}} \\
            from=6.3679e9 to=6.5679e9 mode="lin" points=400
    """,
)
plot = vacask.run(circuit)["hbac1"]

fig, ax = plt.subplots()
for group in plot.split("pump"):
    f = jnp.asarray(group["frequency"].real)
    gain = db(reflection(group["in;0"], I_SIG))
    k = gain.argmax()
    above = f[gain >= gain[k] - 3]
    bandwidth = above.max() - above.min()
    gbw = 10 ** (gain[k] / 20) * bandwidth
    print(
        f"pump {group['pump'][0].real:.3f} Φ0: peak {gain[k]:5.2f} dB at "
        f"{f[k] / 1e9:.4f} GHz, 3 dB bandwidth {bandwidth / 1e6:5.1f} MHz, "
        f"√G·B = {gbw / 1e6:5.1f} MHz"
    )
    ax.plot(f / 1e9, gain, label=f"{group['pump'][0].real:.3f}")
ax.set_xlabel("Signal frequency (GHz)")
ax.set_ylabel("Signal gain (dB)")
ax.legend(title=r"Pump ($\Phi_0$)")
plt.show()

# %% [markdown]
# Below about 3 dB of gain the 3 dB bandwidth is wider than the 200 MHz sweep and
# is not resolved. Above about 10 dB the product $\sqrt{G}\,B$ settles near
# 170 MHz (169.5–173.7 MHz in this run), as expected for the fixed gain–bandwidth product of a single-mode
# degenerate amplifier
# {cite:p}`royIntroductionParametricAmplification2016,eichlerControllingDynamicRange2014`.
#
# ### 3.2 Compression
#
# A two-tone `hb` analysis (pump plus a finite signal) gives the large-signal
# gain. The signal is offset from $f_\text{p}/2$ by 5 MHz, the pump is
# 0.011 $\Phi_0$, and the signal current is swept from 1 pA to 30 nA. The
# input power is $P = (I_\text{s} R / 2)^2 / (2R)$. In SQUID-based JPAs the
# saturation is mostly set by the Kerr shift of the resonance rather than by
# pump depletion {cite:p}`eichlerControllingDynamicRange2014`.

# %%
F_SIG = 6.4729e9
circuit = Netlist(
    "Flux-pumped JPA compression",
    loads=SQUID_LOADS,
    circuit=lumped_circuit(f'type="sine" sinedc=0 ampl=1e-12 freq={F_SIG}', pump=0.011),
    control=f"""
        {OPTIONS}
        sweep sig instance="iin" parameter="ampl" from=1e-12 to=30e-9 mode="dec" points=8
        analysis hb1 hb freq=[{F_PUMP_LUMPED}, {F_SIG}] nharm=[3, 5] immax=7
    """,
)
plot = vacask.run(circuit)["hb1"]

groups = plot.split("sig")
source_current = jnp.array([g["sig"][0].real for g in groups])
v_signal = jnp.array([
    g["in"][jnp.argmin(jnp.abs(jnp.asarray(g["frequency"].real) - F_SIG))]
    for g in groups
])
gain_large = db(v_signal / (-1j * source_current * R_PORT / 2) - 1)
power_dbm = 10 * jnp.log10((source_current * R_PORT / 2) ** 2 / (2 * R_PORT) / 1e-3)
p1db = jnp.interp(gain_large[0] - 1, gain_large[::-1], power_dbm[::-1])
print(f"small-signal gain {gain_large[0]:.2f} dB, input P1dB ≈ {p1db:.1f} dBm")

fig, ax = plt.subplots()
ax.plot(power_dbm, gain_large, "o-")
ax.axhline(gain_large[0] - 1, color="gray", ls=":")
ax.axvline(p1db, color="gray", ls=":")
ax.set_xlabel("Input signal power (dBm)")
ax.set_ylabel("Signal gain (dB)")
plt.show()

# %% [markdown]
# ### 3.3 Quarter-wave CPW resonator
#
# Following {cite:t}`yamamotoFluxdrivenJosephsonParametric2008`, the
# resonator is a CPW whose far end is shorted to ground through the SQUID and
# whose near end is coupled to the port through a capacitor $C_\text{c}$
# (right of the figure at the top). We use the PDK `cpw` cross-section, a
# 4680 µm line ($\lambda/4$ at 6.5 GHz) and $C_\text{c} = 20$ fF.
#
# There are two ways to put a CPW into a VACASK netlist:
#
# - **Route A**: an ideal transmission line, `tline_ideal`, with the
#   characteristic impedance and delay from
#   {func}`~qpdk.simulation.vacask.cpw_tline_params`. This is exact for a
#   lossless TEM line.
# - **Route B**: a rational model of any S-parameter model, fitted by
#   {func}`~qpdk.simulation.vacask.vector_fit` (scikit-rf vector fitting with
#   passivity enforcement
#   {cite:p}`gustavsenRationalApproximationFrequency1999`) and emitted as a
#   VACASK subcircuit by {func}`~qpdk.simulation.vacask.vector_fit_subckt`.
#   This works for any linear SAX model, including ones without a closed-form
#   circuit.
#
# Both are compared against the SAX model of the same coupling capacitor and
# line. `sax.models.rf.coplanar_waveguide` returns S-parameters referenced to
# the line's own impedance, so we renormalize to 50 Ω through ABCD matrices.
# The PDK loss tangent is set to zero so that all three describe the same
# lossless line.

# %%
LENGTH = 4680.0
C_COUPLING = 20e-15
width, gap = get_cpw_dimensions("cpw")
h_sub, t_metal, ep_r, _ = get_cpw_substrate_params()
z0_line, td_line = cpw_tline_params(LENGTH)
print(f"Z0 = {z0_line:.4f} Ω, delay = {td_line * 1e12:.4f} ps")


def s_to_abcd(s: jax.Array, z0: float) -> jax.Array:
    """ABCD matrices of two-port S-matrices ``(..., 2, 2)`` referenced to ``z0``."""
    s11, s12, s21, s22 = s[..., 0, 0], s[..., 0, 1], s[..., 1, 0], s[..., 1, 1]
    a = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    b = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    c = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21 * z0)
    d = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return jnp.stack([jnp.stack([a, b], -1), jnp.stack([c, d], -1)], -2)


def abcd_to_s(abcd: jax.Array, z0: float) -> jax.Array:
    """Two-port S-matrices of ABCD matrices, referenced to ``z0``."""
    a, b, c, d = abcd[..., 0, 0], abcd[..., 0, 1], abcd[..., 1, 0], abcd[..., 1, 1]
    den = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / den
    s12 = 2 * (a * d - b * c) / den
    s22 = (-a + b / z0 - c * z0 + d) / den
    return jnp.stack([jnp.stack([s11, s12], -1), jnp.stack([2 / den, s22], -1)], -2)


def coupled_line_abcd(f: jax.Array) -> jax.Array:
    """ABCD matrix of the coupling capacitor followed by the CPW."""
    line = rf.coplanar_waveguide(
        f=f,
        length=LENGTH,
        width=width,
        gap=gap,
        thickness=t_metal,
        substrate_thickness=h_sub,
        ep_r=ep_r,
        tand=0.0,
    )
    s = jnp.stack(
        [
            jnp.stack([line["o1", "o1"], line["o1", "o2"]], -1),
            jnp.stack([line["o2", "o1"], line["o2", "o2"]], -1),
        ],
        -2,
    )
    one, zero = jnp.ones_like(f), jnp.zeros_like(f)
    z_cap = 1 / (2j * jnp.pi * f * C_COUPLING)
    abcd_cap = jnp.stack([jnp.stack([one, z_cap], -1), jnp.stack([zero, one], -1)], -2)
    return abcd_cap @ s_to_abcd(s, z0_line)


def coupled_line_sdict(f: jax.Array) -> dict:
    """50 Ω S-parameters of the coupling capacitor followed by the CPW."""
    s = abcd_to_s(coupled_line_abcd(f), R_PORT)
    return {(f"o{i + 1}", f"o{j + 1}"): s[:, i, j] for i in range(2) for j in range(2)}


f_fit = jnp.linspace(10e6, 70e9, 3500)
start = time.perf_counter()
fit = vector_fit(coupled_line_sdict(f_fit), f_fit, ports=["o1", "o2"])
reference_fit = coupled_line_sdict(f_fit)
fit_error = max(
    jnp.max(jnp.abs(fit.get_model_response(i, j, f_fit) - reference_fit[pi, pj]))
    for i, pi in enumerate(("o1", "o2"))
    for j, pj in enumerate(("o1", "o2"))
)
print(
    f"rational fit: {len(fit.poles)} poles, max |ΔS| = {fit_error:.1e}, "
    f"{time.perf_counter() - start:.1f} s"
)

# %% [markdown]
# Route A is a hand-written subcircuit; Route B is generated. Both have
# terminals `p1` (port side) and `p2` (far end of the line):

# %%
ROUTE_A = f"""
    subckt ccline (p1 p2)
      cc (p1 1) capacitor c={C_COUPLING}
      t1 (1 0 p2 0) tline_ideal z0={z0_line} td={td_line}
    ends
"""
ROUTE_B = vector_fit_subckt(fit, "ccline")
print("\n".join(ROUTE_B.splitlines()[:12]), "\n...")

# %%
f_check = jnp.linspace(1e9, 20e9, 1901)
LINE_LOADS = ["capacitor.osdi", "resistor.osdi", "tline_ideal.osdi"]
ports = ["vp1", "rp1", "vp2", "rp2"]


def two_port_netlist(route: str) -> Netlist:
    """S-parameters of the coupled line, Route A or Route B."""
    return Netlist(
        f"Coupled CPW, Route {route}",
        loads=LINE_LOADS,
        subckts=[ROUTE_A if route == "A" else ROUTE_B],
        circuit=f"""
            model vsource vsource
            model resistor resistor
            model capacitor capacitor
            model tline_ideal tline_ideal

            x1 (1 2) ccline
            vp1 (a1 0) vsource dc=0
            rp1 (a1 1) resistor r={R_PORT}
            vp2 (a2 0) vsource dc=0
            rp2 (a2 2) resistor r={R_PORT}
        """,
        control=f"""
            {OPTIONS}
            analysis sp acsp ports={format_value(ports)} values={format_value(f_check)}
        """,
    )


results = {route: vacask.run(two_port_netlist(route))["sp"] for route in "AB"}
reference = coupled_line_sdict(f_check)
for key, (i, j) in {"s(1,1)": (0, 0), "s(2,1)": (1, 0), "s(2,2)": (1, 1)}.items():
    ref = reference[f"o{i + 1}", f"o{j + 1}"]
    errors = {
        route: jnp.max(jnp.abs(jnp.asarray(results[route][key]) - ref))
        for route in "AB"
    }
    print(
        f"{key}: Route A max|ΔS| = {errors['A']:.1e}, "
        f"Route B max|ΔS| = {errors['B']:.1e}"
    )

fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10, 4))
ax.plot(f_check / 1e9, db(reference["o2", "o1"]), label="SAX")
ax.plot(f_check / 1e9, db(results["A"]["s(2,1)"]), "--", label="Route A (tline_ideal)")
ax.plot(f_check / 1e9, db(results["B"]["s(2,1)"]), ":", label="Route B (rational fit)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$|S_{21}|$ (dB)")
ax.legend()
for route in "AB":
    error = jnp.abs(jnp.asarray(results[route]["s(2,1)"]) - reference["o2", "o1"])
    ax_err.semilogy(f_check / 1e9, error, label=f"Route {route}")
ax_err.set_xlabel("Frequency (GHz)")
ax_err.set_ylabel(r"$|S_{21} - S_{21}^\text{SAX}|$")
ax_err.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# Route A reproduces SAX to rounding error. Route B differs from SAX by about
# $10^{-3}$ in $S$, which is enough for linear S-parameters but, as shown
# below, not for a high-Q resonator.
#
# ### 3.4 Resonance versus flux
#
# The resonator now ends in a SQUID with $I_\text{c,tot} = 4$ µA. We find the
# resonance from the peak group delay of the reflection,
# $\tau = -\text{d}\arg\Gamma/\text{d}\omega$. For a lossless, overcoupled
# resonator the peak delay is $4/\kappa$, which also gives the linewidth
# $\kappa$. The SAX reference terminates the coupled-line ABCD matrix with the
# admittance of {func}`~qpdk.models.junction.squid_junction`.

# %%
IC_SQUID = 4e-6


def cpw_jpa_netlist(
    control: str,
    *,
    route: str = "A",
    flux: float = 0.3,
    pump: float = 0.0,
    f_pump: float = 1e9,
) -> Netlist:
    """Quarter-wave CPW JPA netlist using Route A or Route B for the line."""
    return Netlist(
        "Quarter-wave CPW JPA",
        loads=[*LINE_LOADS, "squid.va"],
        subckts=[ROUTE_A if route == "A" else ROUTE_B],
        circuit=f"""
            model isource isource
            model vsource vsource
            model resistor resistor
            model capacitor capacitor
            model tline_ideal tline_ideal
            model squid squid

            iin (0 in) isource dc=0 {SMALL_SIGNAL}
            rport (in 0) resistor r={R_PORT}
            x1 (in 2) ccline
            s1 (2 0 fl) squid ic_tot={IC_SQUID}
            vfl (fl 0) vsource type="sine" sinedc={flux} ampl={pump} freq={f_pump}
        """,
        control=f"{OPTIONS}\n{control}",
    )


def resonance(f: jax.Array, gamma: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Resonance frequency and linewidth (Hz) from the group delay of Γ."""
    tau = -jnp.gradient(jnp.unwrap(jnp.angle(gamma)), 2 * jnp.pi * f)
    k = tau.argmax()
    return f[k], 4 / tau[k] / (2 * jnp.pi)


def sax_reflection(f: jax.Array, flux: float) -> jax.Array:
    """Reflection of the SAX coupled line terminated by the SAX SQUID."""
    abcd = coupled_line_abcd(f)
    squid = squid_junction(
        f=f, ic_tot=IC_SQUID, capacitance=0.0, resistance=1e12, flux=flux * Φ_0
    )
    y_load = 1 / squid["o1", "o1"] - 1
    z_in = (abcd[:, 0, 0] / y_load + abcd[:, 0, 1]) / (
        abcd[:, 1, 0] / y_load + abcd[:, 1, 1]
    )
    return (z_in - R_PORT) / (z_in + R_PORT)


fluxes = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
ac = vacask.run(
    cpw_jpa_netlist(f"""
        sweep flux instance="vfl" parameter="sinedc" values={format_value(fluxes)}
        analysis ac1 ac from=5.4e9 to=6.2e9 mode="lin" points=160001
    """)
)["ac1"]
f_grid = jnp.linspace(5.4e9, 6.2e9, 160001)
f_res = jnp.array([
    resonance(jnp.asarray(g["frequency"].real), reflection(g["in"], I_SIG))[0]
    for g in ac.split("flux")
])
f_sax = jnp.array([
    resonance(f_grid, sax_reflection(f_grid, flux))[0] for flux in fluxes
])
for flux, f0, f0_sax in zip(fluxes, f_res, f_sax, strict=True):
    print(f"Φ = {flux:.2f} Φ0: VACASK {f0 / 1e9:.5f} GHz, SAX {f0_sax / 1e9:.5f} GHz")

fluxes_fine = jnp.linspace(0, 0.4, 41)
f_sax_fine = jnp.array([
    resonance(f_grid, sax_reflection(f_grid, flux))[0] for flux in fluxes_fine
])
fig, ax = plt.subplots()
ax.plot(fluxes_fine, f_sax_fine / 1e9, "-", label="SAX")
ax.plot(fluxes, f_res / 1e9, "o", label="VACASK")
ax.set_xlabel(r"Flux bias ($\Phi_0$)")
ax.set_ylabel(r"$f_0$ (GHz)")
ax.legend()
plt.show()

# %% [markdown]
# ### 3.5 Operating point and the single-mode threshold
#
# For a resonator whose frequency is modulated as
# $\omega_0(t) = \omega_0 + \delta\omega \cos(2\omega_0 t)$, single-mode
# theory {cite:p}`royIntroductionParametricAmplification2016` puts the
# parametric threshold at $\delta\omega = \kappa$, above which the device
# becomes a parametric oscillator, and the gain at the centre of the band at
#
# ```{math}
# :label: eq:vacask-3wm-gain
# G_0 = \left(\frac{1 + r^2}{1 - r^2}\right)^2,
# \qquad r = \frac{\delta\omega}{\kappa}
#       = \frac{|\text{d}f_0/\text{d}\Phi|\,\Phi_\text{p}}{\kappa / 2\pi}.
# ```
#
# We measure $f_0$, $\kappa$ and $\text{d}f_0/\text{d}\Phi$ at
# $\Phi = 0.3\,\Phi_0$ from AC sweeps.

# %%
FLUX_BIAS = 0.3
operating_fluxes = [FLUX_BIAS - 1e-3, FLUX_BIAS, FLUX_BIAS + 1e-3]
ac = vacask.run(
    cpw_jpa_netlist(f"""
        sweep flux instance="vfl" parameter="sinedc" values={format_value(operating_fluxes)}
        analysis ac1 ac from=5.85e9 to=6.0e9 mode="lin" points=150001
    """)
)["ac1"]
operating = [
    resonance(jnp.asarray(g["frequency"].real), reflection(g["in"], I_SIG))
    for g in ac.split("flux")
]
(f_lo, _), (F0, KAPPA), (f_hi, _) = (tuple(map(float, x)) for x in operating)
DF_DPHI = (f_hi - f_lo) / 2e-3
PUMP_THRESHOLD = KAPPA / abs(DF_DPHI)
print(f"f0 = {F0 / 1e9:.6f} GHz, κ/2π = {KAPPA / 1e6:.3f} MHz, Q = {F0 / KAPPA:.0f}")
print(f"df0/dΦ = {DF_DPHI / 1e9:.4f} GHz/Φ0")
print(f"single-mode threshold Φp = {PUMP_THRESHOLD:.5f} Φ0")

# %% [markdown]
# ### 3.6 Gain versus pump amplitude
#
# Now pump at $f_\text{p} = 2 f_0$ with amplitudes below the single-mode
# threshold and compare the gain with Eq. {eq}`eq:vacask-3wm-gain`.

# %%
F_PUMP_CPW = 2 * F0
ratios = jnp.array([0.3, 0.5, 0.7, 0.8, 0.9])
plot = vacask.run(
    cpw_jpa_netlist(
        f"""
        sweep pump instance="vfl" parameter="ampl" \\
            values={format_value(ratios * PUMP_THRESHOLD)}
        analysis hbac1 hbac freq=[{F_PUMP_CPW}] nharm=5 outspur={{[0],[-1]}} \\
            from={F0 - 1.0013 * KAPPA} to={F0 + 1.0013 * KAPPA} mode="lin" points=200
        """,
        f_pump=F_PUMP_CPW,
    )
)["hbac1"]

fig, ax = plt.subplots()
for group, r in zip(plot.split("pump"), ratios, strict=True):
    detuning = (jnp.asarray(group["frequency"].real) - F0) / KAPPA
    gain = db(reflection(group["in;0"], I_SIG))
    idler = db(conversion(group["in;-1"], I_SIG))
    k = jnp.argmin(jnp.abs(detuning))
    analytic = 20 * jnp.log10((1 + r**2) / (1 - r**2))
    print(
        f"r = {r:.1f}: peak gain {gain.max():5.2f} dB (single-mode {analytic:5.2f} dB), "
        f"|S_ss|² - |S_is|² = {10 ** (gain[k] / 10) - 10 ** (idler[k] / 10):.3f}"
    )
    (line,) = ax.plot(detuning, gain, label=f"{r:.1f}")
    ax.axhline(analytic, color=line.get_color(), ls=":", lw=1)
ax.set_xlabel(r"Signal detuning $(f_\text{s} - f_0)/(\kappa/2\pi)$")
ax.set_ylabel("Signal gain (dB)")
ax.legend(title=r"$\Phi_\text{p}$ / threshold")
ax.set_title("Solid: VACASK hbac; dotted: single-mode peak gain")
plt.show()

# %% [markdown]
# Unlike the lumped JPA, the CPW JPA gives less gain than
# Eq. {eq}`eq:vacask-3wm-gain` predicts, and the signal–idler Manley–Rowe
# balance falls below 1. Photons are leaving through other sidebands. A
# quarter-wave resonator has its next mode near $3 f_0$, which is close to
# $f_\text{s} + f_\text{p}$, so the pump can up-convert the signal into it.
# {cite:t}`wustmannNondegenerateParametricResonance2017` analyse this
# intermode conversion; the single-mode theory leaves it out.
# The next cell locates that mode and accounts for the photon flux in all
# sidebands up to $\pm 2$. A photon at $f_k$ carries power proportional to
# $f_k$, so the photon-number balance weights $|S_{k\text{s}}|^2$ by
# $f_\text{s}/|f_k|$, with a minus sign for the negative-frequency (idler-like)
# sidebands {cite:p}`clerkIntroductionQuantumNoise2010`.

# %%
ac = vacask.run(
    cpw_jpa_netlist('analysis ac1 ac from=12e9 to=22e9 mode="lin" points=100001')
)["ac1"]
f_ac = jnp.asarray(ac["frequency"].real)
v_mode = jnp.abs(jnp.asarray(ac["2"]))
peaks = jnp.flatnonzero((v_mode[1:-1] > v_mode[:-2]) & (v_mode[1:-1] > v_mode[2:])) + 1
print(
    "modes between 12 and 22 GHz:", ", ".join(f"{f_ac[p] / 1e9:.3f} GHz" for p in peaks)
)
print(f"f_s + f_p at f_s = f_0: {(F0 + F_PUMP_CPW) / 1e9:.3f} GHz")

sidebands = [0, -1, 1, -2, 2]
f_signal = F0 + 0.01 * KAPPA
photon_flux = {}
for r in (0.5, 0.9):
    group = vacask.run(
        cpw_jpa_netlist(
            f"""
            analysis hbac1 hbac freq=[{F_PUMP_CPW}] nharm=5 \\
                outspur={format_value([[k] for k in sidebands])} values=[{f_signal}]
            """,
            pump=r * PUMP_THRESHOLD,
            f_pump=F_PUMP_CPW,
        )
    )["hbac1"]
    photons = []
    for k in sidebands:
        f_k = f_signal + k * F_PUMP_CPW
        amplitude = (reflection if k == 0 else conversion)(group[f"in;{k}"][0], I_SIG)
        photons.append(jnp.sign(f_k) * jnp.abs(amplitude) ** 2 * f_signal / abs(f_k))
    photon_flux[r] = jnp.array(photons)
    parts = ", ".join(
        f"{k:+d}: {p:+.3f}" for k, p in zip(sidebands, photons, strict=True)
    )
    print(f"r = {r}: {parts}; sum = {photon_flux[r].sum():.4f}")

fig, (ax_modes, ax_bars) = plt.subplots(1, 2, figsize=(10, 4))
ax_modes.semilogy(f_ac / 1e9, v_mode)
ax_modes.axvline((F0 + F_PUMP_CPW) / 1e9, color="gray", ls=":")
ax_modes.annotate(
    r"$f_0 + f_\text{p}$",
    ((F0 + F_PUMP_CPW) / 1e9, 0.95),
    xycoords=("data", "axes fraction"),
)
ax_modes.set_xlabel("Frequency (GHz)")
ax_modes.set_ylabel(r"$|V_\text{SQUID}|$ for 1 nA drive (V)")
x = jnp.arange(len(sidebands))
for offset, (r, photons) in zip((-0.2, 0.2), photon_flux.items(), strict=True):
    ax_bars.bar(x + offset, photons, width=0.4, label=f"r = {r}")
ax_bars.set_xticks(x, [f"{k:+d}" for k in sidebands])
ax_bars.axhline(0, color="black", lw=0.8)
ax_bars.set_xlabel("Sideband $k$")
ax_bars.set_ylabel("Weighted photon flux")
ax_bars.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The weighted sum over the sidebands is 1.0003 at r = 0.5 and 1.025 at r = 0.9.
# The gain deficit relative to the single-mode model (10.7 dB against 19.6 dB at
# r = 0.9) therefore goes into the $f_\text{s}+f_\text{p}$ and
# $2f_\text{p}-f_\text{s}$ sidebands near the 3λ/4 mode (17.79 GHz), not into a
# numerical loss. A single-mode model of this resonator overestimates its gain.
#
# ```{warning}
# `hbac` linearizes around the pumped steady state whether or not that state
# is stable. Above the parametric threshold the zero-signal state is unstable
# and the `hbac` "gain" is not physical. Check the pump against the threshold
# before trusting a high gain.
# ```
#
# ### 3.7 Route B with the SQUID attached
#
# The rational subcircuit contains only resistors, capacitors and
# controlled sources, so it runs in harmonic balance too. With the SQUID
# attached, though, its residual fit error appears as loss at the high-Q
# resonance:

# %%
fig, ax = plt.subplots()
for route in "AB":
    ac = vacask.run(
        cpw_jpa_netlist(
            'analysis ac1 ac from=5.85e9 to=6.0e9 mode="lin" points=150001',
            route=route,
        )
    )["ac1"]
    f = jnp.asarray(ac["frequency"].real)
    gamma = reflection(ac["in"], I_SIG)
    f0, kappa = resonance(f, gamma)
    print(
        f"Route {route}: f0 = {f0 / 1e9:.5f} GHz, κ/2π = {kappa / 1e6:.2f} MHz, "
        f"min |Γ| = {jnp.abs(gamma).min():.3f}"
    )
    ax.plot(f / 1e9, jnp.abs(gamma), label=f"Route {route}")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$|\Gamma|$")
ax.legend()
plt.show()

# %% [markdown]
# A lossless resonator reflects everything: Route A gives min $|\Gamma|$ = 1.000.
# Route B gives min $|\Gamma|$ = 0.463 at resonance, and κ/2π = 9.08 MHz instead
# of 10.24 MHz. Its S-parameter fit error is about 1e-3. That is small on its own,
# but once the SQUID turns the line into a resonator with Q ≈ 580 it acts as an
# internal loss comparable to the coupling. Route B suits linear networks and
# low-Q matching sections. Resonators that set the amplifier's Q should use
# Route A, or a fit with tighter accuracy near resonance.
#
# ## 4. Stretch: Josephson-junction ladder TWPA
#
# A TWPA is a chain of junctions with shunt capacitors to ground. Here each
# cell is a junction with $I_\text{c} = 3.29$ µA ($L_\text{J} \approx 100$ pH)
# and a 40 fF shunt, which makes a 50 Ω line. A 6 GHz pump at half the
# critical current is applied through the 50 Ω input; the output is
# terminated in 50 Ω. The ladder is generated in Python, two lines of netlist
# per cell. There is no phase matching, so we do not expect much gain: this is
# a test of how `hbac` scales with circuit size.
#
# ![A ladder of N cells, each a series Josephson junction followed by a shunt capacitor to ground, between a 50 Ω source and a 50 Ω load](figures/vacask-twpa-ladder.svg)

# %%
IC_TWPA, C_TWPA, F_PUMP_TWPA = 3.29e-6, 40e-15, 6e9


def twpa_netlist(cells: int, pump_current: float) -> Netlist:
    """JJ-ladder TWPA netlist with ``cells`` junction–capacitor cells."""
    ladder = "\n".join(
        f"j{i} (n{i} n{i + 1}) jj ic={IC_TWPA}\nc{i} (n{i + 1} 0) capacitor c={C_TWPA}"
        for i in range(cells)
    )
    pump = f'type="sine" sinedc=0 ampl={2 * pump_current} freq={F_PUMP_TWPA}'
    return Netlist(
        f"JJ-ladder TWPA, {cells} cells",
        loads=["josephson_junction.va", "capacitor.osdi", "resistor.osdi"],
        circuit=f"""\
model isource isource
model resistor resistor
model capacitor capacitor
model jj jj

ip (0 n0) isource dc=0 {pump} spur={{[0]}} smag=[{I_SIG}]
rp (n0 0) resistor r={R_PORT}
{ladder}
rl (n{cells} 0) resistor r={R_PORT}
""",
        control=f"""
            {OPTIONS}
            analysis hbac1 hbac freq=[{F_PUMP_TWPA}] nharm=5 outspur={{[0],[-2]}} \\
                from=3e9 to=9e9 mode="lin" points=60
        """,
    )


print("\n".join(twpa_netlist(2, 0.5 * IC_TWPA).render().splitlines()[:16]), "\n...")

# %%
fig, ax = plt.subplots()
for cells in (200, 500, 1000, 2000):
    start = time.perf_counter()
    plot = vacask.run(twpa_netlist(cells, 0.5 * IC_TWPA))["hbac1"]
    elapsed = time.perf_counter() - start
    f = jnp.asarray(plot["frequency"].real)
    s21 = db(conversion(plot[f"n{cells};0"], I_SIG))
    i5 = jnp.argmin(jnp.abs(f - 5e9))
    print(f"{cells:5d} cells: {elapsed:6.1f} s, |S21| at 5 GHz = {s21[i5]:+.2f} dB")
    ax.plot(f / 1e9, s21, label=f"{cells}")
ax.set_xlabel("Signal frequency (GHz)")
ax.set_ylabel(r"$|S_{21}|$ (dB)")
ax.legend(title="Cells")
plt.show()

# %% [markdown]
# The run times above are for this notebook's build machine; they depend on the
# CPU and the BLAS library VACASK was built against. From 200 to 1000 cells the
# run time grows roughly linearly (0.4 s to 1.9 s), and $|S_{21}|$ at 5 GHz falls
# from −0.42 dB to −2.12 dB. At 2000 cells the run time jumps by a factor of about 25
# and $|S_{21}|$ turns positive (+0.51 dB); we have not investigated why.
#
# Without dispersion engineering a junction ladder lets the pump's mixing
# products propagate as freely as the signal, so the pump can convert the signal into
# sidebands at $f_\text{s} \pm 2 f_\text{p}$ and beyond instead of
# amplifying it. A practical TWPA suppresses these products with resonant phase
# matching {cite:p}`macklinNearquantumlimitedJosephsonTravelingwave2015`,
# and its harmonic-balance model must keep enough pump harmonics and
# sidebands {cite:p}`dixonCapturingComplexBehavior2020`.
#
# ## 5. Limitations
#
# - **Classical simulation.** Harmonic balance computes classical gain and
#   conversion. It says nothing about added noise or squeezing; for those use
#   the quantum input–output treatment in
#   {cite:t}`clerkIntroductionQuantumNoise2010` and
#   {cite:t}`royIntroductionParametricAmplification2016`.
# - **Stability.** `hbac` does not check whether the pumped steady state is
#   stable. Above the parametric threshold its "gain" is meaningless.
# - **Junction model.** The RCSJ model keeps only the first Josephson harmonic,
#   although real tunnel junctions show higher ones
#   {cite:p}`willschObservationJosephsonHarmonics2024`. The SQUID model
#   neglects the loop inductance ($\beta_L = 0$) and junction asymmetry.
# - **Route B accuracy.** A rational fit that is accurate to $10^{-3}$ on its
#   own can still add significant loss inside a high-Q resonator.
# - **Transient.** A transient comparison was only made at low gain. Near the
#   gain peak the transient result depends on the simulated time.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
