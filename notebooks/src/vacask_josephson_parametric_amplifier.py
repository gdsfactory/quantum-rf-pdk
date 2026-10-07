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
# {cite:p}`clerkIntroductionQuantumNoise2010,royIntroductionParametricAmplification2016,aumentadoSuperconductingParametricAmplifiers2020`.
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
# parametric gain. JosephsonCircuits.jl solves the same problem in Julia
# {cite:p}`levochkinaModelingFluxTunability2025`, and SPICE-class tools with
# Josephson elements such as WRspice and JoSIM
# {cite:p}`whiteleyJosephsonJunctionsSPICE31991,delportJoSIMSuperconductorSPICE2019`
# have been used for TWPAs {cite:p}`dixonCapturingComplexBehavior2020`. This
# notebook uses VACASK, an open-source SPICE-class simulator with `hb`, `hbac`,
# `tran` and `acsp` (S-parameter) analyses and Verilog-A device models compiled
# with OpenVAF {cite:p}`burmenFreeSoftwareSupport2024`.
#
# ```{note}
# VACASK is licensed under the AGPL-3.0. qpdk does not include or link any
# VACASK code: {mod}`qpdk.simulation.vacask` writes a netlist, runs the
# `vacask` binary as a separate process, and reads the binary `.raw` files it
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

import numpy as np
from matplotlib import pyplot as plt
from sax.models import rf
from skrf.network import a2s, s2a

from qpdk import PDK
from qpdk.models.constants import Φ_0
from qpdk.models.cpw import get_cpw_dimensions, get_cpw_substrate_params
from qpdk.models.junction import josephson_junction, squid_junction
from qpdk.simulation.vacask import (
    VacaskError,
    cpw_tline_params,
    find_vacask,
    run_vacask,
    vacask_model_path,
    vector_fit_subckt,
)

PDK.activate()

# %% [markdown]
# ## Finding VACASK
#
# {func}`~qpdk.simulation.vacask.find_vacask` looks at the `VACASK` and
# `QPDK_VACASK` environment variables, then at `PATH`. The rest of the notebook
# needs the binary, so it stops here if VACASK is not installed.

# %%
try:
    print(f"Using VACASK at {find_vacask()}")
except VacaskError as error:
    print(error)
    raise SystemExit(
        "VACASK is not installed; skipping the rest of this notebook."
    ) from None

# %% [markdown]
# Every netlist below sets `tolscale=1e-6 reltol=1e-6`. With VACASK's default
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
#   $-\mathrm{j}A$, so in harmonic balance the incident phasor is
#   $-\mathrm{j} I_\text{s} R / 2$.
# - `hbac` writes the response at $f_\text{s} + k f_\text{p}$ as `node;k`.

# %%
R_PORT = 50.0
OPTIONS = "  options tolscale=1e-6 reltol=1e-6"
JJ_FILES = {"josephson_junction.va": vacask_model_path("josephson_junction")}
SQUID_FILES = {"squid.va": vacask_model_path("squid")}


def db(x: np.ndarray) -> np.ndarray:
    """Return the power gain in dB of an amplitude ratio."""
    return 20 * np.log10(np.abs(x))


def reflection(v: np.ndarray, source_current: float) -> np.ndarray:
    """Reflection coefficient of a Norton port from its node voltage."""
    return 2 * v / (source_current * R_PORT) - 1


def conversion(v: np.ndarray, source_current: float) -> np.ndarray:
    """Conversion gain into a sideband other than the one driven."""
    return 2 * v / (source_current * R_PORT)


# %% [markdown]
# ## 1. A Josephson junction in Verilog-A
#
# The junction is the resistively and capacitively shunted junction (RCSJ)
# {cite:p}`McCumber1968,stewartCurrentvoltageCharacteristicsJosephson1968`:
#
# ```{math}
# :label: eq:vacask-rcsj
# I = I_\text{c} \sin\varphi + \frac{V}{R} + C \frac{\mathrm{d}V}{\mathrm{d}t},
# \qquad
# V = \frac{\Phi_0}{2\pi} \frac{\mathrm{d}\varphi}{\mathrm{d}t}.
# ```
#
# The model below stores the phase $\varphi$ as the voltage of an internal
# node, so that the supercurrent is an algebraic function of a state variable.

# %%
print(vacask_model_path("josephson_junction").read_text())

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
# We sweep the DC bias and locate the peak of a 90 000-point AC sweep.

# %%
IC, C_SHUNT = 1e-6, 2e-12
BIASES = np.array([0.0, 0.3e-6, 0.5e-6, 0.8e-6])

netlist = f"""JJ LC resonance vs DC bias
load "josephson_junction.va"
load "capacitor.osdi"
model isource isource
model capacitor capacitor
model jj jj
iin (0 1) isource dc=0 mag=1n
j1 (1 0) jj ic={IC} r=1e6
c1 (1 0) capacitor c={C_SHUNT}
control
{OPTIONS}
  sweep bias instance="iin" parameter="dc" values=[{", ".join(map(str, BIASES))}]
  analysis ac1 ac from=1G to=10G mode="lin" points=90000
endc
"""
ac = run_vacask(netlist, JJ_FILES)["ac1"]

lj = Φ_0 / (2 * np.pi * IC * np.sqrt(1 - (BIASES / IC) ** 2))
f_analytic = 1 / (2 * np.pi * np.sqrt(lj * C_SHUNT))
f_peak = np.array([
    g["frequency"].real[np.argmax(np.abs(g["1"]))] for g in ac.split("bias")
])
for ib, fp_, fa in zip(BIASES, f_peak, f_analytic, strict=True):
    print(
        f"I_b = {ib * 1e6:.1f} µA: VACASK {fp_ / 1e9:.5f} GHz, "
        f"analytic {fa / 1e9:.5f} GHz, rel. error {abs(fp_ / fa - 1):.1e}"
    )
np.testing.assert_allclose(f_peak, f_analytic, rtol=1e-5)

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
f_sp = np.linspace(3e9, 9e9, 601)
netlist = f"""JJ + C S-parameters
load "josephson_junction.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model vsource vsource
model capacitor capacitor
model resistor resistor
model jj jj
ib (0 1) isource dc=0.5u
vp (a 0) vsource dc=0
rp (a 1) resistor r={R_PORT}
j1 (1 0) jj ic={IC} r=10k
c1 (1 0) capacitor c={C_SHUNT}
control
{OPTIONS}
  analysis sp acsp ports=["vp","rp"] values=[{",".join(map(str, f_sp))}]
endc
"""
s11 = run_vacask(netlist, JJ_FILES)["sp"]["s(1,1)"]
y_vacask = (1 - s11) / (R_PORT * (1 + s11))

s_sax = josephson_junction(
    f=f_sp, ic=IC, capacitance=C_SHUNT, resistance=10e3, ib=0.5e-6
)
y_sax = 1 / np.asarray(s_sax["o1", "o1"]) - 1
print(f"max |Y_VACASK / Y_SAX - 1| = {np.max(np.abs(y_vacask / y_sax - 1)):.1e}")

fig, ax = plt.subplots()
ax.plot(f_sp / 1e9, y_sax.imag * 1e3, label="SAX")
ax.plot(f_sp / 1e9, y_vacask.imag * 1e3, "--", label="VACASK acsp")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$\mathrm{Im}\,Y$ (mS)")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. Current-pumped Kerr JPA
#
# A junction ($I_\text{c} = 1$ µA) in parallel with 2 pF is coupled to the
# 50 Ω port through 0.185 pF. Its resonance sits slightly above the 5.84 GHz
# pump. A strong pump at $f_\text{p}$ modulates $L_\text{J}$ at $2 f_\text{p}$,
# so a signal at $f_\text{p} + \delta$ is amplified and an idler appears at
# $f_\text{p} - \delta$ (four-wave mixing, $2 f_\text{p} = f_\text{s} +
# f_\text{i}$) {cite:p}`royIntroductionParametricAmplification2016`.
#
# The pump source also injects the small signal on its first-harmonic
# sideband (`spur={[1]}`), so `hbac` offsets are relative to the pump:
# `in;1` is the reflected signal and `in;-1` is the idler.

# %%
F_PUMP_KERR = 5.84e9
I_SIG = 1e-9


def kerr_netlist(pumps: list[float], analysis: str) -> str:
    """Kerr JPA netlist with a pump-amplitude sweep."""
    return f"""Kerr JPA
load "josephson_junction.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model capacitor capacitor
model resistor resistor
model jj jj
ipump (0 in) isource type="sine" sinedc=0 ampl=40n freq={F_PUMP_KERR} spur={{[1]}} smag=[{I_SIG}]
rp (in 0) resistor r={R_PORT}
cc (in 1) capacitor c=0.185p
j1 (1 0) jj ic={IC}
c1 (1 0) capacitor c={C_SHUNT}
control
{OPTIONS}
  sweep pump instance="ipump" parameter="ampl" values=[{", ".join(f"{p:.6g}" for p in pumps)}]
  {analysis}
endc
"""


pumps = [38e-9, 40e-9, 41e-9, 42e-9, 100e-9]
hbac = 'analysis hbac1 hbac freq=[5.84G] nharm=7 outspur={[1],[-1]} from=0.5M to=40M mode="lin" points=79'
start = time.perf_counter()
plot = run_vacask(kerr_netlist(pumps, hbac), JJ_FILES)["hbac1"]
print(f"5 pump amplitudes × 80 offsets in {time.perf_counter() - start:.2f} s")

fig, ax = plt.subplots()
for group in plot.split("pump"):
    offset = group["frequency"].real
    gain = db(reflection(group["in;1"], I_SIG))
    idler = db(conversion(group["in;-1"], I_SIG))
    manley_rowe = 10 ** (gain[0] / 10) - 10 ** (idler[0] / 10)
    i5 = np.argmin(np.abs(offset - 5e6))
    print(
        f"I_p = {group['pump'][0].real * 1e9:5.1f} nA: "
        f"G = {gain[0]:6.2f} dB, idler {idler[0]:6.2f} dB at 0.5 MHz; "
        f"G = {gain[i5]:6.2f} dB at 5 MHz; "
        f"|S_ss|² - |S_is|² = {manley_rowe:.5f}"
    )
    ax.plot(offset / 1e6, gain, label=f"{group['pump'][0].real * 1e9:.0f} nA")
ax.set_xlabel(r"Signal offset $f_\mathrm{s} - f_\mathrm{p}$ (MHz)")
ax.set_ylabel("Signal gain (dB)")
ax.legend(title="Pump")
plt.show()

# %% [markdown]
# For a lossless amplifier the Manley–Rowe relations
# {cite:p}`manleyGeneralPropertiesNonlinear1956` require equal signal and
# idler photon fluxes, so $|S_\text{ss}|^2 - |S_\text{is}|^2 = 1$
# {cite:p}`clerkIntroductionQuantumNoise2010`. The simulation satisfies it to
# $1.4\times10^{-3}$ at 40 nA and 11.8 dB of gain. At 41 nA and 23 dB the
# difference is 1.03; we have not established where the 3 % comes from.
#
# The gain is not monotonic in the pump: the pump also shifts the resonance
# through the Kerr effect, so the operating point moves through and past the
# resonance between 41 nA and 42 nA. At 100 nA the gain has fallen to about
# 1 dB. A stronger drive leads to bifurcation of the driven resonator
# {cite:p}`manucharyanMicrowaveBifurcationJosephson2007`. The model keeps the
# full $\sin\varphi$, so it includes the nonlinear terms beyond Kerr that
# reduce gain and saturation power compared with Kerr-only theory
# {cite:p}`boutinEffectHigherorderNonlinearities2017`.
#
# ### 2.1 Gain map

# %%
pumps_map = np.linspace(30e-9, 50e-9, 41)
hbac = 'analysis hbac1 hbac freq=[5.84G] nharm=7 outspur={[1],[-1]} from=0.5M to=40M mode="lin" points=79'
plot = run_vacask(kerr_netlist(list(pumps_map), hbac), JJ_FILES)["hbac1"]
groups = plot.split("pump")
offset = groups[0]["frequency"].real
gain_map = np.array([db(reflection(g["in;1"], I_SIG)) for g in groups])

fig, ax = plt.subplots()
mesh = ax.pcolormesh(offset / 1e6, pumps_map * 1e9, gain_map, shading="auto")
fig.colorbar(mesh, label="Signal gain (dB)")
ax.set_xlabel(r"Signal offset $f_\mathrm{s} - f_\mathrm{p}$ (MHz)")
ax.set_ylabel("Pump amplitude (nA)")
plt.show()

# %% [markdown]
# ### 2.2 Transient check at low gain
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
netlist = f"""Kerr JPA transient
load "josephson_junction.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model capacitor capacitor
model resistor resistor
model jj jj
ipump (0 in) isource type="sine" sinedc=0 ampl=100n freq={F_PUMP_KERR}
isig (0 in) isource type="sine" sinedc=0 ampl={I_SIG_TRAN} freq={F_PUMP_KERR + F_OFFSET}
rp (in 0) resistor r={R_PORT}
cc (in 1) capacitor c=0.185p
j1 (1 0) jj ic={IC}
c1 (1 0) capacitor c={C_SHUNT}
control
{OPTIONS}
  analysis tr tran stop=600n step=2p maxstep=2p
endc
"""
start = time.perf_counter()
tran = run_vacask(netlist, JJ_FILES)["tr"]
print(
    f"Transient: {len(tran['time'])} time points in {time.perf_counter() - start:.1f} s"
)

t_uniform = np.arange(200e-9, 600e-9, 2e-12)
v_uniform = np.interp(t_uniform, tran["time"], tran["in"])


def phasor(freq: float) -> complex:
    """Cosine-referenced phasor of ``v_uniform`` at ``freq``."""
    return 2 * np.mean(v_uniform * np.exp(-2j * np.pi * freq * t_uniform))


incident = -1j * I_SIG_TRAN * R_PORT / 2
gain_tran = db(phasor(F_PUMP_KERR + F_OFFSET) / incident - 1)
idler_tran = db(phasor(F_PUMP_KERR - F_OFFSET) / incident)

hbac = f'analysis hbac1 hbac freq=[5.84G] nharm=7 outspur={{[1],[-1]}} from={F_OFFSET} to={F_OFFSET} mode="lin" points=1'
plot = run_vacask(kerr_netlist([100e-9], hbac), JJ_FILES)["hbac1"]
gain_hb = db(reflection(plot["in;1"], I_SIG))[0]
idler_hb = db(conversion(plot["in;-1"], I_SIG))[0]
print(f"transient: G = {gain_tran:.3f} dB, idler = {idler_tran:.3f} dB")
print(f"hbac:      G = {gain_hb:.3f} dB, idler = {idler_hb:.3f} dB")

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
# {cite:p}`tescheDcSQUIDNoise1977`; a finite $\beta_L$ skews $f_0(\Phi)$ and can
# make it hysteretic {cite:p}`pogorzalekHystereticFluxResponse2017`. The flux is
# the voltage of the third terminal, in units of $\Phi_0$.

# %%
print(vacask_model_path("squid").read_text())

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


def lumped_netlist(analysis: str, sweep: str = "", source: str = "") -> str:
    """Lumped flux-pumped JPA netlist."""
    source = source or f"iin (0 in) isource dc=0 mag=1n spur={{[0]}} smag=[{I_SIG}]"
    return f"""Flux-pumped JPA
load "squid.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model vsource vsource
model capacitor capacitor
model resistor resistor
model squid squid
{source}
rp (in 0) resistor r={R_PORT}
cc (in 1) capacitor c=0.185p
s1 (1 0 fl) squid ic_tot=2u
c1 (1 0) capacitor c={C_SHUNT}
vfl (fl 0) vsource type="sine" sinedc=0.3 ampl=0.01 freq={F_PUMP_LUMPED}
control
{OPTIONS}
{sweep}
  {analysis}
endc
"""


pumps_flux = [0.002, 0.005, 0.008, 0.010, 0.011, 0.012]
plot = run_vacask(
    lumped_netlist(
        f'analysis hbac1 hbac freq=[{F_PUMP_LUMPED}] nharm=5 outspur={{[0],[-1]}} from=6.3679G to=6.5679G mode="lin" points=400',
        '  sweep pump instance="vfl" parameter="ampl" values=['
        + ", ".join(map(str, pumps_flux))
        + "]",
    ),
    SQUID_FILES,
)["hbac1"]

fig, ax = plt.subplots()
for group in plot.split("pump"):
    f = group["frequency"].real
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
# 170 MHz, as expected for the fixed gain–bandwidth product of a single-mode
# degenerate amplifier
# {cite:p}`royIntroductionParametricAmplification2016,eichlerControllingDynamicRange2014`.
# Impedance-engineered matching networks lift that limit
# {cite:p}`mutusStrongEnvironmentalCoupling2014`.
#
# ### 3.2 Compression
#
# A two-tone `hb` analysis (pump plus a finite signal) gives the large-signal
# gain. The signal is offset from $f_\text{p}/2$ by 5 MHz, the pump is
# 0.011 $\Phi_0$, and the signal current is swept from 1 pA to 30 nA. The
# input power is $P = (I_\text{s} R / 2)^2 / (2R)$. In SQUID-based JPAs the
# saturation is mostly set by the Kerr shift of the resonance rather than by
# pump depletion
# {cite:p}`eichlerControllingDynamicRange2014,planatUnderstandingSaturationPower2019`.

# %%
F_SIG = 6.4729e9
F_IDLER = F_PUMP_LUMPED - F_SIG
netlist = lumped_netlist(
    f"analysis hb1 hb freq=[{F_PUMP_LUMPED}, {F_SIG}] nharm=[3, 5] immax=7",
    '  sweep sig instance="iin" parameter="ampl" from=1p to=30n mode="dec" points=8',
    source=f'iin (0 in) isource type="sine" sinedc=0 ampl=1p freq={F_SIG}',
).replace("ampl=0.01 ", "ampl=0.011 ")
plot = run_vacask(netlist, SQUID_FILES)["hb1"]

power_dbm, gain_large = [], []
for group in plot.split("sig"):
    source_current = group["sig"][0].real
    f = group["frequency"].real
    incident = -1j * source_current * R_PORT / 2
    v_signal = group["in"][np.argmin(np.abs(f - F_SIG))]
    gain_large.append(db(v_signal / incident - 1))
    power_dbm.append(
        10 * np.log10((source_current * R_PORT / 2) ** 2 / (2 * R_PORT) / 1e-3)
    )
power_dbm, gain_large = np.array(power_dbm), np.array(gain_large)
p1db = np.interp(gain_large[0] - 1, gain_large[::-1], power_dbm[::-1])
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
# whose near end is coupled to the port through a capacitor $C_\text{c}$. We
# use the PDK `cpw` cross-section, a 4680 µm line ($\lambda/4$ at 6.5 GHz)
# and $C_\text{c} = 20$ fF.
#
# There are two ways to put a CPW into a VACASK netlist:
#
# - **Route A**: an ideal transmission line, `tline_ideal`, with the
#   characteristic impedance and delay from
#   {func}`~qpdk.simulation.vacask.cpw_tline_params`. This is exact for a
#   lossless TEM line.
# - **Route B**: a rational (vector-fitted) model of any S-parameter model,
#   emitted as a VACASK subcircuit by
#   {func}`~qpdk.simulation.vacask.vector_fit_subckt`, using vector fitting
#   with passivity enforcement
#   {cite:p}`gustavsenRationalApproximationFrequency1999,gustavsenEnforcingPassivityAdmittance2001`.
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


def coupled_line_abcd(f: np.ndarray) -> np.ndarray:
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
    s = np.stack([
        [line["o1", "o1"], line["o1", "o2"]],
        [line["o2", "o1"], line["o2", "o2"]],
    ]).transpose(2, 0, 1)
    abcd_cap = np.zeros((len(f), 2, 2), dtype=complex)
    abcd_cap[:, 0, 0] = abcd_cap[:, 1, 1] = 1
    abcd_cap[:, 0, 1] = 1 / (2j * np.pi * f * C_COUPLING)
    return abcd_cap @ s2a(np.asarray(s), z0_line)


def coupled_line_sdict(f: np.ndarray) -> dict:
    """50 Ω S-parameters of the coupling capacitor followed by the CPW."""
    s = a2s(coupled_line_abcd(f), R_PORT)
    return {(f"o{i + 1}", f"o{j + 1}"): s[:, i, j] for i in range(2) for j in range(2)}


f_fit = np.linspace(10e6, 70e9, 3500)
start = time.perf_counter()
subckt, fit = vector_fit_subckt(coupled_line_sdict(f_fit), f_fit, "ccline")
print(
    f"vector fit: {len(fit.poles)} poles, passive: {fit.is_passive()}, "
    f"RMS error {fit.get_rms_error():.1e}, {time.perf_counter() - start:.1f} s"
)

# %%
f_check = np.linspace(1e9, 20e9, 1901)
netlist = f"""Coupled CPW, two ways
load "resistor.osdi"
load "capacitor.osdi"
load "tline_ideal.osdi"
model resistor resistor
model capacitor capacitor
model tline tline_ideal
model vsource vsource
{subckt}
x1 (1 2) ccline
vp1 (a1 0) vsource dc=0
rp1 (a1 1) resistor r={R_PORT}
vp2 (a2 0) vsource dc=0
rp2 (a2 2) resistor r={R_PORT}
cc (3 4) capacitor c={C_COUPLING}
t1 (4 0 5 0) tline z0={z0_line} td={td_line}
vp3 (a3 0) vsource dc=0
rp3 (a3 3) resistor r={R_PORT}
vp4 (a4 0) vsource dc=0
rp4 (a4 5) resistor r={R_PORT}
control
  analysis spb acsp ports=["vp1","rp1","vp2","rp2"] values=[{",".join(map(str, f_check))}]
  analysis spa acsp ports=["vp3","rp3","vp4","rp4"] values=[{",".join(map(str, f_check))}]
endc
"""
results = run_vacask(netlist)
reference = coupled_line_sdict(f_check)
for key, (i, j) in {"s(1,1)": (0, 0), "s(2,1)": (1, 0), "s(2,2)": (1, 1)}.items():
    ref = reference[f"o{i + 1}", f"o{j + 1}"]
    print(
        f"{key}: Route A max|ΔS| = {np.max(np.abs(results['spa'][key] - ref)):.1e}, "
        f"Route B max|ΔS| = {np.max(np.abs(results['spb'][key] - ref)):.1e}"
    )

fig, ax = plt.subplots()
ax.plot(f_check / 1e9, db(reference["o2", "o1"]), label="SAX")
ax.plot(
    f_check / 1e9, db(results["spa"]["s(2,1)"]), "--", label="Route A (tline_ideal)"
)
ax.plot(f_check / 1e9, db(results["spb"]["s(2,1)"]), ":", label="Route B (vector fit)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$|S_{21}|$ (dB)")
ax.legend()
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
# $\tau = -\mathrm{d}\arg\Gamma/\mathrm{d}\omega$. For a lossless, overcoupled
# resonator the peak delay is $4/\kappa$, which also gives the linewidth
# $\kappa$. The SAX reference terminates the coupled-line ABCD matrix with the
# admittance of {func}`~qpdk.models.junction.squid_junction`.

# %%
IC_SQUID = 4e-6


def cpw_jpa_netlist(
    analysis: str,
    sweep: str = "",
    *,
    route: str = "A",
    flux: float = 0.3,
    pump: float = 0.0,
    f_pump: float = 1e9,
) -> str:
    """Quarter-wave CPW JPA netlist using Route A or Route B for the line."""
    if route == "A":
        line = f"cc (in 1) capacitor c={C_COUPLING}\nt1 (1 0 2 0) tline z0={z0_line} td={td_line}"
    else:
        line = "x1 (in 2) ccline"
    return f"""Quarter-wave CPW JPA
load "squid.va"
load "capacitor.osdi"
load "resistor.osdi"
load "tline_ideal.osdi"
model isource isource
model vsource vsource
model capacitor capacitor
model resistor resistor
model tline tline_ideal
model squid squid
{subckt if route == "B" else ""}
iin (0 in) isource dc=0 mag=1n spur={{[0]}} smag=[{I_SIG}]
rp (in 0) resistor r={R_PORT}
{line}
s1 (2 0 fl) squid ic_tot={IC_SQUID}
vfl (fl 0) vsource type="sine" sinedc={flux} ampl={pump} freq={f_pump}
control
{OPTIONS}
{sweep}
  {analysis}
endc
"""


def resonance(f: np.ndarray, gamma: np.ndarray) -> tuple[float, float]:
    """Resonance frequency and linewidth (Hz) from the group delay of Γ."""
    tau = -np.gradient(np.unwrap(np.angle(gamma)), 2 * np.pi * f)
    k = tau.argmax()
    return f[k], 4 / tau[k] / (2 * np.pi)


def sax_reflection(f: np.ndarray, flux: float) -> np.ndarray:
    """Reflection of the SAX coupled line terminated by the SAX SQUID."""
    abcd = coupled_line_abcd(f)
    squid = squid_junction(
        f=f, ic_tot=IC_SQUID, capacitance=0.0, resistance=1e12, flux=flux * Φ_0
    )
    y_load = 1 / np.asarray(squid["o1", "o1"]) - 1
    z_in = (abcd[:, 0, 0] / y_load + abcd[:, 0, 1]) / (
        abcd[:, 1, 0] / y_load + abcd[:, 1, 1]
    )
    return (z_in - R_PORT) / (z_in + R_PORT)


fluxes = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
ac = run_vacask(
    cpw_jpa_netlist(
        'analysis ac1 ac from=5.4G to=6.2G mode="lin" points=160001',
        '  sweep flux instance="vfl" parameter="sinedc" values=['
        + ", ".join(map(str, fluxes))
        + "]",
    ),
    SQUID_FILES,
)["ac1"]
f_res, f_sax = [], []
f_grid = np.linspace(5.4e9, 6.2e9, 160001)
for group, flux in zip(ac.split("flux"), fluxes, strict=True):
    f0, _ = resonance(group["frequency"].real, reflection(group["in"], I_SIG))
    f0_sax, _ = resonance(f_grid, sax_reflection(f_grid, flux))
    f_res.append(f0)
    f_sax.append(f0_sax)
    print(f"Φ = {flux:.2f} Φ0: VACASK {f0 / 1e9:.5f} GHz, SAX {f0_sax / 1e9:.5f} GHz")

fig, ax = plt.subplots()
ax.plot(fluxes, np.array(f_sax) / 1e9, "-", label="SAX")
ax.plot(fluxes, np.array(f_res) / 1e9, "o", label="VACASK")
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
# becomes a parametric oscillator
# {cite:p}`wustmannParametricResonanceTunable2013,krantzInvestigationNonlinearEffects2013`,
# and the gain at the centre
# of the band at
#
# ```{math}
# :label: eq:vacask-3wm-gain
# G_0 = \left(\frac{1 + r^2}{1 - r^2}\right)^2,
# \qquad r = \frac{\delta\omega}{\kappa}
#       = \frac{|\mathrm{d}f_0/\mathrm{d}\Phi|\,\Phi_\text{p}}{\kappa / 2\pi}.
# ```
#
# We measure $f_0$, $\kappa$ and $\mathrm{d}f_0/\mathrm{d}\Phi$ at
# $\Phi = 0.3\,\Phi_0$ from AC sweeps.

# %%
FLUX_BIAS = 0.3
ac = run_vacask(
    cpw_jpa_netlist(
        'analysis ac1 ac from=5.85G to=6.0G mode="lin" points=150001',
        f'  sweep flux instance="vfl" parameter="sinedc" values=[{FLUX_BIAS - 1e-3}, {FLUX_BIAS}, {FLUX_BIAS + 1e-3}]',
    ),
    SQUID_FILES,
)["ac1"]
operating = [
    resonance(group["frequency"].real, reflection(group["in"], I_SIG))
    for group in ac.split("flux")
]
(f_lo, _), (F0, KAPPA), (f_hi, _) = operating
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
ratios = np.array([0.3, 0.5, 0.7, 0.8, 0.9])
plot = run_vacask(
    cpw_jpa_netlist(
        f'analysis hbac1 hbac freq=[{F_PUMP_CPW}] nharm=5 outspur={{[0],[-1]}} from={F0 - 1.0013 * KAPPA} to={F0 + 1.0013 * KAPPA} mode="lin" points=200',
        '  sweep pump instance="vfl" parameter="ampl" values=['
        + ", ".join(f"{p:.8g}" for p in ratios * PUMP_THRESHOLD)
        + "]",
        f_pump=F_PUMP_CPW,
    ),
    SQUID_FILES,
)["hbac1"]

fig, ax = plt.subplots()
for group, r in zip(plot.split("pump"), ratios, strict=True):
    detuning = (group["frequency"].real - F0) / KAPPA
    gain = db(reflection(group["in;0"], I_SIG))
    idler = db(conversion(group["in;-1"], I_SIG))
    k = np.argmin(np.abs(detuning))
    analytic = 20 * np.log10((1 + r**2) / (1 - r**2))
    print(
        f"r = {r:.1f}: peak gain {gain.max():5.2f} dB (single-mode {analytic:5.2f} dB), "
        f"|S_ss|² - |S_is|² = {10 ** (gain[k] / 10) - 10 ** (idler[k] / 10):.3f}"
    )
    ax.plot(detuning, gain, label=f"{r:.1f}")
ax.set_xlabel(r"Signal detuning $(f_\mathrm{s} - f_0)/(\kappa/2\pi)$")
ax.set_ylabel("Signal gain (dB)")
ax.legend(title=r"$\Phi_\mathrm{p}$ / threshold")
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
ac = run_vacask(
    cpw_jpa_netlist('analysis ac1 ac from=12G to=22G mode="lin" points=100001'),
    SQUID_FILES,
)["ac1"]
f_ac = ac["frequency"].real
v_mode = np.abs(ac["2"])
peaks = np.flatnonzero((v_mode[1:-1] > v_mode[:-2]) & (v_mode[1:-1] > v_mode[2:])) + 1
print(
    "modes between 12 and 22 GHz:", ", ".join(f"{f_ac[p] / 1e9:.3f} GHz" for p in peaks)
)
print(f"f_s + f_p at f_s = f_0: {(F0 + F_PUMP_CPW) / 1e9:.3f} GHz")

sidebands = [0, -1, 1, -2, 2]
outspur = ",".join(f"[{k}]" for k in sidebands)
f_signal = F0 + 0.01 * KAPPA
for r in (0.5, 0.9):
    group = run_vacask(
        cpw_jpa_netlist(
            f'analysis hbac1 hbac freq=[{F_PUMP_CPW}] nharm=5 outspur={{{outspur}}} from={f_signal} to={f_signal} mode="lin" points=1',
            pump=r * PUMP_THRESHOLD,
            f_pump=F_PUMP_CPW,
        ),
        SQUID_FILES,
    )["hbac1"]
    total = 0.0
    parts = []
    for k in sidebands:
        f_k = f_signal + k * F_PUMP_CPW
        amplitude = (reflection if k == 0 else conversion)(group[f"in;{k}"][0], I_SIG)
        photons = np.sign(f_k) * np.abs(amplitude) ** 2 * f_signal / abs(f_k)
        total += photons
        parts.append(f"{k:+d}: {photons:+.3f}")
    print(f"r = {r}: " + ", ".join(parts) + f"; sum = {total:.4f}")

# %% [markdown]
# The weighted sum over the sidebands is close to 1, so the gain deficit is
# accounted for by conversion into the $f_\text{s} + f_\text{p}$ and
# $2f_\text{p} - f_\text{s}$ sidebands near the $3\lambda/4$ mode rather than
# by a numerical loss. A single-mode model of this resonator overestimates its
# gain. Moving the $3\lambda/4$ mode away from $f_\text{s} + f_\text{p}$ should
# reduce the up-conversion; we have not simulated that here.
#
# ```{warning}
# `hbac` linearizes around the pumped steady state whether or not that state
# is stable. Above the parametric threshold the zero-signal state is unstable
# and the `hbac` "gain" is not physical. Check the pump against the threshold
# before trusting a high gain.
# ```
#
# ### 3.7 Gain versus pump frequency
#
# Detuning the pump from $2 f_0$ moves the gain peak to $f_\text{p}/2$ and
# lowers it. We hold the pump at 80 % of the single-mode threshold.

# %%
pump_detunings = np.linspace(-1.0, 1.0, 21) * KAPPA
gain_vs_fp = []
for dfp in pump_detunings:
    fp_ = F_PUMP_CPW + dfp
    group = run_vacask(
        cpw_jpa_netlist(
            f'analysis hbac1 hbac freq=[{fp_}] nharm=5 outspur={{[0],[-1]}} from={F0 - 2.0013 * KAPPA} to={F0 + 2.0013 * KAPPA} mode="lin" points=200',
            pump=0.8 * PUMP_THRESHOLD,
            f_pump=fp_,
        ),
        SQUID_FILES,
    )["hbac1"]
    gain_vs_fp.append(db(reflection(group["in;0"], I_SIG)))
detuning = (group["frequency"].real - F0) / KAPPA
gain_vs_fp = np.array(gain_vs_fp)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(detuning, pump_detunings / KAPPA, gain_vs_fp, shading="auto")
fig.colorbar(mesh, label="Signal gain (dB)")
ax.set_xlabel(r"Signal detuning $(f_\mathrm{s} - f_0)/(\kappa/2\pi)$")
ax.set_ylabel(r"Pump detuning $(f_\mathrm{p} - 2f_0)/(\kappa/2\pi)$")
plt.show()
print(f"peak gain over the map: {gain_vs_fp.max():.2f} dB")

# %% [markdown]
# ### 3.8 Route B in harmonic balance
#
# The vector-fitted subcircuit contains only resistors, capacitors and
# controlled sources, so it runs in harmonic balance too. With the SQUID
# attached, though, its residual fit error appears as loss at the high-Q
# resonance:

# %%
for route in ("A", "B"):
    ac = run_vacask(
        cpw_jpa_netlist(
            'analysis ac1 ac from=5.85G to=6.0G mode="lin" points=150001', route=route
        ),
        SQUID_FILES,
    )["ac1"]
    gamma = reflection(ac["in"], I_SIG)
    f0, kappa = resonance(ac["frequency"].real, gamma)
    print(
        f"Route {route}: f0 = {f0 / 1e9:.5f} GHz, κ/2π = {kappa / 1e6:.2f} MHz, "
        f"min |Γ| = {np.abs(gamma).min():.3f}"
    )

# %% [markdown]
# A lossless resonator reflects everything ($|\Gamma| = 1$); the Route B model
# absorbs a large part of the signal at resonance. A fit error of order
# $10^{-3}$ in $S$ is small on its own but, once the SQUID turns the line into
# a resonator with $Q$ in the hundreds, it acts as an internal loss of the
# same order as the coupling. Route B is therefore suitable for linear
# networks and low-Q matching sections, but resonators that set the
# amplifier's $Q$ should use Route A or a fit with tighter accuracy near
# resonance.
#
# ## 4. Stretch: Josephson-junction ladder TWPA
#
# A TWPA is a chain of junctions with shunt capacitors to ground
# {cite:p}`levochkinaModelingFluxTunability2025`. Here each cell is a junction
# with $I_\text{c} = 3.29$ µA ($L_\text{J} \approx 100$ pH) and a 40 fF shunt,
# which makes a 50 Ω line. A 6 GHz pump at half the critical current is
# applied through the 50 Ω input; the output is terminated in 50 Ω. The
# netlist is generated in Python, one line per element. There is no phase
# matching, so we do not expect much gain: this is a test of how `hbac`
# scales with circuit size.

# %%
IC_TWPA, C_TWPA, F_PUMP_TWPA = 3.29e-6, 40e-15, 6e9


def twpa_netlist(cells: int, pump_current: float) -> str:
    """JJ-ladder TWPA netlist with ``cells`` junction–capacitor cells."""
    ladder = "\n".join(
        f"j{i} (n{i} n{i + 1}) jj ic={IC_TWPA}\nc{i} (n{i + 1} 0) capacitor c={C_TWPA}"
        for i in range(cells)
    )
    return f"""JJ-ladder TWPA, {cells} cells
load "josephson_junction.va"
load "capacitor.osdi"
load "resistor.osdi"
model isource isource
model capacitor capacitor
model resistor resistor
model jj jj
ip (0 n0) isource type="sine" sinedc=0 ampl={2 * pump_current} freq={F_PUMP_TWPA} spur={{[0]}} smag=[{I_SIG}]
rs (n0 0) resistor r={R_PORT}
{ladder}
rl (n{cells} 0) resistor r={R_PORT}
control
{OPTIONS}
  analysis hbac1 hbac freq=[{F_PUMP_TWPA}] nharm=5 outspur={{[0],[-2]}} from=3G to=9G mode="lin" points=60
endc
"""


fig, ax = plt.subplots()
for cells in (200, 500, 1000, 2000):
    start = time.perf_counter()
    plot = run_vacask(twpa_netlist(cells, 0.5 * IC_TWPA), JJ_FILES)["hbac1"]
    elapsed = time.perf_counter() - start
    f = plot["frequency"].real
    s21 = db(plot[f"n{cells};0"] / (I_SIG * R_PORT / 2))
    i5 = np.argmin(np.abs(f - 5e9))
    print(f"{cells:5d} cells: {elapsed:6.1f} s, |S21| at 5 GHz = {s21[i5]:+.2f} dB")
    ax.plot(f / 1e9, s21, label=f"{cells}")
ax.set_xlabel("Signal frequency (GHz)")
ax.set_ylabel(r"$|S_{21}|$ (dB)")
ax.legend(title="Cells")
plt.show()

# %% [markdown]
# The run times above are for this notebook's build machine; they depend on
# the CPU and the BLAS library VACASK was built against. Up to 1000 cells the
# run time grows linearly and the signal transmission falls with length. At
# 2000 cells the run time jumps by a factor of about 25 and the transmission
# turns positive; we have not investigated why.
#
# Without dispersion engineering, a junction ladder lets the pump's mixing
# products propagate as freely as the signal. The next cell shows, for 200
# cells at 5 GHz, where the signal photons go, using the same photon-weighted
# balance as for the CPW JPA.

# %%
cells = 200
netlist = (
    twpa_netlist(cells, 0.5 * IC_TWPA)
    .replace("outspur={[0],[-2]}", "outspur={[0],[-2],[2],[-4],[4]}")
    .replace('from=3G to=9G mode="lin" points=60', 'from=5G to=5G mode="lin" points=1')
)
plot = run_vacask(netlist, JJ_FILES)["hbac1"]
f_s = plot["frequency"].real[0]
incident = I_SIG * R_PORT / 2
total = 0.0
for k in (0, -2, 2, -4, 4):
    f_k = f_s + k * F_PUMP_TWPA
    transmitted = np.abs(plot[f"n{cells};{k}"][0] / incident) ** 2
    reflected = np.abs(plot[f"n0;{k}"][0] / incident - (1 if k == 0 else 0)) ** 2
    photons = np.sign(f_k) * (transmitted + reflected) * f_s / abs(f_k)
    total += photons
    print(
        f"k = {k:+d} ({abs(f_k) / 1e9:5.1f} GHz): transmitted {transmitted:.4f}, "
        f"reflected {reflected:.4f}, photons {photons:+.4f}"
    )
print(f"photon-weighted sum = {total:.4f}")

# %% [markdown]
# The sum over sidebands is 1 to within $10^{-3}$, so photons are conserved.
# The signal photons that are not transmitted at 5 GHz are converted into
# $f_\text{s} + 2 f_\text{p}$ and $f_\text{s} + 4 f_\text{p}$, partly balanced
# by the idler-like sidebands at $2 f_\text{p} - f_\text{s}$ and
# $4 f_\text{p} - f_\text{s}$. A practical TWPA suppresses these products
# with dispersion engineering, such as resonant phase matching
# {cite:p}`obrienResonantPhaseMatching2014,macklinNearquantumlimitedJosephsonTravelingwave2015`,
# which this ladder does not have. Without it, Kerr phase mismatch limits the
# gain {cite:p}`yaakobiParametricAmplificationJosephson2013`. Pump harmonics and
# sidebands carry real power in TWPAs, so a harmonic-balance model must keep
# enough of them
# {cite:p}`dixonCapturingComplexBehavior2020,pengFloquetmodeTravelingwaveParametric2022`.

# %% [markdown]
# ## 5. Limitations
#
# - **Classical simulation.** Harmonic balance computes classical gain and
#   conversion. It says nothing about added noise or squeezing; for those use
#   the quantum input–output treatment in
#   {cite:t}`clerkIntroductionQuantumNoise2010` and
#   {cite:t}`royIntroductionParametricAmplification2016`. Quantum efficiency can
#   be derived from the linearized network
#   {cite:p}`pengFloquetmodeTravelingwaveParametric2022`, but this notebook does
#   not do that.
# - **Stability.** `hbac` does not check whether the pumped steady state is
#   stable. Above the parametric threshold its "gain" is meaningless.
# - **Junction model.** The RCSJ model keeps only the first Josephson harmonic,
#   although real tunnel junctions show higher ones
#   {cite:p}`willschObservationJosephsonHarmonics2024`. The SQUID model
#   neglects the loop inductance ($\beta_L = 0$) and junction asymmetry.
# - **Route B accuracy.** A vector fit that is accurate to $10^{-3}$ on its own
#   can still add significant loss inside a high-Q resonator.
# - **Manley–Rowe residual.** In the Kerr JPA the signal–idler balance
#   deviates from 1 by up to 3 % at the highest gain; the cause has not been
#   identified.
# - **Transient.** A transient comparison was only made at low gain. Near the
#   gain peak the transient result depends on the simulated time.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
