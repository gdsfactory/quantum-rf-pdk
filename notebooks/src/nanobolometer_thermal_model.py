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
# # Nanobolometer thermal model from the layout
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra:
#
# ```bash
# uv add "qpdk[models]"
# # or, from a checkout of this repository:
# uv sync --extra models
# # or with pip:
# pip install "qpdk[models]"
# ```
#
# See the {ref}`extras reference <notebook-extras>` for what each extra installs.
# ::::
#
# A normal-metal hot-electron bolometer absorbs RF power in a nanowire between
# superconducting leads. Andreev reflection at the clean NS contacts confines the
# heat to the wire electrons, which relax to the phonon bath through
# electron-phonon coupling {cite:p}`giazottoOpportunitiesMesoscopicsThermometry2006`.
# A single electron reservoir relaxes with $\tau = \gamma / (5 \Sigma T^3)$.
#
# Measured relaxation traces of both an Aalto AuPd nanobolometer
# {cite:p}`gunyhoZeptojouleCalorimetry2026` and a Pekola Cu calorimeter
# {cite:p}`gasparinettiFastElectronThermometry2015` instead fit a double exponential
# whose second time constant has no established origin. This notebook asks whether
# the drawn geometry alone, without per-device tuning, explains it. The second
# reservoir is the normal metal under and beyond the superconducting leads, linked to
# the absorber by Wiedemann-Franz conduction.

# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing QPDK...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qpdk import PDK
from qpdk.cells import nanobolometer, nanocalorimeter
from qpdk.models.bolometer import nanobolometer_thermal_modes

PDK.activate()

# %% [markdown]
# ## Layout
#
# The default {func}`~qpdk.cells.nanobolometer` follows the AuPd nanowire of
# {cite:t}`girardCryogenicSensorEnabling2023`: a $1\,\text{µm}$ absorber with a
# $135\,\text{nm} \times 30\,\text{nm}$ cross-section, terminating a CPW input. The
# normal metal (`NM_DRAW`) runs `contact_overlap` under each Al lead (`JJ_AREA`) and
# `extension_length` beyond it. Two Al probes form NIS thermometer junctions.

# %%
bolometer = nanobolometer()
bolometer.plot()
{k: v for k, v in bolometer.info.model_dump().items() if k.endswith("volume")}

# %% [markdown]
# {func}`~qpdk.cells.nanocalorimeter` is the same cell with the
# $20\,\text{µm} \times 100\,\text{nm} \times 25\,\text{nm}$ Cu wire and single
# central probe of {cite:t}`gasparinettiFastElectronThermometry2015`.

# %%
nanocalorimeter().plot()

# %% [markdown]
# ## Predictions against two measured devices
#
# The inputs come from the papers alone. Where a paper is silent, the values are
# choices, listed here so they can be challenged:
#
# - **Aalto device** {cite:p}`gunyhoZeptojouleCalorimetry2026,girardCryogenicSensorEnabling2023`.
#   All uncovered AuPd (the $1\,\text{µm}$ absorber plus eight $0.3\,\text{µm}$ SNS
#   junctions) is lumped into reservoir 1, and the Al-covered metal (about
#   $10 \times 0.3\,\text{µm}$) into reservoir 2. The width of $135\,\text{nm}$ is
#   chosen to reproduce $V_0 \approx (240\,\text{nm})^3$ of
#   {cite:t}`goveniusDetectionZeptojouleMicrowave2016`. No electron temperature is
#   reported, so a range is shown.
# - **Pekola calorimeter** {cite:p}`gasparinettiFastElectronThermometry2015`, at the
#   reported $T_\text{e} \approx 85\,\text{mK}$, with $\gamma = 71$ and
#   $\Sigma = 2 \times 10^9$ in SI units. The NS overlap is not reported;
#   $0.2\,\text{µm}$ under and beyond each lead is assumed.

# %%
gunyho = {
    "absorber_length": 3.4,
    "absorber_width": 0.135,
    "normal_metal_thickness": 0.03,
    "contact_overlap": 1.5,
    "extension_length": 0.0,
}
gasparinetti = {
    "absorber_length": 20.0,
    "absorber_width": 0.1,
    "normal_metal_thickness": 0.025,
    "contact_overlap": 0.2,
    "extension_length": 0.2,
    "heat_capacity_coefficient": 71.0,
    "electron_phonon_coupling": 2e9,
    "resistivity": 4.5e-8,
}

rows = [
    (
        "Aalto",
        f"{t * 1e3:.0f} mK",
        *nanobolometer_thermal_modes(**gunyho, temperature=t),
    )
    for t in (0.03, 0.05, 0.07)
]
rows.append((
    "Pekola",
    "85 mK",
    *nanobolometer_thermal_modes(**gasparinetti, temperature=0.085),
))

print(
    f"{'device':8} {'T_e':>7} {'tau_fast (s)':>12} {'tau_slow (s)':>12} {'a_slow/a_fast':>14}"
)
for device, temperature, taus, amplitudes in rows:
    print(
        f"{device:8} {temperature:>7} {taus[0]:>12.2e} {taus[1]:>12.2e} "
        f"{amplitudes[1] / amplitudes[0]:>14.3g}"
    )
print(f"{'measured':8} {'Aalto':>7} {18e-6:>12.2e} {150e-6:>12.2e} {1.5:>14.3g}")
print(f"{'measured':8} {'Pekola':>7} {97e-6:>12.2e} {410e-6:>12.2e} {0.018:>14.3g}")

# %% [markdown]
# ## Relaxation traces
#
# Normalised absorber temperature after a heat pulse, predicted and measured.

# %%
t = jnp.logspace(-10, -2.5, 400)


def trace(taus: jnp.ndarray, amplitudes: jnp.ndarray) -> jnp.ndarray:
    """Sum of exponentials normalised to one at t = 0."""
    return jnp.sum(amplitudes[:, None] * jnp.exp(-t[None, :] / taus[:, None]), axis=0)


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
measured = {
    "Aalto": (jnp.array([18e-6, 150e-6]), jnp.array([1.0, 1.5]) / 2.5),
    "Pekola": (jnp.array([97e-6, 410e-6]), jnp.array([1.0, 0.018]) / 1.018),
}
for ax, (device, temperature, taus, amplitudes) in zip(
    axes, (rows[1], rows[3]), strict=True
):
    ax.semilogx(t * 1e6, trace(taus, amplitudes), label=f"layout model, {temperature}")
    ax.semilogx(t * 1e6, trace(*measured[device]), "--", label="measured fit")
    ax.set_title(device)
    ax.set_xlabel(r"$t$ ($\text{µs}$)")
    ax.legend()
axes[0].set_ylabel(r"$\theta_1(t) / \theta_1(0)$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What the drawn geometry does and does not explain
#
# - **Amplitude ratio.** In the strong-link limit $a_\text{slow}/a_\text{fast} \to
#   C_1/C_2$, the volume ratio. For the Aalto layout that gives about 1.1 against the
#   measured 1.5, the right order.
# - **Fast time.** A Wiedemann-Franz link over a few hundred nanometres of metal
#   conducts some $10^{-10}\,\text{W/K}$, against about $10^{-15}\,\text{W/K}$ for
#   electron-phonon coupling. The two reservoirs equilibrate in nanoseconds, four
#   orders of magnitude faster than the measured $18\,\text{µs}$.
# - **Slow time.** This is the single-reservoir electron-phonon time. It reaches
#   $150\,\text{µs}$ only near $T_\text{e} \approx 64\,\text{mK}$, which the paper does
#   not report, so the agreement is not a test.
# - **Calorimeter.** The dominant mode is again the electron-phonon time, about eight
#   times faster than measured (the paper's own heat-capacity excess). The layout
#   reservoir adds a picosecond mode of weight 0.04, not a slow tail of weight 0.018.
#
# A bare Wiedemann-Franz link to drawn normal metal therefore cannot produce either
# measured second time constant: that would need a coupling of order the
# electron-phonon conductance. Candidates are proximity-suppressed heat transport
# into the Al-covered metal {cite:p}`goveniusMicrowaveNanobolometerBased2014`, an
# internal material reservoir such as weakly coupled grains
# {cite:p}`wangDynamicThermalRelaxation2026`, or substrate and TLS heat capacity.
# The model exposes the material reservoir as an optional third node,
# `material_heat_capacity`, off by default. The cell's `contact_overlap` and
# `extension_length` parameters allow drawing a design of experiments that varies
# the covered volume, which would tell these hypotheses apart.
