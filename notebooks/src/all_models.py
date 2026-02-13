# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qpdk
#     language: python
#     name: qpdk
# ---

# %% [markdown]
# ## QPDK Models

# %% [markdown]
# ## Imports

# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
import skrf
from gdsfactory.typings import CrossSectionSpec
from tqdm.notebook import tqdm

from qpdk import PDK

PDK.activate()

# ruff: disable[E402]

# %% [markdown]
# ## Constants

# %%
from qpdk.models.constants import DEFAULT_FREQUENCY, TEST_FREQUENCY

# %% [markdown]
# ## Media
# %%
from qpdk.models.media import cross_section_to_media

cross_section_to_media("cpw")

# %% [markdown]
# ## Generic

# %%
from qpdk.models.generic import gamma_0_load

gamma_0_load(f=TEST_FREQUENCY, gamma_0=1, n_ports=2)

# %%
from qpdk.models.generic import short

short(f=TEST_FREQUENCY, n_ports=2)

# %%
from qpdk.models.generic import short_2_port

short_2_port(f=TEST_FREQUENCY)

# %%
from qpdk.models.generic import open

open(f=TEST_FREQUENCY, n_ports=2)

# %%
from qpdk.models.generic import tee

tee(f=TEST_FREQUENCY)

# %%
from qpdk.models.generic import single_impedance_element

single_impedance_element(f=TEST_FREQUENCY)

# %%
from qpdk.models.generic import single_admittance_element

single_admittance_element()

# %%
from qpdk.models.generic import capacitor

capacitor(f=TEST_FREQUENCY)

# %%
from qpdk.models.generic import inductor

inductor(f=TEST_FREQUENCY)

# %%
from qpdk.models.generic import josephson_junction

josephson_junction(f=TEST_FREQUENCY)

# %%
f = jnp.linspace(1e9, 25e9, 201)
S = gamma_0_load(f=f, gamma_0=0.5 + 0.5j, n_ports=2)
for key in S:
    plt.plot(f / 1e9, abs(S[key]) ** 2, label=key)
plt.ylim(-0.05, 1.05)
plt.xlabel("Frequency [GHz]")
plt.ylabel("S")
plt.grid(True)
plt.legend()
plt.show(block=False)

S_cap = capacitor(f=f, capacitance=(capacitance := 100e-15))
# print(S_cap)
plt.figure()
# Polar plot of S21 and S11
plt.subplot(121, projection="polar")
plt.plot(jnp.angle(S_cap[("o1", "o1")]), abs(S_cap[("o1", "o1")]), label="$S_{11}$")
plt.plot(jnp.angle(S_cap[("o1", "o2")]), abs(S_cap[("o2", "o1")]), label="$S_{21}$")
plt.title("S-parameters capacitor")
plt.legend()
# Magnitude and phase vs frequency
ax1 = plt.subplot(122)
ax1.plot(f / 1e9, abs(S_cap[("o1", "o1")]), label="|S11|", color="C0")
ax1.plot(f / 1e9, abs(S_cap[("o1", "o2")]), label="|S21|", color="C1")
ax1.set_xlabel("Frequency [GHz]")
ax1.set_ylabel("Magnitude [unitless]")
ax1.grid(True)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(
    f / 1e9,
    jnp.angle(S_cap[("o1", "o1")]),
    label="∠S11",
    color="C0",
    linestyle="--",
)
ax2.plot(
    f / 1e9,
    jnp.angle(S_cap[("o1", "o2")]),
    label="∠S21",
    color="C1",
    linestyle="--",
)
ax2.set_ylabel("Phase [rad]")
ax2.legend(loc="upper right")

plt.title(f"Capacitor $S$-parameters ($C={capacitance * 1e15}\\,$fF)")
plt.show(block=False)

S_ind = inductor(f=f, inductance=(inductance := 1e-9))
# print(S_ind)
plt.figure()
plt.subplot(121, projection="polar")
plt.plot(jnp.angle(S_ind[("o1", "o1")]), abs(S_ind[("o1", "o1")]), label="$S_{11}$")
plt.plot(jnp.angle(S_ind[("o1", "o2")]), abs(S_ind[("o2", "o1")]), label="$S_{21}$")
plt.title("S-parameters inductor")
plt.legend()
ax1 = plt.subplot(122)
ax1.plot(f / 1e9, abs(S_ind[("o1", "o1")]), label="|S11|", color="C0")
ax1.plot(f / 1e9, abs(S_ind[("o1", "o2")]), label="|S21|", color="C1")
ax1.set_xlabel("Frequency [GHz]")
ax1.set_ylabel("Magnitude [unitless]")
ax1.grid(True)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(
    f / 1e9,
    jnp.angle(S_ind[("o1", "o1")]),
    label="∠S11",
    color="C0",
    linestyle="--",
)
ax2.plot(
    f / 1e9,
    jnp.angle(S_ind[("o1", "o2")]),
    label="∠S21",
    color="C1",
    linestyle="--",
)
ax2.set_ylabel("Phase [rad]")
ax2.legend(loc="upper right")

plt.title(f"Inductor $S$-parameters ($L={inductance * 1e9}\\,$nH)")
plt.show()

# %% [markdown]
# ## Waveguides

# %%
from qpdk.models.waveguides import straight

straight(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import straight_shorted

straight_shorted(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import bend_circular

bend_circular(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import bend_euler

bend_euler(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import bend_s

bend_s(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import rectangle

rectangle(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import taper_cross_section

taper_cross_section(f=TEST_FREQUENCY)

# %%
from qpdk.models.waveguides import launcher

launcher(f=TEST_FREQUENCY)

# %%
from qpdk.tech import coplanar_waveguide

cpw_cs = coplanar_waveguide(width=10, gap=6)


def straight_no_jit(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    length: int | float = 1000,
    cross_section: CrossSectionSpec = "cpw",
) -> sax.SType:
    """Version of straight without just-in-time compilation."""
    media = cross_section_to_media(cross_section)
    skrf_media = media(frequency=skrf.Frequency.from_f(f, unit="Hz"))
    transmission_line = skrf_media.line(d=length, unit="um")
    sdict = {
        ("o1", "o1"): jnp.array(transmission_line.s[:, 0, 0]),
        ("o1", "o2"): jnp.array(transmission_line.s[:, 0, 1]),
        ("o2", "o2"): jnp.array(transmission_line.s[:, 1, 1]),
    }
    return sax.reciprocal(sdict)


test_freq = jnp.linspace(0.5e9, 9e9, 200001)
test_length = 1000

print("Benchmarking jitted vs non-jitted performance…")

n_runs = 10

jit_times = []
for _ in tqdm(range(n_runs), desc="With jax.jit", ncols=80, unit="run"):
    start_time = time.perf_counter()
    S_jit = straight(f=test_freq, length=test_length, cross_section=cpw_cs)
    _ = S_jit["o2", "o1"].block_until_ready()
    end_time = time.perf_counter()
    jit_times.append(end_time - start_time)

no_jit_times = []
for _ in tqdm(range(n_runs), desc="Without jax.jit", ncols=80, unit="run"):
    start_time = time.perf_counter()
    S_no_jit = straight_no_jit(f=test_freq, length=test_length, cross_section=cpw_cs)
    _ = S_no_jit["o2", "o1"].block_until_ready()
    end_time = time.perf_counter()
    no_jit_times.append(end_time - start_time)

jit_times_steady = jit_times[1:]
avg_jit = sum(jit_times_steady) / len(jit_times_steady)
avg_no_jit = sum(no_jit_times) / len(no_jit_times)
speedup = avg_no_jit / avg_jit

print(f"Jitted: {avg_jit:.4f}s avg (excl. first), {jit_times[0]:.3f}s first run")
print(f"Non-jitted: {avg_no_jit:.4f}s avg")
print(f"Speedup: {speedup:.1f}x")

S_jit = straight(f=test_freq, length=test_length, cross_section=cpw_cs)
S_no_jit = straight_no_jit(f=test_freq, length=test_length, cross_section=cpw_cs)
max_diff = jnp.max(jnp.abs(S_jit["o2", "o1"] - S_no_jit["o2", "o1"]))
print(f"Max absolute difference in results: {max_diff:.2e}")

try:
    s21_array = S_jit["o2", "o1"]
    s21_gpu = jax.device_put(s21_array, jax.devices("gpu")[0])
    print(f"GPU available: {s21_gpu.device}")
except Exception:
    print("GPU not available, using CPU")

# Test tapers
S = taper_cross_section(
    f=test_freq,
    length=1000,
    cross_section_1=cpw_cs,
    cross_section_2=coplanar_waveguide(width=12, gap=10),
)
print("Taper S-parameters computed successfully.")
print(f"S-parameter keys: {list(S.keys())}")

# %% [markdown]
# ## Couplers

# %%
from qpdk.models.couplers import cpw_cpw_coupling_capacitance

cpw_cpw_coupling_capacitance(TEST_FREQUENCY, 100, 100, "cpw")

# %%
from qpdk.models.couplers import coupler_straight

coupler_straight(f=TEST_FREQUENCY)

# %%
# Define frequency range from 1 GHz to 10 GHz with 201 points
f = jnp.linspace(1e9, 10e9, 201)

# Calculate coupler S-parameters for a 20 um straight coupler with 0.27 um gap
coupler = coupler_straight(f=f, length=20, gap=0.27)

# Create figure with single plot for comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define S-parameters to plot
s_params = [
    (("o1", "o1"), "$S_{11}$ Reflection"),
    (("o1", "o2"), "$S_{12}$ Coupled branch 1"),
    (("o1", "o3"), "$S_{13}$ Coupled branch 2"),
    (("o1", "o4"), "$S_{14}$ Insertion loss (direct through)"),
]

# Plot each S-parameter for both coupler implementations
default_color_cycler = plt.cm.tab10.colors
for idx, (ports, label) in enumerate(s_params):
    color = default_color_cycler[idx % len(default_color_cycler)]
    # Plot both implementations with same color but different linestyles
    ax.plot(
        f / 1e9,
        20 * jnp.log10(jnp.abs(coupler[ports])),
        linestyle="-",
        color=color,
        label=f"{label} coupler_straight",
    )

# Configure plot
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("$S$-parameter [dB]")
ax.set_title(r"$S$-parameters: $\mathtt{coupler\_straight}$")
ax.grid(True, which="both")
ax.legend()

plt.tight_layout()
plt.show()

# Example calculation of coupling capacitance
from qpdk.tech import coplanar_waveguide

cs = coplanar_waveguide(width=10, gap=6)
coupling_capacitance = cpw_cpw_coupling_capacitance(
    length=20.0, gap=0.27, cross_section=cs, f=f
)
print(
    "Coupling capacitance for 20 um length and 0.27 um gap:",
    coupling_capacitance,
    "F",
)

# %% [markdown]
# ## Resonators

# %%
from qpdk.models.resonator import quarter_wave_resonator_coupled

quarter_wave_resonator_coupled(f=TEST_FREQUENCY)

# %%
from qpdk.models.resonator import resonator_frequency

cs = coplanar_waveguide(width=10, gap=6)
cpw = cross_section_to_media(cs)(frequency=skrf.Frequency(2, 9, 101, unit="GHz"))
print(f"{cpw=!r}")
print(f"{cpw.z0.mean().real=!r}")  # Characteristic impedance

res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
print("Resonance frequency (quarter-wave):", res_freq / 1e9, "GHz")

# Plot resonator_coupled example
f = jnp.linspace(0.1e9, 9e9, 1001)
resonator = quarter_wave_resonator_coupled(
    f=f,
    cross_section=cs,
    coupling_gap=0.27,
    length=4000,
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for key in [
    ("coupling_o2", "resonator_o1"),
    ("coupling_o1", "coupling_o2"),
    ("coupling_o1", "resonator_o1"),
]:
    ax.plot(f / 1e9, 20 * jnp.log10(jnp.abs(resonator[key])), label=f"$S${key}")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Magnitude [dB]")
ax.set_title(r"$S$-parameters: $\mathtt{resonator\_coupled}$ (3-port)")
ax.grid(True, which="both")
ax.legend()

plt.show()
# ruff: enable[E402]
