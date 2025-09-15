# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Resonator frequency estimation models
#
# This example demonstrates estimating resonance frequencies of superconducting microwave resonators using scikit-rf and Jax.

# %%
from functools import partial

import jax.numpy as jnp
import sax
import skrf

from qpdk.models.resonator import (
    cpw_media_skrf,
    quarter_wave_resonator_coupled_to_probeline,
    resonator_frequency,
)

# %% [markdown]
# ## Probelines weakly coupled to $\lambda/4$ resonator
#
# Creates a probelines weakly coupled to a quarter-wave resonator.
# The resonance frequency is first estimated using the `resonator_frequency` function and then compared to the frequency in the coupled case.

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cpw = cpw_media_skrf(width=10, gap=6)(
        frequency=skrf.Frequency(2, 9, 101, unit="GHz")
    )
    print(f"{cpw=!r}")
    print(f"{cpw.z0.mean().real=!r}")  # Characteristic impedance

    res_freq = resonator_frequency(length=4000, media=cpw, is_quarter_wave=True)
    print("Resonance frequency (quarter-wave):", res_freq / 1e9, "GHz")

    circuit, info = sax.circuit(
        netlist={
            "instances": {
                "R1": "quarter_wave_resonator",
            },
            "connections": {},
            "ports": {
                "in": "R1,o1",
                "out": "R1,o2",
            },
        },
        models={
            "quarter_wave_resonator": partial(
                quarter_wave_resonator_coupled_to_probeline,
                media=cpw_media_skrf(width=10, gap=6),
                length=4000,
                coupling_capacitance=15e-15,
            )
        },
    )

    frequencies = jnp.linspace(1e9, 10e9, 5001)
    S = circuit(f=frequencies)
    print(info)
    plt.plot(frequencies / 1e9, abs(S["in", "out"]) ** 2)
    plt.xlabel("f [GHz]")
    plt.ylabel("$S_{21}$")

    def _mark_resonance_frequency(x_value: float, color: str, label: str):
        """Draws a vertical dashed line on the current matplotlib plot to mark a resonance frequency."""
        plt.axvline(
            x_value / 1e9,  # Convert frequency from Hz to GHz for plotting
            color=color,
            linestyle="--",
            label=label,
        )

    _mark_resonance_frequency(res_freq, "red", "Predicted resonance Frequency")
    actual_freq = frequencies[jnp.argmin(abs(S["in", "out"]))]
    print("Coupled resonance frequency:", actual_freq / 1e9, "GHz")
    _mark_resonance_frequency(actual_freq, "green", "Coupled resonance Frequency")

    plt.legend()
    plt.show()
