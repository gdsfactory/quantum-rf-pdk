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

# %% tags=["hide-input", "hide-output"]
import math
import os
from functools import partial

import jax.numpy as jnp
import sax
import skrf

from qpdk.models.media import cpw_media_skrf
from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator_frequency,
)
from qpdk.tech import coplanar_waveguide

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
                "in": "R1,coupling_o1",
                "out": "R1,coupling_o2",
            },
        },
        models={
            "quarter_wave_resonator": partial(
                quarter_wave_resonator_coupled,
                cross_section=coplanar_waveguide(width=10, gap=6),
                length=4000,
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


# %% [markdown]
# ## Optimizer for given resonance frequency
#
# Find the resonator length that gives a desired resonance frequency using an optimizer.

# %% [markdown]
#
# ```python
#
if __name__ == "__main__":
    import ray
    import ray.tune
    import ray.tune.search.optuna

    frequencies = jnp.linspace(0.5e9, 10e9, 1001)
    TARGET_FREQUENCY = 6e9  # Target resonance frequency in Hz

    def loss_fn(config: dict[str, float]) -> dict[str, float]:
        """Loss function to minimize the difference between the actual and target resonance frequencies.

        Args:
            config: Dictionary containing the resonator length in micrometers.
        """
        length = config["length"]
        # Setup model
        S = circuit(f=frequencies, length=length)
        # Get frequency at minimum S21
        coupled_freq = frequencies[jnp.argmin(abs(S["in", "out"]))]
        return {
            "l1_loss_ghz": abs(float(coupled_freq) - TARGET_FREQUENCY) / 1e9,
            "mse": (float(coupled_freq) - TARGET_FREQUENCY) ** 2,
        }

    # Test loss function
    print(f"{loss_fn(dict(length=4000.0))=}")
    print(f"{loss_fn(dict(length=5900.0))=}")

    # Initialize Ray (possibly with cluster)
    ray.init()

    # Optimize length using Ray Tune
    tuner = ray.tune.Tuner(
        loss_fn,
        param_space={
            "length": ray.tune.uniform(1000.0, 9000.0),
        },
        tune_config=ray.tune.TuneConfig(
            metric="mse",
            mode="min",
            num_samples=50,
            max_concurrent_trials=math.ceil((os.cpu_count() or 1) / 4),
            reuse_actors=True,
            search_alg=ray.tune.search.optuna.OptunaSearch(),
        ),
    )
    results = tuner.fit()
    best_trial = results.get_best_result()
    length = best_trial.config["length"]
    print(f"Best trial config: {best_trial.config}")

    # Initialize optimizer
    print(f"Optimized Length: {length:.2f} µm")
    optimal_S = circuit(f=frequencies, length=length)
    optimal_freq = frequencies[jnp.argmin(abs(optimal_S["in", "out"]))]
    print(f"Achieved Resonance Frequency: {optimal_freq / 1e9:.2f} GHz")

    # Plot
    plt.close()
    plt.plot(frequencies / 1e9, abs(optimal_S["in", "out"]) ** 2)
    plt.xlabel("f [GHz]")
    plt.ylabel("$S_{21}$")
    _mark_resonance_frequency(optimal_freq, "blue", "Optimized resonance Frequency")
    _mark_resonance_frequency(TARGET_FREQUENCY, "orange", "Target resonance Frequency")
    plt.legend()
    plt.show()
# ```
