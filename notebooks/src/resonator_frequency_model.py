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
# This example demonstrates estimating resonance frequencies of superconducting microwave resonators using Jax.

# %% tags=["hide-input", "hide-output"]
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sax

from qpdk import PDK, logger
from qpdk.models.cpw import cpw_parameters

PDK.activate()

# ruff: disable[E402]
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
ep_eff, z0 = cpw_parameters(width=10, gap=6)
logger.info(f"{ep_eff=!r}")
logger.info(f"{z0=!r}")  # Characteristic impedance

res_freq = resonator_frequency(
    length=4000,
    epsilon_eff=float(jnp.real(ep_eff)),
    is_quarter_wave=True,
)
logger.info(f"Resonance frequency (quarter-wave): {float(res_freq) / 1e9} GHz")

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
        )
    },
)

frequencies = jnp.linspace(1e9, 10e9, 5001)
S = circuit(f=frequencies, length=4000.0)
logger.info(info)
plt.plot(frequencies / 1e9, abs(S["in", "out"]) ** 2)
plt.xlabel("f [GHz]")
plt.ylabel("$S_{21}$")


def _mark_resonance_frequency(x_value: float, color: str, label: str):
    """Draws a vertical dashed line on the current matplotlib plot to mark a resonance frequency."""
    plt.axvline(
        float(x_value) / 1e9,  # Convert frequency from Hz to GHz for plotting
        color=color,
        linestyle="--",
        label=label,
    )


_mark_resonance_frequency(res_freq, "red", "Predicted resonance Frequency")
actual_freq = frequencies[jnp.argmin(abs(S["in", "out"]))]
logger.info(f"Coupled resonance frequency: {float(actual_freq) / 1e9} GHz")
_mark_resonance_frequency(actual_freq, "green", "Coupled resonance Frequency")

plt.legend()


# %% [markdown]
# ## Optimizer for given resonance frequency
#
# Find the resonator length that gives a desired resonance frequency using an optimizer.
# Here we use Optax and JAX's automatic differentiation. Instead of evaluating the entire frequency band in every iteration, we can simply minimize the transmission $|S_{21}|^2$ exactly at the target frequency. JAX automatically computes the analytical gradient of the transmission with respect to the resonator length, making the optimization remarkably fast and precise.

# %%
TARGET_FREQUENCY = 6e9  # Target resonance frequency in Hz


@jax.jit
def loss_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Loss function to minimize the S21 transmission at the target frequency.

    Args:
        params: Dictionary containing the resonator length in micrometers.
    """
    length = params["length"]
    # Setup model using the Jittable `circuit` function
    S = circuit(f=jnp.array([TARGET_FREQUENCY]), length=length)
    # S is evaluated at TARGET_FREQUENCY
    # Minimize S21 magnitude at the target frequency
    s21 = S["in", "out"][0]
    return jnp.real(s21 * jnp.conj(s21))


# Test loss function
logger.info(f"Loss at 4000 um: {float(loss_fn({'length': jnp.array(4000.0)}))}")
logger.info(f"Loss at 5900 um: {float(loss_fn({'length': jnp.array(5900.0)}))}")

# To ensure we start within the narrow resonance dip, we first evaluate a coarse array of lengths
expected_length = float(4000.0 * (res_freq / TARGET_FREQUENCY))
coarse_lengths = jnp.linspace(expected_length - 1000, expected_length + 1000, 2001)


@jax.jit
@jax.vmap
def sweep_loss(length_val: jnp.ndarray) -> jnp.ndarray:
    """Evaluate loss over a single length value during the initial vmap sweep.

    Args:
        length_val: The length of the resonator in micrometers.
    """
    return loss_fn({"length": length_val})


coarse_losses = sweep_loss(coarse_lengths)
best_initial_length = float(coarse_lengths[jnp.argmin(coarse_losses)])
logger.info(f"Best initial guess from sweep: {best_initial_length:.2f} µm")

# Initialize optimizer
params = {"length": jnp.array(best_initial_length)}
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)


@jax.jit
def step(params, opt_state):
    """Perform a single Optax optimization step to update parameters.

    Args:
        params: Dictionary containing current parameters (e.g., 'length').
        opt_state: Current state of the optimizer.
    """
    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


logger.info("Starting optimization…")
for i in range(100):
    params, opt_state, loss_value = step(params, opt_state)
    if i % 10 == 0:
        length_val = float(jax.device_get(params["length"]))
        loss_val = float(jax.device_get(loss_value))
        logger.info(f"Step {i}, Length: {length_val:.2f} µm, Loss: {loss_val:.6f}")

length_val = float(jax.device_get(params["length"]))
logger.info(f"Optimized Length: {length_val:.2f} µm")

# Evaluate over a range of frequencies to verify
verification_frequencies = jnp.linspace(0.5e9, 10e9, 1001)
optimal_S = circuit(f=verification_frequencies, length=params["length"])
optimal_freq = verification_frequencies[jnp.argmin(abs(optimal_S["in", "out"]))]
optimal_freq_val = float(jax.device_get(optimal_freq))
logger.info(f"Achieved Resonance Frequency: {optimal_freq_val / 1e9:.2f} GHz")

# Plot
plt.close()
plt.plot(verification_frequencies / 1e9, abs(optimal_S["in", "out"]) ** 2)
plt.xlabel("f [GHz]")
plt.ylabel("$S_{21}$")
_mark_resonance_frequency(optimal_freq_val, "blue", "Optimized resonance Frequency")
_mark_resonance_frequency(TARGET_FREQUENCY, "orange", "Target resonance Frequency")
plt.legend()
plt.show()
# ruff: enable[E402]
