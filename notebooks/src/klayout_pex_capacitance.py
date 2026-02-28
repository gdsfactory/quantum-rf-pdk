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
# # KLayout-PEX Capacitance Extraction
#
# This notebook demonstrates parasitic capacitance extraction using
# [KLayout-PEX](https://github.com/iic-jku/klayout-pex) {cite:p}`klayoutpex2024`,
# an open-source parasitic extraction tool for KLayout.
#
# We will use the interdigitated capacitor component from QPDK and extract
# the mutual capacitance between its conductors using field solver methods.
#
# ## Prerequisites
#
# KLayout-PEX must be installed separately:
#
# ```bash
# pip install klayout-pex
# ```
#
# For the FasterCap engine (3D field solver), you also need to install
# [FasterCap](https://github.com/iic-jku/FasterCap).
#
# The 2.5D analytical engine is built into klayout-pex and doesn't require
# additional dependencies.

# %% tags=["hide-input", "hide-output"]
import tempfile
from pathlib import Path

import gdsfactory as gf
import numpy as np
from IPython.display import Math, display
from matplotlib import pyplot as plt

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.models.pex import (
    PEXResult,
    is_kpex_available,
    parse_capacitance_matrix_from_log,
    run_capacitance_extraction,
)
from qpdk.tech import LAYER

PDK.activate()

# %% [markdown]
# ## Check KLayout-PEX Availability
#
# First, let's check if `kpex` is available on the system.

# %%
kpex_available = is_kpex_available()
print(f"KLayout-PEX (kpex) available: {kpex_available}")

if not kpex_available:
    print("\nTo install KLayout-PEX, run:")
    print("  pip install klayout-pex")
    print("\nFor FasterCap 3D solver, also install FasterCap.")

# %% [markdown]
# ## Interdigitated Capacitor Component
#
# The interdigitated capacitor is a planar capacitor formed by interleaving
# metal fingers from two electrodes. The capacitance depends on:
#
# - Number of fingers
# - Finger length
# - Finger gap (spacing between fingers)
# - Finger thickness (width)
# - Substrate properties (permittivity)
#
# Let's create several interdigitated capacitors with different parameters.

# %%
# Create interdigitated capacitors with varying parameters
capacitors = {
    "4_fingers": interdigital_capacitor(
        fingers=4, finger_length=20.0, finger_gap=2.0, thickness=5.0
    ),
    "6_fingers": interdigital_capacitor(
        fingers=6, finger_length=20.0, finger_gap=2.0, thickness=5.0
    ),
    "8_fingers": interdigital_capacitor(
        fingers=8, finger_length=20.0, finger_gap=2.0, thickness=5.0
    ),
}

# Display one of them
cap_example = capacitors["6_fingers"]
cap_example.plot()
plt.title("6-Finger Interdigitated Capacitor")
plt.show()

# %% [markdown]
# ## Capacitor Geometry Analysis
#
# Before running extraction, let's analyze the capacitor geometry.

# %%
for name, cap in capacitors.items():
    width = cap.xsize
    height = cap.ysize
    print(f"{name}: {width:.1f} × {height:.1f} µm")

# %% [markdown]
# ## Parasitic Extraction Concept
#
# KLayout-PEX extracts parasitic capacitances by analyzing the layout geometry
# and computing the Maxwell capacitance matrix. The matrix elements represent:
#
# - **Diagonal elements ($C_{ii}$)**: Self-capacitance of each conductor
# - **Off-diagonal elements ($C_{ij}$)**: Negative of mutual capacitance between conductors
#
# The mutual capacitance between two conductors is:
#
# ```{math}
# C_{\text{mutual},ij} = -C_{ij}
# ```
#
# For an interdigitated capacitor, we expect significant mutual capacitance
# between the two electrode sets (connected to ports o1 and o2).

# %% [markdown]
# ## Running Capacitance Extraction
#
# The following cell demonstrates how to run capacitance extraction.
# Since KLayout-PEX requires specific PDK technology files, we provide
# both a demonstration using mock data (for documentation) and
# the actual extraction code (for when kpex is properly configured).

# %%
# Demo: Parse a sample capacitance matrix from FasterCap output
sample_fastercap_output = """
Running FasterCap...

Capacitance matrix is:
Dimension 2 x 2
g1_PORT1  2.5e-14 -2.1e-14
g2_PORT2  -2.1e-14 2.8e-14

Done.
"""

# Parse the sample output
matrix, net_names = parse_capacitance_matrix_from_log(sample_fastercap_output)

print("Parsed capacitance matrix (F):")
print(f"Nets: {net_names}")
print(f"Matrix:\n{matrix}")

# Calculate mutual capacitance
c_mutual = -matrix[0, 1]
print(f"\nMutual capacitance: {c_mutual * 1e15:.2f} fF")

# %% [markdown]
# ## Working with PEXResult
#
# The `PEXResult` dataclass provides convenient methods for accessing
# capacitance values and generating summaries.

# %%
# Create a PEXResult from the parsed data
result = PEXResult(
    capacitance_matrix=matrix,
    net_names=net_names,
    success=True,
)

# Print summary
print(result.summary())

# %% [markdown]
# ## Accessing Individual Capacitances

# %%
# Get mutual capacitance between two nets
c_12 = result.get_mutual_capacitance("PORT1", "PORT2")
print(f"Mutual capacitance (PORT1 <-> PORT2): {c_12 * 1e15:.2f} fF")

# Get self-capacitances
c_11 = result.get_self_capacitance("PORT1")
c_22 = result.get_self_capacitance("PORT2")
print(f"Self-capacitance PORT1: {c_11 * 1e15:.2f} fF")
print(f"Self-capacitance PORT2: {c_22 * 1e15:.2f} fF")

# %% [markdown]
# ## Analytical Capacitance Estimation
#
# For comparison, we can estimate the interdigitated capacitor capacitance
# using analytical formulas. A commonly used approximation for interdigitated
# capacitors is based on conformal mapping techniques
# {cite:p}`leizhuAccurateCircuitModel2000`.
#
# A simplified formula for the total capacitance is:
#
# ```{math}
# C \approx \varepsilon_0 \varepsilon_{\text{eff}} \cdot (n-1) \cdot \frac{l}{g} \cdot K(k) / K'(k)
# ```
#
# where:
# - $n$ is the number of fingers
# - $l$ is the finger length
# - $g$ is the gap between fingers
# - $K(k)$ is the complete elliptic integral of the first kind
# - $\varepsilon_{\text{eff}}$ is the effective permittivity

# %%
from scipy import constants
from scipy.special import ellipk


def estimate_interdigital_capacitance(
    fingers: int,
    finger_length: float,  # µm
    finger_gap: float,  # µm
    finger_width: float,  # µm
    epsilon_r_substrate: float = 11.45,  # Silicon
) -> float:
    """Estimate interdigitated capacitor capacitance.

    Uses simplified conformal mapping approach.

    Args:
        fingers: Number of fingers.
        finger_length: Length of each finger in µm.
        finger_gap: Gap between adjacent fingers in µm.
        finger_width: Width of each finger in µm.
        epsilon_r_substrate: Relative permittivity of substrate.

    Returns:
        Estimated capacitance in Farads.
    """
    # Effective permittivity (air on top, substrate below)
    epsilon_eff = (1 + epsilon_r_substrate) / 2

    # Modulus for elliptic integral
    # k = sin(pi * finger_width / (2 * (finger_width + finger_gap)))
    a = finger_width / 2
    b = (finger_width + finger_gap) / 2
    k = a / b
    k_prime = np.sqrt(1 - k**2)

    # Elliptic integrals
    K_k = ellipk(k**2)
    K_k_prime = ellipk(k_prime**2)

    # Capacitance per unit length (F/m)
    # Using standard CPW capacitance formula adapted for IDC
    c_per_length = 2 * constants.epsilon_0 * epsilon_eff * K_k / K_k_prime

    # Total capacitance
    # (n-1) gaps contribute, each with length l
    n_effective_gaps = fingers - 1
    length_m = finger_length * 1e-6  # Convert µm to m

    capacitance = c_per_length * length_m * n_effective_gaps

    return capacitance


# Estimate capacitance for our example capacitors
print("Analytical Capacitance Estimates:")
print("-" * 40)

for name, params in [
    ("4_fingers", {"fingers": 4, "finger_length": 20.0, "finger_gap": 2.0, "finger_width": 5.0}),
    ("6_fingers", {"fingers": 6, "finger_length": 20.0, "finger_gap": 2.0, "finger_width": 5.0}),
    ("8_fingers", {"fingers": 8, "finger_length": 20.0, "finger_gap": 2.0, "finger_width": 5.0}),
]:
    c_est = estimate_interdigital_capacitance(**params)
    print(f"{name}: {c_est * 1e15:.2f} fF")

# %% [markdown]
# ## Running Actual Extraction (When Available)
#
# The following code shows how to run actual extraction when KLayout-PEX
# is properly installed and configured with the appropriate PDK technology files.
#
# Note that KLayout-PEX requires technology definition files that describe
# the process stack (metal thicknesses, dielectric layers, etc.). For the
# QPDK, a custom technology file would need to be created to match the
# layer stack defined in `qpdk/tech.py`.

# %%
# This cell runs actual extraction if kpex is available
# and properly configured for the target PDK

if kpex_available:
    print("Running capacitance extraction with KLayout-PEX...")
    print("Note: This requires proper PDK technology configuration.")

    # Create a test capacitor
    test_cap = interdigital_capacitor(
        fingers=4, finger_length=20.0, finger_gap=2.0, thickness=5.0
    )

    # Run extraction (will fail gracefully if PDK not configured)
    result = run_capacitance_extraction(
        component=test_cap,
        engine="2.5D",  # Use built-in analytical engine
        cleanup=True,
    )

    if result.success:
        print("\nExtraction successful!")
        print(result.summary())
    else:
        print(f"\nExtraction failed: {result.error_message}")
        print("\nThis is expected if the QPDK technology files are not")
        print("configured for KLayout-PEX. See the KLayout-PEX documentation")
        print("for creating custom technology definitions.")
else:
    print("KLayout-PEX not available. Skipping actual extraction.")
    print("Install with: pip install klayout-pex")

# %% [markdown]
# ## Visualization: Capacitance vs. Number of Fingers
#
# Let's visualize how capacitance scales with the number of fingers
# using our analytical estimates.

# %%
# Sweep number of fingers
n_fingers = np.arange(2, 16)
capacitances = []

for n in n_fingers:
    c = estimate_interdigital_capacitance(
        fingers=n, finger_length=20.0, finger_gap=2.0, finger_width=5.0
    )
    capacitances.append(c * 1e15)  # Convert to fF

plt.figure(figsize=(8, 5))
plt.plot(n_fingers, capacitances, "o-", linewidth=2, markersize=8)
plt.xlabel("Number of Fingers")
plt.ylabel("Capacitance (fF)")
plt.title("Interdigitated Capacitor: Capacitance vs. Number of Fingers")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display the scaling relationship
display(
    Math(
        r"C \propto (n - 1) \cdot l / g"
    )
)

# %% [markdown]
# ## Visualization: Capacitance vs. Finger Gap
#
# The capacitance is inversely related to the finger gap.

# %%
# Sweep finger gap
gaps = np.linspace(1.0, 10.0, 20)
capacitances_gap = []

for gap in gaps:
    c = estimate_interdigital_capacitance(
        fingers=6, finger_length=20.0, finger_gap=gap, finger_width=5.0
    )
    capacitances_gap.append(c * 1e15)

plt.figure(figsize=(8, 5))
plt.plot(gaps, capacitances_gap, "o-", linewidth=2, markersize=6)
plt.xlabel("Finger Gap (µm)")
plt.ylabel("Capacitance (fF)")
plt.title("Interdigitated Capacitor: Capacitance vs. Finger Gap (6 fingers)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **KLayout-PEX Integration**: How to use the `qpdk.models.pex` module
#    to interface with KLayout-PEX for parasitic extraction.
#
# 2. **Capacitance Matrix Parsing**: How to parse FasterCap output to
#    obtain the Maxwell capacitance matrix.
#
# 3. **PEXResult Usage**: How to work with extraction results using
#    the `PEXResult` dataclass.
#
# 4. **Analytical Estimation**: Comparison with analytical formulas
#    for interdigitated capacitor capacitance.
#
# 5. **Parameter Sweeps**: Visualizing how capacitance scales with
#    geometric parameters.
#
# For production use, you would need to:
#
# - Install KLayout-PEX and its dependencies (FasterCap or MAGIC)
# - Create a QPDK technology definition file for KLayout-PEX
# - Run extraction on actual layouts with proper net connectivity

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
