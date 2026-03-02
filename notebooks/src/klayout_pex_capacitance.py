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
# We use the KPEX 2.5D analytical engine to extract the mutual capacitance
# of an interdigitated capacitor component from QPDK.
#
# ## Prerequisites
#
# KLayout-PEX must be installed:
#
# ```bash
# pip install klayout-pex
# ```
#
# The 2.5D analytical engine is built into klayout-pex and doesn't require
# additional external dependencies like FasterCap or MAGIC.

# %% tags=["hide-input", "hide-output"]
from matplotlib import pyplot as plt

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.models.pex import (
    generate_kpex_tech_json,
    is_kpex_available,
    run_capacitance_extraction,
)
from qpdk.tech import LAYER_STACK

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

# %%
# Create an interdigitated capacitor
cap = interdigital_capacitor(
    fingers=6, finger_length=20.0, finger_gap=2.0, thickness=5.0
)

cap.plot()
plt.title("6-Finger Interdigitated Capacitor")
plt.show()

# Print geometry info
print(f"Capacitor size: {cap.xsize:.1f} × {cap.ysize:.1f} µm")

# %% [markdown]
# ## Generated KPEX Technology Definition
#
# The `qpdk.models.pex` module automatically generates KLayout-PEX technology
# files from the QPDK LayerStack. Let's examine the generated technology JSON:

# %%
import json

tech_json = generate_kpex_tech_json(LAYER_STACK, name="qpdk")
print("Generated KPEX Technology JSON:")
print(json.dumps(tech_json, indent=2))

# %% [markdown]
# ## Running Capacitance Extraction with KPEX 2.5D Engine
#
# The KPEX 2.5D engine uses analytical formulas based on MAGIC's parasitic
# extraction concepts, implemented using KLayout methods. This provides
# fast extraction without requiring external field solvers.

# %%
if kpex_available:
    print("Running capacitance extraction with KPEX 2.5D engine...")

    # Run extraction
    result = run_capacitance_extraction(
        component=cap,
        engine="2.5D",
        layer_stack=LAYER_STACK,
        cleanup=False,  # Keep output for inspection
    )

    if result.success:
        print("\n✓ Extraction successful!")
        print(result.summary())
    else:
        print(f"\n✗ Extraction failed: {result.error_message}")
        if result.log_output:
            print("\nLog output (last 2000 chars):")
            print(result.log_output[-2000:])
else:
    print("KLayout-PEX not available. Install with: pip install klayout-pex")

# %% [markdown]
# ## Extracting Multiple Capacitors
#
# Let's extract capacitances for capacitors with different numbers of fingers
# and compare the results.

# %%
if kpex_available:
    results = {}

    for n_fingers in [4, 6, 8]:
        cap_n = interdigital_capacitor(
            fingers=n_fingers, finger_length=20.0, finger_gap=2.0, thickness=5.0
        )

        result_n = run_capacitance_extraction(
            component=cap_n,
            engine="2.5D",
            layer_stack=LAYER_STACK,
            cleanup=True,
        )

        if result_n.success:
            results[n_fingers] = result_n
            print(f"{n_fingers} fingers: Extraction successful")
        else:
            print(f"{n_fingers} fingers: {result_n.error_message}")

    # Display results comparison
    if results:
        print("\n" + "=" * 50)
        print("Capacitance Comparison:")
        print("=" * 50)
        for n, res in results.items():
            print(f"\n{n} fingers:")
            print(res.summary())

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
#
# 1. **KLayout-PEX Integration**: Using the `qpdk.models.pex` module
#    to perform parasitic extraction with the KPEX 2.5D engine.
#
# 2. **Automatic Technology Generation**: The module generates KPEX
#    technology definition files from the QPDK LayerStack.
#
# 3. **Capacitance Extraction**: Running extraction on interdigitated
#    capacitor components and accessing the results.
#
# The KPEX 2.5D engine provides fast analytical extraction suitable for
# quick design iterations. For higher accuracy, the FasterCap 3D field
# solver engine can be used (requires separate FasterCap installation).

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
