# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RSFQ Circuit Simulation with JoSIM
#
# This notebook demonstrates how to simulate **Rapid Single Flux Quantum (RSFQ)**
# circuits using [JoSIM](https://github.com/JoeyDelp/JoSIM)
# {cite:p}`delportJoSIMSupercondutor2019` and connect the results to layout
# components in **qpdk**.
#
# ## What is RSFQ?
#
# Rapid Single Flux Quantum (RSFQ) logic is a digital electronics technology based
# on superconducting Josephson junctions. Information is encoded as single magnetic
# flux quanta (:math:`\Phi_0 = h / 2e \approx 2.07 \times 10^{-15}\,\text{Wb}`)
# propagating through circuits of Josephson junctions and inductors. RSFQ circuits
# operate at cryogenic temperatures (typically 4 K) and achieve clock speeds of
# tens to hundreds of GHz with extremely low power dissipation
# {cite:p}`likharevRSFQLogicMemory1991`.
#
# ## JoSIM
#
# JoSIM is an open-source superconducting circuit simulator that reads SPICE-like
# netlists containing Josephson junction models and performs transient analysis. It
# is the standard tool for verifying RSFQ cell designs before fabrication.
#
# ### Prerequisites
#
# JoSIM must be compiled from source or obtained as a pre-built binary. See the
# [JoSIM documentation](https://joeydelp.github.io/JoSIM/) for installation
# instructions:
#
# ```bash
# git clone https://github.com/JoeyDelp/JoSIM
# cd JoSIM && mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)
# ```
#
# The resulting `josim-cli` binary should be on your `PATH` or specified
# explicitly below.

# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing quantum-rf-pdk...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])
    # JoSIM must be installed separately — see instructions above.

# %% [markdown]
# ## Imports and Configuration

# %%
import shutil
import subprocess
import tempfile
from pathlib import Path

import gdsfactory as gf
import numpy as np
from matplotlib import pyplot as plt

from qpdk.cells import meander_inductor, single_josephson_junction

# Path to JoSIM CLI binary — adjust if not on PATH
JOSIM_CLI = shutil.which("josim-cli") or "josim-cli"

# %% [markdown]
# ## Helper: Run JoSIM and Parse Results
#
# We define a small utility function that writes a netlist to a temporary file,
# invokes `josim-cli`, and returns the results as a NumPy array with column names.


# %%
def run_josim(
    netlist: str, josim_path: str = JOSIM_CLI
) -> tuple[list[str], np.ndarray]:
    """Run a JoSIM simulation and return (column_names, data_array).

    Args:
        netlist: SPICE-format netlist string for JoSIM.
        josim_path: Path to the josim-cli binary.

    Returns:
        Tuple of (list of column header strings, 2-D NumPy array of results).

    Raises:
        RuntimeError: If JoSIM exits with an error.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cir_path = Path(tmpdir) / "circuit.cir"
        out_path = Path(tmpdir) / "output.csv"
        cir_path.write_text(netlist)

        result = subprocess.run(  # noqa: S603
            [josim_path, "-o", str(out_path), str(cir_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = f"JoSIM failed:\n{result.stderr}\n{result.stdout}"
            raise RuntimeError(msg)

        # Parse CSV output
        lines = out_path.read_text().splitlines()
        headers = [h.strip().strip('"') for h in lines[0].split(",")]
        data = np.loadtxt(out_path, delimiter=",", skiprows=1)

    return headers, data


# %% [markdown]
# ## Example 1: Josephson Transmission Line (JTL)
#
# The Josephson Transmission Line (JTL) is the most fundamental RSFQ component.
# It propagates SFQ pulses from one junction to the next using a chain of
# Josephson junctions biased just below their critical current
# {cite:p}`likharevRSFQLogicMemory1991`.
#
# ### Circuit Schematic
#
# ```
# VIN ─── L01 ─── B01 ─── L02 ─── L03 ─── B02 ─── L04 ─── ROUT
#                   │                │                │
#                  LP01            LPR01            LP02
#                   │                │                │
#                  GND             IB01             GND
# ```
#
# ### Netlist

# %%
jtl_netlist = """\
* Josephson Transmission Line (JTL) — MITLL process
B01        3          7          jmitll     area=2.16
B02        6          8          jmitll     area=2.16
IB01       0          1          pwl(0      0 5p 280u)
L01        4          3          2p
L02        3          2          2.425p
L03        2          6          2.425p
L04        6          5          2.031p
LP01       0          7          0.086p
LP02       0          8          0.096p
LPR01      2          1          0.278p
LRB01      7          9          0.086p
LRB02      8          10         0.086p
RB01       9          3          5.23
RB02       10         6          5.23
ROUT       5          0          2
VIN        4          0          pwl(0 0 300p 0 302.5p 827.13u 305p 0 600p 0 602.5p 827.13u 605p 0)
.model jmitll jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)
.tran 0.25p 1000p 0 0.25p
.print DEVV VIN
.print DEVI ROUT
.print PHASE B01
.print PHASE B02
.end
"""

print(jtl_netlist)

# %% [markdown]
# ### Simulation

# %%
headers, data = run_josim(jtl_netlist)
time_ps = data[:, 0] * 1e12  # Convert seconds to picoseconds

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Input voltage pulse
axes[0].plot(time_ps, data[:, 1] * 1e6, "b-", linewidth=0.8)
axes[0].set_ylabel("Input voltage (µV)")
axes[0].set_title("JTL Simulation — SFQ Pulse Propagation")

# Output current through load resistor
axes[1].plot(time_ps, data[:, 2] * 1e6, "r-", linewidth=0.8)
axes[1].set_ylabel("Output current (µA)")

# Junction phases — each 2π jump represents one SFQ pulse
axes[2].plot(time_ps, data[:, 3] / (2 * np.pi), "g-", linewidth=0.8, label="B01")
axes[2].plot(time_ps, data[:, 4] / (2 * np.pi), "m-", linewidth=0.8, label="B02")
axes[2].set_ylabel(r"Phase ($\Phi / 2\pi$)")
axes[2].set_xlabel("Time (ps)")
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# The phase of each junction advances by :math:`2\pi` when an SFQ pulse passes
# through it, confirming that single flux quanta are propagating along the
# transmission line.

# %% [markdown]
# ## Example 2: DC-to-SFQ Converter with JTL Chain
#
# The DC-to-SFQ (DC/SFQ) converter transforms a DC current pulse into a train of
# SFQ pulses — the standard interface between conventional electronics and RSFQ
# logic. Here we connect it to a chain of JTLs and a pulse sink.

# %%
dcsfq_netlist = """\
* DC-SFQ converter driving a JTL chain into a sink
.SUBCKT JTL 4 5
B01        3          7          jj1     area=2.16
B02        6          8          jj1     area=2.16
IB01       0          1          pwl(0 0 5p 280u)
L01        4          3          2.031p
L02        3          2          2.425p
L03        2          6          2.425p
L04        6          5          2.031p
LP01       0          7          0.086p
LP02       0          8          0.096p
LPR01      2          1          0.278p
LRB01      7          9          1p
LRB02      8          10         1p
RB01       9          3          5.23
RB02       10         6          5.23
.model jj1 jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)
.ends JTL
.SUBCKT DCSFQ 2 17
B01        5          3          jj1     area=1.32
B02        5          6          jj1     area=1
B03        9          10         jj1     area=1.5
B04        13         14         jj1     area=1.96
B05        15         16         jj1     area=1.96
IB01       0          8          pwl(0 0 5p 162.5u)
IB02       0          12         pwl(0 0 5p 260u)
L01        2          1          0.848p
L02        0          1          7.712p
L03        1          3          1.778p
L04        5          7          0.543p
L05        7          9          3.149p
L06        9          11         1.323p
L07        11         13         1.095p
L08        13         15         2.951p
L09        15         17         1.63p
LP01       0          6          0.398p
LP02       0          10         0.211p
LP03       0          14         0.276p
LP04       0          16         0.224p
LPR01      7          8          0.915p
LPR02      11         12         0.307p
LRB01      4          5          1p
LRB02      18         6          1p
LRB03      19         10         1p
LRB04      20         14         1p
LRB05      21         16         1p
RB01       3          4          8.56
RB02       18         5          11.30
RB03       19         9          7.53
RB04       20         13         5.77
RB05       21         15         5.77
.model jj1 jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)
.ends DCSFQ
.SUBCKT SINK 2
B01        1          4          jj1     area=2.16
IB01       0          5          pwl(0 0 5p 280u)
L01        2          1          0.517p
L02        1          3          5.307p
LP01       0          4          0.086p
LPR01      1          5          0.265p
LRB01      4          6          1p
RB01       6          1          5.23
ROUT       0          3          4.02
.model jj1 jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160, rN=16, icrit=0.1mA)
.ends SINK
IA         0          1         pwl(0 0 170p 0 176p 600u 182p 0 370p 0 376p 600u 382p 0 600p 0 606p 600u 612p 0 700p 0 706p 600u 712p 0)
X01        DCSFQ      1         2
X02        JTL        2         3
X03        JTL        3         4
X04        JTL        4         5
X05        SINK       5
.tran 0.25p 1000p 0 0.25p
.print nodev 1 0
.print nodep 3 0
.print nodev 5 0
.end
"""

headers2, data2 = run_josim(dcsfq_netlist)
time_ps2 = data2[:, 0] * 1e12

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time_ps2, data2[:, 1] * 1e6, "b-", linewidth=0.8)
axes[0].set_ylabel("Input node (µV)")
axes[0].set_title("DC-SFQ Converter → JTL Chain → Sink")

axes[1].plot(time_ps2, data2[:, 2] / (2 * np.pi), "g-", linewidth=0.8)
axes[1].set_ylabel(r"Phase at node 3 ($\Phi / 2\pi$)")

axes[2].plot(time_ps2, data2[:, 3] * 1e6, "r-", linewidth=0.8)
axes[2].set_ylabel("Sink node voltage (µV)")
axes[2].set_xlabel("Time (ps)")

plt.tight_layout()
plt.show()

# %% [markdown]
# Each DC current pulse applied to the input of the DC/SFQ converter produces an
# SFQ pulse that propagates through the JTL chain. The phase staircase at node 3
# confirms that discrete flux quanta traverse the circuit, and the voltage spikes
# at the sink resistor show the output SFQ pulses.

# %% [markdown]
# ## Connecting to QPDK Layout
#
# The RSFQ components simulated above are built from Josephson junctions and
# superconducting inductors — both available in qpdk. Below we show how the
# physical layout components correspond to the circuit elements in the JoSIM
# netlist.
#
# ### Josephson Junction
#
# The `single_josephson_junction` component in qpdk implements the physical
# junction structure (two overlapping superconducting wires separated by an
# oxide barrier).

# %%
jj = single_josephson_junction()
jj.plot()

# %% [markdown]
# ### Meander Inductor
#
# Inductors in RSFQ circuits are typically implemented as narrow superconducting
# traces whose kinetic inductance dominates. The `meander_inductor` component
# provides a compact serpentine layout.

# %%
ind = meander_inductor(n_turns=3, turn_length=50.0)
ind.plot()

# %% [markdown]
# ### Conceptual JTL Layout
#
# A JTL cell in layout consists of two junctions connected by inductors, with a
# bias feed inductor. We can compose these qpdk primitives to sketch the layout
# structure.

# %%
c = gf.Component("jtl_layout_concept")

# Place two junctions
jj1_ref = c << single_josephson_junction()
jj2_ref = c << single_josephson_junction()
jj2_ref.d.movex(100)

# Connect with a straight waveguide (representing the coupling inductor)
route = gf.routing.route_single(
    c,
    port1=jj1_ref.ports["o2"],
    port2=jj2_ref.ports["o1"],
)

c.plot()

# %% [markdown]
# ## Summary
#
# This notebook demonstrated how to:
#
# 1. **Write RSFQ circuit netlists** in SPICE format for JoSIM.
# 2. **Run transient simulations** of Josephson Transmission Lines and DC-SFQ
#    converters.
# 3. **Visualize SFQ pulse propagation** through junction phase and voltage
#    waveforms.
# 4. **Connect simulation results to physical layout** using qpdk's Josephson
#    junction and inductor components.
#
# The combination of JoSIM for circuit-level verification and qpdk/gdsfactory
# for physical layout provides a complete design flow for RSFQ digital circuits.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
