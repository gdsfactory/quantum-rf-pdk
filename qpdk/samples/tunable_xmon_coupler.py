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
# # Tunable Xmon Qubit Coupler Example
#
# This example demonstrates a tunable coupler architecture with two Xmon transmon qubits
# coupled through a third tunable Xmon qubit. The design is based on
# :cite:`li2020TunableCouplerRealizing`.
#
# The architecture consists of:
# - Two main Xmon qubits (Q1 and Q2) with coupled resonators
# - One tunable coupler Xmon qubit (QC) positioned between them
# - Flux lines for frequency tuning of all three qubits
# - Readout through coupled resonators connected to probelines

# %%
import gdsfactory as gf

from qpdk import PDK, tech
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import resonator_coupled
from qpdk.cells.transmon import xmon_transmon
from qpdk.cells.waveguides import rectangle
from qpdk.tech import coplanar_waveguide, route_single_cpw

# %% [markdown]
# ## Tunable Xmon Coupler Sample Chip
#
# Creates a sample chip with two main Xmon qubits coupled through a tunable coupler Xmon.


# %%
@gf.cell
def tunable_xmon_coupler_chip(
    qubit_spacing: float = 1500.0,
    resonator_length: float = 4000.0,
    coupling_gap: float = 15.0,
) -> gf.Component:
    """Creates a tunable Xmon coupler test chip.

    Architecture based on :cite:`li2020TunableCouplerRealizing`, featuring two main Xmon
    qubits coupled through a tunable coupler Xmon qubit with flux control lines.

    Args:
        qubit_spacing: Horizontal spacing between main qubits in μm.
        resonator_length: Length of coupled resonators in μm.
        coupling_gap: Gap between resonator and probeline for coupling in μm.

    Returns:
        Component: A complete chip with tunable Xmon coupler architecture.
    """
    PDK.activate()  # Activate the PDK to access cross sections
    c = gf.Component()

    # Create main Xmon qubits (Q1 and Q2)
    # Q1 - Left qubit
    q1_ref = c << xmon_transmon(
        arm_width=(40.0, 40.0, 40.0, 40.0),  # top, right, bottom, left
        arm_lengths=(200.0, 180.0, 200.0, 180.0),  # top, right, bottom, left
        gap_width=8.0,
    )
    q1_ref.move((-qubit_spacing / 2, 0))

    # Q2 - Right qubit
    q2_ref = c << xmon_transmon(
        arm_width=(40.0, 40.0, 40.0, 40.0),  # top, right, bottom, left
        arm_lengths=(200.0, 180.0, 200.0, 180.0),  # top, right, bottom, left
        gap_width=8.0,
    )
    q2_ref.move((qubit_spacing / 2, 0))

    # QC - Tunable coupler qubit (smaller, in the center)
    qc_ref = c << xmon_transmon(
        arm_width=(30.0, 30.0, 30.0, 30.0),  # top, right, bottom, left
        arm_lengths=(120.0, 100.0, 120.0, 100.0),  # smaller for tunable coupler
        gap_width=6.0,
    )
    qc_ref.move((0, 0))

    # Create coupled resonators for Q1 and Q2
    resonator_params = {
        "resonator_params": {"length": resonator_length, "meanders": 4},
        "coupling_gap": coupling_gap,
        "coupling_straight_length": 200.0,
    }

    # Q1 resonator positioned below Q1
    q1_resonator_ref = c << resonator_coupled(**resonator_params)
    q1_resonator_ref.move((-qubit_spacing / 2 - 200, -800))

    # Q2 resonator positioned below Q2
    q2_resonator_ref = c << resonator_coupled(**resonator_params)
    q2_resonator_ref.move((qubit_spacing / 2 - 200, -800))

    # Add launchers for readout
    launcher_q1_in = c << launcher()
    launcher_q1_in.move((-qubit_spacing / 2 - 600, -800))

    launcher_q1_out = c << launcher()
    launcher_q1_out.rotate(180)
    launcher_q1_out.move((-qubit_spacing / 2 + 200, -800))

    launcher_q2_in = c << launcher()
    launcher_q2_in.move((qubit_spacing / 2 - 600, -800))

    launcher_q2_out = c << launcher()
    launcher_q2_out.rotate(180)
    launcher_q2_out.move((qubit_spacing / 2 + 200, -800))

    # Route the probelines with simple connections
    # Q1 probeline
    route_single_cpw(
        component=c,
        port1=launcher_q1_in.ports["o1"],
        port2=q1_resonator_ref.ports["coupling_o1"],
        cross_section=coplanar_waveguide(width=10, gap=6),
    )

    route_single_cpw(
        component=c,
        port1=q1_resonator_ref.ports["coupling_o2"],
        port2=launcher_q1_out.ports["o1"],
        cross_section=coplanar_waveguide(width=10, gap=6),
    )

    # Q2 probeline
    route_single_cpw(
        component=c,
        port1=launcher_q2_in.ports["o1"],
        port2=q2_resonator_ref.ports["coupling_o1"],
        cross_section=coplanar_waveguide(width=10, gap=6),
    )

    route_single_cpw(
        component=c,
        port1=q2_resonator_ref.ports["coupling_o2"],
        port2=launcher_q2_out.ports["o1"],
        cross_section=coplanar_waveguide(width=10, gap=6),
    )

    # Create simple flux lines as rectangles
    flux_line_layer = tech.LAYER.M1_DRAW

    # Q1 flux line (extending upward)
    q1_flux = c << rectangle(
        size=(5.0, 300.0),
        layer=flux_line_layer,
    )
    q1_flux.move((-qubit_spacing / 2 + 10, 250))

    # Q2 flux line (extending upward)
    q2_flux = c << rectangle(
        size=(5.0, 300.0),
        layer=flux_line_layer,
    )
    q2_flux.move((qubit_spacing / 2 + 10, 250))

    # QC flux line (extending to the left)
    qc_flux = c << rectangle(
        size=(200.0, 5.0),
        layer=flux_line_layer,
    )
    qc_flux.move((-300, 10))

    # Add component metadata
    c.info["chip_type"] = "tunable_xmon_coupler"
    c.info["qubit_count"] = 3
    c.info["main_qubits"] = 2
    c.info["coupler_qubits"] = 1
    c.info["reference"] = "li2020TunableCouplerRealizing"

    return c


# %%
if __name__ == "__main__":
    chip = tunable_xmon_coupler_chip()
    chip.show()
    print("Tunable Xmon coupler chip created successfully!")
    print(f"Chip info: {chip.info}")
