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
# # Xmon Tunable Coupler Chip
#
# This sample demonstrates creating a chip with two Xmon transmon qubits
# coupled through a third Xmon qubit that acts as a tunable coupler.
#
# The design is inspired by {cite:p}`liTunableCouplerRealizing2020` which demonstrates
# a controlled-phase gate using a tunable coupler in superconducting circuits.

# %%
from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from klayout.db import DCplxTrans

from qpdk.cells.capacitor import plate_capacitor_single
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import resonator_quarter_wave
from qpdk.cells.transmon import xmon_transmon
from qpdk.cells.waveguides import straight, tee
from qpdk.tech import (
    LAYER,
    route_single_cpw,
)

# %% [markdown]
# ## Xmon with Resonator Function
#
# Creates an Xmon transmon qubit with a coupled quarter-wave resonator for readout.


# %%
@gf.cell
def xmon_with_resonator(
    arm_width: tuple[float, float, float, float] = (30.0, 20.0, 30.0, 20.0),
    arm_lengths: tuple[float, float, float, float] = (160.0, 120.0, 160.0, 120.0),
    gap_width: float = 10.0,
    resonator_length: float = 5000.0,
    resonator_meanders: int = 5,
    resonator_meander_start: tuple[float, float] = (-500, -500),
    coupler_offset: tuple[float, float] = (-45, 0),
) -> Component:
    """Creates an Xmon transmon qubit with a coupled quarter-wave resonator.

    The Xmon design follows {cite:p}`barendsCoherentJosephsonQubit2013a`.

    Args:
        arm_width: Tuple of (top, right, bottom, left) arm widths in μm.
        arm_lengths: Tuple of (top, right, bottom, left) arm lengths in μm.
        gap_width: Width of the etched gap around arms in μm.
        resonator_length: Length of the resonator in µm.
        resonator_meanders: Number of meander sections for the resonator.
        resonator_meander_start: (x, y) position of the start of the resonator meander.
        coupler_offset: (x, y) offset for the coupler position.

    Returns:
        Component: An Xmon qubit with coupled resonator.
    """
    c = Component()

    # Create Xmon qubit
    xmon_ref = c << xmon_transmon(
        arm_width=arm_width,
        arm_lengths=arm_lengths,
        gap_width=gap_width,
    )

    # Create coupling capacitor
    coupler = plate_capacitor_single(width=20, length=200)
    coupler_ref = c << coupler

    # Position coupler near the left arm of the Xmon
    coupler_ref.transform(
        xmon_ref.ports["left_arm"].dcplx_trans
        * DCplxTrans.R180
        * DCplxTrans(*coupler_offset)
    )

    # Route to resonator input
    resonator_input_port = gf.Port(
        name="resonator_input",
        center=resonator_meander_start,
        orientation=0,
        layer=LAYER.M1_DRAW,
        width=10.0,
    )
    route = route_single_cpw(
        component=c,
        port1=resonator_input_port,
        port2=coupler_ref.ports["o1"],
        steps=[{"x": coupler_ref.ports["o1"].x}],
        auto_taper=False,
    )

    # Create resonator
    resonator_ref = c << resonator_quarter_wave(
        length=resonator_length - route.length * c.kcl.dbu,
        meanders=resonator_meanders,
        open_end=True,
    )
    resonator_ref.rotate(180)
    resonator_ref.transform(resonator_input_port.dcplx_trans)

    # Add ports from Xmon (all arms except left which is used for resonator coupling)
    c.add_port(port=xmon_ref.ports["top_arm"], name="top_arm")
    c.add_port(port=xmon_ref.ports["right_arm"], name="right_arm")
    c.add_port(port=xmon_ref.ports["bottom_arm"], name="bottom_arm")
    # Add left_arm as placement port (not for waveguide routing since it's near resonator)
    c.add_port(port=xmon_ref.ports["left_arm"], name="left_arm", port_type="placement")
    c.add_ports(xmon_ref.ports.filter(regex=r"junction"))

    # Add resonator output port
    res_port = resonator_ref.ports["o1"]
    c.add_port(
        center=res_port.center,
        cross_section=res_port.cross_section,
        layer=res_port.layer,
        name="resonator_o1",
        orientation=res_port.orientation,
        port_type="placement",
        width=res_port.width,
    )

    c.info["qubit_type"] = "xmon"
    c.info["resonator_type"] = "quarter_wave"

    return c


# %% [markdown]
# ## Tunable Coupler Chip Function
#
# Creates a chip with two Xmon qubits coupled through a tunable coupler Xmon.
# Each qubit has a readout resonator coupled to a common probeline, and each
# qubit has an individual flux line for frequency tuning.


# %%
@gf.cell
def xmon_tunable_coupler_chip(
    qubit_arm_lengths: tuple[float, float, float, float] = (160.0, 120.0, 160.0, 120.0),
    coupler_arm_lengths: tuple[float, float, float, float] = (100.0, 80.0, 100.0, 80.0),
    qubit_spacing: float = 600.0,
    resonator1_length: float = 5000.0,
    resonator2_length: float = 4800.0,
    resonator_meanders: int = 5,
    probeline_length: float = 3000.0,
) -> Component:
    """Creates a tunable coupler chip with two Xmon qubits and a central coupler.

    This design follows the architecture described in
    {cite:p}`liTunableCouplerRealizing2020` for realizing controlled-phase gates
    with dynamically decoupled regime in superconducting circuits.

    The chip features:
    - Two Xmon transmon qubits (Q1 and Q2) for computation
    - A central Xmon qubit (Coupler) acting as a tunable coupler
    - Quarter-wave resonators for readout of Q1 and Q2
    - A common probeline for resonator readout
    - Individual flux lines for each qubit and the coupler

    Args:
        qubit_arm_lengths: Arm lengths for the computational qubits in μm.
        coupler_arm_lengths: Arm lengths for the tunable coupler qubit in μm.
        qubit_spacing: Horizontal spacing between qubits in μm.
        resonator1_length: Length of Q1's readout resonator in µm.
        resonator2_length: Length of Q2's readout resonator in µm.
        resonator_meanders: Number of meander sections for the resonators.
        probeline_length: Length of the probeline in µm.

    Returns:
        Component: Complete tunable coupler chip layout.
    """
    c = Component()

    # Create qubit Q1 with resonator (left side)
    q1 = xmon_with_resonator(
        arm_lengths=qubit_arm_lengths,
        resonator_length=resonator1_length,
        resonator_meanders=resonator_meanders,
        resonator_meander_start=(-500, -600),
    )
    q1_ref = c << q1
    q1_ref.move((-qubit_spacing, 0))

    # Create tunable coupler (center) - smaller Xmon, no resonator needed
    coupler = xmon_transmon(
        arm_lengths=coupler_arm_lengths,
        arm_width=(20.0, 15.0, 20.0, 15.0),  # Smaller widths for coupler
        gap_width=8.0,
    )
    coupler_ref = c << coupler

    # Create qubit Q2 with resonator (right side)
    # Mirror Y so resonators of Q1 and Q2 are on opposite sides for symmetry
    q2 = xmon_with_resonator(
        arm_lengths=qubit_arm_lengths,
        resonator_length=resonator2_length,
        resonator_meanders=resonator_meanders,
        resonator_meander_start=(-500, -600),
    )
    q2_ref = c << q2
    q2_ref.mirror_y()  # Mirror along Y axis - left_arm now faces coupler
    q2_ref.move((qubit_spacing, 0))

    # Create coupling capacitors between qubits and coupler
    coupling_cap = partial(plate_capacitor_single, width=15, length=80)

    # Q1 to Coupler coupling (Q1's right_arm faces coupler's left_arm)
    q1_coupler_cap_ref = c << coupling_cap()
    q1_coupler_cap_ref.rotate(90)
    q1_coupler_cap_ref.move(
        (
            (q1_ref.ports["right_arm"].x + coupler_ref.ports["left_arm"].x) / 2,
            q1_ref.ports["right_arm"].y,
        )
    )

    # Coupler to Q2 coupling (coupler's right_arm faces Q2's left_arm after mirror)
    coupler_q2_cap_ref = c << coupling_cap()
    coupler_q2_cap_ref.rotate(90)
    coupler_q2_cap_ref.move(
        (
            (coupler_ref.ports["right_arm"].x + q2_ref.ports["left_arm"].x) / 2,
            coupler_ref.ports["right_arm"].y,
        )
    )

    # Create probeline with launchers
    launcher_in = c << launcher()
    launcher_in.move((-(probeline_length / 2 + 300), -1000))

    launcher_out = c << launcher()
    launcher_out.mirror_x()
    launcher_out.move((probeline_length / 2 + 300, -1000))

    # Add straight probeline section
    probeline_xs = "cpw"
    probeline_straight = c << straight(
        length=probeline_length, cross_section=probeline_xs
    )
    probeline_straight.move((-probeline_length / 2, -1000))

    # Connect launchers to probeline
    route_single_cpw(
        c,
        port1=launcher_in.ports["o1"],
        port2=probeline_straight.ports["o1"],
        auto_taper=False,
    )
    route_single_cpw(
        c,
        port1=probeline_straight.ports["o2"],
        port2=launcher_out.ports["o1"],
        auto_taper=False,
    )

    # Create flux lines with tee structures for each qubit
    # Q1 flux line (top launcher)
    flux_tee_q1 = c << tee()
    flux_tee_q1.rotate(90)
    flux_tee_q1.move((q1_ref.ports["top_arm"].x, q1_ref.ports["top_arm"].y + 100))

    flux_launcher_q1 = c << launcher()
    flux_launcher_q1.rotate(-90)
    flux_launcher_q1.move((q1_ref.ports["top_arm"].x, 800))

    route_single_cpw(
        c,
        port1=q1_ref.ports["top_arm"],
        port2=flux_tee_q1.ports["o3"],
        auto_taper=False,
        allow_width_mismatch=True,
    )
    route_single_cpw(
        c,
        port1=flux_tee_q1.ports["o2"],
        port2=flux_launcher_q1.ports["o1"],
        auto_taper=False,
    )

    # Coupler flux line (top launcher)
    flux_tee_coupler = c << tee()
    flux_tee_coupler.rotate(90)
    flux_tee_coupler.move(
        (coupler_ref.ports["top_arm"].x, coupler_ref.ports["top_arm"].y + 100)
    )

    flux_launcher_coupler = c << launcher()
    flux_launcher_coupler.rotate(-90)
    flux_launcher_coupler.move((coupler_ref.ports["top_arm"].x, 800))

    route_single_cpw(
        c,
        port1=coupler_ref.ports["top_arm"],
        port2=flux_tee_coupler.ports["o3"],
        auto_taper=False,
        allow_width_mismatch=True,
    )
    route_single_cpw(
        c,
        port1=flux_tee_coupler.ports["o2"],
        port2=flux_launcher_coupler.ports["o1"],
        auto_taper=False,
    )

    # Q2 flux line (top launcher)
    flux_tee_q2 = c << tee()
    flux_tee_q2.rotate(90)
    flux_tee_q2.move((q2_ref.ports["top_arm"].x, q2_ref.ports["top_arm"].y + 100))

    flux_launcher_q2 = c << launcher()
    flux_launcher_q2.rotate(-90)
    flux_launcher_q2.move((q2_ref.ports["top_arm"].x, 800))

    route_single_cpw(
        c,
        port1=q2_ref.ports["top_arm"],
        port2=flux_tee_q2.ports["o3"],
        auto_taper=False,
        allow_width_mismatch=True,
    )
    route_single_cpw(
        c,
        port1=flux_tee_q2.ports["o2"],
        port2=flux_launcher_q2.ports["o1"],
        auto_taper=False,
    )

    # Add chip info
    c.info["description"] = "Xmon tunable coupler chip"
    c.info["reference"] = "10.1103/PhysRevApplied.14.024070"

    return c


# %% [markdown]
# ## Example Usage

# %%
if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # Create and display the tunable coupler chip
    chip = xmon_tunable_coupler_chip()
    chip.show()
