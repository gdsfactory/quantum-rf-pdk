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
# # Export a double pad qubit for simulation
#

# %%

import gdsfactory as gf
import numpy as np
from gdsfactory.export import to_stl

from qpdk.cells.transmon import double_pad_transmon
from qpdk.config import PATH
from qpdk.tech import LAYER

# %% [markdown]
# ## System geometry
#
# Create a simulation layout with a double pad transmon qubit.
#
# The layout consists of:
#   - A double pad transmon qubit with Josephson junction
#   - A simulation area layer enlarged around the layout
#   - Ports added for both left and right pads with prefixes


# %%
@gf.cell
def double_pad_qubit_simulation(
    pad_size: tuple[float, float] = (250.0, 400.0),
    pad_gap: float = 15.0,
) -> gf.Component:
    """Create a double pad qubit simulation layout.

    Args:
        pad_size: (width, height) of each capacitor pad in μm.
        pad_gap: Gap between the two capacitor pads in μm.

    Returns:
        Component with double pad transmon and simulation ports.
    """
    c = gf.Component()

    qubit_ref = c << double_pad_transmon(
        pad_size=pad_size,
        pad_gap=pad_gap,
    )

    # Add simulation area layer enlarged around the layout
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(100, 100))

    # Add ports for both pads
    c.add_port(
        name="left_pad_port",
        center=qubit_ref.ports["left_pad"].center,
        width=qubit_ref.ports["left_pad"].width,
        orientation=qubit_ref.ports["left_pad"].orientation,
        layer=qubit_ref.ports["left_pad"].layer,
    )
    c.add_port(
        name="right_pad_port",
        center=qubit_ref.ports["right_pad"].center,
        width=qubit_ref.ports["right_pad"].width,
        orientation=qubit_ref.ports["right_pad"].orientation,
        layer=qubit_ref.ports["right_pad"].layer,
    )

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # Create and display the component
    c = gf.Component("double_pad_qubit_simulation")
    qubit_ref = c << double_pad_qubit_simulation()
    c.add_ports(qubit_ref.ports)
    c.plot(
        pixel_buffer_options=dict(width=1300, height=1000, oversampling=2, linewidth=3)
    )
    c.show()

    # Export 3D model
    to_stl(
        c,
        PATH.simulation / "double_pad_qubit_simulation.stl",
        layer_stack=PDK.layer_stack,
        hull_invalid_polygons=True,
    )

    material_spec = {
        "Si": {"relative_permittivity": 11.45},
        "Nb": {"relative_permittivity": np.inf},
        "vacuum": {"relative_permittivity": 1},
    }

    # TODO implement running simulations here

    # from gplugins.palace import run_scattering_simulation_palace
    #
    # results = run_scattering_simulation_palace(
    #     c,
    #     layer_stack=PDK.layer_stack,
    #     material_spec=material_spec,
    #     only_one_port=False,
    #     driven_settings={
    #         "MinFreq": 0.1,
    #         "MaxFreq": 8,
    #         "FreqStep": 5,
    #     },
    #     n_processes=1,
    #     simulation_folder=Path().cwd() / "temporary",
    #     mesh_parameters=dict(
    #         background_tag="vacuum",
    #         background_padding=(0,) * 5 + (700,),
    #         port_names=[port.name for port in c.ports],
    #         default_characteristic_length=200,
    #         resolutions={
    #             "M1": {
    #                 "resolution": 15,
    #             },
    #             "Silicon": {
    #                 "resolution": 40,
    #             },
    #             "vacuum": {
    #                 "resolution": 40,
    #             },
    #             **{
    #                 f"M1__{port}": {  # `__` is used as the layer to port delimiter for Elmer
    #                     "resolution": 20,
    #                     "DistMax": 30,
    #                     "DistMin": 10,
    #                     "SizeMax": 14,
    #                     "SizeMin": 3,
    #                 }
    #                 for port in c.ports
    #             },
    #         },
    #     ),
    # )
    #
    # display(results)
