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
# # Export a resonator for simulation
#

# %%
from pathlib import Path

import gdsfactory as gf
import numpy as np
from IPython import display

from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.cells.launcher import launcher
from qpdk.cells.resonator import resonator_coupled
from qpdk.tech import LAYER, route_bundle_sbend_cpw

# %% [markdown]
# ## System geometry


# %%
@gf.cell
def resonator_simulations() -> gf.Component:
    """Creates a simulation layout with a resonator coupled to a probeline."""
    c = gf.Component()

    res_ref = c << resonator_coupled({"open_start": True}, coupling_straight_length=300)
    res_ref.movex(-res_ref.size_info.width / 4)

    launcher_left = c << launcher()
    launcher_right = c << launcher()
    launcher_right.mirror()

    width_offset = res_ref.size_info.width + 700
    launcher_left.move((-width_offset, 0))
    launcher_right.move((width_offset, 0))

    route_bundle_sbend_cpw(
        c,
        [launcher_left["o1"], launcher_right["o1"]],
        [res_ref["coupling_o1"], res_ref["coupling_o2"]],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=250.0, airbridge_padding=20.0
        ),
    )

    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 400))

    c.add_ports(launcher_left.ports, prefix="left_")
    c.add_ports(launcher_right.ports, prefix="right_")

    return c


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # Create and display the filled version
    c = gf.Component("capacitance")
    res_ref = c << resonator_simulations()
    c.add_ports(res_ref.ports)
    # c.flatten()
    c.show()
    # to_stl(
    #     c,
    #     "resonator_simulations.stl",
    #     layer_stack=PDK.layer_stack,
    #     hull_invalid_polygons=True,
    # )

    from gplugins.palace import run_scattering_simulation_palace

    material_spec = {
        "Si": {"relative_permittivity": 11.45},
        "Nb": {"relative_permittivity": np.inf},
        "vacuum": {"relative_permittivity": 1},
    }

    results = run_scattering_simulation_palace(
        c,
        layer_stack=PDK.layer_stack,
        material_spec=material_spec,
        only_one_port=True,
        driven_settings={
            "MinFreq": 0.1,
            "MaxFreq": 5,
            "FreqStep": 5,
        },
        n_processes=1,
        simulation_folder=Path().cwd() / "temporary",
        mesh_parameters=dict(
            background_tag="vacuum",
            background_padding=(0,) * 5 + (700,),
            port_names=[port.name for port in c.ports],
            default_characteristic_length=200,
            resolutions={
                "M1": {
                    "resolution": 15,
                },
                "Silicon": {
                    "resolution": 40,
                },
                "vacuum": {
                    "resolution": 40,
                },
                **{
                    f"M1__{port}": {  # `__` is used as the layer to port delimiter for Elmer
                        "resolution": 20,
                        "DistMax": 30,
                        "DistMin": 10,
                        "SizeMax": 14,
                        "SizeMin": 3,
                    }
                    for port in c.ports
                },
            },
        ),
    )
    display(results)
    import skrf
    df = results.scattering_matrix
    df.columns = df.columns.str.strip()
    s_complex = 10 ** df["|S[2][1]| (dB)"].values * np.exp(
        1j * skrf.degree_2_radian(df["arg(S[2][1]) (deg.)"].values)
    )
    ntw = skrf.Network(f=df["f (GHz)"].values, s=s_complex, z0=50)
    cap = np.imag(ntw.y.flatten()) / (ntw.f * 2 * np.pi)
    display(cap)

    plt.plot(ntw.f, cap * 1e15)
    plt.xlabel("Freq (GHz)")
    plt.ylabel("C (fF)")

    if results.field_file_locations:
        pv.start_xvfb()
        pv.set_jupyter_backend("trame")
        field = pv.read(results.field_file_locations[0])
        slice = field.slice_orthogonal(z=layer_stack.layers["bw"].zmin * 1e-6)

        p = pv.Plotter()
        p.add_mesh(slice, scalars="Ue", cmap="turbo")
        p.show_grid()
        p.camera_position = "xy"
        p.enable_parallel_projection()
        p.show()

    # trimesh_scene = to_3d(c,layer_stack=PDK.layer_stack)
    # trimesh_scene.export("resonator_simulations.stl")
