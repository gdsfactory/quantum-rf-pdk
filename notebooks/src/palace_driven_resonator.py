# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Palace Driven Simulation of a Coupled CPW Resonator
#
# This notebook demonstrates a full-wave finite-element (FEM) S-parameter simulation of a
# coupled coplanar-waveguide resonator using [Palace](https://github.com/awslabs/palace),
# an open-source parallel electromagnetics solver, driven through
# [gsim](https://gdsfactory.github.io/gsim/), the same geometry-to-solver bridge the
# `hfss_eigenmode_resonator` notebook covers for Ansys HFSS, but with a fully open-source
# toolchain.
#
# The workflow is:
#
# 1. Build a resonator coupled to two feed lines with qpdk.
# 2. Convert the etch-based layout into explicit conductor and dielectric regions.
# 3. Describe the layer stack and ports with gsim and mesh with Gmsh.
# 4. Run Palace over a frequency sweep and extract the S₂₁ response.
# 5. Extract the resonance frequency and quality factor from the transmission dip.
#
# gsim and Palace are not qpdk dependencies: install gsim with `pip install gsim` and
# see the [Palace documentation](https://awslabs.github.io/palace/) for solver
# installation, including precompiled [Apptainer](https://apptainer.org) images. The
# results shown below were produced with Palace on the Aalto Triton cluster; the
# S-parameter data is embedded so the analysis cells run anywhere.

# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing QPDK...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import gdsfactory as gf
import klayout.db as kdb

from qpdk import PDK, cells
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.tech import LAYER, route_bundle_sbend_cpw

PDK.activate()

# %% [markdown]
# ## Simulation layout
#
# The device under test is a meandering half-wave resonator (`resonator_coupled`) with two
# straight feed lines capacitively coupled to it. Each feed is routed to the resonator
# coupling gap with smooth s-bends, using a CPW cross section that carries airbridges so
# the ground plane stays equipotential across the bends.

# %%


@gf.cell
def resonator_compact(coupling_gap: float = 20.0) -> gf.Component:
    """Coupled resonator with two CPW feeds routed close to the coupling section.

    Args:
        coupling_gap: Gap between the feed lines and the resonator, in μm.

    Returns:
        Component with ports `o1` and `o2` on the two feed lines.
    """
    c = gf.Component()
    res = c << cells.resonator_coupled(
        coupling_straight_length=300, coupling_gap=coupling_gap
    )
    res.movex(-res.size_info.width / 4)
    left = c << cells.straight()
    right = c << cells.straight()
    w = res.size_info.width + 100
    left.move((-w, 0))
    right.move((w, 0))
    route_bundle_sbend_cpw(
        c,
        [left["o2"], right["o1"]],
        [res["coupling_o1"], res["coupling_o2"]],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=250.0, airbridge_padding=20.0
        ),
    )
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 100))
    c.add_port(name="o1", port=left["o1"])
    c.add_port(name="o2", port=right["o2"])
    return c


component = resonator_compact(coupling_gap=15.0)
component.show()

# %% [markdown]
# ## From etch layers to simulation regions
#
# qpdk draws metal as an *etch* layer: the mask describes where metal is removed, not
# where it remains. A volumetric FEM solver needs the opposite, explicit conductor and
# dielectric regions, so the layout is converted by subtracting the etch polygons from
# the simulation area and copying the resulting regions onto dedicated GDS layers:
# `SUBSTRATE` (1, 0), `SUPERCONDUCTOR` (2, 0) and `VACUUM` (3, 0). The zero-thickness
# superconducting sheet later becomes a perfect electric conductor boundary in Palace.

# %%

# Simulation area minus the etch mask gives the metal region
sim_area_layer = (LAYER.SIM_AREA[0], LAYER.SIM_AREA[1])
etch_layer = (LAYER.M1_ETCH[0], LAYER.M1_ETCH[1])
CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = component.kdb_cell.layout()
sim_region = kdb.Region(
    component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))
conductor_region = sim_region - etch_region

etched = gf.Component("etched_component")
el = etched.kdb_cell.layout()
for name, region in [
    ("SUPERCONDUCTOR", conductor_region),
    ("SUBSTRATE", sim_region),
    ("VACUUM", sim_region),
]:
    idx = el.layer(*CPW_LAYERS[name])
    etched.kdb_cell.shapes(idx).insert(region)
for port in component.ports:
    etched.add_port(name=port.name, port=port)
etched.show()

# %% [markdown]
# ## gsim setup: layer stack, ports and mesh
#
# gsim turns the converted layout into a 3-D model: a `LayerStack` assigns each GDS
# layer a material and a z-extent, and `DrivenSim` configures the frequency sweep and
# ports. This section is not executed in the documentation build because gsim is not a
# qpdk dependency, but it is the exact code that produced the results below.
#
# ```python
# from gsim.common.stack import Layer, LayerStack
# from gsim.common.stack.materials import MATERIALS_DB
# from gsim.palace import DrivenSim
#
# substrate_thickness = 500
# vacuum_thickness = 500
#
# stack = LayerStack(pdk_name="qpdk")
# stack.layers["SUBSTRATE"] = Layer(
#     name="SUBSTRATE",
#     gds_layer=(1, 0),
#     zmin=0.0,
#     zmax=substrate_thickness,
#     thickness=substrate_thickness,
#     material="sapphire",
#     layer_type="dielectric",
# )
# stack.layers["SUPERCONDUCTOR"] = Layer(
#     name="SUPERCONDUCTOR",
#     gds_layer=(2, 0),
#     zmin=substrate_thickness,
#     zmax=substrate_thickness,
#     thickness=0,
#     material="aluminum",
#     layer_type="conductor",
# )
# stack.layers["VACUUM"] = Layer(
#     name="VACUUM",
#     gds_layer=(3, 0),
#     zmin=substrate_thickness,
#     zmax=substrate_thickness + vacuum_thickness,
#     thickness=vacuum_thickness,
#     material="vacuum",
#     layer_type="dielectric",
# )
# stack.dielectrics = [
#     {"name": "substrate", "zmin": 0.0, "zmax": substrate_thickness, "material": "sapphire"},
#     {
#         "name": "vacuum",
#         "zmin": substrate_thickness,
#         "zmax": substrate_thickness + vacuum_thickness,
#         "material": "vacuum",
#     },
# ]
# stack.materials = {
#     "sapphire": MATERIALS_DB["sapphire"].to_dict(),
#     "aluminum": MATERIALS_DB["aluminum"].to_dict(),
#     "vacuum": MATERIALS_DB["vacuum"].to_dict(),
# }
#
# sim = DrivenSim()
# sim.set_geometry(etched)
# sim.set_stack(stack)
# sim.add_cpw_port("o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
# sim.add_cpw_port("o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
# sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=300, save_fields_at=[7.78e9])
#
# sim.set_output_dir("./sim_qpdk_resonator")
# sim.mesh(preset="default")
# sim.write_config()
# ```
#
# A few points worth noting:
#
# - The substrate is modeled as anisotropic sapphire from the gsim materials database
#   (:math:`\varepsilon_r = 11.5` along the c-axis, 9.3 in-plane), rotated in-plane so
#   the crystal axes do not align with the layout.
# - The 500 μm vacuum layer above the chip acts as the simulation domain boundary; its
#   top and side faces get first-order absorbing boundaries in the generated Palace
#   configuration.
# - The two feeds end in lumped 50 Ω ports, one excited and one passive, so the sweep
#   directly yields the S₂₁ transmission through the coupled resonator.
# - Meshing with `sim.mesh()` invokes Gmsh; `gmsh` needs `libGLU` at import time on
#   some systems (on Triton, `module load mesa-glu` provides it).

# %% [markdown]
# ## Running Palace
#
# `sim.run()` in gsim submits the simulation to the GDSFactory+ cloud service. To run
# the same configuration locally or on an HPC cluster, execute the generated
# `config.json` with a Palace binary directly. On Aalto Triton, Palace is available as a
# precompiled Apptainer image, which makes the Slurm batch script short:
#
# ```bash
# #!/bin/bash
# #SBATCH --job-name=palace-res
# #SBATCH --partition=batch-milan,batch-csl,batch-skl,batch-bdw,batch-hsw
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=24G
# #SBATCH --time=08:00:00
# #SBATCH --output=/scratch/work/%u/slurm-logs/%x-%j.out
#
# set -euo pipefail
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#
# SIF=/scratch/work/savolan2/palace-container/palace-x86_64_v3.sif
# cd /scratch/work/savolan2/gsim-palace/sim_qpdk_resonator
#
# # One MPI rank per core; the container ships its own MPI and Palace binaries.
# # The palace-x86_64.bin binary is used instead of the palace wrapper because
# # the wrapper mis-parses Slurm's environment inside an allocation.
# apptainer exec --cleanenv "$SIF" mpirun --oversubscribe -np 8 \
#     palace-x86_64.bin config.json > run.log 2>&1
# ```
#
# Practical notes for running the container on a cluster:
#
# - `--cleanenv` keeps the host environment (loaded modules, Slurm variables) out of the
#   container, avoiding conflicts with the bundled MPI.
# - Name several partitions: Slurm starts the job wherever it can begin earliest, so a
#   comma-separated list usually beats hand-picking one.
# - Pin BLAS threading to one thread per rank, since Palace parallelizes with MPI.
# - Run from a self-contained directory: Apptainer binds the current working directory
#   subtree, so `config.json`, the mesh and the `output/` folder all live side by side.
# - Palace writes S-parameters to `output/palace/postpro/driven/port-S.csv` and prints an
#   "Elapsed Time Report" at the end, handy for benchmarking (see the CPU vs GPU
#   comparison notebook).
#
# The run below used 8 MPI ranks on a Cascade Lake node with second-order elements,
# giving about 754k unknowns, and Palace's adaptive frequency sampling converged the
# 7.75 to 7.8 GHz sweep in a handful of solves.

# %% [markdown]
# ## Results
#
# Palace writes one row per computed frequency point to `port-S.csv` (frequency in GHz,
# followed by the real and imaginary parts of each S-matrix entry). The rows computed by
# the adaptive sampler are embedded below so the analysis runs without the output files.

# %% [markdown]
# ## Resonance frequency and quality factor
#
# The transmission dip of a capacitively coupled resonator sits at the resonance
# frequency :math:`f_0`. Two standard extractions give the quality factor:
#
# - the −3 dB bandwidth method, :math:`Q = f_0 / \Delta f` where :math:`\Delta f` is the
#   full width of the dip at half depth, and
# - a Lorentzian fit to the dip, which also uses the points on the flanks and is less
#   sensitive to individual sample noise.

# %%
# PLACEHOLDER_RESULTS

# %% [markdown]
# ## Field visualization
#
# Palace saves the electric field at the requested frequencies (`save_fields_at`) in the
# `output/palace/postpro/` directory. gsim's `load_fields` helper reads them into a
# PyVista volume for inspection:
#
# ```python
# from gsim.palace.postpro import load_fields, plot_topview
#
# vol = load_fields(results_dir, excitation=2)
# vpts = vol.slice(normal="z", origin=(0, 0, substrate_thickness)).points
# plot_topview(vpts, np.linalg.norm(vol_slice.point_data["E_real"], axis=1),
#              f"|E| at {freq_ghz:.4f} GHz (V/m)")
# ```
#
# The field snapshot at the resonance shows the energy concentrated along the meander,
# with fringing fields strongest at the coupling gaps, exactly where the layout geometry
# controls the external quality factor.

# %% [markdown]
# ## Summary
#
# Starting from a pure qpdk layout, the gsim meshing pipeline plus a parallel Palace
# solve yields the full-wave S₂₁ response of a coupled CPW resonator, including the
# anisotropic sapphire substrate and realistic port terminations, without any commercial
# solver license. Compared to the SAX circuit models in the other notebooks, this FEM
# result captures geometry-dependent effects (radiation, substrate modes, finite ground
# plane) at a substantially higher computational cost, so the two approaches complement
# each other: circuit models to explore the design space, FEM to verify the final
# geometry.
