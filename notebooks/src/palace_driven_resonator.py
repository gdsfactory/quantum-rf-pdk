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
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
# - Palace writes S-parameters to `output/palace/port-S.csv` and prints an "Elapsed Time
#   Report" at the end, handy for benchmarking (see the CPU vs GPU comparison notebook).
#
# The run below used 8 MPI ranks on a Cascade Lake node with second-order elements,
# giving about 754k unknowns. Palace's adaptive frequency sampling converged the
# 7.75 to 7.8 GHz sweep with 14 sampled points, and the whole job, including meshing
# and the PROM (projection-based reduced-order model) evaluation of all 300 sweep
# points, took about 8 minutes of wall time.

# %% [markdown]
# ## Results
#
# Palace writes `port-S.csv` with one row per frequency point: the frequency in GHz
# followed by the magnitude (dB) and phase (degrees) of each S-matrix entry. The
# adaptive sampler converged after 14 solve samples; the reduced-order model then
# evaluates all 300 requested sweep points from those samples at negligible cost. The
# :math:`|S_{11}|` and :math:`|S_{21}|` columns of the run are embedded below so the
# analysis runs without the Palace output files.

# %%
freq_ghz = np.array([
    7.75000000,
    7.75016722,
    7.75033445,
    7.75050167,
    7.75066890,
    7.75083612,
    7.75100334,
    7.75117057,
    7.75133779,
    7.75150502,
    7.75167224,
    7.75183946,
    7.75200669,
    7.75217391,
    7.75234114,
    7.75250836,
    7.75267559,
    7.75284281,
    7.75301003,
    7.75317726,
    7.75334448,
    7.75351171,
    7.75367893,
    7.75384615,
    7.75401338,
    7.75418060,
    7.75434783,
    7.75451505,
    7.75468227,
    7.75484950,
    7.75501672,
    7.75518395,
    7.75535117,
    7.75551839,
    7.75568562,
    7.75585284,
    7.75602007,
    7.75618729,
    7.75635452,
    7.75652174,
    7.75668896,
    7.75685619,
    7.75702341,
    7.75719064,
    7.75735786,
    7.75752508,
    7.75769231,
    7.75785953,
    7.75802676,
    7.75819398,
    7.75836120,
    7.75852843,
    7.75869565,
    7.75886288,
    7.75903010,
    7.75919732,
    7.75936455,
    7.75953177,
    7.75969900,
    7.75986622,
    7.76003344,
    7.76020067,
    7.76036789,
    7.76053512,
    7.76070234,
    7.76086957,
    7.76103679,
    7.76120401,
    7.76137124,
    7.76153846,
    7.76170569,
    7.76187291,
    7.76204013,
    7.76220736,
    7.76237458,
    7.76254181,
    7.76270903,
    7.76287625,
    7.76304348,
    7.76321070,
    7.76337793,
    7.76354515,
    7.76371237,
    7.76387960,
    7.76404682,
    7.76421405,
    7.76438127,
    7.76454849,
    7.76471572,
    7.76488294,
    7.76505017,
    7.76521739,
    7.76538462,
    7.76555184,
    7.76571906,
    7.76588629,
    7.76605351,
    7.76622074,
    7.76638796,
    7.76655518,
    7.76672241,
    7.76688963,
    7.76705686,
    7.76722408,
    7.76739130,
    7.76755853,
    7.76772575,
    7.76789298,
    7.76806020,
    7.76822742,
    7.76839465,
    7.76856187,
    7.76872910,
    7.76889632,
    7.76906355,
    7.76923077,
    7.76939799,
    7.76956522,
    7.76973244,
    7.76989967,
    7.77006689,
    7.77023411,
    7.77040134,
    7.77056856,
    7.77073579,
    7.77090301,
    7.77107023,
    7.77123746,
    7.77140468,
    7.77157191,
    7.77173913,
    7.77190635,
    7.77207358,
    7.77224080,
    7.77240803,
    7.77257525,
    7.77274247,
    7.77290970,
    7.77307692,
    7.77324415,
    7.77341137,
    7.77357860,
    7.77374582,
    7.77391304,
    7.77408027,
    7.77424749,
    7.77441472,
    7.77458194,
    7.77474916,
    7.77491639,
    7.77508361,
    7.77525084,
    7.77541806,
    7.77558528,
    7.77575251,
    7.77591973,
    7.77608696,
    7.77625418,
    7.77642140,
    7.77658863,
    7.77675585,
    7.77692308,
    7.77709030,
    7.77725753,
    7.77742475,
    7.77759197,
    7.77775920,
    7.77792642,
    7.77809365,
    7.77826087,
    7.77842809,
    7.77859532,
    7.77876254,
    7.77892977,
    7.77909699,
    7.77926421,
    7.77943144,
    7.77959866,
    7.77976589,
    7.77993311,
    7.78010033,
    7.78026756,
    7.78043478,
    7.78060201,
    7.78076923,
    7.78093645,
    7.78110368,
    7.78127090,
    7.78143813,
    7.78160535,
    7.78177258,
    7.78193980,
    7.78210702,
    7.78227425,
    7.78244147,
    7.78260870,
    7.78277592,
    7.78294314,
    7.78311037,
    7.78327759,
    7.78344482,
    7.78361204,
    7.78377926,
    7.78394649,
    7.78411371,
    7.78428094,
    7.78444816,
    7.78461538,
    7.78478261,
    7.78494983,
    7.78511706,
    7.78528428,
    7.78545151,
    7.78561873,
    7.78578595,
    7.78595318,
    7.78612040,
    7.78628763,
    7.78645485,
    7.78662207,
    7.78678930,
    7.78695652,
    7.78712375,
    7.78729097,
    7.78745819,
    7.78762542,
    7.78779264,
    7.78795987,
    7.78812709,
    7.78829431,
    7.78846154,
    7.78862876,
    7.78879599,
    7.78896321,
    7.78913043,
    7.78929766,
    7.78946488,
    7.78963211,
    7.78979933,
    7.78996656,
    7.79013378,
    7.79030100,
    7.79046823,
    7.79063545,
    7.79080268,
    7.79096990,
    7.79113712,
    7.79130435,
    7.79147157,
    7.79163880,
    7.79180602,
    7.79197324,
    7.79214047,
    7.79230769,
    7.79247492,
    7.79264214,
    7.79280936,
    7.79297659,
    7.79314381,
    7.79331104,
    7.79347826,
    7.79364548,
    7.79381271,
    7.79397993,
    7.79414716,
    7.79431438,
    7.79448161,
    7.79464883,
    7.79481605,
    7.79498328,
    7.79515050,
    7.79531773,
    7.79548495,
    7.79565217,
    7.79581940,
    7.79598662,
    7.79615385,
    7.79632107,
    7.79648829,
    7.79665552,
    7.79682274,
    7.79698997,
    7.79715719,
    7.79732441,
    7.79749164,
    7.79765886,
    7.79782609,
    7.79799331,
    7.79816054,
    7.79832776,
    7.79849498,
    7.79866221,
    7.79882943,
    7.79899666,
    7.79916388,
    7.79933110,
    7.79949833,
    7.79966555,
    7.79983278,
    7.80000000,
])

s11_db = np.array([
    -7.697517,
    -7.696736,
    -7.695937,
    -7.695119,
    -7.694283,
    -7.693427,
    -7.692551,
    -7.691654,
    -7.690735,
    -7.689794,
    -7.688830,
    -7.687843,
    -7.686831,
    -7.685794,
    -7.684730,
    -7.683640,
    -7.682521,
    -7.681374,
    -7.680196,
    -7.678988,
    -7.677748,
    -7.676474,
    -7.675166,
    -7.673822,
    -7.672442,
    -7.671023,
    -7.669564,
    -7.668064,
    -7.666521,
    -7.664933,
    -7.663300,
    -7.661618,
    -7.659886,
    -7.658103,
    -7.656265,
    -7.654371,
    -7.652419,
    -7.650406,
    -7.648329,
    -7.646186,
    -7.643974,
    -7.641690,
    -7.639331,
    -7.636894,
    -7.634374,
    -7.631770,
    -7.629076,
    -7.626288,
    -7.623403,
    -7.620416,
    -7.617321,
    -7.614115,
    -7.610791,
    -7.607345,
    -7.603769,
    -7.600058,
    -7.596206,
    -7.592204,
    -7.588047,
    -7.583725,
    -7.579232,
    -7.574557,
    -7.569692,
    -7.564627,
    -7.559352,
    -7.553857,
    -7.548129,
    -7.542156,
    -7.535927,
    -7.529428,
    -7.522646,
    -7.515565,
    -7.508171,
    -7.500448,
    -7.492381,
    -7.483952,
    -7.475144,
    -7.465941,
    -7.456324,
    -7.446277,
    -7.435781,
    -7.424822,
    -7.413382,
    -7.401449,
    -7.389009,
    -7.376052,
    -7.362574,
    -7.348571,
    -7.334048,
    -7.319015,
    -7.303491,
    -7.287505,
    -7.271096,
    -7.254321,
    -7.237248,
    -7.219966,
    -7.202585,
    -7.185236,
    -7.168072,
    -7.151274,
    -7.135046,
    -7.119614,
    -7.105228,
    -7.092151,
    -7.080658,
    -7.071024,
    -7.063516,
    -7.058381,
    -7.055834,
    -7.056044,
    -7.059126,
    -7.065131,
    -7.074040,
    -7.085765,
    -7.100148,
    -7.116973,
    -7.135973,
    -7.156841,
    -7.179248,
    -7.202857,
    -7.227330,
    -7.252347,
    -7.277611,
    -7.302853,
    -7.327841,
    -7.352376,
    -7.376295,
    -7.399469,
    -7.421800,
    -7.443216,
    -7.463674,
    -7.483146,
    -7.501626,
    -7.519118,
    -7.535640,
    -7.551218,
    -7.565884,
    -7.579673,
    -7.592626,
    -7.604783,
    -7.616188,
    -7.626880,
    -7.636903,
    -7.646295,
    -7.655097,
    -7.663345,
    -7.671075,
    -7.678320,
    -7.685113,
    -7.691483,
    -7.697458,
    -7.703065,
    -7.708329,
    -7.713273,
    -7.717917,
    -7.722282,
    -7.726386,
    -7.730248,
    -7.733882,
    -7.737304,
    -7.740528,
    -7.743567,
    -7.746432,
    -7.749136,
    -7.751687,
    -7.754096,
    -7.756373,
    -7.758524,
    -7.760558,
    -7.762483,
    -7.764304,
    -7.766029,
    -7.767662,
    -7.769210,
    -7.770678,
    -7.772070,
    -7.773390,
    -7.774644,
    -7.775834,
    -7.776965,
    -7.778039,
    -7.779060,
    -7.780031,
    -7.780955,
    -7.781834,
    -7.782670,
    -7.783467,
    -7.784225,
    -7.784948,
    -7.785637,
    -7.786294,
    -7.786920,
    -7.787517,
    -7.788087,
    -7.788631,
    -7.789150,
    -7.789646,
    -7.790120,
    -7.790572,
    -7.791004,
    -7.791417,
    -7.791812,
    -7.792190,
    -7.792551,
    -7.792897,
    -7.793227,
    -7.793544,
    -7.793846,
    -7.794136,
    -7.794414,
    -7.794680,
    -7.794934,
    -7.795178,
    -7.795412,
    -7.795636,
    -7.795851,
    -7.796056,
    -7.796254,
    -7.796443,
    -7.796625,
    -7.796799,
    -7.796966,
    -7.797126,
    -7.797280,
    -7.797428,
    -7.797569,
    -7.797705,
    -7.797836,
    -7.797962,
    -7.798082,
    -7.798198,
    -7.798309,
    -7.798416,
    -7.798519,
    -7.798617,
    -7.798712,
    -7.798803,
    -7.798891,
    -7.798975,
    -7.799056,
    -7.799134,
    -7.799209,
    -7.799282,
    -7.799351,
    -7.799418,
    -7.799482,
    -7.799544,
    -7.799604,
    -7.799661,
    -7.799717,
    -7.799770,
    -7.799822,
    -7.799871,
    -7.799919,
    -7.799965,
    -7.800009,
    -7.800052,
    -7.800093,
    -7.800133,
    -7.800172,
    -7.800209,
    -7.800245,
    -7.800280,
    -7.800313,
    -7.800345,
    -7.800377,
    -7.800407,
    -7.800436,
    -7.800465,
    -7.800492,
    -7.800519,
    -7.800545,
    -7.800570,
    -7.800594,
    -7.800617,
    -7.800640,
    -7.800663,
    -7.800684,
    -7.800705,
    -7.800725,
    -7.800745,
    -7.800765,
    -7.800783,
    -7.800802,
    -7.800820,
    -7.800837,
    -7.800854,
    -7.800871,
    -7.800887,
    -7.800903,
    -7.800919,
    -7.800934,
    -7.800949,
    -7.800964,
    -7.800979,
    -7.800993,
    -7.801007,
    -7.801021,
    -7.801034,
    -7.801048,
])

s21_db = np.array([
    -5.882616,
    -5.882738,
    -5.882865,
    -5.882998,
    -5.883138,
    -5.883284,
    -5.883437,
    -5.883598,
    -5.883765,
    -5.883941,
    -5.884125,
    -5.884317,
    -5.884519,
    -5.884729,
    -5.884950,
    -5.885181,
    -5.885422,
    -5.885675,
    -5.885940,
    -5.886218,
    -5.886508,
    -5.886812,
    -5.887130,
    -5.887464,
    -5.887813,
    -5.888179,
    -5.888562,
    -5.888964,
    -5.889385,
    -5.889827,
    -5.890290,
    -5.890775,
    -5.891284,
    -5.891818,
    -5.892379,
    -5.892967,
    -5.893584,
    -5.894233,
    -5.894914,
    -5.895630,
    -5.896382,
    -5.897173,
    -5.898005,
    -5.898879,
    -5.899800,
    -5.900769,
    -5.901790,
    -5.902865,
    -5.903998,
    -5.905192,
    -5.906451,
    -5.907780,
    -5.909183,
    -5.910664,
    -5.912228,
    -5.913881,
    -5.915630,
    -5.917479,
    -5.919436,
    -5.921509,
    -5.923705,
    -5.926032,
    -5.928500,
    -5.931118,
    -5.933898,
    -5.936851,
    -5.939989,
    -5.943326,
    -5.946876,
    -5.950655,
    -5.954681,
    -5.958970,
    -5.963544,
    -5.968424,
    -5.973633,
    -5.979195,
    -5.985139,
    -5.991492,
    -5.998286,
    -6.005555,
    -6.013335,
    -6.021662,
    -6.030580,
    -6.040129,
    -6.050357,
    -6.061309,
    -6.073036,
    -6.085587,
    -6.099014,
    -6.113367,
    -6.128696,
    -6.145047,
    -6.162460,
    -6.180970,
    -6.200599,
    -6.221358,
    -6.243237,
    -6.266207,
    -6.290206,
    -6.315146,
    -6.340896,
    -6.367284,
    -6.394092,
    -6.421053,
    -6.447854,
    -6.474135,
    -6.499503,
    -6.523536,
    -6.545805,
    -6.565890,
    -6.583400,
    -6.597995,
    -6.609404,
    -6.617444,
    -6.622024,
    -6.623153,
    -6.620935,
    -6.615556,
    -6.607274,
    -6.596399,
    -6.583274,
    -6.568258,
    -6.551707,
    -6.533968,
    -6.515359,
    -6.496174,
    -6.476666,
    -6.457059,
    -6.437538,
    -6.418257,
    -6.399339,
    -6.380879,
    -6.362952,
    -6.345609,
    -6.328886,
    -6.312804,
    -6.297373,
    -6.282594,
    -6.268460,
    -6.254959,
    -6.242075,
    -6.229789,
    -6.218078,
    -6.206921,
    -6.196293,
    -6.186170,
    -6.176529,
    -6.167346,
    -6.158597,
    -6.150261,
    -6.142316,
    -6.134741,
    -6.127517,
    -6.120624,
    -6.114045,
    -6.107762,
    -6.101760,
    -6.096024,
    -6.090539,
    -6.085291,
    -6.080268,
    -6.075458,
    -6.070850,
    -6.066432,
    -6.062196,
    -6.058131,
    -6.054229,
    -6.050481,
    -6.046880,
    -6.043418,
    -6.040088,
    -6.036883,
    -6.033798,
    -6.030827,
    -6.027964,
    -6.025204,
    -6.022542,
    -6.019974,
    -6.017495,
    -6.015102,
    -6.012789,
    -6.010554,
    -6.008393,
    -6.006304,
    -6.004282,
    -6.002325,
    -6.000430,
    -5.998594,
    -5.996815,
    -5.995091,
    -5.993420,
    -5.991798,
    -5.990225,
    -5.988698,
    -5.987215,
    -5.985775,
    -5.984376,
    -5.983017,
    -5.981695,
    -5.980410,
    -5.979161,
    -5.977945,
    -5.976761,
    -5.975609,
    -5.974487,
    -5.973395,
    -5.972330,
    -5.971293,
    -5.970282,
    -5.969297,
    -5.968336,
    -5.967398,
    -5.966483,
    -5.965591,
    -5.964720,
    -5.963869,
    -5.963039,
    -5.962227,
    -5.961435,
    -5.960661,
    -5.959904,
    -5.959164,
    -5.958441,
    -5.957734,
    -5.957042,
    -5.956366,
    -5.955704,
    -5.955056,
    -5.954422,
    -5.953801,
    -5.953193,
    -5.952598,
    -5.952015,
    -5.951444,
    -5.950885,
    -5.950337,
    -5.949800,
    -5.949273,
    -5.948757,
    -5.948251,
    -5.947754,
    -5.947268,
    -5.946790,
    -5.946322,
    -5.945862,
    -5.945411,
    -5.944968,
    -5.944534,
    -5.944107,
    -5.943688,
    -5.943277,
    -5.942873,
    -5.942476,
    -5.942087,
    -5.941704,
    -5.941328,
    -5.940958,
    -5.940595,
    -5.940238,
    -5.939887,
    -5.939541,
    -5.939202,
    -5.938869,
    -5.938541,
    -5.938218,
    -5.937901,
    -5.937588,
    -5.937281,
    -5.936979,
    -5.936682,
    -5.936389,
    -5.936101,
    -5.935818,
    -5.935539,
    -5.935264,
    -5.934993,
    -5.934727,
    -5.934465,
    -5.934207,
    -5.933952,
    -5.933702,
    -5.933455,
    -5.933212,
    -5.932973,
    -5.932737,
    -5.932504,
    -5.932275,
    -5.932049,
    -5.931827,
    -5.931607,
    -5.931391,
    -5.931178,
    -5.930968,
    -5.930761,
    -5.930557,
    -5.930355,
    -5.930156,
    -5.929961,
    -5.929767,
    -5.929577,
])

plt.figure(figsize=(10, 6))
plt.plot(freq_ghz, s11_db, label="|S11|")
plt.plot(freq_ghz, s21_db, label="|S21|")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.legend()
plt.show()

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
# Baseline transmission away from the dip, taken as the median of the sweep edges
baseline = np.median(np.concatenate([s21_db[:20], s21_db[-20:]]))
i_min = int(np.argmin(s21_db))
f_dip = freq_ghz[i_min]
depth = baseline - s21_db[i_min]
print(f"Baseline |S21|: {baseline:.4f} dB")
print(f"Dip: {f_dip:.6f} GHz, depth {depth:.4f} dB below baseline")

# Half-depth crossings, linearly interpolated between sweep points
half = baseline - depth / 2.0
i0 = np.where(s21_db[:i_min] >= half)[0][-1]
i1 = i_min + np.where(s21_db[i_min:] >= half)[0][0]
f_lo = np.interp(half, s21_db[i0 : i0 + 2][::-1], freq_ghz[i0 : i0 + 2][::-1])
f_hi = np.interp(half, s21_db[i1 - 1 : i1 + 1], freq_ghz[i1 - 1 : i1 + 1])
q_3db = f_dip / (f_hi - f_lo)
print(f"Half-depth crossings: {f_lo:.6f} / {f_hi:.6f} GHz")
print(f"Q (3 dB bandwidth): {q_3db:.1f}")


def lorentzian_dip(f, baseline, depth, f0, q):
    """Lorentzian transmission dip of a resonator with quality factor q at f0."""
    return baseline - depth / (1.0 + (2.0 * q * (f - f0) / f0) ** 2)


mask = np.abs(freq_ghz - f_dip) <= 0.01
popt, _ = curve_fit(
    lorentzian_dip, freq_ghz[mask], s21_db[mask], p0=[baseline, depth, f_dip, 1000.0]
)
fit_baseline, fit_depth, fit_f0, fit_q = popt
print(f"Lorentzian fit: f0 = {fit_f0:.6f} GHz, Q = {fit_q:.1f}")

# %% [markdown]
# The dip is shallow, about 0.7 dB on a -5.9 dB baseline, because the 15 μm coupling gap
# only weakly couples the feeds to the resonator. The same gap causes the reflection to
# stay below -7 dB everywhere, with |S11| peaking near the dip instead of dropping, a sign
# of mismatch between the lumped port terminations and the feed lines rather than of
# resonant behavior. Both quality-factor extractions nevertheless agree to within about
# 8% (:math:`Q_{3\mathrm{dB}} \approx 1126` against :math:`Q_{\mathrm{fit}} \approx 1045`),
# consistent with a low-Q, externally dominated resonance.

# %% [markdown]
# ## Field visualization
#
# Palace saves the electric field at the requested frequencies (`save_fields_at`) in the
# `output/palace/paraview/` directory. gsim's `load_fields` helper reads them into a
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
