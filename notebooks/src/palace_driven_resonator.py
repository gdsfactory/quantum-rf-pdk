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
# gsim and Palace are not qpdk dependencies: install gsim with
# `pip install "gsim @ git+https://github.com/gdsfactory/gsim.git"` (requires Python
# 3.12; the PyPI release is outdated) and `pip install palace-toolkit`, whose
# `install_palace_runtime()` helper downloads a precompiled CPU Palace binary. This
# notebook runs Palace live with that runtime, the same path the repository's CI smoke
# test uses; see the [Palace documentation](https://awslabs.github.io/palace/) for other
# installation options, including precompiled [Apptainer](https://apptainer.org)
# images. The simulation cells need a real Palace run and are not executed in the
# documentation build; everything from the layout onward runs end to end locally.

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
        "scipy",
        "gsim @ git+https://github.com/gdsfactory/gsim.git",
        "palace-toolkit",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import subprocess
import sys
from pathlib import Path

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
# the ground plane stays equipotential across the bends. The airbridge metal itself is
# left out of the FEM geometry below; only the planar M1 conductor layer is simulated.

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
# ports. These cells are not executed in the documentation build because gsim is not a
# qpdk dependency, but they run as-is in a local Python 3.12 environment.

# %%
from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace import DrivenSim

substrate_thickness = 500
vacuum_thickness = 500

stack = LayerStack(pdk_name="qpdk")
stack.layers["SUBSTRATE"] = Layer(
    name="SUBSTRATE",
    gds_layer=CPW_LAYERS["SUBSTRATE"],
    zmin=0.0,
    zmax=substrate_thickness,
    thickness=substrate_thickness,
    material="sapphire",
    layer_type="dielectric",
)
stack.layers["SUPERCONDUCTOR"] = Layer(
    name="SUPERCONDUCTOR",
    gds_layer=CPW_LAYERS["SUPERCONDUCTOR"],
    zmin=substrate_thickness,
    zmax=substrate_thickness,
    thickness=0,
    material="aluminum",
    layer_type="conductor",
)
stack.layers["VACUUM"] = Layer(
    name="VACUUM",
    gds_layer=CPW_LAYERS["VACUUM"],
    zmin=substrate_thickness,
    zmax=substrate_thickness + vacuum_thickness,
    thickness=vacuum_thickness,
    material="vacuum",
    layer_type="dielectric",
)
stack.dielectrics = [
    {
        "name": "substrate",
        "zmin": 0.0,
        "zmax": substrate_thickness,
        "material": "sapphire",
    },
    {
        "name": "vacuum",
        "zmin": substrate_thickness,
        "zmax": substrate_thickness + vacuum_thickness,
        "material": "vacuum",
    },
]
stack.materials = {
    "sapphire": MATERIALS_DB["sapphire"].to_dict(),
    "aluminum": MATERIALS_DB["aluminum"].to_dict(),
    "vacuum": MATERIALS_DB["vacuum"].to_dict(),
}

sim = DrivenSim()
sim.set_geometry(etched)
sim.set_stack(stack)
sim.add_cpw_port("o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.add_cpw_port("o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=300, save_fields_at=[7.78e9])

# %% [markdown]
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

# %%
out_dir = Path("sim_qpdk_resonator")
sim.set_output_dir(str(out_dir))

# Meshing invokes Gmsh; `gmsh` needs `libGLU` at import time on some systems
# (`libglu1-mesa` on Debian/Ubuntu provides it).
sim.mesh(preset="default")
sim.write_config()

# %% [markdown]
# ## Running Palace
#
# `sim.run()` in gsim submits the simulation to the GDSFactory+ cloud service. To run
# the same configuration locally, execute the generated `config.json` with a Palace
# binary directly.
#
# The easiest route on a laptop is the precompiled CPU runtime that ships with
# `palace-toolkit`: `install_palace_runtime()` returns a self-contained Palace binary
# (Linux x86_64, glibc 2.38 or newer), and `get_palace_runtime_env()` the environment it
# needs. The `--serial` flag runs the singleton (one-rank) build, the same path the
# repository's CI smoke test uses. This cell runs Palace for real, so the sweep results
# below come straight from the solver rather than being hardcoded. On the default mesh
# the serial run takes about 40 minutes on a single core and peaks at roughly 5 GB of
# memory. The `coarse` mesh preset saves only about a third of that time, and the
# accuracy cost is not small: it shifts the resonance down by about 55 MHz, moving the
# dip outside the 7.75-7.8 GHz sweep entirely, so the default preset is the right
# choice here.

# %%
from palacetoolkit.palace_runtime import install_palace_runtime
from palacetoolkit.simulation import get_palace_runtime_env

palace = install_palace_runtime()
result = subprocess.run(
    [str(palace), "--serial", "config.json"],
    cwd=out_dir,
    env=get_palace_runtime_env(palace),
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(result.stdout[-4000:])
    print(result.stderr[-4000:], file=sys.stderr)
    raise RuntimeError(f"Palace exited with code {result.returncode}")
(out_dir / "run.log").write_text(result.stdout)

# %% [markdown]
# For larger runs, the precompiled [Apptainer](https://apptainer.org) images bundle
# Palace with its own MPI and need no system installation. From the directory holding
# `config.json` and the mesh:
#
# ```bash
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#
# # One MPI rank per core; use the raw palace-x86_64.bin binary instead of the
# # palace wrapper if the wrapper mis-parses the host environment in a batch
# # allocation.
# apptainer exec --cleanenv palace.sif mpirun -np 8 \
#     palace-x86_64.bin config.json > run.log 2>&1
# ```
#
# Practical notes for running the container:
#
# - `--cleanenv` keeps the host environment out of the container, avoiding conflicts
#   with the bundled MPI.
# - Pin BLAS threading to one thread per rank, since Palace parallelizes with MPI.
# - Run from a self-contained directory: Apptainer binds the current working directory
#   subtree, so `config.json`, the mesh and the `output/` folder all live side by side.
# - Palace writes S-parameters to `output/palace/port-S.csv` and prints an "Elapsed Time
#   Report" at the end, handy for benchmarking (see the CPU vs GPU comparison notebook).
#
# The same invocation works unchanged in an HPC environment such as a Slurm cluster,
# where a batch script requests the cores and then runs the `mpirun` line inside the
# allocation. The benchmark notebook records one such MPI run on 8 CPU ranks, which
# solved this configuration with 14 adaptive samples in about 8 minutes of wall time,
# alongside a single-GPU run of the same model.

# %% [markdown]
# ## Results
#
# Palace writes `port-S.csv` with one row per frequency point: the frequency in GHz
# followed by the magnitude (dB) and phase (degrees) of each S-matrix entry. The
# adaptive sampler converges after a handful of solve samples; the reduced-order model
# then evaluates all 300 requested sweep points from those samples at negligible cost.
# Column 0 is the frequency, and for the two-port S-matrix the :math:`|S_{11}|` and
# :math:`|S_{21}|` magnitudes in dB are columns 1 and 5.

# %%
port_csv = out_dir / "output" / "palace" / "port-S.csv"
s_data = np.loadtxt(port_csv, delimiter=",", skiprows=1)
freq_ghz = s_data[:, 0]
s11_db = s_data[:, 1]
s21_db = s_data[:, 5]
s_data.shape

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
# 9% (:math:`Q_{3\mathrm{dB}} \approx 1133` against :math:`Q_{\mathrm{fit}} \approx 1034`),
# consistent with a low-Q, externally dominated resonance.

# %% [markdown]
# ## Field visualization
#
# Palace saves the electric field at the requested frequencies (`save_fields_at`) in the
# `output/palace/paraview/` directory. gsim's `load_fields` helper reads them into a
# PyVista volume for inspection, and `plot_topview` slices it at a given height and
# renders a top view of the named field:
#
# ```python
# from gsim.palace.results import load_fields
# from gsim.viz import plot_topview
#
# vol = load_fields("sim_qpdk_resonator", excitation=2)
# plot_topview(vol, field="E_real", z=substrate_thickness,
#              title="|E| at 7.7800 GHz (V/m)")
# ```
#
# The snapshot is taken at 7.78 GHz, within a megahertz of the resonance dip, so the
# fields are effectively those of the resonant mode. They show the energy concentrated
# along the meander, with fringing fields strongest at the coupling gaps, exactly where
# the layout geometry controls the external quality factor.

# %% [markdown]
# ## Summary
#
# Starting from a pure qpdk layout, the gsim meshing pipeline plus a parallel Palace
# solve yields the full-wave S₂₁ response of a coupled CPW resonator, including the
# anisotropic sapphire substrate, without any commercial solver license. The geometry is
# the planar M1 conductor only; the airbridges drawn on their own layers are not part of
# the mesh, so the ground plane is a single etched sheet with no cross-connections.
# Compared to the SAX circuit models in the other notebooks, this FEM result captures
# geometry-dependent effects (radiation, substrate modes) at a substantially higher
# computational cost, so the two approaches complement each other: circuit models to
# explore the design space, FEM to verify the final geometry.
