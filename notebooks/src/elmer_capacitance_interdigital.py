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
# # Elmer Capacitance Extraction of an Interdigital Capacitor
#
# ::::{admonition} Required extras
# :class: tip
#
# This notebook needs the `models` extra and the Elmer driver from `gplugins`:
#
# ```bash
# uv add "qpdk[models]" "gplugins[elmer] @ git+https://github.com/gdsfactory/gplugins.git@4ac6063ebfc01473a2f514b460d6f3ff04f5a17d"
# # or with pip:
# pip install "qpdk[models]" "gplugins[elmer] @ git+https://github.com/gdsfactory/gplugins.git@4ac6063ebfc01473a2f514b460d6f3ff04f5a17d"
# ```
#
# The Elmer driver changes are currently in the companion
# [gplugins pull request](https://github.com/gdsfactory/gplugins/pull/781); the install
# commands pin the revision tested with this notebook until a release includes it.
#
# **Elmer is an external solver.** `ElmerGrid` and `ElmerSolver` must be available on your
# `PATH`; they are not pip-installable. See the
# [Elmer FEM installation guide](https://www.elmerfem.org/blog/binaries/) for binaries and
# container options.
#
# See the {ref}`extras reference <notebook-extras>` for what each qpdk extra installs.
# ::::
#
# This notebook runs a quasi-static electrostatic solve with Elmer FEM to extract the
# capacitance of a QPDK interdigital capacitor. Elmer solves
# $\nabla \cdot (\epsilon \nabla \phi) = 0$ with a fixed potential on each metal
# terminal. Elmer's `.dat` result is the lumped (circuit) capacitance matrix;
# `gplugins` converts it to a Maxwell matrix, with negative off-diagonal entries, in
# `ElectrostaticResults`.
#
# The reported number comes from a mesh-convergence study. The geometry, the layer stack
# and the simulation domain are held fixed, the mesh is refined over four factors, and
# the finest mesh supplies the final value. A separate lateral-pad comparison checks how
# much that value depends on the outer boundary of the finite domain.

# %% [markdown]
# ## Physics
#
# An interdigital capacitor (IDC) is two interleaved combs of metal fingers. Each comb is
# a separate conductor (terminal), and the capacitance we care about is the coupling
# between them.
#
# For $N$ conductors Elmer's `.dat` result is the lumped (circuit) capacitance
# matrix. `gplugins` converts it to the Maxwell form $C$ reported in
# `ElectrostaticResults`, defined by $Q_i = \sum_j C_{ij} V_j$ with negative
# off-diagonal entries:
#
# $$ C_{ij} = -C_{ij}^{\text{mutual}} \quad (i \neq j). $$
#
# So for a two-terminal device the mutual capacitance is
# $C_{12}^{\text{mutual}} = -C_{12}$, and it shows up as a **negative** off-diagonal
# entry. This model has **no grounded conductor**: the diagonal entries are not independent
# capacitances-to-ground, and $C_{11}$ and $C_{22}$ are effectively equal to the
# mutual $|C_{12}|$. All Elmer output is in SI units (farads); we convert to
# femtofarads below.
#
# This is a 3D FEM result only. It is **not benchmarked against an analytic IDC model**,
# and no analytic formula is evaluated here. The check below compares the last two
# mesh results; their difference is not a bound on absolute model error.

# %% [markdown]
# ## Setup and Imports

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
        "gplugins[elmer] @ git+https://github.com/gdsfactory/gplugins.git@4ac6063ebfc01473a2f514b460d6f3ff04f5a17d",
    ])

# %% tags=["hide-input", "hide-output"]
# CI executes this notebook headless with MPLBACKEND=Agg, which would drop the
# convergence figure from the saved output, so pin the backend that renders figures
# inline.
import matplotlib

matplotlib.use("module://matplotlib_inline.backend_inline")

# %% tags=["hide-input", "hide-output"]
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np
from gdsfactory.technology import LayerLevel, LayerStack
from gplugins.elmer import run_capacitive_simulation_elmer
from matplotlib import pyplot as plt
from meshwell.resolution import ConstantInField

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.tech import LAYER, material_properties

PDK.activate()

# %% [markdown]
# ## Simulation Geometry
#
# We use the QPDK `interdigital_capacitor` geometry with
# a small number of fingers to keep the mesh and solve affordable.
#
# Two details matter for a valid capacitance extraction:
#
# 1. **Two isolated terminals.** The capacitor must present two disconnected metal
#    polygons. QPDK's IDC draws its metal on `M1_DRAW` (the additive mask) and an
#    enclosing rectangle on `M1_ETCH` (the subtractive mask). We pass `etch_layer=None` so
#    the component contains only the two `M1_DRAW` combs: nothing can merge them into one
#    conductor.
# 2. **A domain outline for the dielectrics.** The substrate and air prisms are built from
#    an explicit outline, drawn on the non-fabrication `SIM_AREA` layer.
#
# We deliberately do **not** reuse `PDK.layer_stack` here. In QPDK the `M1` level is a
# `DerivedLayer` built from `SIM_AREA`, `M1_ETCH` and `M1_DRAW` (metal is reconstructed
# wherever `SIM_AREA` is not etched). Fed a bare capacitor component that derived layer
# either resolves to nothing (no `SIM_AREA` shape) or rebuilds the whole `SIM_AREA` region
# as one conductor, which would short both terminals into a full-plane `M1`. A minimal
# stack whose metal level *is* `LAYER.M1_DRAW` keeps exactly the two terminal polygons the
# component draws.
#
# The lateral pad is fixed for the whole study at `domain_pad=90.0` μm, which puts the
# outer boundary well away from the finger gaps where the coupling lives. The outer
# substrate and vacuum faces use Elmer's natural zero-normal-flux boundary condition, so
# the finite domain still affects the extracted value; the lateral-pad comparison below
# quantifies that effect separately from mesh convergence.


# %%
@gf.cell
def interdigital_capacitor_for_elmer(
    fingers: int = 4,
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
    domain_pad: float = 90.0,
) -> gf.Component:
    """Two-terminal IDC plus a `SIM_AREA` outline for the dielectric prisms.

    Args:
        fingers: Total number of interleaved fingers.
        finger_length: Length of each finger in μm.
        finger_gap: Gap between adjacent fingers in μm.
        thickness: Finger width in μm.
        domain_pad: Lateral padding of the simulation outline around the metal in μm.

    Returns:
        Component with two isolated `M1_DRAW` terminals (ports ``o1`` and ``o2``) and a
        `SIM_AREA` rectangle marking the substrate/air footprint.
    """
    c = gf.Component()
    idc = c << interdigital_capacitor(
        fingers=fingers,
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        etch_layer=None,  # keep the two combs as separate M1_DRAW polygons
    )
    c.add_ports(idc.ports)

    # Flatten so the terminals are plain polygons in one cell, as the mesher expects.
    c.flatten()
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(domain_pad, domain_pad))
    return c


component = interdigital_capacitor_for_elmer()
component.plot()
print(f"Bounding box: {component.bbox()}")
print(f"Terminals: {[port.name for port in component.ports]}")

# %% [markdown]
# ## Layer Stack and Materials
#
# The stack uses a thin Nb film on a $60\,\mu m$ Si substrate with a $40\,\mu m$
# vacuum prism above it. The vacuum starts at the substrate surface to fill the gaps
# beside the film; meshwell cuts the higher-priority metal out of that prism. The
# substrate is thick enough that the field has room below the fingers, and the vacuum
# prism gives it room above.
#
# Material permittivities come from the QPDK technology definition
# (`qpdk.tech.material_properties`): Si uses $\epsilon_r = 11.45$, and the niobium
# film is treated as a perfect conductor.

# %%
metal_thickness = 0.2  # µm, a 200 nm Nb film
substrate_thickness = 60.0  # µm
air_thickness = 40.0  # µm

layer_stack = LayerStack(
    layers={
        "metal": LayerLevel(
            name="metal",
            layer=LAYER.M1_DRAW,
            thickness=metal_thickness,
            zmin=substrate_thickness,
            material="Nb",
            mesh_order=2,
        ),
        "substrate": LayerLevel(
            name="substrate",
            layer=LAYER.SIM_AREA,
            thickness=substrate_thickness,
            zmin=0.0,
            material="Si",
            mesh_order=4,
        ),
        "vacuum": LayerLevel(
            name="vacuum",
            layer=LAYER.SIM_AREA,
            thickness=air_thickness,
            zmin=substrate_thickness,
            material="vacuum",
            mesh_order=99,
        ),
    }
)

material_spec = material_properties

print("Layer stack:")
for name, level in layer_stack.layers.items():
    print(
        f"  {name:>9}: z = {level.zmin:+.2f} … {level.zmin + level.thickness:+.2f} µm, "
        f"material = {level.material}"
    )

# %% [markdown]
# ## Mesh Settings
#
# `mesh_parameters` is forwarded to `meshwell.mesh.mesh`. Resolution is controlled with the
# `resolution_specs` API: each key is a physical prism name, and each value is a list of
# resolution objects. The driver splits the metal into `metal@o1` and `metal@o2`.
# `ConstantInField` is the simplest one and pins a uniform element size. The
# metal surfaces are meshed finely because the finger gaps (2 μm) carry most of the
# coupling; the bulk dielectric only needs to resolve the field far from the metal.
#
# `BASE_MESH_LENGTHS_UM` holds the nominal (factor 1.0) lengths, and
# `mesh_parameters_for_factor` scales every one of them by the same factor: a smaller
# factor refines the whole mesh uniformly, while the geometry, the layer stack and the
# domain stay untouched.

# %%
BASE_MESH_LENGTHS_UM = {
    "default": 8.0,  # μm, everywhere not covered by a more specific spec
    "terminal": 0.5,  # μm, terminal edges and faces
    "dielectric_surfaces": 3.0,  # μm, substrate and vacuum interfaces
    "dielectric_volumes": 8.0,  # μm, substrate and vacuum bulk
}


def mesh_parameters_for_factor(mesh_factor: float) -> dict[str, Any]:
    """Mesh parameters with every base length scaled by ``mesh_factor``.

    Args:
        mesh_factor: Multiplier applied to all mesh lengths. ``1.0`` is the nominal
            mesh, smaller values refine it.

    Returns:
        Keyword arguments for :func:`meshwell.mesh.mesh`.

    Raises:
        ValueError: If ``mesh_factor`` is not positive.
    """
    if not mesh_factor > 0:
        raise ValueError(f"mesh_factor must be positive, got {mesh_factor}")

    scaled = {
        name: length * mesh_factor for name, length in BASE_MESH_LENGTHS_UM.items()
    }
    resolution_specs: dict[str, list[ConstantInField]] = {}
    for terminal in ("metal@o1", "metal@o2"):
        resolution_specs[terminal] = [
            ConstantInField(resolution=scaled["terminal"], apply_to="curves"),
            ConstantInField(resolution=scaled["terminal"], apply_to="surfaces"),
        ]
    for dielectric in ("substrate", "vacuum"):
        resolution_specs[dielectric] = [
            ConstantInField(
                resolution=scaled["dielectric_surfaces"], apply_to="surfaces"
            ),
            ConstantInField(
                resolution=scaled["dielectric_volumes"], apply_to="volumes"
            ),
        ]
    return {
        "default_characteristic_length": scaled["default"],
        "resolution_specs": resolution_specs,
    }


nominal_mesh = mesh_parameters_for_factor(1.0)
print(
    f"Nominal default characteristic length: "
    f"{nominal_mesh['default_characteristic_length']:.2f} µm"
)
for prism, specs in nominal_mesh["resolution_specs"].items():
    sizes = ", ".join(f"{spec.resolution:.2f} µm ({spec.apply_to})" for spec in specs)
    print(f"  {prism:>10}: {sizes}")

# %% [markdown]
# ## Solve at Each Mesh Factor
#
# Every solve below uses the same component, layer stack, materials and domain; only the
# mesh factor changes. Two solver settings are worth calling out:
#
# - `element_order=2` uses second-order (quadratic) basis functions, which resolve the
#   potential far better per element than the first-order default at the same element
#   count. A first-order solve on a coarse mesh is not a reliable capacitance number.
# - `n_processes=1` keeps the solve serial, so it runs on a plain CPU runner without MPI.
#
# `solve_idc` runs one solve into a fresh scratch directory and returns the Maxwell
# matrix in fF together with the mutual capacitance. Each port becomes a terminal held at
# a fixed potential; both combs carry a port, so no metal surface is grounded and the
# model has no grounded conductor.

# %%
ELEMENT_ORDER = 2
N_PROCESSES = 1
MESH_FACTORS = (1.0, 0.75, 0.6, 0.5)
# Require two successive refinements to change the result by less than 2%.
CONVERGENCE_TOLERANCE = 0.02


@dataclass(slots=True, frozen=True)
class MeshSolve:
    """One independently remeshed electrostatic solve."""

    label: str
    mesh_factor: float
    capacitance_ff: np.ndarray
    mutual_ff: float


def solve_idc(component: gf.Component, mesh_factor: float, label: str) -> MeshSolve:
    """Solve the capacitor on a mesh scaled by ``mesh_factor``.

    Args:
        component: Two-terminal IDC component carrying a `SIM_AREA` outline.
        mesh_factor: Multiplier on every base mesh length; smaller is finer.
        label: Name for this solve, used in tables and error messages.

    Returns:
        The Maxwell capacitance matrix in fF and the extracted mutual capacitance.

    Raises:
        ValueError: If the component ports are not named ``o1`` and ``o2``.
    """
    simulation_folder = Path(
        tempfile.mkdtemp(prefix="qpdk_elmer_interdigital_capacitor_")
    )
    print(f"{label}: solving")
    results = run_capacitive_simulation_elmer(
        component,
        element_order=ELEMENT_ORDER,
        n_processes=N_PROCESSES,
        layer_stack=layer_stack,
        material_spec=material_spec,
        simulation_folder=simulation_folder,
        mesh_parameters=mesh_parameters_for_factor(mesh_factor),
    )

    terminals = tuple(port.name for port in component.ports)
    if terminals != ("o1", "o2"):
        raise ValueError(f"{label}: unexpected terminals {terminals}")
    capacitance_ff = (
        np.array([
            [results.capacitance_matrix[i, j] for j in terminals] for i in terminals
        ])
        * 1e15
    )  # F -> fF
    return MeshSolve(
        label=label,
        mesh_factor=mesh_factor,
        capacitance_ff=capacitance_ff,
        mutual_ff=-float(capacitance_ff[0, 1]),
    )


# %%
mesh_results = [
    solve_idc(component, factor, f"mesh factor {factor:g}") for factor in MESH_FACTORS
]

# %% [markdown]
# ## Capacitance Matrix
#
# The Maxwell capacitance matrix is assembled in the component's port order, so the
# terminals are exactly `o1` and `o2`. The final reported value is the one from the
# finest mesh, which is the last element of `mesh_results`.

# %%
finest = mesh_results[-1]
terminals = tuple(port.name for port in component.ports)

print("Final Maxwell capacitance matrix (fF):")
print(finest.capacitance_ff)
print(f"\nCapacitance matrix indexed by {terminals}")

# %% [markdown]
# The off-diagonal entry is negative by the Maxwell convention
# ($C_{ij} = -C_{ij}^{\text{mutual}}$), so the mutual capacitance between the two
# combs is its negative. There is no grounded conductor, so $C_{11}$ and
# $C_{22}$ are not independent capacitances-to-ground; we report the mutual value.

# %%
print(f"Final off-diagonal C12: {finest.capacitance_ff[0, 1]:.3f} fF")
print(f"Final mutual capacitance C12_mutual = -C12: {finest.mutual_ff:.3f} fF")
print(
    f"Final reported mutual capacitance: {finest.mutual_ff:.3f} fF "
    f"(element_order={ELEMENT_ORDER}, mesh factor {finest.mesh_factor:g})"
)

# %% [markdown]
# ## Mesh Convergence
#
# The four solves above are **independent remeshes**, not Elmer nonlinear iteration
# counts: pass 1 is the nominal mesh (factor 1.0), followed by factors 0.75, 0.6,
# and 0.5. Nothing is continued from one solve to the next, and no field
# solution is reused, so the pass number is only an index into the refinement sequence.
#
# The upper panel shows the extracted mutual capacitance, the lower panel the absolute
# relative change from the previous pass, and the table lists the same two numbers. A
# the final two changes below `CONVERGENCE_TOLERANCE` are a practical refinement check,
# not an error bound. Independent remeshes need not change the result monotonically.

# %%
passes = np.arange(1, len(mesh_results) + 1)
mutual_ff = np.array([result.mutual_ff for result in mesh_results])
relative_change = np.full(mutual_ff.shape, np.nan)
relative_change[1:] = np.abs(np.diff(mutual_ff)) / mutual_ff[:-1]

fig, (ax_value, ax_change) = plt.subplots(
    2, 1, sharex=True, figsize=(6.0, 5.0), layout="constrained"
)
ax_value.plot(passes, mutual_ff, marker="o")
ax_value.set_ylabel("Mutual capacitance [fF]")
ax_value.grid(True)

ax_change.plot(passes[1:], 100 * relative_change[1:], marker="s", linestyle="--")
ax_change.axhline(
    100 * CONVERGENCE_TOLERANCE,
    color="tab:red",
    linestyle=":",
    label=f"tolerance {100 * CONVERGENCE_TOLERANCE:.0f} %",
)
ax_change.set_xticks(passes)
ax_change.set_xlabel("Pass number (independently remeshed solve)")
ax_change.set_ylabel("Change to\nprevious pass [%]")
ax_change.legend()
plt.show()

print(f"{'pass':>4} {'factor':>7} {'C_mutual_fF':>12} {'change_pct':>11}")
for index, result in enumerate(mesh_results, start=1):
    change = relative_change[index - 1]
    change_text = "n/a" if np.isnan(change) else f"{100 * change:.3f}"
    print(
        f"{index:>4} {result.mesh_factor:>7g} {result.mutual_ff:>12.3f} {change_text:>11}"
    )

final_changes = relative_change[-2:]
if not (final_changes <= CONVERGENCE_TOLERANCE).all():
    raise ValueError(
        f"the last two mesh refinements changed the mutual capacitance by "
        f"{100 * final_changes[0]:.3f} % and {100 * final_changes[1]:.3f} %, "
        f"above CONVERGENCE_TOLERANCE = "
        f"{100 * CONVERGENCE_TOLERANCE:.1f} %; refine the mesh further"
    )
print(
    f"Mesh convergence check passed: final changes "
    f"{100 * final_changes[0]:.3f} % and {100 * final_changes[1]:.3f} % "
    f"<= tolerance {100 * CONVERGENCE_TOLERANCE:.1f} %"
)

# %% [markdown]
# ## Lateral Domain Sensitivity
#
# Mesh convergence above holds the domain fixed at `domain_pad=90.0` μm. This separate
# check asks how much the extracted value depends on the outer boundary: we rebuild the
# same capacitor with a 60 μm lateral pad, keep second-order elements and the finest
# mesh factor 0.5, and compare against the pad-90 μm result from the final pass.
#
# Changing the pad also changes the mesh, so the difference mixes the domain effect with
# a discretization effect. This is a sensitivity check between two finite domains: it is
# **not** a proof that the 90 μm boundary is converged to an infinite domain, and it says
# nothing about absolute accuracy.

# %%
component_narrow = interdigital_capacitor_for_elmer(domain_pad=60.0)
narrow = solve_idc(component_narrow, finest.mesh_factor, "pad 60 μm, finest mesh")
wide = finest

print(f"Mutual capacitance at domain_pad=60.0 μm: {narrow.mutual_ff:.3f} fF")
print(f"Mutual capacitance at domain_pad=90.0 μm: {wide.mutual_ff:.3f} fF")
domain_change = abs(wide.mutual_ff - narrow.mutual_ff) / narrow.mutual_ff
print(f"Relative change: {domain_change:.2%}")
if domain_change > 0.01:
    raise ValueError(f"lateral-pad check changed capacitance by {domain_change:.2%}")
print("Lateral-domain sensitivity check passed: change <= 1 %")

# %% [markdown]
# ## Sanity Checks
#
# Every solve must give a finite, symmetric Maxwell matrix with a positive diagonal,
# negative off-diagonal and near-zero row sums. With no grounded conductor, `C11` and
# `C22` duplicate the mutual term. The broad 5-40 fF range is a regression guard against
# order-of-magnitude errors, not a claim about model accuracy.

# %%
for result in (*mesh_results, narrow):
    matrix = result.capacitance_ff
    if not np.isfinite(matrix).all():
        raise ValueError(f"{result.label}: matrix has non-finite entries")
    if not np.allclose(matrix, matrix.T, rtol=1e-2, atol=1e-3):
        raise ValueError(f"{result.label}: matrix is not symmetric")
    if not (np.diag(matrix) > 0).all():
        raise ValueError(f"{result.label}: matrix diagonal must be positive")
    if matrix[0, 1] >= 0 or matrix[1, 0] >= 0:
        raise ValueError(f"{result.label}: Maxwell off-diagonals must be negative")
    if not 5.0 < result.mutual_ff < 40.0:
        raise ValueError(
            f"{result.label}: mutual capacitance outside 5-40 fF: {result.mutual_ff}"
        )
    if (np.abs(matrix.sum(axis=1)) > 0.01 * result.mutual_ff).any():
        raise ValueError(f"{result.label}: matrix has nonzero row sums")

print("All checks passed.")

# %% [markdown]
# ## Summary
#
# We extracted the quasi-static capacitance of a QPDK interdigital capacitor with Elmer
# and report the value from the finest mesh of the convergence study:
#
# - Built a two-terminal geometry from `qpdk.cells.capacitor.interdigital_capacitor`,
#   disabling its `M1_ETCH` mask so the two `M1_DRAW` combs stay isolated, and a minimal
#   `M1_DRAW`-based layer stack to sidestep QPDK's derived `M1` level.
# - Fixed the domain at a 90 μm lateral pad with a 60 μm substrate and a 40 μm vacuum,
#   and solved with second-order elements on four independently generated meshes
#   (factors 1.0, 0.75, 0.6, 0.5).
# - Converted Elmer's lumped result to the 2x2 Maxwell matrix and took the mutual
#   capacitance $-C_{12}$ from each solve. With no grounded conductor, `C11` and `C22`
#   are not independent capacitances-to-ground.
# - Required both final successive changes to stay below the named 2 % tolerance, and
#   reported the finest-mesh value as the result.
# - Required the 60 μm against 90 μm lateral-pad comparison at the finest mesh to change
#   the value by no more than 1 %, as a finite-domain sensitivity check.
#
# The result is a 3D FEM number only, not benchmarked against an analytic IDC model. The
# refinement tolerance measures change between meshes, not absolute accuracy.
#
# The same layout can be swept or optimized by varying `fingers`, `finger_length` and
# `finger_gap` in `interdigital_capacitor_for_elmer` and re-running the solve.
