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
# # Palace Eigenmode Simulation: Transmon Qubit with Readout Resonator
#
# This notebook demonstrates a full eigenmode simulation workflow for a
# **double-pad transmon qubit** capacitively coupled to a **quarter-wave
# readout resonator** and read through a **probeline**. The simulation
# uses [Palace](https://awslabs.github.io/palace/), an open-source 3-D
# finite-element electromagnetic solver, via the
# [gsim](https://gdsfactory.github.io/gsim/) interface.
#
# The workflow covers:
#
# 1. Creating the layout with ``qpdk`` cells.
# 2. Converting etch layers to conductor geometry for Palace.
# 3. Computing an analytical resonator frequency estimate with the
#    ``qpdk`` semi-analytical models and comparing it to the eigenmode
#    result.
# 4. A compact **Optuna** optimization loop that tunes the resonator
#    length towards a target frequency.
#
# **Prerequisites**
#
# * ``qpdk`` — ``uv pip install qpdk``
# * ``gsim`` — ``uv pip install gsim``
# * A local [Palace](https://awslabs.github.io/palace/) installation
#   (the notebook calls ``sim.run()`` which executes Palace locally).
#
# **References**
#
# * :cite:`kochChargeinsensitiveQubitDesign2007a` — Charge-insensitive
#   qubit design (transmon).
# * :cite:`gopplCoplanarWaveguideResonators2008a` — CPW resonator
#   design and characterization.
# * :cite:`blaisCircuitQuantumElectrodynamics2021` — Circuit quantum
#   electrodynamics review.
# * :cite:`krantzQuantumEngineersGuide2019` — Quantum engineer's guide
#   to superconducting qubits.

# %% tags=["hide-input", "hide-output"]
import warnings

import gdsfactory as gf
import klayout.db as kdb

from qpdk import PDK
from qpdk.cells import transmon_with_resonator_and_probeline
from qpdk.cells.helpers import apply_additive_metals
from qpdk.tech import LAYER

PDK.activate()

# ruff: disable[E402]
from qpdk.models.cpw import cpw_parameters, get_cpw_dimensions
from qpdk.models.resonator import resonator_frequency

# %% [markdown]
# ## 1. Create the layout
#
# We use
# :func:`~qpdk.cells.derived.transmon_with_resonator_and_probeline.transmon_with_resonator_and_probeline`
# which places a double-pad transmon, a quarter-wave CPW resonator, and a
# probeline coupler in a single component. The probeline exposes two CPW
# ports (``coupling_o1`` and ``coupling_o2``) that act as the input and
# output of the readout feedline.

# %%
RESONATOR_LENGTH = 5000.0  # µm
RESONATOR_MEANDERS = 5
RESONATOR_CROSS_SECTION = "cpw"


@gf.cell
def qubit_resonator_sim_component(
    resonator_length: float = RESONATOR_LENGTH,
    resonator_meanders: int = RESONATOR_MEANDERS,
) -> gf.Component:
    """Transmon + resonator + probeline wrapped with a simulation area.

    Args:
        resonator_length: Length of the quarter-wave resonator in µm.
        resonator_meanders: Number of meander sections.

    Returns:
        Component with the simulation layout including ports.
    """
    c = gf.Component()

    ref = c << transmon_with_resonator_and_probeline(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=resonator_length,
        resonator_meanders=resonator_meanders,
        qubit_rotation=90,
    )
    c.add_ports(ref.ports)

    # Add simulation area layer around the component with some margin
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(100, 100))
    return c


component = qubit_resonator_sim_component()
_c = component.copy()
_c.draw_ports()
_c  # noqa: B018

# %% [markdown]
# ## 2. Convert QPDK etch layers to conductor geometry
#
# QPDK uses a *subtractive* layer convention: the ``M1_ETCH`` layer
# defines where metal is removed from the ground plane. Palace needs
# explicit conductor polygons, so we subtract the etch from the
# simulation area.
#
# First we apply
# :func:`~qpdk.cells.helpers.apply_additive_metals` to handle any
# additive metal (``M1_DRAW``) layers, then we perform the Boolean
# operation ``SIM_AREA − M1_ETCH`` to obtain the conductor region.

# %%
from gsim.common.polygon_utils import decimate

# Apply additive metals processing (flattens internally)
processed = apply_additive_metals(component.copy())

sim_area_layer = (LAYER.SIM_AREA[0], LAYER.SIM_AREA[1])
etch_layer = (LAYER.M1_ETCH[0], LAYER.M1_ETCH[1])

CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = processed.kdb_cell.layout()
sim_region = kdb.Region(
    processed.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(processed.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))

# Simplify polygon vertices for a cleaner mesh
etch_polys = decimate(list(etch_region.each()))
etch_region = kdb.Region()
for poly in etch_polys:
    etch_region.insert(poly)

if sim_region.is_empty():
    warnings.warn("No polygons found on SIM_AREA", stacklevel=2)
if etch_region.is_empty():
    warnings.warn("No polygons found on M1_ETCH", stacklevel=2)

conductor_region = sim_region - etch_region

# Build etched component with layers expected by gsim
etched = gf.Component("etched_component")
el = etched.kdb_cell.layout()
for name, region in [
    ("SUPERCONDUCTOR", conductor_region),
    ("SUBSTRATE", sim_region),
    ("VACUUM", sim_region),
]:
    idx = el.layer(*CPW_LAYERS[name])
    etched.kdb_cell.shapes(idx).insert(region)

for port in processed.ports:
    etched.add_port(name=port.name, port=port)

etched  # noqa: B018

# %% [markdown]
# ## 3. Analytical resonator frequency estimate
#
# Before running the full FEM simulation, we compute a quick analytical
# estimate of the resonator frequency using
# :func:`~qpdk.models.resonator.resonator_frequency`. The formula
# for a quarter-wave resonator is :math:`f_r = v_p / (4 L)` where
# :math:`v_p = c_0 / \sqrt{\varepsilon_{\text{eff}}}`.

# %%
import jax.numpy as jnp

from qpdk import logger

width, gap = get_cpw_dimensions(RESONATOR_CROSS_SECTION)
ep_eff, z0 = cpw_parameters(width, gap)

logger.info(f"CPW width = {width} µm, gap = {gap} µm")
logger.info(f"ε_eff = {float(jnp.real(ep_eff)):.4f}, Z₀ = {float(jnp.real(z0)):.2f} Ω")

analytical_freq = resonator_frequency(
    length=RESONATOR_LENGTH,
    epsilon_eff=float(jnp.real(ep_eff)),
    is_quarter_wave=True,
)
logger.info(
    f"Analytical quarter-wave resonator frequency: {analytical_freq / 1e9:.4f} GHz"
)

# %% [markdown]
# ## 4. Configure eigenmode simulation
#
# We set up an eigenmode simulation using
# :class:`~gsim.palace.EigenmodeSim`. The Josephson junction is
# modelled as a lumped port with a characteristic inductance of 10 nH
# (a typical value for a transmon qubit
# :cite:`kochChargeinsensitiveQubitDesign2007a`). The probeline ports
# are configured as CPW wave-ports.

# %%
from gsim.palace import EigenmodeSim

sim = EigenmodeSim()
sim.set_geometry(etched)
sim.set_stack(substrate_thickness=500, air_above=500)

# Junction port with 10 nH inductance (models the Josephson junction)
sim.add_port("junction", layer="SUPERCONDUCTOR", length=5.0, inductance=10e-9)

# CPW feed ports for the readout probeline
sim.add_cpw_port(
    "coupling_o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)
sim.add_cpw_port(
    "coupling_o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0
)

# Search for eigenmodes near the analytical estimate
sim.set_eigenmode(target=analytical_freq, num_modes=5)

# %% [markdown]
# ## 5. Mesh and run
#
# Generate the finite-element mesh and launch Palace locally. The
# ``coarse`` preset gives a fast turnaround suitable for design
# exploration; use ``fine`` or ``graded`` for production accuracy.

# %%
sim.set_output_dir("./sim_palace_qubit_resonator")
sim.mesh(preset="coarse")
sim.plot_mesh()

# %%
results = sim.run()

if results.ok:
    logger.info("Eigenvalues:")
    logger.info(f"{'Mode':<6} {'Freq (GHz)':<16} {'Q':<16}")
    for i, ev in enumerate(results.eigenvalues, start=1):
        logger.info(f"{i:<6} {ev.freq:<16.4f} {ev.quality_factor:<16.4f}")
else:
    logger.error(f"Simulation failed: {results.error_msg}")

# %% [markdown]
# ## 6. Compare eigenmode result with analytical model
#
# The first eigenmode of the system typically corresponds to the
# fundamental resonance of the quarter-wave readout resonator (dressed
# by the qubit coupling). We compare this against the semi-analytical
# prediction.

# %%
import matplotlib.pyplot as plt

if results.ok:
    eigenfreqs_ghz = [ev.freq for ev in results.eigenvalues]

    # Identify the mode closest to the analytical prediction
    analytical_ghz = analytical_freq / 1e9
    closest_idx = min(
        range(len(eigenfreqs_ghz)),
        key=lambda i: abs(eigenfreqs_ghz[i] - analytical_ghz),
    )
    palace_res_freq_ghz = eigenfreqs_ghz[closest_idx]

    logger.info(f"Analytical resonator frequency: {analytical_ghz:.4f} GHz")
    logger.info(f"Palace eigenmode frequency:     {palace_res_freq_ghz:.4f} GHz")
    logger.info(
        f"Relative difference:            "
        f"{abs(palace_res_freq_ghz - analytical_ghz) / analytical_ghz * 100:.2f}%"
    )

    # Bar chart comparing all eigenmodes with the analytical estimate
    fig, ax = plt.subplots(figsize=(8, 4))
    mode_labels = [f"Mode {i}" for i in range(1, len(eigenfreqs_ghz) + 1)]
    ax.barh(mode_labels, eigenfreqs_ghz, color="steelblue", label="Palace eigenmode")
    ax.axvline(
        analytical_ghz,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Analytical f_r = {analytical_ghz:.3f} GHz",
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_title("Eigenmode frequencies vs. analytical resonator estimate")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Optimization loop: tune resonator length for a target frequency
#
# In practice, designers need to adjust the resonator geometry so that
# the readout resonance falls at a specific frequency. Here we wrap the
# eigenmode simulation in an [Optuna](https://optuna.org/) objective
# function that minimizes the squared frequency error.
#
# .. note::
#
#    Each trial launches a full Palace eigenmode solve, so the
#    optimization is computationally expensive.  We keep ``n_trials``
#    small for demonstration. Increase it when running on a
#    high-performance machine.

# %%
import optuna


def eigenmode_objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective: minimize squared frequency error from target.

    Args:
        trial: Optuna trial object with parameter suggestions.

    Returns:
        Squared difference between Palace resonator frequency and
        target frequency, in GHz².
    """
    target_ghz = 6.0  # target resonator frequency

    res_length = trial.suggest_float("resonator_length", 3000.0, 8000.0)

    try:
        # 1. Build layout
        comp = qubit_resonator_sim_component(resonator_length=res_length)
        proc = apply_additive_metals(comp.copy())

        # 2. Extract conductor geometry
        lay = proc.kdb_cell.layout()
        sr = kdb.Region(proc.kdb_cell.begin_shapes_rec(lay.layer(*sim_area_layer)))
        er = kdb.Region(proc.kdb_cell.begin_shapes_rec(lay.layer(*etch_layer)))
        er_dec = kdb.Region()
        for p in decimate(list(er.each())):
            er_dec.insert(p)
        cond = sr - er_dec

        etch_comp = gf.Component(f"opt_{trial.number}")
        el_ = etch_comp.kdb_cell.layout()
        for name_, region_ in [
            ("SUPERCONDUCTOR", cond),
            ("SUBSTRATE", sr),
            ("VACUUM", sr),
        ]:
            etch_comp.kdb_cell.shapes(el_.layer(*CPW_LAYERS[name_])).insert(region_)
        for port_ in proc.ports:
            etch_comp.add_port(name=port_.name, port=port_)

        # 3. Configure and run eigenmode simulation
        opt_sim = EigenmodeSim()
        opt_sim.set_geometry(etch_comp)
        opt_sim.set_stack(substrate_thickness=500, air_above=500)
        opt_sim.add_port(
            "junction", layer="SUPERCONDUCTOR", length=5.0, inductance=10e-9
        )
        opt_sim.add_cpw_port(
            "coupling_o1",
            layer="SUPERCONDUCTOR",
            s_width=10.0,
            gap_width=6.0,
            length=5.0,
        )
        opt_sim.add_cpw_port(
            "coupling_o2",
            layer="SUPERCONDUCTOR",
            s_width=10.0,
            gap_width=6.0,
            length=5.0,
        )

        # Quick analytical guess to guide the eigenmode search
        ep_eff_opt, _ = cpw_parameters(width, gap)
        target_analytical = resonator_frequency(
            length=res_length,
            epsilon_eff=float(jnp.real(ep_eff_opt)),
            is_quarter_wave=True,
        )
        opt_sim.set_eigenmode(target=target_analytical, num_modes=3)

        opt_sim.set_output_dir(f"./sim_opt_trial_{trial.number}")
        opt_sim.mesh(preset="coarse")
        opt_res = opt_sim.run()

        if not opt_res.ok:
            return 1e6  # penalty for failed simulation

        # Find mode closest to analytical estimate
        freqs = [ev.freq for ev in opt_res.eigenvalues]
        an_ghz = target_analytical / 1e9
        best_mode = min(freqs, key=lambda f: abs(f - an_ghz))

        trial.set_user_attr("palace_freq_ghz", best_mode)
        trial.set_user_attr("analytical_freq_ghz", an_ghz)
        trial.set_user_attr("resonator_length_um", res_length)

        return (best_mode - target_ghz) ** 2

    except Exception as exc:
        logger.warning(f"Trial {trial.number} failed: {exc}")
        return 1e6


# %% [markdown]
#
# The optimization block below is wrapped so that it only executes when
# run interactively (it requires a local Palace installation).
#
# ```python

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="resonator_length_optimization",
    )

    logger.info("Starting Optuna optimization for 6 GHz resonator frequency...")
    study.optimize(eigenmode_objective, n_trials=3, n_jobs=1, show_progress_bar=True)

    successful = [t for t in study.trials if t.value < 1e5]
    if successful:
        best = study.best_trial
        logger.info(f"Best trial: {best.number}")
        logger.info(f"  Resonator length: {best.params['resonator_length']:.1f} µm")
        logger.info(
            f"  Palace frequency: {best.user_attrs.get('palace_freq_ghz', 'N/A')} GHz"
        )
        logger.info(f"  Objective (freq error²): {best.value:.6f} GHz²")
    else:
        logger.warning("No successful trials. Check Palace installation.")

# ```

# ruff: enable[E402]
