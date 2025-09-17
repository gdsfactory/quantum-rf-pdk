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
# # Optuna Optimization of Interdigital Capacitor
#
# This example demonstrates using Optuna to optimize an interdigital capacitor
# to achieve a target capacitance of 40 fF. The optimization is constrained to
# use exactly 5 interdigital fingers.

# %%

from typing import Any

import gdsfactory as gf
import optuna
from meshwell.resolution import ConstantInField, ExponentialField

from qpdk import PDK
from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.waveguides import straight
from qpdk.config import PATH
from qpdk.tech import LAYER, material_properties

# %% [markdown]
# ## Simulation Layout Setup
#
# Create a simulation layout with an interdigital capacitor,
# some extended straights and an etch for capacitive simulation.


# %%
@gf.cell
def capacitor_simulation(
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
) -> gf.Component:
    """Create a capacitor simulation layout with launchers and direct connections.

    Args:
        finger_length: Length of each finger in μm.
        finger_gap: Gap between adjacent fingers in μm.
        thickness: Thickness of fingers and base section in μm.

    Returns:
        Component with the simulation layout including ports.
    """
    c = gf.Component()

    # Create interdigital capacitor with fixed 5 fingers as specified
    cap_ref = c << interdigital_capacitor(
        fingers=5,  # Fixed to 5 fingers as per requirements
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        etch_layer="M1_ETCH",
        etch_bbox_margin=5.0,
        cross_section="cpw",
        half=False,
    )

    # Add straights for larger area
    straight_left = c << straight(length=20.0, cross_section="cpw")
    straight_right = c << straight(length=20.0, cross_section="cpw")
    straight_left.connect("o2", cap_ref.ports["o1"])
    straight_right.connect("o1", cap_ref.ports["o2"])

    # Add etched end at the straights
    etch_left = c << straight(length=6.0, cross_section="etch")
    etch_right = c << straight(length=6.0, cross_section="etch")
    etch_left.connect("o2", straight_left.ports["o1"], allow_layer_mismatch=True)
    etch_right.connect("o1", straight_right.ports["o2"], allow_layer_mismatch=True)

    # Add simulation area layer around the layout
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(50, 50))

    # Add ports for marking capacitor terminals
    c.add_port(name="M1_left", port=straight_left.ports["o1"])
    c.add_port(name="M1_right", port=straight_right.ports["o2"])

    return c


# %% [markdown]
# ## Optuna Objective Function
#
# Define the objective function that Optuna will optimize. The goal is to minimize
# the difference between the simulated capacitance and the target of 40 fF.


# %%
def _objective_function(trial: optuna.trial.Trial) -> float:
    """Optuna objective function to optimize capacitor for target capacitance.

    Args:
        trial: Optuna trial object with parameter suggestions.

    Returns:
        Objective value (difference from target capacitance).
    """
    target_capacitance = 40.0  # in fF

    # Define parameter search space
    finger_length = trial.suggest_float("finger_length", 5.0, 50.0)  # μm
    finger_gap = trial.suggest_float("finger_gap", 1.0, 10.0)  # μm
    thickness = trial.suggest_float("thickness", 2.0, 10.0)  # μm

    try:
        c = capacitor_simulation(
            finger_length=finger_length,
            finger_gap=finger_gap,
            thickness=thickness,
        )
        simulated_capacitance = _run_capacitive_simulation(c)

        # Calculate objective: minimize mean squared error from target capacitance
        objective_value = (simulated_capacitance - target_capacitance) ** 2

        # Store additional info for analysis
        trial.set_user_attr("simulated_capacitance", simulated_capacitance)
        trial.set_user_attr("target_capacitance", target_capacitance)

        return objective_value

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 1000.0  # Large penalty value


# %% [markdown]
# ## Palace imulation settings
#
# This section shows how the Palace simulation is set up.


# %%
def _setup_palace_simulation() -> dict[str, Any]:
    """Setup configuration for Palace capacitive simulation.

    The mesh_parameters section is provided as keyword arguments
    to :func:`~meshwell.mesh.mesh`.

    The mesh parameters here are not optimized but serve as a reasonable
    starting point while demonstrating how te set up the mesh in different ways.

    Args:
        component: The gdsfactory component to simulate.

    Returns:
        Dictionary with simulation configuration.
    """
    simulation_folder = PATH.simulation / "capacitor_simulation"
    simulation_folder.mkdir(exist_ok=True)

    # Palace simulation configuration
    return {
        "layer_stack": PDK.layer_stack,
        "material_spec": material_properties,
        "simulation_folder": simulation_folder,
        "mesh_parameters": {
            "default_characteristic_length": 30,  # μm
            "resolution_specs": {
                "M1@M1_left": [
                    ExponentialField(
                        sizemin=0.3, lengthscale=2, growth_factor=2.0, apply_to="curves"
                    ),
                    ConstantInField(resolution=0.3, apply_to="surfaces"),
                ],
                "M1@M1_right": [
                    ExponentialField(
                        sizemin=0.3, lengthscale=2, growth_factor=2.0, apply_to="curves"
                    ),
                    ConstantInField(resolution=0.3, apply_to="surfaces"),
                ],
                "M1": [ConstantInField(resolution=8.0, apply_to="volumes")],
                "Substrate": [
                    ConstantInField(resolution=5.0, apply_to="curves"),
                    ConstantInField(resolution=8.0, apply_to="surfaces"),
                    ConstantInField(resolution=15.0, apply_to="volumes"),
                ],
                "Vacuum": [
                    ConstantInField(resolution=5.0, apply_to="curves"),
                    ConstantInField(resolution=15.0, apply_to="surfaces"),
                    ConstantInField(resolution=25.0, apply_to="volumes"),
                ],
            },
            "verbosity": 10,
        },
    }


def _run_capacitive_simulation(component: gf.Component) -> float:
    """Run Palace capacitive simulation (requires system dependencies)."""
    from gplugins.palace import run_capacitive_simulation_palace

    config = _setup_palace_simulation()
    results = run_capacitive_simulation_palace(component, n_processes=4, **config)
    return results.capacitance_matrix[tuple(p.name for p in component.ports)] * 1e15


# %% [markdown]
# ## Main Execution and Optuna Study
#
# Run the optimization study using Optuna to find parameters that achieve
# the target capacitance of 40 fF.

# %% [markdown]
#
# This section is not run in the documentation because Palace
# requires an installation and the optimization may take time.
#
# ```python
#
if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    # First ensure a single simulation runs correctly
    c = capacitor_simulation()
    c.show()
    simulated_capacitance = _run_capacitive_simulation(c)
    print(f"Single simulation capacitance: {simulated_capacitance:.3f} fF")

    # Create an Optuna study for minimization
    study = optuna.create_study(
        direction="minimize",
        study_name="interdigital_capacitor_optimization",
    )

    print("Starting Optuna optimization for 40 fF interdigital capacitor...")
    print("Target: 40 fF capacitance with 5 fingers")
    print("Optimizing: finger_length, finger_gap, thickness")
    print()
    study.optimize(_objective_function, n_trials=5, n_jobs=1, show_progress_bar=True)

    # Check if we have any successful trials
    successful_trials = [t for t in study.trials if t.value < 1000.0]

    if successful_trials:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best objective value: {study.best_value:.3f} fF difference from target")
        print("\nBest parameters:")
        for param, value in study.best_trial.params.items():
            print(f"  {param}: {value:.3f} μm")

        if "simulated_capacitance" in study.best_trial.user_attrs:
            print(
                f"\nSimulated capacitance: {study.best_trial.user_attrs['simulated_capacitance']:.3f} fF"
            )
            print(
                f"Target capacitance: {study.best_trial.user_attrs['target_capacitance']:.3f} fF"
            )

        # Create and show the optimized component
        best_params = study.best_trial.params
        optimized_component = capacitor_simulation(
            finger_length=best_params["finger_length"],
            finger_gap=best_params["finger_gap"],
            thickness=best_params["thickness"],
        )

        print("\nOptimized component created with:")
        print("  - 5 fingers (fixed)")
        print(f"  - finger_length: {best_params['finger_length']:.3f} μm")
        print(f"  - finger_gap: {best_params['finger_gap']:.3f} μm")
        print(f"  - thickness: {best_params['thickness']:.3f} μm")
    else:
        print(
            "No successful trials found. All trials resulted in component generation errors."
        )
        print(
            "This suggests there may be issues with the component geometry or routing."
        )
        # Still try to create a component with default parameters for debugging
        print("\nTrying to create component with default parameters for debugging...")
        try:
            test_component = capacitor_simulation()
            print("Default component created successfully")
        except Exception as e:
            print(f"Error creating default component: {e}")

    # Display the component (optional - requires display backend)
    try:
        if successful_trials:
            optimized_component.show()
        else:
            test_component.show()
    except Exception:
        print("(Component display not available in this environment)")

    # Save optimization history for analysis
    if not PATH.simulation.exists():
        PATH.simulation.mkdir()
    results_file = PATH.simulation / "capacitor_optimization_results.csv"
    study.trials_dataframe().to_csv(results_file, index=False)
    print(f"\nOptimization results saved to: {results_file}")

# ```
