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
# use exactly 5 interdigital fingers as specified in the requirements.

# %%

from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np
from gdsfactory.export import to_stl

from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.launcher import launcher
from qpdk.config import PATH
from qpdk.tech import LAYER

# %% [markdown]
# ## Simulation Layout Setup
#
# Create a simulation layout with an interdigital capacitor between two probe launchers
# for capacitive simulation. This follows the same pattern as the resonator simulation
# but is designed for extracting capacitance rather than S-parameters.


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

    # Add launchers for probing
    launcher_left = c << launcher()
    launcher_right = c << launcher()
    launcher_right.mirror()

    # Position launchers with proper spacing from capacitor ports
    # Capacitor ports are at ±width/2, launcher port is at (300, 0) from launcher origin
    spacing = 300  # μm spacing between port and capacitor

    # Left launcher: position so its o1 port is to the left of capacitor's o1 port
    launcher_left.move((cap_ref["o1"].center[0] - spacing - 300, 0))

    # Right launcher: position so its o1 port is to the right of capacitor's o2 port
    launcher_right.move((cap_ref["o2"].center[0] + spacing, 0))

    # Connect launchers directly to capacitor ports using straight routes
    from qpdk.cells.waveguides import straight

    # Left connection
    left_length = abs(launcher_left["o1"].center[0] - cap_ref["o1"].center[0])
    if left_length > 0:
        left_connection = c << straight(length=left_length, cross_section="cpw")
        left_connection.connect("o1", launcher_left["o1"])
        # Note: In a real design we'd properly route these, but for simulation we keep it simple

    # Right connection
    right_length = abs(cap_ref["o2"].center[0] - launcher_right["o1"].center[0])
    if right_length > 0:
        right_connection = c << straight(length=right_length, cross_section="cpw")
        right_connection.connect("o1", cap_ref["o2"])

    # Add simulation area layer around the layout
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(50, 50))

    # Add ports for external connections
    c.add_ports(launcher_left.ports, prefix="left_")
    c.add_ports(launcher_right.ports, prefix="right_")

    return c


# %% [markdown]
# ## Mock Capacitive Simulation Function
#
# Since the actual Palace simulation may not run due to system dependencies,
# we create a mock function that simulates the capacitance extraction process.
# In a real implementation, this would call `run_capacitive_simulation_palace`.


# %%
def run_mock_capacitive_simulation(
    component: gf.Component,  # noqa: ARG001
    finger_length: float,
    finger_gap: float,
    thickness: float,
) -> float:
    """Mock capacitive simulation that estimates capacitance based on geometry.

    This function provides a realistic but simplified capacitance model for
    interdigital capacitors based on the geometric parameters. In practice,
    this would be replaced by Palace electromagnetic simulation.

    Args:
        component: The gdsfactory component to simulate.
        finger_length: Length of capacitor fingers in μm.
        finger_gap: Gap between fingers in μm.
        thickness: Thickness of fingers in μm.

    Returns:
        Estimated capacitance in fF (femtofarads).
    """
    # Simple analytical model for interdigital capacitor capacitance
    # Based on parallel plate and fringing field contributions

    # Constants for the material stack (typical values for superconducting qubits)
    epsilon_r = 11.45  # Silicon relative permittivity
    epsilon_0 = 8.854e-12  # F/m, vacuum permittivity

    # Number of fingers is fixed at 5
    n_fingers = 5

    # Convert from μm to m for calculations
    length_m = finger_length * 1e-6
    gap_m = finger_gap * 1e-6
    thickness_m = thickness * 1e-6

    # Parallel plate capacitance between adjacent fingers
    # C = ε₀ × ε_r × A / d where A is the overlap area
    overlap_area = length_m * thickness_m  # Area of one finger face
    n_gaps = n_fingers - 1  # Number of gaps between fingers

    parallel_plate_cap = epsilon_0 * epsilon_r * overlap_area * n_gaps / gap_m

    # Fringing field correction (empirical formula)
    # Adds ~20-50% depending on geometry
    fringing_factor = 1 + 0.3 * np.log(1 + thickness_m / gap_m)

    # Total capacitance in Farads
    total_cap_f = parallel_plate_cap * fringing_factor

    # Convert to femtofarads (fF)
    total_cap_ff = total_cap_f * 1e15

    # Add some realistic noise/variation
    rng = np.random.default_rng()
    noise_factor = 1 + 0.05 * (rng.random() - 0.5)  # ±2.5% variation

    return total_cap_ff * noise_factor


# %% [markdown]
# ## Optuna Objective Function
#
# Define the objective function that Optuna will optimize. The goal is to minimize
# the difference between the simulated capacitance and the target of 40 fF.


# %%
def objective_function(trial) -> float:
    """Optuna objective function to optimize capacitor for target capacitance.

    Args:
        trial: Optuna trial object with parameter suggestions.

    Returns:
        Objective value (difference from target capacitance).
    """
    # Target capacitance in fF
    target_capacitance = 40.0

    # Define parameter search space
    finger_length = trial.suggest_float("finger_length", 5.0, 50.0)  # μm
    finger_gap = trial.suggest_float("finger_gap", 1.0, 10.0)  # μm
    thickness = trial.suggest_float("thickness", 2.0, 10.0)  # μm

    try:
        # Create the simulation layout
        c = capacitor_simulation(
            finger_length=finger_length,
            finger_gap=finger_gap,
            thickness=thickness,
        )

        # Run mock simulation (in practice, this would be Palace)
        simulated_capacitance = run_mock_capacitive_simulation(
            c, finger_length, finger_gap, thickness
        )

        # Calculate objective: minimize absolute difference from target
        objective_value = abs(simulated_capacitance - target_capacitance)

        # Store additional info for analysis
        trial.set_user_attr("simulated_capacitance", simulated_capacitance)
        trial.set_user_attr("target_capacitance", target_capacitance)

        return objective_value

    except Exception as e:
        # Return large penalty for invalid geometries
        print(f"Trial failed with error: {e}")
        return 1000.0  # Large penalty value


# %% [markdown]
# ## Real Palace Simulation Setup (commented)
#
# This section shows how the actual Palace simulation would be set up,
# but is commented out since it requires system dependencies that may not be available.


# %%
def setup_palace_simulation(component: gf.Component) -> dict[str, Any]:
    """Setup configuration for Palace capacitive simulation.

    This function demonstrates how to configure a real Palace simulation
    for capacitance extraction. It's currently set up as a mock since
    Palace requires system dependencies that may not be available.

    Args:
        component: The gdsfactory component to simulate.

    Returns:
        Dictionary with simulation configuration.
    """
    from qpdk import PDK

    # Material specifications for superconducting quantum devices
    material_spec = {
        "Si": {"relative_permittivity": 11.45},  # Silicon substrate
        "Nb": {
            "relative_permittivity": np.inf
        },  # Superconducting metal (perfect conductor)
        "vacuum": {"relative_permittivity": 1},  # Air/vacuum gaps
    }

    # Export 3D model for simulation
    simulation_folder = Path().cwd() / "capacitor_simulation"
    simulation_folder.mkdir(exist_ok=True)

    to_stl(
        component,
        simulation_folder / "capacitor.stl",
        layer_stack=PDK.layer_stack,
        hull_invalid_polygons=True,
    )

    # Palace simulation configuration
    return {
        "material_spec": material_spec,
        "simulation_folder": simulation_folder,
        "mesh_parameters": {
            "background_tag": "vacuum",
            "background_padding": (0,) * 5 + (200,),  # 200 μm padding in z
            "port_names": [port.name for port in component.ports],
            "default_characteristic_length": 50,  # μm
            "resolutions": {
                "M1": {"resolution": 5},  # Fine mesh on metal
                "Silicon": {"resolution": 20},  # Coarser on substrate
                "vacuum": {"resolution": 30},  # Coarsest in air
                # Port-specific mesh refinement
                **{
                    f"M1__{port.name}": {
                        "resolution": 3,
                        "DistMax": 15,
                        "DistMin": 2,
                        "SizeMax": 8,
                        "SizeMin": 1,
                    }
                    for port in component.ports
                },
            },
        },
    }


# Commented out real Palace simulation call:
# def run_real_capacitive_simulation(component: gf.Component) -> float:
#     """Run actual Palace capacitive simulation (requires system dependencies)."""
#     from gplugins.palace import run_capacitive_simulation_palace
#     from qpdk import PDK
#
#     config = setup_palace_simulation(component)
#
#     results = run_capacitive_simulation_palace(
#         component,
#         layer_stack=PDK.layer_stack,
#         material_spec=config["material_spec"],
#         simulation_folder=config["simulation_folder"],
#         mesh_parameters=config["mesh_parameters"],
#     )
#
#     # Extract capacitance from simulation results
#     # This would depend on the specific format of Palace results
#     capacitance_ff = extract_capacitance_from_results(results)
#
#     return capacitance_ff


# %% [markdown]
# ## Main Execution and Optuna Study
#
# Run the optimization study using Optuna to find parameters that achieve
# the target capacitance of 40 fF.

# %%
if __name__ == "__main__":
    import optuna

    from qpdk import PDK

    # Activate the PDK
    PDK.activate()

    # Create an Optuna study for minimization
    study = optuna.create_study(
        direction="minimize",
        study_name="interdigital_capacitor_optimization",
    )

    print("Starting Optuna optimization for 40 fF interdigital capacitor...")
    print("Target: 40 fF capacitance with 5 fingers")
    print("Optimizing: finger_length, finger_gap, thickness")
    print()

    # Run optimization
    n_trials = 50
    study.optimize(objective_function, n_trials=n_trials)

    # Print results
    print(f"\nOptimization completed after {n_trials} trials!")

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
    results_file = PATH.simulation / "capacitor_optimization_results.csv"
    study.trials_dataframe().to_csv(results_file, index=False)
    print(f"\nOptimization results saved to: {results_file}")
