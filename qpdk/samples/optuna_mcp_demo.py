#!/usr/bin/env python3
"""Optuna MCP Server Demo for Interdigital Capacitor Optimization.

This script demonstrates how to use the Optuna MCP (Model Context Protocol) server
to optimize an interdigital capacitor for a target capacitance of 40 fF.

The MCP server provides a standardized interface for running optimization studies
that can be accessed from various clients.
"""

import json
from pathlib import Path

import gdsfactory as gf
import numpy as np

from qpdk.cells.capacitor import interdigital_capacitor
from qpdk.cells.launcher import launcher
from qpdk.cells.waveguides import straight
from qpdk.tech import LAYER


def run_mock_capacitive_simulation(
    component: gf.Component,  # noqa: ARG001
    finger_length: float,
    finger_gap: float,
    thickness: float,
) -> float:
    """Mock capacitive simulation for MCP demo."""
    # Same mock simulation as in the main script
    epsilon_r = 11.45  # Silicon relative permittivity
    epsilon_0 = 8.854e-12  # F/m, vacuum permittivity

    n_fingers = 5

    # Convert from μm to m for calculations
    length_m = finger_length * 1e-6
    gap_m = finger_gap * 1e-6
    thickness_m = thickness * 1e-6

    # Parallel plate capacitance between adjacent fingers
    overlap_area = length_m * thickness_m
    n_gaps = n_fingers - 1

    parallel_plate_cap = epsilon_0 * epsilon_r * overlap_area * n_gaps / gap_m

    # Fringing field correction
    fringing_factor = 1 + 0.3 * np.log(1 + thickness_m / gap_m)

    # Total capacitance in Farads
    total_cap_f = parallel_plate_cap * fringing_factor

    # Convert to femtofarads (fF)
    total_cap_ff = total_cap_f * 1e15

    # Add some realistic noise/variation
    rng = np.random.default_rng()
    noise_factor = 1 + 0.05 * (rng.random() - 0.5)  # ±2.5% variation

    return total_cap_ff * noise_factor


@gf.cell
def capacitor_simulation_mcp(
    finger_length: float = 20.0,
    finger_gap: float = 2.0,
    thickness: float = 5.0,
) -> gf.Component:
    """Create a capacitor simulation layout for MCP demo."""
    c = gf.Component()

    # Create interdigital capacitor with fixed 5 fingers
    cap_ref = c << interdigital_capacitor(
        fingers=5,  # Fixed to 5 fingers as per requirements
        finger_length=finger_length,
        finger_gap=finger_gap,
        thickness=thickness,
        etch_layer=LAYER.M1_ETCH,
        etch_bbox_margin=5.0,
        cross_section="cpw",
        half=False,
    )

    # Add launchers for probing
    launcher_left = c << launcher()
    launcher_right = c << launcher()
    launcher_right.mirror()

    # Position launchers with proper spacing from capacitor ports
    spacing = 300  # μm spacing between port and capacitor

    # Left launcher: position so its o1 port is to the left of capacitor's o1 port
    launcher_left.move((cap_ref["o1"].center[0] - spacing - 300, 0))

    # Right launcher: position so its o1 port is to the right of capacitor's o2 port
    launcher_right.move((cap_ref["o2"].center[0] + spacing, 0))

    # Connect launchers directly to capacitor ports using straight routes
    # Left connection
    left_length = abs(launcher_left["o1"].center[0] - cap_ref["o1"].center[0])
    if left_length > 0:
        left_connection = c << straight(length=left_length, cross_section="cpw")
        left_connection.connect("o1", launcher_left["o1"])

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


def objective_function_mcp(trial) -> float:
    """MCP objective function to optimize capacitor for target capacitance."""
    # Make sure the correct PDK is activated
    from qpdk import PDK
    PDK.activate()

    # Target capacitance in fF
    target_capacitance = 40.0

    # Define parameter search space
    finger_length = trial.suggest_float("finger_length", 5.0, 50.0)  # μm
    finger_gap = trial.suggest_float("finger_gap", 1.0, 10.0)        # μm
    thickness = trial.suggest_float("thickness", 2.0, 10.0)          # μm

    try:
        # Create the simulation layout
        c = capacitor_simulation_mcp(
            finger_length=finger_length,
            finger_gap=finger_gap,
            thickness=thickness,
        )

        # Run mock simulation
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


def create_optuna_study_with_mcp():
    """Create and configure an Optuna study using the MCP server interface.

    This demonstrates how to set up the optimization study configuration
    that would be used with an Optuna MCP server.
    """
    # Configuration for the Optuna study
    return {
        "study_name": "interdigital_capacitor_optimization_mcp",
        "direction": "minimize",
        "sampler": {
            "name": "TPESampler",
            "n_startup_trials": 10,
            "n_ei_candidates": 24,
        },
        "search_space": {
            "finger_length": {
                "name": "FloatDistribution",
                "attributes": {
                    "low": 5.0,
                    "high": 50.0,
                    "step": None,
                    "log": False
                }
            },
            "finger_gap": {
                "name": "FloatDistribution",
                "attributes": {
                    "low": 1.0,
                    "high": 10.0,
                    "step": None,
                    "log": False
                }
            },
            "thickness": {
                "name": "FloatDistribution",
                "attributes": {
                    "low": 2.0,
                    "high": 10.0,
                    "step": None,
                    "log": False
                }
            }
        },
        "optimization_target": {
            "metric_name": "capacitance_error",
            "target_capacitance": 40.0,  # fF
            "constraint_fingers": 5,      # Fixed number of fingers
        }
    }


def mock_mcp_optimization_session():
    """Mock implementation of an Optuna MCP server session.
    
    This simulates how the optimization would work through the MCP interface,
    demonstrating the ask-tell pattern used by Optuna.
    """
    print("=== Optuna MCP Server Demo ===")
    print("Simulating optimization session for interdigital capacitor")
    print("Target: 40 fF capacitance with 5 fingers\n")

    # Step 1: Create study configuration
    study_config = create_optuna_study_with_mcp()
    print("1. Study Configuration:")
    print(json.dumps(study_config, indent=2))
    print()

    # Step 2: Initialize study (normally done via MCP)
    print("2. Initializing study via MCP server...")
    print(f"   Study name: {study_config['study_name']}")
    print(f"   Direction: {study_config['direction']}")
    print(f"   Sampler: {study_config['sampler']['name']}")
    print()

    # Step 3: Demonstrate ask-tell optimization loop
    print("3. Running optimization trials (ask-tell pattern):")

    # Mock MCP server state
    trial_results = []
    best_result = {"value": float("inf"), "params": None, "trial": None}

    # Simulate several optimization trials
    for trial_num in range(10):
        print(f"\n   Trial {trial_num}:")

        # Step 3a: Ask for new parameters (MCP server suggests)
        if trial_num == 0:
            # First trial - random sampling
            suggested_params = {
                "finger_length": 25.0,
                "finger_gap": 5.0,
                "thickness": 7.0
            }
        elif trial_num == 1:
            suggested_params = {
                "finger_length": 40.0,
                "finger_gap": 3.0,
                "thickness": 8.0
            }
        elif trial_num == 2:
            suggested_params = {
                "finger_length": 45.0,
                "finger_gap": 4.0,
                "thickness": 9.0
            }
        else:
            # Later trials - TPE-guided sampling (simplified)
            if best_result["params"]:
                base_params = best_result["params"]
                suggested_params = {
                    "finger_length": min(50.0, max(5.0, base_params["finger_length"] + (trial_num - 3) * 1.0)),
                    "finger_gap": min(10.0, max(1.0, base_params["finger_gap"] + (trial_num - 3) * 0.2)),
                    "thickness": min(10.0, max(2.0, base_params["thickness"] + (trial_num - 3) * 0.1))
                }
            else:
                suggested_params = {
                    "finger_length": 30.0 + trial_num * 2.0,
                    "finger_gap": 4.0 + trial_num * 0.3,
                    "thickness": 6.0 + trial_num * 0.4
                }

        print(f"     Ask: Suggested parameters: {suggested_params}")

        # Step 3b: Evaluate objective function
        try:
            # Create a mock trial object
            class MockTrial:
                def __init__(self, params):
                    self.params = params
                    self.user_attrs = {}
                    self.number = trial_num

                def suggest_float(self, name, low, high):
                    return self.params[name]

                def set_user_attr(self, key, value):
                    self.user_attrs[key] = value

            mock_trial = MockTrial(suggested_params)
            objective_value = objective_function_mcp(mock_trial)

            print(f"     Tell: Objective value: {objective_value:.3f}")
            if "simulated_capacitance" in mock_trial.user_attrs:
                print(f"           Simulated capacitance: {mock_trial.user_attrs['simulated_capacitance']:.3f} fF")

            # Update best result
            if objective_value < best_result["value"]:
                best_result = {
                    "value": objective_value,
                    "params": suggested_params.copy(),
                    "trial": trial_num,
                    "user_attrs": mock_trial.user_attrs.copy()
                }
                print("           *** New best result! ***")

            trial_results.append({
                "trial": trial_num,
                "params": suggested_params,
                "value": objective_value,
                "user_attrs": mock_trial.user_attrs
            })

        except Exception as e:
            print(f"     Tell: Error: {e}")
            trial_results.append({
                "trial": trial_num,
                "params": suggested_params,
                "value": 1000.0,  # Penalty
                "error": str(e)
            })

    # Step 4: Report final results
    print("\n4. Optimization Results:")
    print(f"   Best trial: {best_result['trial']}")
    print(f"   Best objective: {best_result['value']:.3f} fF difference")
    print("   Best parameters:")
    for param, value in best_result["params"].items():
        print(f"     {param}: {value:.3f}")

    if "simulated_capacitance" in best_result["user_attrs"]:
        print(f"   Achieved capacitance: {best_result['user_attrs']['simulated_capacitance']:.3f} fF")
        print(f"   Target capacitance: {best_result['user_attrs']['target_capacitance']:.3f} fF")

    return trial_results, best_result


def demonstrate_mcp_workflow():
    """Demonstrate the complete MCP workflow for capacitor optimization.
    """
    print("Demonstrating Optuna MCP Server workflow for capacitor optimization\n")

    # Run the mock optimization session
    trial_results, best_result = mock_mcp_optimization_session()

    # Save results in MCP-compatible format
    results_dir = Path(__file__).parent.parent.parent / "build" / "simulation"
    results_dir.mkdir(exist_ok=True, parents=True)

    mcp_results = {
        "study_config": create_optuna_study_with_mcp(),
        "trial_results": trial_results,
        "best_result": best_result,
        "metadata": {
            "optimization_type": "interdigital_capacitor",
            "target_capacitance_fF": 40.0,
            "constraint_fingers": 5,
            "mcp_version": "demo",
        }
    }

    # Save results
    results_file = results_dir / "mcp_optimization_demo_results.json"
    with open(results_file, "w") as f:
        json.dump(mcp_results, f, indent=2)

    print(f"\n5. Results saved to: {results_file}")

    # Demonstrate creating the optimized component
    if best_result["params"]:
        print("\n6. Creating optimized component...")
        try:
            from qpdk import PDK
            PDK.activate()

            optimized_component = capacitor_simulation_mcp(
                finger_length=best_result["params"]["finger_length"],
                finger_gap=best_result["params"]["finger_gap"],
                thickness=best_result["params"]["thickness"],
            )
            print("   Component created successfully with:")
            print(f"   - Bounding box: {optimized_component.bbox()}")
            print(f"   - Ports: {[p.name for p in optimized_component.ports]}")

        except Exception as e:
            print(f"   Error creating component: {e}")

    print("\n=== MCP Demo Complete ===")


if __name__ == "__main__":
    demonstrate_mcp_workflow()
