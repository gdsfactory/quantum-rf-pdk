###########
 Notebooks
###########

.. meta::
    :description: Jupyter notebooks demonstrating simulation approaches for superconducting quantum devices.

These notebooks demonstrate integration with relevant tools for design and simulation of
superconducting quantum devices. Each notebook addresses a different stage of the design
flow and uses a different simulation method, allowing users to choose the tools that
best fit their needs.

*************************************
 Why multiple simulation approaches?
*************************************

Designing a superconducting quantum chip involves physics at many scales. No single
simulation tool covers all of them efficiently, so a practical design flow combines
several complementary methods
:cite:`krantzQuantumEngineersGuide2019,blaisCircuitQuantumElectrodynamics2021`.

The notebooks in this collection are organized around **four simulation categories**:

1. **Scattering-parameter (S-parameter) circuit models** — fast, analytical or
   semi-analytical models for passive microwave components.
2. **FEM-based electromagnetic simulations** — full-wave or quasi-static solvers that
   capture geometry-dependent effects beyond simple analytical formulas.
3. **Hamiltonian analysis** — numerical or perturbative diagonalization of the quantum
   Hamiltonian to extract qubit parameters such as frequency, anharmonicity, and
   dispersive shift.
4. **Pulse-level simulations** — time-domain simulation of control pulses acting on the
   quantum system, including gate fidelity, leakage, and decoherence.

Any of these methods can be wrapped in an automated optimization loop (e.g. with `Optuna
<https://optuna.org/>`_) for design-space exploration.

*******************************************
 Where each method fits in the design flow
*******************************************

The typical workflow when creating a chip with **qpdk / gdsfactory** can be summarized
as follows. Each stage may loop back to earlier stages as the design is refined.

.. only:: html

    .. mermaid::

        flowchart TB
            A["Physical requirements<br>(qubit frequency, coupling, T₁, …)"]
            B["Hamiltonian / perturbation analysis<br>(map requirements → circuit parameters)"]
            C["Circuit / S-parameter models<br>(design passive components)"]
            D["Layout with gdsfactory<br>(draw the chip in qpdk)"]
            E["FEM verification<br>(validate geometry with a full-wave solver)"]
            F["Pulse-level simulation<br>(predict gate performance)"]
            G["Fabrication & measurement"]
            A --> B --> C --> D --> E --> F --> G
            F -.-> A
            E -.-> C

.. only:: latex

    Design flow: Physical requirements → Hamiltonian analysis → Circuit/S-parameter models
    → Layout (gdsfactory/qpdk) → FEM verification → Pulse-level simulation → Fabrication.
    FEM results feed back to circuit models; pulse simulations feed back to requirements.

****************************
 S-parameter circuit models
****************************

S-parameter circuit models treat microwave components as linear, frequency-dependent
networks described by their scattering matrices. In **qpdk** these models are
implemented with `JAX <https://jax.readthedocs.io/>`_ and composed into circuits using
`SAX <https://flaport.github.io/sax/>`_
:cite:`blaisCircuitQuantumElectrodynamics2021,gopplCoplanarWaveguideResonators2008a`.

**Typical use cases:**

- Choosing coplanar-waveguide (CPW) resonator, capacitor, and coupling structure
  geometries to meet target parameters.
- Predicting resonance frequencies and quality factors of passive components.
- Simulating complete test chips with many resonators from a gdsfactory netlist.

**Notebooks:**

- :doc:`notebooks/all_models` — Comprehensive overview of all S-parameter models
  available in the qpdk model library (capacitors, inductors, waveguides, couplers,
  resonators).
- :doc:`notebooks/circuit_simulation_demo` — Builds and simulates composite circuits
  with SAX, starting from individual components and assembling a quarter-wave resonator.
- :doc:`notebooks/resonator_frequency_model` — Compares analytical resonance-frequency
  estimates with SAX circuit simulations.
- :doc:`notebooks/resonator_test_chip_simulation` — Loads a multi-resonator test chip
  from a YAML netlist and simulates the full S₂₁ response with SAX.
- :doc:`notebooks/model_comparison_to_qucs` — Validates qpdk S-parameter models against
  Qucs-S reference data for various passive components.
- :doc:`notebooks/jax_backend_comparison` — Benchmarks SAX circuit evaluation on CPU,
  GPU (CUDA), and NPU (OpenVINO) backends.

***************************************
 FEM-based electromagnetic simulations
***************************************

Finite-element method (FEM) and full-wave solvers discretize Maxwell's equations over
the physical geometry of the device. They capture effects such as radiation, surface
currents, and substrate modes that analytical models may miss
:cite:`gopplCoplanarWaveguideResonators2008a,chenCompactInductorcapacitorResonators2023`.

**Typical use cases:**

- Extracting characteristic impedance and effective permittivity of CPW cross-sections.
- Computing eigenmode frequencies and quality factors of resonators from their physical
  geometry.
- Running driven-modal (port-based) S-parameter simulations of capacitors and other
  structures.
- Optimizing component geometry against a target specification (e.g. a desired
  capacitance value).

**Notebooks:**

- :doc:`notebooks/hfss_q2d_cpw_impedance` — Uses the Ansys Q2D quasi-static solver to
  extract CPW impedance from the cross-section geometry and compares the result with
  analytical conformal-mapping estimates.
- :doc:`notebooks/hfss_eigenmode_resonator` — Eigenmode analysis of a meandering CPW
  resonator in Ansys HFSS to find resonant frequencies and Q-factors.
- :doc:`notebooks/hfss_driven_capacitor` — Driven-modal S-parameter simulation of an
  interdigital capacitor in Ansys HFSS.
- :doc:`notebooks/optimize_capacitor_optuna` — Couples Optuna optimization with the
  Palace FEM solver to optimize an interdigital capacitor towards a target capacitance.
- :doc:`notebooks/palace_eigenmode_qubit_resonator` — Eigenmode simulation of a
  double-pad transmon qubit coupled to a quarter-wave readout resonator using `gsim
  <https://gdsfactory.github.io/gsim/>`_ and Palace, including comparison with
  semi-analytical frequency estimates and an Optuna optimization loop.

.. note::

    **gsim — additional FEM and FDTD simulation examples**

    The `gsim <https://gdsfactory.github.io/gsim/>`_ project provides a collection of
    example notebooks that demonstrate FEM (finite-element method) and FDTD
    (finite-difference time-domain) electromagnetic simulations built on top of
    GDSFactory. These notebooks cover solvers such as **Palace** (FEM) and **Meep**
    (FDTD), showing how to go from a GDSFactory layout to a full 3-D electromagnetic
    simulation. They are a valuable complement to the Ansys-based notebooks above and
    are especially useful for users looking for open-source solver workflows.

    Topics covered in the gsim notebooks include:

    - Eigenmode and driven-port simulations with Palace.
    - FDTD simulations with Meep, including S-parameter extraction.
    - Geometry preparation and meshing pipelines starting from GDSFactory components.
    - Post-processing and visualization of electromagnetic field results.

    See the `gsim documentation <https://gdsfactory.github.io/gsim/>`_ for the full list
    of available notebooks.

**********************
 Hamiltonian analysis
**********************

Superconducting qubits are nonlinear quantum circuits whose behavior is governed by a
Hamiltonian. Diagonalizing this Hamiltonian yields qubit frequencies, anharmonicities,
and coupling strengths that feed back into the layout design
:cite:`kochChargeinsensitiveQubitDesign2007a,blaisCircuitQuantumElectrodynamics2021`.

**Typical use cases:**

- Computing transmon qubit frequency (:math:`\omega_{01}`) and anharmonicity
  (:math:`\alpha`) from Josephson energy :math:`E_J` and charging energy :math:`E_C`.
- Calculating the dispersive shift :math:`\chi` of a transmon–resonator system for
  readout design.
- Translating Hamiltonian-level parameters into physical layout dimensions.

**Notebooks:**

- :doc:`notebooks/scqubits_parameter_calculation` — Full numerical diagonalization of
  the transmon–resonator Hamiltonian with scQubits
  :cite:`groszkowskiScqubitsPythonPackage2021`, compared against analytical perturbation
  theory.
- :doc:`notebooks/pymablock_dispersive_shift` — Perturbative block-diagonalization with
  Pymablock :cite:`arayaDayPymablockAlgorithmPackage2025` to compute the dispersive
  shift symbolically and map the result to layout parameters.
- :doc:`notebooks/netket_transmon_design` — Transmon Hamiltonian analysis with NetKet
  (exact diagonalization and variational methods) including extraction of qubit
  parameters and conversion to layout dimensions.

*************************
 Pulse-level simulations
*************************

Once the qubit parameters are known, pulse-level simulations model the time-domain
evolution of the quantum state under microwave control pulses. These simulations predict
gate fidelities, leakage to non-computational states, and the impact of decoherence
:cite:`liBoshlomQutipqipPulselevel2022,motzoi2009DRAGpulse`.

**Typical use cases:**

- Simulating single-qubit gates (e.g. X, Y) and two-qubit gates (e.g. Bell-state
  preparation) with realistic pulse shapes.
- Estimating leakage to higher transmon levels.
- Evaluating the effect of :math:`T_1` and :math:`T_2` decoherence on gate fidelity.
- Connecting physical layout parameters (frequency, anharmonicity) to gate performance.

**Notebooks:**

- :doc:`notebooks/qutip_qip_pulse_simulation` — Pulse-level simulation of transmon gates
  with QuTiP-QIP :cite:`liBoshlomQutipqipPulselevel2022`, including population dynamics,
  leakage analysis, and decoherence effects.

***************
 Summary table
***************

.. list-table::
    :header-rows: 1
    :widths: 40 30 30

    - - Notebook
      - Category
      - Key tools
    - - :doc:`notebooks/all_models`
      - S-parameter models
      - qpdk, JAX
    - - :doc:`notebooks/circuit_simulation_demo`
      - S-parameter models
      - SAX, JAX
    - - :doc:`notebooks/resonator_frequency_model`
      - S-parameter models
      - SAX
    - - :doc:`notebooks/resonator_test_chip_simulation`
      - S-parameter models
      - SAX, gdsfactory
    - - :doc:`notebooks/model_comparison_to_qucs`
      - S-parameter models
      - SAX, Qucs-S
    - - :doc:`notebooks/jax_backend_comparison`
      - S-parameter models
      - SAX, JAX, OpenVINO
    - - :doc:`notebooks/hfss_q2d_cpw_impedance`
      - FEM electromagnetics
      - Ansys Q2D, PyAEDT
    - - :doc:`notebooks/hfss_eigenmode_resonator`
      - FEM electromagnetics
      - Ansys HFSS, PyAEDT
    - - :doc:`notebooks/hfss_driven_capacitor`
      - FEM electromagnetics
      - Ansys HFSS, PyAEDT
    - - :doc:`notebooks/optimize_capacitor_optuna`
      - FEM optimization
      - Optuna, Palace
    - - :doc:`notebooks/palace_eigenmode_qubit_resonator`
      - FEM electromagnetics
      - gsim, Palace, Optuna
    - - :doc:`notebooks/scqubits_parameter_calculation`
      - Hamiltonian analysis
      - scQubits
    - - :doc:`notebooks/pymablock_dispersive_shift`
      - Hamiltonian analysis
      - Pymablock, SymPy
    - - :doc:`notebooks/netket_transmon_design`
      - Hamiltonian analysis
      - NetKet, JAX
    - - :doc:`notebooks/qutip_qip_pulse_simulation`
      - Pulse-level simulation
      - QuTiP-QIP, JAX

.. toctree::
    :hidden:
    :glob:

    notebooks/*
