###########
 Notebooks
###########

.. meta::
    :description: Jupyter notebooks demonstrating simulation approaches for superconducting quantum devices.

These notebooks demonstrate integration with relevant tools for design and simulation of
superconducting quantum devices. Each notebook addresses a different stage of the design
flow and uses a different simulation method, allowing users to choose the tools that
best fit their needs.

Because each notebook pulls in a different simulation backend, ``qpdk`` keeps those
backends behind optional dependencies rather than installing all of them by default. The
:ref:`summary table <notebook-summary>` at the bottom lists which :ref:`extras
<notebook-extras>` each notebook needs, and how to install them.

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

.. only:: typst or typstpdf

    Design flow: Physical requirements → Hamiltonian analysis → Circuit/S-parameter models
    → Layout (gdsfactory/qpdk) → FEM verification → Pulse-level simulation → Fabrication.
    FEM results feed back to circuit models; pulse simulations feed back to requirements.

****************************
 S-parameter circuit models
****************************

S-parameter circuit models treat microwave components as linear, frequency-dependent
networks described by their scattering matrices. In **qpdk** these models are
implemented with `JAX <https://jax.readthedocs.io/>`_ and composed into circuits using
`SAX <https://gdsfactory.github.io/sax/>`_
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
- :doc:`notebooks/monte_carlo_fabrication_tolerance` — Monte Carlo fabrication tolerance
  analysis that loads a multi-resonator test chip from a YAML netlist, simulates the
  full S₂₁ response with SAX, and varies CPW width and gap to quantify resonance
  frequency spread.
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
- :doc:`notebooks/comsol_cpw_resonator` — Builds and solves a coupled quarter-wave CPW
  resonator in COMSOL with MPh: extends both source feeds to open port planes, extracts
  the QPDK metal, builds the air/silicon sheet model, and adds PEC, two numeric TEM
  ports with voltage integration lines, boundary mode analysis, and a frequency-domain
  study. Ships a real 5-10 GHz transmission sweep and a saved field map from a COMSOL
  6.3 solve, and notes that the 0.25 GHz sweep spacing is too coarse to resolve the
  resonance notch. The feed extension changes the coupling geometry relative to the
  unextended reference cell.
- :doc:`notebooks/comsol_qubit_capacitance` — Solves the pad capacitance of an EM-only
  QPDK double-pad transmon in COMSOL with MPh: removes the Josephson-junction mask,
  extracts the pads and ground plane, builds the air/silicon sheet model, and adds an
  Electrostatics study driving one pad with a voltage terminal and grounding the other
  pad and the ground. Reports the saved es.C11 and stored energy from a COMSOL 6.3 solve
  with a potential and field map, and labels the LC frequency formed from a chosen L_J
  an estimate rather than an eigenmode or the anharmonic f01.
- :doc:`notebooks/optimize_capacitor_optuna` — Couples Optuna optimization with the
  Palace FEM solver to optimize an interdigital capacitor towards a target capacitance.

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

***********************************
 Differentiable circuit simulation
***********************************

Differentiable circuit simulators formulate the circuit as a system of Differential
Algebraic Equations (DAEs) and solve them with automatic differentiation support. This
enables gradient-based optimization of physical parameters directly from simulation
outputs—without finite-difference approximations.

**Typical use cases:**

- Optimizing Josephson junction parameters (critical current, shunt capacitance) to meet
  target qubit frequency and anharmonicity.
- Simulating time-domain response of coupled qubit circuits to fast control pulses.
- Computing gradients of crosstalk metrics with respect to layout geometry for automated
  design refinement.
- Harmonic-balance analysis of nonlinear superconducting circuits under periodic
  microwave drives.

**Notebooks:**

- :doc:`notebooks/circulax_transmon_optimization` — Demonstrates Circulax's harmonic
  balance and transient solvers applied to a transmon qubit circuit: optimizes junction
  parameters via ``jax.grad`` and simulates crosstalk between coupled qubits, analyzing
  its sensitivity to the coupling capacitance.

**********************
 External integration
**********************

Notebooks that demonstrate driving qpdk from outside Python — useful for users whose
primary tooling lives in another environment.

**Notebooks:**

- :doc:`notebooks/matlab_integration` — Calls qpdk **directly from MATLAB** via MATLAB's
  built-in Python interface (`py.module.function(...)`). Demonstrates GDS generation,
  parameter sweeps over `resonator_frequency`, inverse design with `fzero`, and a
  parametric chip variant grid summarised in a MATLAB `table`. The notebook uses the
  MATLAB Jupyter kernel from `jupyter-matlab-proxy
  <https://github.com/mathworks/jupyter-matlab-proxy>`_.

.. _notebook-extras:

****************************
 Installing optional extras
****************************

``pip install qpdk`` gives you the layout PDK and nothing else: the analytical models,
FEM drivers, and Hamiltonian/pulse solvers all live in *extras*, declared under
``[project.optional-dependencies]`` in ``pyproject.toml``.

.. list-table::
    :header-rows: 1
    :widths: 18 40 42

    - - Extra
      - Installs
      - What it is for
    - - ``models``
      - ``sax``, ``jaxellip``, ``optax``, ``optuna``, ``gplugins[meshwell]``,
        ``scikit-rf``, ``sympy``, ``polars``, ``pandas[parquet]``
      - The analytical and S-parameter model library (``qpdk.models``), SAX circuit
        simulation, meshing, and the DataFrame display helpers. **This is the baseline
        extra — nearly every notebook needs it.**
    - - ``hfss``
      - ``pyaedt[graphics]``, ``polars``
      - Ansys AEDT drivers (HFSS, Q2D, Q3D) behind ``qpdk.simulation``. Also requires a
        local Ansys installation and a license, which are not pip-installable.
    - - ``comsol``
      - ``MPh``
      - MPh-based COMSOL geometry and study builders in ``qpdk.simulation``. Building
        and solving require a COMSOL installation and license; RF solves also need the
        RF Module. MPh itself is only a client and installs no solver.
    - - ``circulax``
      - ``circulax``, ``optax``
      - Differentiable (JAX/DAE) circuit simulation: harmonic-balance and transient
        solvers with gradients.
    - - ``netket``
      - ``netket``, ``flax``, ``optax``
      - Exact-diagonalization and variational Hamiltonian analysis with NetKet.
    - - ``pymablock``
      - ``pymablock``
      - Symbolic perturbative block-diagonalization of the qubit–resonator Hamiltonian.
    - - ``qutip``
      - ``qutip-jax``, ``qutip-qip``
      - Pulse-level, time-domain simulation of gates, leakage, and decoherence.
    - - ``scqubits``
      - ``scqubits``
      - Numerical diagonalization of transmon and transmon–resonator Hamiltonians.
    - - ``ray``
      - ``ray[default]``, ``tqdm``
      - Parallel and distributed parameter sweeps, used for Monte Carlo tolerance runs.
    - - ``graphics``
      - ``trimesh``, ``pyglet``
      - Interactive 3-D viewing of component meshes. Not needed by any notebook.
    - - ``gdsfactoryplus``
      - ``doroutes``, ``elvis-lvs``, ``httpx``, ``inspice``, ``jaxellip``,
        ``kfnetlist``, ``sax``
      - Dependencies of the GDSFactory+ v2 SDK exercised by ``just test-gfp``. Not
        needed by any notebook.

Extras compose, so install them together in one command. Always quote the brackets —
``zsh`` and ``fish`` treat them as globs.

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    # add qpdk with extras to the current project (writes pyproject.toml)
    uv add "qpdk[models]"
    uv add "qpdk[models,netket]"

    # install into the active environment without touching pyproject.toml
    uv pip install "qpdk[models,netket]"

    # in a checkout of this repository, sync the locked environment
    uv sync --extra models --extra netket
    uv sync --all-extras

    # run a notebook in a throwaway environment, no install step
    uvx --with "qpdk[models,netket]" --from jupyterlab jupyter lab

With ``pip``:

.. code-block:: bash

    pip install "qpdk[models]"
    pip install "qpdk[models,netket]"
    pip install "qpdk[circulax,comsol,graphics,hfss,models,netket,pymablock,qutip,ray,scqubits]"

.. note::

    Two notebook dependencies are deliberately *not* extras:

    - ``openvino``, used by :doc:`notebooks/jax_backend_comparison` for the NPU
      benchmark, is optional and platform-specific — install it with ``pip install
      openvino``. The notebook skips that section if it is missing.
    - MATLAB and `jupyter-matlab-proxy
      <https://github.com/mathworks/jupyter-matlab-proxy>`_, needed by
      :doc:`notebooks/matlab_integration`, are not Python packages managed by ``qpdk``.

    If you only want to reproduce the rendered documentation, ``uv sync --group docs``
    installs the ``docs`` dependency group, which already pulls in every backend the
    notebooks execute with.

.. _notebook-summary:

***************
 Summary table
***************

The **Extras** column lists the ``qpdk`` extras required to run each notebook; see
:ref:`notebook-extras` for what each one installs.

.. list-table::
    :header-rows: 1
    :widths: 31 22 23 24

    - - Notebook
      - Category
      - Key tools
      - Extras
    - - :doc:`notebooks/all_models`
      - S-parameter models
      - qpdk, JAX
      - ``models``
    - - :doc:`notebooks/circuit_simulation_demo`
      - S-parameter models
      - SAX, JAX
      - ``models``
    - - :doc:`notebooks/resonator_frequency_model`
      - S-parameter models
      - SAX
      - ``models``
    - - :doc:`notebooks/monte_carlo_fabrication_tolerance`
      - S-parameter models
      - SAX, JAX, gdsfactory
      - ``models``, ``ray``
    - - :doc:`notebooks/model_comparison_to_qucs`
      - S-parameter models
      - SAX, Qucs-S
      - ``models``
    - - :doc:`notebooks/jax_backend_comparison`
      - S-parameter models
      - SAX, JAX, OpenVINO
      - ``models`` (+ ``openvino``)
    - - :doc:`notebooks/hfss_q2d_cpw_impedance`
      - FEM electromagnetics
      - Ansys Q2D, PyAEDT
      - ``models``, ``hfss``
    - - :doc:`notebooks/hfss_eigenmode_resonator`
      - FEM electromagnetics
      - Ansys HFSS, PyAEDT
      - ``models``, ``hfss``
    - - :doc:`notebooks/hfss_driven_capacitor`
      - FEM electromagnetics
      - Ansys HFSS, PyAEDT
      - ``models``, ``hfss``
    - - :doc:`notebooks/comsol_cpw_resonator`
      - FEM electromagnetics
      - COMSOL, MPh
      - ``comsol``
    - - :doc:`notebooks/comsol_qubit_capacitance`
      - FEM electromagnetics
      - COMSOL, MPh
      - ``comsol``
    - - :doc:`notebooks/optimize_capacitor_optuna`
      - FEM optimization
      - Optuna, Palace
      - ``models``
    - - :doc:`notebooks/scqubits_parameter_calculation`
      - Hamiltonian analysis
      - scQubits
      - ``models``, ``scqubits``
    - - :doc:`notebooks/pymablock_dispersive_shift`
      - Hamiltonian analysis
      - Pymablock, SymPy
      - ``models``, ``pymablock``
    - - :doc:`notebooks/netket_transmon_design`
      - Hamiltonian analysis
      - NetKet, JAX
      - ``models``, ``netket``
    - - :doc:`notebooks/qutip_qip_pulse_simulation`
      - Pulse-level simulation
      - QuTiP-QIP, JAX
      - ``models``, ``qutip``
    - - :doc:`notebooks/circulax_transmon_optimization`
      - Differentiable circuit simulation
      - Circulax, JAX, Optax
      - ``models``, ``circulax``
    - - :doc:`notebooks/matlab_integration`
      - External integration
      - MATLAB, jupyter-matlab-proxy
      - ``models``

************
 References
************

.. bibliography::
    :filter: docname in docnames

.. toctree::
    :hidden:
    :glob:

    notebooks/*
