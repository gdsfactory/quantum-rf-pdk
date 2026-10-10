###########
 Notebooks
###########

.. meta::
    :description: Jupyter notebooks demonstrating simulation approaches for superconducting quantum devices.

Worked, executable examples that take a **qpdk** component from requirements to a
simulated result. Each notebook covers one stage of the design flow with one simulation
method, so pick the one that fits your question and tooling.

.. only:: html

    .. grid:: 1 2 3 3
        :gutter: 3
        :class-container: notebook-categories

        .. grid-item-card:: :octicon:`graph;1.5em;sd-text-primary` S-parameter models
            :link: notebooks-sparameter
            :link-type: ref
            :class-card: notebook-category

            Fast JAX/SAX circuit models for resonators, capacitors, and couplers.

            +++

            :bdg-primary-line:`6 notebooks`

        .. grid-item-card:: :octicon:`zap;1.5em;sd-text-primary` FEM electromagnetics
            :link: notebooks-fem
            :link-type: ref
            :class-card: notebook-category

            Full-wave and quasi-static solvers: HFSS, Q2D, COMSOL, Elmer, Palace.

            +++

            :bdg-primary-line:`7 notebooks`

        .. grid-item-card:: :octicon:`beaker;1.5em;sd-text-primary` Hamiltonian analysis
            :link: notebooks-hamiltonian
            :link-type: ref
            :class-card: notebook-category

            Qubit frequency, anharmonicity, and dispersive shift from the circuit
            Hamiltonian.

            +++

            :bdg-primary-line:`3 notebooks`

        .. grid-item-card:: :octicon:`pulse;1.5em;sd-text-primary` Pulse-level simulation
            :link: notebooks-pulse
            :link-type: ref
            :class-card: notebook-category

            Gate fidelity, leakage, and decoherence under realistic control pulses.

            +++

            :bdg-primary-line:`1 notebook`

        .. grid-item-card:: :octicon:`workflow;1.5em;sd-text-primary` Differentiable circuits
            :link: notebooks-differentiable
            :link-type: ref
            :class-card: notebook-category

            Harmonic-balance and transient solvers with gradients through ``jax.grad``.

            +++

            :bdg-primary-line:`1 notebook`

        .. grid-item-card:: :octicon:`plug;1.5em;sd-text-primary` External integration
            :link: notebooks-external
            :link-type: ref
            :class-card: notebook-category

            Drive qpdk from MATLAB and its RF Toolbox.

            +++

            :bdg-primary-line:`1 notebook`

.. tip::

    Each backend sits behind an optional *extra*, so ``pip install qpdk`` stays light.
    The badges on every notebook card name the extras it needs; see
    :ref:`notebook-extras` for how to install them.

*******************************************
 Where each method fits in the design flow
*******************************************

No single tool covers every physical scale of a superconducting chip, so a practical
flow combines complementary methods
:cite:`krantzQuantumEngineersGuide2019,blaisCircuitQuantumElectrodynamics2021`. Each
stage may loop back as the design is refined, and any of them can sit inside an
optimization loop (e.g. with `Optuna <https://optuna.org/>`_).

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

.. _notebooks-sparameter:

****************************
 S-parameter circuit models
****************************

Microwave components as linear, frequency-dependent scattering matrices, implemented in
`JAX <https://jax.readthedocs.io/>`_ and composed into circuits with `SAX
<https://gdsfactory.github.io/sax/>`_
:cite:`blaisCircuitQuantumElectrodynamics2021,gopplCoplanarWaveguideResonators2008a`.

.. dropdown:: Typical use cases
    :icon: light-bulb
    :color: primary

    - Choosing coplanar-waveguide (CPW) resonator, capacitor, and coupling structure
      geometries to meet target parameters.
    - Predicting resonance frequencies and quality factors of passive components.
    - Simulating complete test chips with many resonators from a gdsfactory netlist.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Model library tour
        :link: notebooks/all_models
        :link-type: doc
        :class-card: notebook-card

        Every S-parameter model in ``qpdk.models``: capacitors, inductors, waveguides,
        couplers, resonators.

        +++

        :bdg-secondary-line:`qpdk` :bdg-secondary-line:`JAX` :bdg-primary:`models`

    .. grid-item-card:: Circuit simulation with SAX
        :link: notebooks/circuit_simulation_demo
        :link-type: doc
        :class-card: notebook-card

        From single components to an assembled quarter-wave resonator.

        +++

        :bdg-secondary-line:`SAX` :bdg-secondary-line:`JAX` :bdg-primary:`models`

    .. grid-item-card:: Resonator frequency model
        :link: notebooks/resonator_frequency_model
        :link-type: doc
        :class-card: notebook-card

        Analytical resonance-frequency estimates against SAX circuit simulation.

        +++

        :bdg-secondary-line:`SAX` :bdg-primary:`models`

    .. grid-item-card:: Monte Carlo fabrication tolerance
        :link: notebooks/monte_carlo_fabrication_tolerance
        :link-type: doc
        :class-card: notebook-card

        Vary CPW width and gap on a YAML-netlist test chip and measure the spread in
        S₂₁ resonances.

        +++

        :bdg-secondary-line:`SAX` :bdg-secondary-line:`gdsfactory` :bdg-primary:`models`
        :bdg-primary:`ray`

    .. grid-item-card:: Validation against Qucs-S
        :link: notebooks/model_comparison_to_qucs
        :link-type: doc
        :class-card: notebook-card

        qpdk models checked against Qucs-S reference data for passive components.

        +++

        :bdg-secondary-line:`SAX` :bdg-secondary-line:`Qucs-S` :bdg-primary:`models`

    .. grid-item-card:: JAX backend benchmark
        :link: notebooks/jax_backend_comparison
        :link-type: doc
        :class-card: notebook-card

        SAX circuit evaluation on CPU, GPU (CUDA), and NPU (OpenVINO).

        +++

        :bdg-secondary-line:`JAX` :bdg-secondary-line:`OpenVINO` :bdg-primary:`models`

.. _notebooks-fem:

***************************************
 FEM-based electromagnetic simulations
***************************************

Finite-element and full-wave solvers discretize Maxwell's equations over the real
geometry, capturing radiation, surface currents, and substrate modes that analytical
models miss
:cite:`gopplCoplanarWaveguideResonators2008a,chenCompactInductorcapacitorResonators2023`.

.. dropdown:: Typical use cases
    :icon: light-bulb
    :color: primary

    - Extracting characteristic impedance and effective permittivity of CPW
      cross-sections.
    - Computing eigenmode frequencies and quality factors of resonators from their
      physical geometry.
    - Running driven-modal (port-based) S-parameter simulations of capacitors and other
      structures.
    - Optimizing component geometry against a target specification (e.g. a desired
      capacitance value).

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Q2D CPW impedance
        :link: notebooks/hfss_q2d_cpw_impedance
        :link-type: doc
        :class-card: notebook-card

        CPW impedance from the cross-section, against conformal-mapping estimates.

        +++

        :bdg-secondary-line:`Ansys Q2D` :bdg-primary:`models` :bdg-primary:`hfss`
        :bdg-warning:`licensed solver`

    .. grid-item-card:: HFSS eigenmode resonator
        :link: notebooks/hfss_eigenmode_resonator
        :link-type: doc
        :class-card: notebook-card

        Resonant frequencies and Q-factors of a meandering CPW resonator.

        +++

        :bdg-secondary-line:`Ansys HFSS` :bdg-primary:`models` :bdg-primary:`hfss`
        :bdg-warning:`licensed solver`

    .. grid-item-card:: HFSS driven capacitor
        :link: notebooks/hfss_driven_capacitor
        :link-type: doc
        :class-card: notebook-card

        Driven-modal S-parameters of an interdigital capacitor.

        +++

        :bdg-secondary-line:`Ansys HFSS` :bdg-primary:`models` :bdg-primary:`hfss`
        :bdg-warning:`licensed solver`

    .. grid-item-card:: Elmer capacitance extraction
        :link: notebooks/elmer_capacitance_interdigital
        :link-type: doc
        :class-card: notebook-card

        Quasi-static capacitance of an interdigital capacitor with the open-source Elmer
        solver.

        +++

        :bdg-secondary-line:`Elmer` :bdg-secondary-line:`meshwell` :bdg-primary:`models`
        :bdg-success:`open source`

    .. grid-item-card:: COMSOL CPW resonator
        :link: notebooks/comsol_cpw_resonator
        :link-type: doc
        :class-card: notebook-card

        Ported resonator layout, adaptive S-parameter sweep, and field map.

        +++

        :bdg-secondary-line:`COMSOL` :bdg-secondary-line:`MPh` :bdg-primary:`comsol`
        :bdg-warning:`licensed solver`

    .. grid-item-card:: COMSOL qubit capacitance
        :link: notebooks/comsol_qubit_capacitance
        :link-type: doc
        :class-card: notebook-card

        Electrostatic extraction of transmon pad capacitance, with field map.

        +++

        :bdg-secondary-line:`COMSOL` :bdg-secondary-line:`MPh` :bdg-primary:`comsol`
        :bdg-warning:`licensed solver`

    .. grid-item-card:: Capacitor optimization with Optuna
        :link: notebooks/optimize_capacitor_optuna
        :link-type: doc
        :class-card: notebook-card

        Optuna drives the Palace FEM solver towards a target capacitance.

        +++

        :bdg-secondary-line:`Optuna` :bdg-secondary-line:`Palace` :bdg-primary:`models`
        :bdg-success:`open source`

    .. grid-item-card:: :octicon:`link-external` More FEM and FDTD in gsim
        :link: https://gdsfactory.github.io/gsim/
        :class-card: notebook-card notebook-card-external

        The `gsim <https://gdsfactory.github.io/gsim/>`_ notebooks take a GDSFactory
        layout to a 3-D simulation with open-source solvers: Palace eigenmode and
        driven-port runs, Meep FDTD with S-parameter extraction, meshing, and field
        post-processing.

        +++

        :bdg-secondary-line:`Palace` :bdg-secondary-line:`Meep` :bdg-success:`open source`

.. dropdown:: Why the licensed-solver notebooks show a recorded run
    :icon: info
    :color: warning

    COMSOL and Ansys AEDT are not pip-installable and do not run without a license, so
    :doc:`notebooks/hfss_q2d_cpw_impedance`, :doc:`notebooks/hfss_eigenmode_resonator`,
    :doc:`notebooks/hfss_driven_capacitor`, :doc:`notebooks/comsol_cpw_resonator`, and
    :doc:`notebooks/comsol_qubit_capacitance` are published with the outputs of a real
    solver run stored in the notebook. Those stored figures and numbers are the
    artifact: the pages are not re-executed here, so what you see is the recorded run.
    The two COMSOL notebooks additionally run without a license, skipping the solver
    cells and reporting how to supply exported results, so their code can be read and
    executed up to the point where a license is needed.

.. _notebooks-hamiltonian:

**********************
 Hamiltonian analysis
**********************

Diagonalizing the circuit Hamiltonian yields qubit frequencies, anharmonicities, and
coupling strengths that feed back into the layout
:cite:`kochChargeinsensitiveQubitDesign2007a,blaisCircuitQuantumElectrodynamics2021`.

.. dropdown:: Typical use cases
    :icon: light-bulb
    :color: primary

    - Computing transmon qubit frequency (:math:`\omega_{01}`) and anharmonicity
      (:math:`\alpha`) from Josephson energy :math:`E_\text{J}` and charging energy
      :math:`E_\text{C}`.
    - Calculating the dispersive shift :math:`\chi` of a transmon–resonator system for
      readout design.
    - Translating Hamiltonian-level parameters into physical layout dimensions.

.. grid:: 1 1 3 3
    :gutter: 3

    .. grid-item-card:: scQubits parameters
        :link: notebooks/scqubits_parameter_calculation
        :link-type: doc
        :class-card: notebook-card

        Full numerical diagonalization of the transmon–resonator Hamiltonian
        :cite:`groszkowskiScqubitsPythonPackage2021`, against perturbation theory.

        +++

        :bdg-secondary-line:`scQubits` :bdg-primary:`models` :bdg-primary:`scqubits`

    .. grid-item-card:: Dispersive shift with Pymablock
        :link: notebooks/pymablock_dispersive_shift
        :link-type: doc
        :class-card: notebook-card

        Symbolic block-diagonalization :cite:`arayaDayPymablockAlgorithmPackage2025`
        mapped to layout parameters.

        +++

        :bdg-secondary-line:`Pymablock` :bdg-secondary-line:`SymPy`
        :bdg-primary:`models` :bdg-primary:`pymablock`

    .. grid-item-card:: Transmon design with NetKet
        :link: notebooks/netket_transmon_design
        :link-type: doc
        :class-card: notebook-card

        Exact diagonalization and variational methods, converted to layout dimensions.

        +++

        :bdg-secondary-line:`NetKet` :bdg-secondary-line:`JAX` :bdg-primary:`models`
        :bdg-primary:`netket`

.. _notebooks-pulse:

*************************
 Pulse-level simulations
*************************

With qubit parameters in hand, time-domain simulation under microwave control pulses
predicts gate fidelity, leakage to non-computational states, and the impact of
decoherence :cite:`liBoshlomQutipqipPulselevel2022,motzoi2009DRAGpulse`.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Transmon gates with QuTiP-QIP
        :link: notebooks/qutip_qip_pulse_simulation
        :link-type: doc
        :class-card: notebook-card

        Single-qubit (X, Y) and two-qubit (Bell-state) gates with realistic pulse
        shapes: population dynamics, leakage to higher levels, and :math:`T_1`/:math:`T_2`
        effects on fidelity.

        +++

        :bdg-secondary-line:`QuTiP-QIP` :bdg-secondary-line:`JAX` :bdg-primary:`models`
        :bdg-primary:`qutip`

.. _notebooks-differentiable:

***********************************
 Differentiable circuit simulation
***********************************

The circuit as a system of differential-algebraic equations, solved with automatic
differentiation: gradients of any simulated output with respect to physical parameters,
without finite differences.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Transmon optimization with Circulax
        :link: notebooks/circulax_transmon_optimization
        :link-type: doc
        :class-card: notebook-card

        Harmonic-balance and transient solvers on a transmon: optimize junction
        parameters with ``jax.grad`` and trace crosstalk sensitivity to the coupling
        capacitance.

        +++

        :bdg-secondary-line:`Circulax` :bdg-secondary-line:`Optax`
        :bdg-primary:`models` :bdg-primary:`circulax`

.. _notebooks-external:

**********************
 External integration
**********************

Driving qpdk from outside Python, for teams whose primary tooling lives elsewhere.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: qpdk from MATLAB
        :link: notebooks/matlab_integration
        :link-type: doc
        :class-card: notebook-card

        GDS generation, parameter sweeps, inverse design with ``fzero``, and Touchstone
        round-trips through the RF Toolbox, via MATLAB's built-in Python interface.

        +++

        :bdg-secondary-line:`MATLAB` :bdg-secondary-line:`RF Toolbox`
        :bdg-primary:`models`

.. dropdown:: How the MATLAB notebook calls qpdk
    :icon: code

    The notebook calls qpdk **directly from MATLAB** via its built-in Python interface
    (``py.module.function(...)``): GDS generation, parameter sweeps over
    ``resonator_frequency``, inverse design with ``fzero``, and a parametric chip variant
    grid summarised in a MATLAB ``table``. It then exports SAX models as Touchstone files
    with ``sax.write_sdict_touchstone`` and consumes them from MATLAB's `RF Toolbox
    <https://se.mathworks.com/help/rf/index.html>`_ as ``sparameters``/``nport`` boxes —
    Smith charts, cascades in a ``circuit``, rational fitting and transient response, and
    back into SAX with ``sax.read_sdict_touchstone``.

    Those sections need the RF Toolbox and skip themselves when it is unavailable or when
    ``QPDK_SKIP_RF_TOOLBOX`` is set; the rendered page shows a saved execution that had
    the toolbox available. The notebook uses the MATLAB Jupyter kernel from
    `jupyter-matlab-proxy <https://github.com/mathworks/jupyter-matlab-proxy>`_.

.. _notebook-extras:

****************************
 Installing optional extras
****************************

``pip install qpdk`` gives you the layout PDK and nothing else: the analytical models,
FEM drivers, and Hamiltonian/pulse solvers all live in *extras*, declared under
``[project.optional-dependencies]`` in ``pyproject.toml``. Extras compose, so install
them together in one command, and always quote the brackets — ``zsh`` and ``fish`` treat
them as globs.

.. tab-set::
    :sync-group: installer

    .. tab-item:: uv
        :sync: uv

        .. code-block:: bash

            # add qpdk with extras to the current project (writes pyproject.toml)
            uv add "qpdk[models,netket]"

            # install into the active environment without touching pyproject.toml
            uv pip install "qpdk[models,netket]"

            # run a notebook in a throwaway environment, no install step
            uvx --with "qpdk[models,netket]" --from jupyterlab jupyter lab

    .. tab-item:: pip
        :sync: pip

        .. code-block:: bash

            pip install "qpdk[models]"
            pip install "qpdk[models,netket]"
            pip install "qpdk[circulax,comsol,graphics,hfss,models,netket,pymablock,qutip,ray,scqubits]"

    .. tab-item:: From a checkout
        :sync: checkout

        .. code-block:: bash

            # sync the locked environment with the extras you need
            uv sync --extra models --extra netket
            uv sync --all-extras

            # or reproduce the rendered documentation, every backend included
            uv sync --group docs

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

.. note::

    Two notebook dependencies are deliberately *not* extras:

    - ``openvino``, used by :doc:`notebooks/jax_backend_comparison` for the NPU
      benchmark, is optional and platform-specific — install it with ``pip install
      openvino``. The notebook skips that section if it is missing.
    - MATLAB and `jupyter-matlab-proxy
      <https://github.com/mathworks/jupyter-matlab-proxy>`_, needed by
      :doc:`notebooks/matlab_integration`, are not Python packages managed by ``qpdk``.

    The Elmer notebook also needs ``gplugins[elmer]`` installed from Git until the next
    gplugins release.

.. _notebook-summary:

***************
 Summary table
***************

Every notebook at a glance; the **Extras** column links back to :ref:`notebook-extras`.

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
    - - :doc:`notebooks/elmer_capacitance_interdigital`
      - FEM electromagnetics
      - Elmer, meshwell
      - ``models`` (+ ``gplugins[elmer]`` from Git until release)
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
      - MATLAB, RF Toolbox (optional), jupyter-matlab-proxy
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
