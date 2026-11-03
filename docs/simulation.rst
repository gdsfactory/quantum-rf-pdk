############
 Simulation
############

.. meta::
    :description: Circuit-model hierarchy and field-solver simulation helpers for QPDK components.

*******************************
 Circuit-level model hierarchy
*******************************

SAX simulations expand unmodeled assemblies until they reach cells with declared models.
Model declarations therefore define hierarchy boundaries instead of relying on a fixed
depth.

Coupled resonators declare their own models so the coupling is preserved. Chip
assemblies remain unmodeled and expand to these boundaries and other modeled primitives.

For new components, attach the compact model through ``schematic_function`` and keep its
``port_order`` consistent with the layout ports.

.. automodule:: qpdk.simulation

*****
 FEM
*****

.. automodule:: qpdk.simulation.fem
    :members:
    :show-inheritance:

.. _subtractive-to-positive:

****************************************
 From etch layers to simulation regions
****************************************

The PDK and a volumetric solver describe metal in opposite ways, and every FEM workflow
in qpdk starts by converting between them.

qpdk draws metal **subtractively**: ``M1_ETCH`` marks where metal is removed from a full
sheet, and additive shapes on ``M1_DRAW`` punch back through it, so the conductor is
``SIM_AREA - (M1_ETCH - M1_DRAW)``. This matches ``get_layer_stack()``'s M1 definition
and is how the transmon pads are drawn.

A volumetric solver needs the **positive** form instead: explicit conductor and
dielectric bodies. :func:`~qpdk.simulation.to_fem_regions` performs that conversion and
copies conductor, substrate and vacuum onto dedicated GDS layers that the FEM layer
stack maps to materials. The zero-thickness conductor sheet becomes a
perfect-electric-conductor boundary.

Two consequences are worth seeing rather than reading: everything the etch mask covers
turns into vacuum, and an additive shape sitting inside an etched gap survives as an
*isolated* metal island.

.. plot::
    :include-source: false

    import klayout.db as kdb
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    import gdsfactory as gf

    def region_patch(region, **kwargs):
        dbu = gf.kcl.dbu
        vertices, codes = [], []
        for polygon in region.merged().each():
            loops = [polygon.each_point_hull()]
            loops += [polygon.each_point_hole(i) for i in range(polygon.holes())]
            for loop in loops:
                points = [(pt.x * dbu, pt.y * dbu) for pt in loop]
                vertices += [*points, points[0]]
                codes += [MplPath.MOVETO,
                          *[MplPath.LINETO] * (len(points) - 1),
                          MplPath.CLOSEPOLY]
        return PathPatch(MplPath(vertices, codes), **kwargs)

    def box(x0, y0, x1, y1):
        return kdb.Region(kdb.DBox(x0, y0, x1, y1).to_itype(gf.kcl.dbu))

    sim = box(0, 0, 100, 60)
    etch = box(10, 10, 90, 22) + box(10, 38, 90, 50) + box(40, 22, 60, 38)
    draw = box(46, 26, 54, 34)
    conductor = (sim - etch) | (draw & sim)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    axes[0].add_patch(region_patch(sim, facecolor="none", edgecolor="0.45",
                                   linestyle="--", linewidth=1.2,
                                   label="SIM_AREA"))
    axes[0].add_patch(region_patch(etch, facecolor="#d1495b", alpha=0.75,
                                   edgecolor="none", label="M1_ETCH (removed)"))
    axes[0].add_patch(region_patch(draw, facecolor="#1b7f9e", edgecolor="none",
                                   label="M1_DRAW (added back)"))
    axes[0].set_title("Subtractive: what qpdk draws")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=3,
                   frameon=False, fontsize=8)
    axes[1].add_patch(region_patch(conductor, facecolor="#c9a227",
                                   edgecolor="black", linewidth=0.6))
    axes[1].set_title("Positive: what the solver meshes")
    for ax in axes:
        ax.set_xlim(-5, 105)
        ax.set_ylim(-8, 65)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()

The same rule applies per metal level for flip-chip stacks, where
:func:`~qpdk.simulation.to_flip_chip_regions` converts both levels and copies the indium
bump layer to its own region.

***************************************************
 Batched layout optimisation and cluster execution
***************************************************

The Palace optimisation notebook defines its transmon layout and loss objective. The
generic runner keeps trials in flight and sends that callable to Slurm or Ray workers.
gsim provides single-node local execution and text-result loading; these helpers add
multi-node MPI launching, domain-energy requests and port-connectivity checks.

.. automodule:: qpdk.simulation.palace_run
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.study
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.trial
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.cluster
    :members:
    :show-inheritance:

********
 Common
********

.. automodule:: qpdk.simulation.layout
    :members:

*************************
 Ansys HFSS, Q3D and Q2D
*************************

QPDK does not wrap Ansys AEDT. Use :mod:`gplugins.ansys` (installed by the ``hfss``
extra) directly. Pass ``layer_stack=qpdk.tech.LAYER_STACK`` explicitly to its ``HFSS``,
``Q3D`` and ``Q2D`` classes. The PDK default is :data:`~qpdk.tech.LAYER_STACK_3D`, which
adds an implicit ground plane for mask previews and must not be used for simulations.
Pass ``material_properties=qpdk.tech.material_properties`` to use QPDK's material table,
and prepare components with :func:`~qpdk.simulation.layout.prepare_metal_layout`. The
HFSS notebooks show the full workflow.

********
 VACASK
********

The external simulator runs :doc:`notebooks/vacask_lumped_parametric_amplifier` and
:doc:`notebooks/vacask_cpw_parametric_amplifier` with released VACASK. Native
cold-network and subprocess checks also use the release. The TWPA notebook,
:doc:`notebooks/josephsoncircuits_twpa`, uses JosephsonCircuits.jl instead.

.. automodule:: qpdk.simulation.vacask
    :members:
    :imported-members:

********
 COMSOL
********

Build COMSOL models with :mod:`gplugins.comsol` directly: its ``COMSOL`` class
subclasses :class:`mph.Model`, holds the layout, and chains the sheet or metal builder,
the RF or electrostatic study, and the mesh helpers. Plotting and result helpers live in
``gplugins.comsol.plotting`` and ``gplugins.comsol.results``.

:mod:`qpdk.simulation.comsol` adds only what is specific to QPDK:
:func:`~qpdk.simulation.comsol.prepare_comsol_layout`, which inverts the QPDK
``M1_ETCH`` mask into metal with a ground margin before the gplugins extraction, and the
technology permittivities :data:`~qpdk.simulation.comsol.SILICON_RELATIVE_PERMITTIVITY`
and :data:`~qpdk.simulation.comsol.AIR_RELATIVE_PERMITTIVITY`, which are passed
explicitly to ``COMSOL.create_sheet``.

The module needs the ``comsol`` extra, which installs ``gplugins[comsol]``; building and
solving a model also needs COMSOL. :mod:`qpdk.simulation` does not import it, so the
rest of the package works without the extra.

.. automodule:: qpdk.simulation.comsol
    :members:

**************
 FEM datasets
**************

A dataset is nothing but its files: Parquet parts, or a Delta Lake table, each carrying
the dataset metadata (units, terminals, quantities, provenance) inside it. Lookups read
the table with Polars and return jittable SAX models. Requires the ``models`` extra. See
:doc:`notebooks/fem_dataset_lookup` for a walkthrough.

Curated datasets ship in ``qpdk/models/datasets/data/`` as Parquet in Git LFS. From a
checkout, fetch only these with ``git lfs pull
--include="qpdk/models/datasets/data/**"``. Wheels on PyPI ship the resolved Parquet
files, so installed users need no Git LFS.

Results can also live outside Git, in a `Delta Lake <https://delta.io>`_ table on a
cloud bucket (``gs://``, ``s3://``, ``az://``) or a local path, optionally pinned to a
table version. Credentials are passed when opening the dataset and never stored in it.
Requires the ``delta`` extra.

.. code-block:: python

    from qpdk.models.datasets import Dataset

    options = {"google_service_account": "key.json"}
    dataset = Dataset("gs://my-bucket/plate_capacitor", delta=True, storage_options=options)
    dataset.append(new_rows)  # one atomic Delta commit

    pinned = Dataset(
        "gs://my-bucket/plate_capacitor", delta=True, version=3, storage_options=options
    )

Datasets and models
===================

Every model below takes geometry in µm and frequency ``f`` in Hz. Values outside the
stored grid return NaN, an omitted geometry argument falls back to the model's stored
default, usually the middle grid coordinate, and discrete variants such as the turn
count are fixed when the model is created. The generation recipe for each dataset lives
in ``qpdk/models/datasets/recipes/``.

.. list-table::
    :header-rows: 1
    :widths: 22 16 24 18 20

    - - Model
      - Dataset
      - Gives you
      - Geometry inputs
      - Notes
    - - ``flipmon_unbonded_model()``
      - ``flipmon_unbonded_palace``
      - Maxwell matrix, ports ``inner_pad``, ``outer_ring``, ``top_pad``
      - ``chip_gap``, ``alignment``, ``inner_radius``, ``top_radius``
      - Separate electrodes between two 300 µm chips, no junction or bonding bump
    - - ``flipmon_ideal_bond_model(inductance, junction_capacitance=0.0)``
      - ``flipmon_unbonded_palace``
      - Ports ``inner_pad`` and ``outer_ring`` with a linearized junction
      - same as above
      - Ideal wire from the inner pad to the top pad; no bump field or inductance
    - - ``double_pad_transmon_capacitance()``
      - ``double_pad_transmon_palace``
      - Differential pad capacitance :math:`C_\text{diff}` in F
      - ``pad_width``, ``pad_height``, ``pad_gap``, ``ground_clearance``
      - Junction excluded; add :math:`C_\text{J}` in parallel before computing
        :math:`E_\text{C} = e^2/[2(C_\text{diff}+C_\text{J})]`
    - - ``double_pad_transmon_fem_model(inductance, junction_capacitance=0.0)``
      - ``double_pad_transmon_palace``
      - Ports ``left_pad`` and ``right_pad`` with a linearized junction
      - same as above
      - Both ports use the coupon's finite 300 µm ground frame; no readout, package or
        backside ground
    - - ``gatemon_capacitances()``
      - ``gatemon_palace``
      - :math:`C_\Sigma` in F, which sets :math:`E_\text{C}`; :math:`C_\text{g}`, which
        sets :math:`n_\text{g} = C_\text{g}V_\text{g}/2e`; :math:`C_\text{c}`
      - ``island_size``, ``pocket_gap``, ``pocket_gap_top``, ``gate_tip_gap``
      - Only M1 sheets, so :math:`C_\text{g}` is the M1 feed coupling, not the nanotube
        gate lever arm. :math:`C_\Sigma` already includes both branches
    - - ``gatemon_fem_model(delta_ghz=50.0, transmissions=(0.8,))``
      - ``gatemon_palace``
      - Ports ``island``, ``gate``, ``readout``, with the ABS junction linearized
        between island and ground
      - same as above, plus ``delta_ghz``, ``transmissions``, ``junction_capacitance``
      - Electrostatic solve, since :math:`E_\text{J}` is gate-tuned rather than set by
        the geometry. ``readout`` is open, so do not add its self-capacitance to a
        distributed resonator
    - - ``flipmon_readout_coupon_model()``
      - ``flipmon_readout_coupon_palace``
      - Ports ``island_a``, ``island_b``, ``readout``, ``feedline``, ``top_pad``
      - ``pad_length``, ``pad_offset``, ``feedline_gap``, ``coupling_length``
      - Feedline cut open and the whole conductor at one potential, so this is not a
        resonator mode. Needs RF validation
    - - ``transmon_readout_coupon_model()``
      - ``transmon_readout_coupon_palace``
      - Ports ``island_a``, ``island_b``, ``readout``, ``feedline``
      - same as above
      - Same caveat. Feedline gaps of 14 to 20 µm retain the ground strip
    - - ``plate_capacitor()``, registered as ``qpdk.models.plate_capacitor``
      - ``coupling_pad_palace``
      - Ports ``o1`` and ``o2``, with both pad-to-ground branches and the coupling
        between the pads
      - ``length``, ``width``, ``gap``, ``offset``
      - Matches ``qpdk.cells.plate_capacitor`` and its finite ground frame. Connect the
        pair directly without adding the ground branches again.
        ``plate_capacitor_analytical`` is the closed form without them
    - - ``meander_inductor_ground_capacitance(turn_length=200, wire_width=2, wire_gap=4,
        n_turns=3)``
      - ``meander_ground_capacitance_palace``
      - Winding-to-ground capacitance in F, from 54 extractions
      - ``turn_length``, ``wire_width``, ``wire_gap``
      - Both winding ends at one potential, so this is common mode only: no differential
        parasitics, inductance or self-resonance
    - - ``meander_inductor_fullwave_model(n_turns=5)``
      - ``meander_sparameters_palace``
      - Complex S-parameters on ports ``o1`` and ``o2``, 0.5 to 20 GHz, 50 ohm, no
        de-embedding, from 243 sweeps
      - ``turn_length`` (100 to 300), ``wire_width`` (1 to 2), ``wire_gap`` (2 to 8)
      - Fixed 0.4 pH per square on both winding and frame, no material loss, radiation
        allowed. Use ``meander_inductor`` for an analytical model with adjustable film
        properties
    - - ``qpdk.models.couplers.cpw_coupling_model()``
      - ``cpw_coupling_ground_strip_palace`` by default, ``cpw_coupling_palace`` for a
        fully etched gap
      - Four-port quasi-TEM coupled section of a uniform slice
      - ``width``, ``cpw_gap``, and ``gap`` or ``ground_strip_width``
      - Interpolates logarithmically once a ground strip is present.
        ``coupler_straight`` uses the same lookup
    - - ``drum_capacitances()``
      - ``drum_lc_resonator_palace``
      - Coupling :math:`C_\text{c}`, parasitic :math:`C_\text{p}` and drum
        :math:`C_\text{d}` in F, from 54 electrostatic solves
      - ``coupling_gap``, ``coupling_pad_size``, ``radius``, ``turn_length``
      - The other cell arguments are fixed at their defaults, including 75 meander runs.
        Values outside the grid return NaN
    - - ``drum_lc_resonator_fem_model()``
      - ``drum_lc_resonator_palace``
      - Two-port lumped SAX model with the extracted capacitances in place of the
        analytical ones
      - same as above, plus ``gap`` and ``sheet_inductance``
      - The extracted :math:`C_\text{d}` replaces the analytical value at the stack gap,
        so ``gap`` still moves only the analytical parallel-plate term

``flipmon_ideal_bond_model`` and ``double_pad_transmon_fem_model`` build on
:func:`~qpdk.models.datasets.junction_model`, which documents the circuit convention.
The winding and ground frame of the full-wave coupon share its sheet inductance, so the
terminal response includes both. For a straight strip with nearly uniform current, sheet
inductance times length over width estimates only the strip's contribution; compare
against a control solve with a PEC ground return, and keep the frame's sheet inductance
on when generating a device dataset.

For the hybrid process, use ``qpdk.tech.LAYER_STACK_GATEMON`` and measure the flake
thickness before meshing a local electrostatics solve.

Gatemon validation
==================

The bundled ``gatemon_palace`` dataset holds all 144 geometries and 3024 records,
including the complete three-terminal Maxwell matrices and solver diagnostics. A
representative convergence study re-solved four M1 island geometries on a finer 0.6 µm /
40 µm mesh with fourth-order elements and a larger domain: :math:`C_\Sigma`,
:math:`C_\text{g}` and :math:`C_\text{c}` moved by at most 0.74%, inside their 1%, 5%
and 1% tolerances. Three independent holdout solves interpolate to within 2.61%, inside
the 3%, 10% and 5% limits. Every matrix gives a finite, reciprocal, lossless response
over 1 to 10 GHz. At the default geometry the M1 values are 51.93 fF, 1.582 fF and 5.731
fF, for the finite extraction coupon without the hybrid process's local capacitances.
Full-grid coverage and interpolation are checked separately.

Both reports sit in each Parquet part's ``qpdk.dataset`` metadata under
``provenance.validation``, so read them through the dataset API:

.. code-block:: python

    from qpdk.models.datasets import Dataset

    validation = Dataset("gatemon_palace").metadata.provenance["validation"]
    convergence = validation["convergence"]
    verification = validation["verification"]

The verification report hashes the result rows independently of the metadata. Its source
provenance retains the original solver-file hashes, while Git LFS identifies the
published Parquet file including the embedded reports.

The gatemon test chip
=====================

The ``gatemon_test_chip`` sample follows Fig. 1a of
:cite:`riechertCarbonNanotubeGatemon2025`: two independent transmission resonators with
series coupling capacitors at both ends, both targeting device A's 5.056 GHz bare-cavity
frequency. A capacitor-to-capacitor CPW path is a 600 µm coupling straight, a 9946.135
µm meander and a 1400 µm tail, giving about 5.14 fF of end capacitance each. The
capacitor geometry and the 10 µm / 6 µm CPW width and gap are QPDK design choices, not
published dimensions. The targets are :math:`E_\text{C}/h = 260\,\text{MHz}` and
:math:`g \approx 50\,\text{MHz}` for device A, and 330 MHz and about 120 MHz for device
B, which stay targets for electromagnetic calibration of the island and readout gap.

``gatemon_coupler`` models the coupling CPW with a lumped 5.4 fF branch to the island,
its third port referencing the island edge across the etched gap without adding metal
there. ``gatemon_island`` adds the grounded Josephson inductance, a 49.1 fF island
shunt, 20 fF island-to-gate and 35 fF gate-to-ground. Each contact overlaps the top gate
by 18 µm², and 30 nm hBN of relative permittivity 3.4 puts a parallel-plate contact near
18 fF. These are nominal hybrid estimates that still need a measured flake thickness and
a full-stack extraction.

The junction is linearized, so the simulation gives the qubit's loading and the
gate-tuned readout response but not the nonlinear spectrum or decoherence. The gate
ports use a 50 ohm reference, so a matched gate loads the qubit; the paper's bias tee
grounds its RF branch and acts as a reflective filter, so use an ideal RF-grounded gate
for frequency and coupling checks. The nonlinear ABS Hamiltonian instead gives an
isolated :math:`f_{01}` near 4.447 GHz for the nominal charging energy, and reproducing
the paper's quantum-dot parity transitions needs its extended Hamiltonian
:cite:`riechertCarbonNanotubeGatemon2025`. Broad sweeps can miss the maxima of narrow
resonances, so use a local frequency sweep before interpreting their heights.

Drum LC resonator
=================

``drum_capacitances`` returns, in F, the coupling :math:`C_\text{c}`, the parasitic
:math:`C_\text{p}` and the drum capacitance :math:`C_\text{d}` of a
``drum_lc_resonator`` layout, from ``coupling_gap``, ``coupling_pad_size``, ``radius``
and ``turn_length`` in um. ``drum_lc_resonator_fem_model()`` returns the lumped SAX
model with these values in place of its ``coupling_capacitance`` and
``parasitic_capacitance`` inputs. The extracted :math:`C_\text{d}` replaces the
analytical one at the stack gap, and the ``gap`` argument still tunes the analytical
parallel-plate term. The ``drum_lc_resonator_palace`` dataset is not generated yet;
until it is, both functions return NaN and warn once.

The recipe ``qpdk/models/datasets/recipes/drum.py`` is a 3D electrostatic solve of the
whole cell with its feedline in a finite coplanar ground. Its terminals are the node
(pad, wiring and bottom electrode), the ``VG_TOP`` membrane one vacuum gap above it, the
feedline cut open beyond the ports, and five blocks of meander runs separated by 1 um
slits. The recipe weights the Maxwell matrix with the node at 1, the membrane and feed
at 0 and each meander block at its mean position along the wire: the feed and membrane
charges give :math:`C_\text{c}` and :math:`C_\text{d}`, and the remaining energy gives
:math:`C_\text{p}`. The full matrix is stored so other weightings can be applied. A 2D
sheet solve cannot hold the membrane, so the recipe is 3D. Conductors are zero-thickness
sheets by default; ``conductors="solid"`` extrudes M1 and the membrane. A gmsh box field
sets 0.07 um elements in the 70 nm gap. The 1 um near-metal and 60 um far-field sizes,
600 um padding and third-order basis functions are starting guesses.

Run ``uv run --script qpdk/models/datasets/recipes/drum.py convergence`` before the
54-point grid to compare gap and near-metal mesh, element order, padding and conductor
model at the default geometry. The linear voltage profile is a lumped approximation: the
default meander is about 23 mm of wire, longer than the guided wavelength near 7 GHz, so
the driven meander recipe is the reference for its distributed response.

Generating Palace datasets
==========================

The bundled ``plate_capacitor_palace`` and ``cpw_coupling_palace`` datasets contain real
Palace electrostatic extractions. Their standalone scripts in
``qpdk/models/datasets/recipes/`` define the grid, settings, metadata and solver
function. Inline dependencies select the Python 3.12 environment required by gsim;
generation scripts are excluded from the installed package.

Preview or run the bundled 27-point sweep from a checkout:

::

    uv run --script qpdk/models/datasets/recipes/plate_capacitor.py --dry-run
    uv run --script qpdk/models/datasets/recipes/plate_capacitor.py --processes 4
    uv run --script qpdk/models/datasets/recipes/plate_capacitor.py --sif /path/to/palace.sif

Copy the script beside the original in ``qpdk/models/datasets/recipes/`` and edit
``GRID`` and ``SETTINGS`` for another experiment. Meshwell meshes labelled conductor
sheets and the surrounding air and silicon volumes. The shared experiment helper uses
gsim to write the Palace configuration, run ``ElectrostaticSim`` and load capacitance
matrices. Lookup models need none of these simulation dependencies.

Each geometry retains its inputs, mesh, config, solver log and results under
``--workdir``. Rerunning reuses matching completed solves. Extend the grid while
retaining the work directory to solve only new points. ``--output`` is replaced only
after the sweep succeeds. The output Parquet files contain all dataset metadata, so
lookups need no generator or solver. Container runs fingerprint the complete SIF; native
runs fingerprint only the supplied executable.

Set ``save_fields=True`` for ParaView outputs. Compare representative geometries at
finer ``near_mesh`` sizes and larger ``domain_pad`` before relying on small differences.
Every solve stores ``solver_relative_residual``, ``solver_iterations`` and
``fem_error_indicator_norm`` alongside its capacitance. Missing diagnostics and
unconverged terminal solves prevent publication. The FEM indicator is Palace's
energy-normalized recovered-flux estimate, not a capacitance error bound. Scan and rank
it to select refinement candidates. The CPW script also refines every geometry until all
raw matrix entries change by at most ``--mesh-tolerance`` (default 1%) and the identical
traces agree within 1%. It stores ``mesh_relative_change``, ``mesh_refinement_level``
and the accepted ``mesh_near_size`` / ``mesh_far_size`` in SI units. Exhausting
``--max-refinements`` blocks publication. This refinement measures mesh sensitivity
only; check larger domains and held-out-geometry interpolation separately. The notebook
demonstrates a lazy diagnostic scan and outlier flag.

The plate example has a separate ground frame. The CPW example is a uniform slice of two
identical traces with outer ground rails, sweeping trace width, outer slot width and
inter-trace gap. Use Palace 0.17 or newer for its two-dimensional transverse mesh.
``Model.Lc`` sets the implicit depth to ``slice_length_um``, so the stored capacitances
remain in F. Divide by the depth in meters for capacitance per length.

Select ``--topology as-drawn`` to retain the ground strip of width ``max(gap - 2 *
cpw_gap, 0)`` left between the component's CPW etch masks. This topology produces
``cpw_coupling_ground_strip_palace``, used by ``coupler_straight`` and the default
distributed model. The default ``fully-etched`` experiment produces
``cpw_coupling_palace`` for comparison with the unshielded analytical ECCPW formula. The
two geometries agree when the CPW slots touch or overlap. Above that gap, their
difference measures the shielding from the intervening ground strip. The grounded sweep
uses ``ground_strip_width = gap - 2 * cpw_gap`` as its third axis, with logarithmic
sampling from 0.01 to 248 µm. The CPW models interpolate coordinates and capacitance
magnitudes logarithmically in this regime and use the fully etched grid where the slots
touch or overlap. Positive strips narrower than the sampled minimum return NaN.
``cpw_coupling_model()`` converts its capacitance lookup into a jittable four-port
quasi-TEM SAX model; see the notebook for geometry heatmaps and S-parameters.

Both scripts support disjoint rectangular shards with ``--shard INDEX --shards COUNT``.
Use separate output and work directories for each worker, then merge after all succeed.
For the smaller CPW development grid, use two workers with the generic Slurm helper:

::

    generator=qpdk/models/datasets/recipes/cpw_coupling.py
    sbatch --array=0-1 --cpus-per-task=4 qpdk/models/datasets/recipes/slurm_array.sh "$generator" build/cpw-shards build/cpw-runs --sif /path/to/palace.sif
    shards=()
    for i in {0..1}; do shards+=(--merge-shards "build/cpw-shards/shard-$i"); done
    uv run --script "$generator" "${shards[@]}" --output build/datasets/cpw_coupling_palace

Supply your scheduler's partition and resource options to ``sbatch``. Merging checks
matching provenance, successful results and exact coverage before publishing. Keep the
experiment, runtime and array size fixed when resuming.

Without Slurm, run the generator directly to solve the full grid sequentially. The
notebook also shows a local shard loop with the same merge step. Each Palace solve
defaults to four MPI ranks; the Slurm helper reserves four CPUs and sets one thread per
rank. Supply the array range explicitly at submission, starting at zero.

``qpdk.models.datasets.s_parameters_model`` loads a stored complex scattering matrix as
a jittable SAX model with any number of labelled ports. ``capacitance_model`` converts a
real Maxwell matrix into a lumped N-port model, including capacitances to ground.
Distributed CPW physics lives in ``qpdk.models.couplers.cpw_coupling_model``;
``coupler_straight`` uses the same Palace mutual-capacitance lookup through
``cpw_cpw_coupling_capacitance``.

For a full-wave dataset, copy ``meander_inductor.py`` and use the shared
``_driven.Recipe`` and ``_driven.run_cli`` helpers. Supply your geometry grid,
frequencies in Hz and port names in matrix order. The ``prepare`` function builds and
meshes a driven simulation, sets an independent excitation at each port, writes the
config, and returns the simulation. Record the reference impedances, reference planes
and materials in the recipe provenance.

The helper runs the complete frequency sweep once per geometry, retains all complex
matrix entries and resumes completed solves. Geometry shards retain the full frequency
axis. Missing ports, frequencies or failed solves cannot be merged into a complete
dataset. Passivity and optional reciprocity checks apply to any number of ports. These
checks do not replace mesh refinement or held-out interpolation checks. For
nonreciprocal media, set ``reciprocal=False`` in the recipe.

Load the results with ``s_parameters_model(path, **variants)``. The returned SAX model
interpolates frequency and continuous geometry axes together and returns NaN outside the
stored domain. Select discrete variants when loading. Complex linear interpolation can
introduce apparent loss between sample points, so check withheld frequencies near
resonances before choosing the frequency spacing.

Dataset API
===========

.. automodule:: qpdk.models.datasets

.. automodule:: qpdk.models.datasets.metadata
    :members:

.. automodule:: qpdk.models.datasets.table
    :members:

.. automodule:: qpdk.models.datasets.generate
    :members:

.. automodule:: qpdk.models.datasets.store
    :members:

.. automodule:: qpdk.models.datasets.interpolation
    :members:

.. automodule:: qpdk.models.datasets.capacitance
    :members:

.. automodule:: qpdk.models.datasets.models
    :members:
