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

COMSOL models are built with :mod:`gplugins.comsol` directly: its ``COMSOL`` class
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

Reusable FEM extraction results, read with Polars and looked up in jittable SAX models.
A dataset is nothing but its files: Parquet parts, or a Delta Lake table, each carrying
the dataset metadata (units, terminals, quantities, provenance) inside it. Requires the
``models`` extra. See :doc:`notebooks/fem_dataset_lookup`.

Curated datasets ship in ``qpdk/models/datasets/data/`` as Parquet in Git LFS. From a
checkout, fetch only these with ``git lfs pull
--include="qpdk/models/datasets/data/**"``. Wheels on PyPI ship the resolved Parquet
files, so installed users need no Git LFS.

``flipmon_unbonded_model()`` returns a three-port SAX model for separate inner, ring and
top electrodes between two silicon chips. The returned model accepts ``chip_gap``,
``alignment``, ``inner_radius`` and ``top_radius`` in um, and frequency ``f`` in Hz.
``flipmon_ideal_bond_model(inductance=10e-9)`` joins the inner and top electrodes with
an ideal wire and adds a linearized Josephson junction. This reduction omits finite bump
fields and inductance. The generation recipe is
``qpdk/models/datasets/recipes/flipmon.py``.

``double_pad_transmon_capacitance`` looks up the differential pad capacitance in F,
including the two pad-to-ground branches. With junction capacitance :math:`C_\text{J}`,
the charging frequency is :math:`E_\text{C}/h = e^2/[2h(C_\text{diff}+C_\text{J})]` in
Hz. For a linearized circuit, create ``double_pad_transmon_fem_model(inductance=10e-9,
junction_capacitance=2e-15)`` and pass geometry dimensions in um to the returned SAX
model. Both ports reference the coupon's finite ground frame. The dataset omits the
readout, package and Josephson junction; its recipe is
``qpdk/models/datasets/recipes/double_pad_transmon.py``.

``gatemon_capacitances`` looks up the island's total capacitance :math:`C_\Sigma`, its
gate capacitance :math:`C_\text{g}` and its readout coupling :math:`C_\text{c}` in F,
from ``island_size``, ``pocket_gap``, ``pocket_gap_top`` and ``gate_tip_gap`` in um.
:math:`C_\Sigma` sets :math:`E_\text{C}` for ``gatemon_frequency_and_anharmonicity`` and
:math:`C_\text{g}` the induced island offset charge :math:`n_\text{g} = C_\text{g}
V_\text{g}/2e`. ``gatemon_fem_model(delta_ghz=50.0, transmissions=(0.8,))`` returns a
three-port SAX model (``island``, ``gate``, ``readout``) with the Andreev-bound-state
junction linearized between the island and ground. The readout port is an open CPW
segment, so do not add its self-capacitance to a distributed resonator. The recipe is
``qpdk/models/datasets/recipes/gatemon.py``; it is an electrostatic solve because the
device is lumped and :math:`E_\text{J}` is gate-tuned rather than fixed by the geometry.
The current recipe includes only M1 sheets. It omits the Nb/Au contacts, local top gate,
hBN and thermal oxide, so its gate capacitance describes the M1 feed's coupling to the
island, not the nanotube's gate lever arm. Use ``qpdk.tech.LAYER_STACK_GATEMON`` for the
hybrid process and measure the flake thickness before constructing a mesh for local
electrostatics.

The gatemon recipe uses 0.8 µm near-metal and 60 µm far-field mesh sizes, a 1200 µm
domain padding and height, third-order elements and a :math:`10^{-9}` linear solver
tolerance. The embedded representative convergence study records four geometries with a
finer 0.6 µm / 40 µm mesh, fourth-order elements and a larger 1800 µm domain. The
largest relative changes in :math:`C_\Sigma`, :math:`C_\text{g}` and :math:`C_\text{c}`
were 0.652%, 0.736% and 0.476%, within the respective 1%, 5% and 1% tolerances. These
checks apply to the M1 island row. Full-grid coverage and interpolation are validated
separately.

The bundled ``gatemon_palace`` dataset contains all 144 geometries and 3024 records,
including the complete three-terminal Maxwell matrices and solver diagnostics. The
embedded production verification report includes three independent holdout solves. Their
largest interpolation errors in :math:`C_\Sigma`, :math:`C_\text{g}` and
:math:`C_\text{c}` are 1.65%, 2.61% and 1.39%, below the respective 3%, 10% and 5%
limits. All matrices produce finite, reciprocal, lossless SAX responses over 1 to 10
GHz. At the default geometry, the M1 values are 51.93 fF, 1.582 fF and 5.731 fF. These
values describe the finite extraction coupon and do not include the hybrid process's
local capacitances.

Both reports are stored in each Parquet part's ``qpdk.dataset`` metadata under
``provenance.validation``. Read them through the dataset API without separate files:

.. code-block:: python

    from qpdk.models.datasets import Dataset

    validation = Dataset("gatemon_palace").metadata.provenance["validation"]
    convergence = validation["convergence"]
    verification = validation["verification"]

The verification report hashes the result rows independently of the metadata. Its source
provenance retains the original solver-file hashes, while Git LFS identifies the
published Parquet file including the embedded reports.

The ``gatemon_test_chip`` sample uses two independent transmission resonators with
series coupling capacitors at both ends, as drawn in Fig. 1a of
:cite:`riechertCarbonNanotubeGatemon2025`. Both channels target device A's reported
bare-cavity frequency of 5.056 GHz. The paper does not specify an exact resonance
frequency for device B, so the second channel is an A-style companion.

Each capacitor-to-capacitor CPW path contains a 600 µm coupling straight, a 9946.135 µm
meander and a 1400 µm tail. The lengths include the loading of the default interdigital
capacitors and the nominal island with its Josephson junction open. The estimated end
capacitance is about 5.14 fF each. The capacitor geometry and 10 µm / 6 µm CPW width/gap
are QPDK design choices, not dimensions reported by the paper.

The reported device A targets are :math:`E_\text{C}/h = 260\,\text{MHz}` and :math:`g
\approx 50\,\text{MHz}`. Device B has 330 MHz and about 120 MHz, respectively. These
remain targets for electromagnetic calibration of the island and readout gap. The
sample's analytical frequency check excludes the package and the hybrid process's
detailed dielectric topography.

The sample's ``gatemon_coupler`` is a three-port model of the complete coupling CPW and
a lumped 5.4 fF branch to the island. Its third layout port references the island edge
across the etched gap, without adding metal there. The schematic wires this port to the
gatemon's ``readout`` terminal. The ``gatemon_island`` model adds a grounded Josephson
inductance, a 49.1 fF island shunt, a 20 fF island-to-gate capacitance and a 35 fF
gate-to-ground capacitance. Each contact overlaps the top gate by 18 µm². The nominal 30
nm hBN with relative permittivity 3.4 gives about 18 fF per contact in a parallel-plate
estimate. The gate capacitances include this overlap and the M1 feed's contribution. The
island shunt retains the 260 MHz charging-energy target including both mutual
capacitances. These are nominal hybrid estimates that still need measured flake
thickness and full-stack extraction. The complete schematic simulation includes the
qubit's loading and gate-tuned readout response. It uses a linearized junction, so it
does not predict the nonlinear qubit spectrum or decoherence.

The nonlinear ABS Hamiltonian uses the short-junction, even-parity approximation, with
channel transmissions supplied as inputs. Reproducing the paper's quantum-dot parity
transitions and anharmonicities below the short-junction bound requires an extended
Hamiltonian :cite:`riechertCarbonNanotubeGatemon2025`.

The exported gate ports use a 50 ohm S-parameter reference. For comparison with
measurements, connect the experimental bias network to those ports. The paper's setup
uses a bias tee with its RF branch grounded as a reflective filter
:cite:`riechertCarbonNanotubeGatemon2025`. The sample's frequency and coupling checks
use an ideal RF-grounded gate. A matched 50 ohm gate loads the qubit and gives a
different transmission spectrum.

The complete test chip is also checked directly with gdsfactory and SAX, without
GDSFactory+.

The schematic source reads the YAML connections and treats routed links as ideal wires.
The layout source builds the YAML geometry and calls
``get_netlist(serialization_max_digits=18)`` to include every routed CPW segment. The
precision preserves SI capacitances such as the 5.4 fF coupling, which default netlist
rounding would erase. Both sources bind the ``gatemon`` layout to ``gatemon_island``,
whose ports are ``readout`` and ``gate``. The older standalone ``gatemon`` SAX model has
different ports. Both responses are checked against their corresponding GFP simulation
paths.

With default transmissions and matched gates, the schematic cavity is near 5.06033 GHz,
its next harmonic near 10.11284 GHz and the broad qubit feature in gate transfer near
4.55891 GHz. Opening the Josephson junction with zero transmission returns the
fundamental to 5.05600 GHz. Tuning transmission to 0.98 gives two hybrid peaks about 100
MHz apart, consistent with the nominal 50 MHz coupling. Setting the coupling capacitance
to zero removes the qubit's effect on RF transmission. The nonlinear ABS Hamiltonian
instead gives an isolated :math:`f_{01}` near 4.447 GHz for the nominal charging energy
and default junction. That quantum transition is distinct from the linear oscillator
resonance plotted by SAX.

An ideal RF short at each gate launcher's external reference plane retains the
intervening gate CPW in the layout source, so this differs from grounding the island
model's gate terminal directly. The matched six-port circuit is a device
characterization, not the complete bias network used in the paper. Power sent to a
matched gate port reduces RF transmission, and the CPW model also includes the
substrate's dielectric loss. Broad sweeps can miss the maxima of narrow resonances; use
a local frequency sweep before interpreting their heights.

``flipmon_readout_coupon_model()`` creates a 5-port static SAX coupon model. Pass
``pad_length``, ``pad_offset``, ``feedline_gap`` and ``coupling_length`` in um, and
frequency ``f`` in Hz. The whole readout conductor shares one potential, and feedline
reference cuts are open. This dataset requires RF validation before replacing a
distributed resonator circuit. Its recipe is
``qpdk/models/datasets/recipes/flipmon_readout_assembly.py``.

``transmon_readout_coupon_model()`` creates a 4-port static SAX coupon model. Pass
``pad_length``, ``pad_offset``, ``feedline_gap`` and ``coupling_length`` in um, and
frequency ``f`` in Hz. The whole readout conductor shares one potential, and feedline
reference cuts are open. Sampled feedline gaps of 14 to 20 um retain the ground strip
between the coupled lines. This dataset requires RF validation before replacing a
distributed resonator circuit. Its recipe is
``qpdk/models/datasets/recipes/readout_assembly.py``.

``qpdk.cells.plate_capacitor`` combines two ``plate_capacitor_single`` pads with an
optional vertical offset. It includes the finite ground frame used by the
``coupling_pad_palace`` dataset. Its matching SAX model is registered as
``qpdk.models.plate_capacitor`` and retains both pad-to-ground branches as well as the
coupling. The generation recipe uses this same layout cell. The closed-form
approximation without ground branches remains available as
``qpdk.models.plate_capacitor_analytical``.

``meander_inductor_ground_capacitance(turn_length=200, wire_width=2, wire_gap=4,
n_turns=3)`` returns the winding-to-ground capacitance in F from 54 Palace extractions.
Dimensions are in um; turn counts 3, 5 and 7 are separate variants. Both winding ends
share one potential, so this lookup does not determine differential parasitics,
inductance or self-resonance. The generation recipe is
``qpdk/models/datasets/recipes/meander_ground_capacitance.py``.

``meander_inductor_fullwave_model(n_turns=5)`` creates a two-port SAX model from 243
Palace frequency sweeps. The returned model accepts ``f`` from 0.5 to 20 GHz in Hz,
``turn_length`` from 100 to 300 um, ``wire_width`` from 1 to 2 um and ``wire_gap`` from
2 to 8 um. Turn counts 3, 5 and 7 are separate variants. Omitted geometry arguments use
the middle stored coordinates, and values outside the grid return NaN. Its ports are the
winding ends, with 50 ohm reference impedance and no de-embedding. The generation recipe
is ``qpdk/models/datasets/recipes/meander_inductor.py``.

The full-wave coupon includes the finite ground frame and a fixed kinetic sheet
inductance of 0.4 pH per square. It omits material loss and allows radiation through the
outer boundary. Use ``meander_inductor`` for the analytical lumped model with adjustable
film properties.

The winding and ground frame share the specified sheet inductance, so the terminal
response includes both contributions. For a straight strip with nearly uniform current,
sheet inductance times length divided by width estimates only the strip's contribution.
To check the material units against that estimate, use a PEC ground return in the
control simulation. Keep the ground frame's sheet inductance enabled when generating the
device dataset.

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
``--max-refinements`` blocks publication. This measures mesh sensitivity; compare larger
domains separately. Check interpolation with held-out geometries separately. The
notebook demonstrates a lazy diagnostic scan and outlier flag. The plate example has a
separate ground frame. The CPW example is a uniform slice of two identical traces with
outer ground rails, sweeping trace width, outer slot width and inter-trace gap. Use
Palace 0.17 or newer for its two-dimensional transverse mesh. ``Model.Lc`` sets the
implicit depth to ``slice_length_um``, so the stored capacitances remain in F. Divide by
the depth in meters for capacitance per length. Select ``--topology as-drawn`` to retain
the ground strip of width ``max(gap - 2 * cpw_gap, 0)`` left between the component's CPW
etch masks. This produces ``cpw_coupling_ground_strip_palace``, used by
``coupler_straight`` and the default distributed model. The default ``fully-etched``
experiment produces ``cpw_coupling_palace`` for comparison with the unshielded
analytical ECCPW formula. The two geometries agree when the CPW slots touch or overlap.
Above that gap, their difference measures the shielding from the intervening ground
strip. The grounded sweep uses ``ground_strip_width = gap - 2 * cpw_gap`` as its third
axis, with logarithmic sampling from 0.01 to 248 µm. The CPW models interpolate
coordinates and capacitance magnitudes logarithmically in this regime and use the fully
etched grid where the slots touch or overlap. Positive strips narrower than the sampled
minimum return NaN. ``cpw_coupling_model()`` converts its capacitance lookup into a
jittable four-port quasi-TEM SAX model; see the notebook for geometry heatmaps and
S-parameters.

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
stored domain. Discrete variants are selected when loading. Complex linear interpolation
can introduce apparent loss between sample points, so check withheld frequencies near
resonances before choosing the frequency spacing.
