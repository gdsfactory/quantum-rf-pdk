############
 Simulation
############

.. meta::
    :description: Circuit-model hierarchy and AEDT simulation wrappers for QPDK components.

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

.. automodule:: qpdk.simulation.aedt_base
    :members:
    :show-inheritance:

******
 HFSS
******

.. automodule:: qpdk.simulation.hfss
    :members:
    :show-inheritance:

*************
 Q3D and Q2D
*************

.. automodule:: qpdk.simulation.q3d
    :members:
    :show-inheritance:

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
``qpdk/models/datasets/data/flipmon.py``.

``double_pad_transmon_capacitance`` looks up the differential pad capacitance in F,
including the two pad-to-ground branches. With junction capacitance :math:`C_\text{J}`,
the charging frequency is :math:`E_\text{C}/h = e^2/[2h(C_\text{diff}+C_\text{J})]` in
Hz. For a linearized circuit, create ``double_pad_transmon_fem_model(inductance=10e-9,
junction_capacitance=2e-15)`` and pass geometry dimensions in um to the returned SAX
model. Both ports reference the coupon's finite ground frame. The dataset omits the
readout, package and Josephson junction; its recipe is
``qpdk/models/datasets/data/double_pad_transmon.py``.

``flipmon_readout_coupon_model()`` creates a 5-port static SAX coupon model. Pass
``pad_length``, ``pad_offset``, ``feedline_gap`` and ``coupling_length`` in um, and
frequency ``f`` in Hz. The whole readout conductor shares one potential, and feedline
reference cuts are open. This dataset requires RF validation before replacing a
distributed resonator circuit. Its recipe is
``qpdk/models/datasets/data/flipmon_readout_assembly.py``.

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
``qpdk/models/datasets/data/`` define the grid, settings, metadata and solver function.
Inline dependencies select the Python 3.12 environment required by gsim; generation
scripts are excluded from the installed package.

Preview or run the bundled 27-point sweep from a checkout:

::

    uv run --script qpdk/models/datasets/data/plate_capacitor.py --dry-run
    uv run --script qpdk/models/datasets/data/plate_capacitor.py --processes 4
    uv run --script qpdk/models/datasets/data/plate_capacitor.py --sif /path/to/palace.sif

Copy the script beside the original in ``qpdk/models/datasets/data/`` and edit ``GRID``
and ``SETTINGS`` for another experiment. Meshwell meshes labelled conductor sheets and
the surrounding air and silicon volumes. The shared experiment helper uses gsim to write
the Palace configuration, run ``ElectrostaticSim`` and load capacitance matrices. Lookup
models need none of these simulation dependencies.

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

    generator=qpdk/models/datasets/data/cpw_coupling.py
    sbatch --array=0-1 --cpus-per-task=4 qpdk/models/datasets/data/slurm_array.sh "$generator" build/cpw-shards build/cpw-runs --sif /path/to/palace.sif
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
