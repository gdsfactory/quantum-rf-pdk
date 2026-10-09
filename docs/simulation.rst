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
 COMSOL
********

The builders below return a plain MPh model. Study and local refinement helpers take
that model and its layout; absolute mesh helpers use the model and named selections.
:class:`~qpdk.simulation.comsol.model.COMSOL` wraps the two in one object: it subclasses
:class:`mph.Model`, so MPh solves, saves, and evaluates it as usual, and it holds the
layout, so a setup reads as a chain.

1. Build with :meth:`~qpdk.simulation.comsol.model.COMSOL.create_sheet` or
   :meth:`~qpdk.simulation.comsol.model.COMSOL.create_metal`
2. Add a study with :meth:`~qpdk.simulation.comsol.model.COMSOL.add_cpw_rf_study` or
   :meth:`~qpdk.simulation.comsol.model.COMSOL.add_capacitance_study`
3. Mesh it with :meth:`~qpdk.simulation.comsol.model.COMSOL.refine_metal_plane_mesh`,
   :meth:`~qpdk.simulation.comsol.model.COMSOL.pin_absolute_mesh_sizes`, or
   :meth:`~qpdk.simulation.comsol.model.COMSOL.pin_absolute_edge_mesh_sizes`
4. Solve, save, and evaluate with MPh's own ``solve``, ``save``, and ``evaluate``

Constructing the class needs the ``comsol`` extra; geometry and result helpers remain
importable without it.

.. automodule:: qpdk.simulation.comsol.model
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.comsol.layout
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.comsol.metal
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.comsol.sheet
    :members:

.. automodule:: qpdk.simulation.comsol.rf
    :members:

.. automodule:: qpdk.simulation.comsol.capacitance
    :members:

.. automodule:: qpdk.simulation.comsol.mesh
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
outer ground rails, sweeping trace width, outer slot width and inter-trace gap. Select
``--topology as-drawn`` to retain the ground strip of width ``max(gap - 2 * cpw_gap,
0)`` left between the component's CPW etch masks. This produces
``cpw_coupling_ground_strip_palace``, used by ``coupler_straight`` and the default
distributed model. The default ``fully-etched`` experiment produces
``cpw_coupling_palace`` for comparison with the unshielded analytical ECCPW formula. The
two geometries agree when the CPW slots touch or overlap. Above that gap, their
difference measures the shielding from the intervening ground strip.
``cpw_coupling_model()`` converts its capacitance lookup into a jittable four-port
quasi-TEM SAX model; see the notebook for geometry heatmaps and S-parameters.

Both scripts support disjoint rectangular shards with ``--shard INDEX --shards COUNT``.
Use separate output and work directories for each worker, then merge after all succeed.
For the CPW grid, the generic Slurm helper groups 195 geometries per task:

::

    generator=qpdk/models/datasets/data/cpw_coupling.py
    sbatch --array=0-34 --cpus-per-task=4 qpdk/models/datasets/data/slurm_array.sh "$generator" build/cpw-shards build/cpw-runs --sif /path/to/palace.sif
    shards=()
    for i in {0..34}; do shards+=(--merge-shards "build/cpw-shards/shard-$i"); done
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
