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

The bundled ``plate_capacitor_palace`` dataset contains real electrostatic extractions.
``datasets/plate_capacitor.py`` defines a complete experiment with its grid, settings,
metadata and solver function. Its inline dependencies select the Python 3.12 environment
required by gsim; generation code is not installed with QPDK.

Preview or run the bundled 27-point sweep from a checkout:

::

    uv run --script datasets/plate_capacitor.py --dry-run
    uv run --script datasets/plate_capacitor.py --processes 4
    uv run --script datasets/plate_capacitor.py --sif /path/to/palace.sif

Copy the script beside the original in ``datasets/`` and edit ``GRID`` and ``SETTINGS``
for another experiment. It uses gsim's ``ElectrostaticSim`` for execution and
``load_capacitance`` for reading results. A small geometry-specific Gmsh mesher in the
script preserves separate terminals on the same physical layer and the coplanar ground
frame. The current gsim layer-based terminal API cannot distinguish these electrodes,
and its planar mesher fails on this geometry. The reusable dataset package contains no
solver wrapper.

Each geometry retains its inputs, mesh, config, solver log and results under
``--workdir``. Rerunning reuses matching completed solves. Extend the grid while
retaining the work directory to solve only new points. ``--output`` is replaced only
after the sweep succeeds. The output Parquet files contain all dataset metadata, so
lookups need no generator or solver. Container runs fingerprint the complete SIF; native
runs fingerprint only the supplied executable.

Set ``save_fields=True`` for ParaView outputs. Compare representative geometries at
finer ``near_mesh`` sizes and larger ``domain_pad`` before relying on small differences.
This example models perfect conductor sheets on silicon with a separate ground frame.

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
