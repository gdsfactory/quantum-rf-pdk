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
A TOML generation recipe selects an importable Python generator, its parameter grid and
Palace settings. It is a generation input, not a dataset manifest: the output Parquet
files remain self-describing and need no recipe to be read.

Preview or run the bundled 27-point sweep:

::

    just generate-dataset datasets/plate_capacitor.toml --dry-run
    just generate-dataset datasets/plate_capacitor.toml

Outside a checkout, use ``python -m qpdk.models.datasets.recipe recipe.toml``. Copy and
edit the recipe to choose a different grid or mesh. Its output and work paths are
relative to the recipe file; ``--output`` and ``--workdir`` overrides are relative to
the current directory. ``--palace-command "palace -np 8"`` overrides the MPI or
container command without changing the file. Commands are argument vectors, and the
config filename is appended without invoking a shell.

The recipe's ``generator`` is a ``module:function`` accepting ``runner``, ``grid`` and
``output`` keyword arguments. That function owns the geometry, units, terminal
conventions and metadata, and returns a :class:`~qpdk.models.datasets.Dataset`. Use
:func:`~qpdk.models.datasets.plate_capacitor.generate` as a starting point for another
geometry. No generator registry or separate command target is needed.

Each geometry gets its own mesh, config, solver log and results beneath ``workdir``.
Running the same command resumes completed solves; geometry, solver and mesh changes get
separate run directories. Extend a grid while retaining its work directory to solve only
new points. The dataset is published after all solves and validation succeed. A failed
solve raises with its log path and preserves the existing dataset.

Set ``save_fields = true`` in ``[palace.settings]`` for ParaView outputs. ``near_mesh``
and ``domain_pad`` control mesh and domain refinement; compare representative points
before using a new parameter range. The bundled recipe uses zero-thickness perfect
conductor sheets on silicon and a separate coplanar ground frame, not a finite-thickness
process-stack extraction. The notebook demonstrates a small recipe, optional live
solving, inspecting the output and refinement checks.

.. automodule:: qpdk.models.datasets

.. automodule:: qpdk.models.datasets.metadata
    :members:

.. automodule:: qpdk.models.datasets.table
    :members:

.. automodule:: qpdk.models.datasets.generate
    :members:

.. automodule:: qpdk.models.datasets.plate_capacitor
    :members:

.. automodule:: qpdk.models.datasets.store
    :members:

.. automodule:: qpdk.models.datasets.interpolation
    :members:

.. automodule:: qpdk.models.datasets.capacitance
    :members:

.. automodule:: qpdk.simulation.palace
    :members:

.. automodule:: qpdk.models.datasets.recipe
    :members:
