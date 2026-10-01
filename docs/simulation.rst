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

Reusable FEM extraction results: a ``manifest.toml`` in Git plus Parquet results in Git
LFS, read with Polars and looked up in jittable SAX models. Requires the ``models``
extra. See :doc:`notebooks/fem_dataset_lookup`.

From a checkout, fetch only the dataset files with ``git lfs pull
--include="qpdk/datasets/data/**"``. Wheels on PyPI ship the resolved Parquet files, so
installed users need no Git LFS.

Results can also live outside Git, in a `Delta Lake <https://delta.io>`_ table on a
cloud bucket (``gs://``, ``s3://``, ``az://``) or a local path. The manifest stays in
Git and points at the table, optionally pinned to a table version; credentials are
passed when opening the dataset, never stored in the manifest. Requires the ``delta``
extra.

.. code-block:: toml

    # manifest.toml
    [storage]
    format = "delta"
    uri = "gs://my-bucket/plate_capacitor"
    version = 3  # optional: read exactly this commit

.. code-block:: python

    from qpdk.datasets import Dataset

    dataset = Dataset(
        "path/to/plate_capacitor", storage_options={"google_service_account": "key.json"}
    )
    dataset.append(new_rows)  # one atomic Delta commit

.. automodule:: qpdk.datasets

.. automodule:: qpdk.datasets.manifest
    :members:

.. automodule:: qpdk.datasets.table
    :members:

.. automodule:: qpdk.datasets.store
    :members:

.. automodule:: qpdk.datasets.interpolation
    :members:

.. automodule:: qpdk.datasets.capacitance
    :members:
