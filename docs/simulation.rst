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

The builders below return a plain MPh model. Study and local refinement helpers
take that model and its layout; absolute mesh helpers use the model and named selections.
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

Constructing the class needs the ``comsol`` extra; geometry and result helpers
remain importable without it.

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
