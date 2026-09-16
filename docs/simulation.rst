############
 Simulation
############

.. meta::
    :description: Circuit-model hierarchy and AEDT simulation wrappers for QPDK components.

*******************************
 Circuit-level model hierarchy
*******************************

QPDK declares SAX models on each cell's ``schematic_function``. A declared model is a
semantic hierarchy boundary: gdsfactoryplus must simulate that cell with the compact
model and must not replace it with the cell's geometric children. This is especially
important for coupling structures, because a layout connectivity graph cannot represent
non-galvanic capacitive coupling.

``resonator_coupled`` and ``quarter_wave_resonator_coupled`` therefore carry their own
SAX models. Their model ports match their layout ports, including the resonator
endpoint. Higher-level assemblies such as ``resonator_test_chip_python`` intentionally
carry no whole-chip model. They expand until gdsfactoryplus reaches the modeled couplers
and ordinary modeled primitives on each branch; the stopping level is determined by
model metadata, not by a fixed hierarchy depth.

When adding a new coupling component, attach its compact model through the cell's
``schematic_function`` and keep the model's ``port_order`` consistent with the layout
ports. Do not add a pass-through whole-chip model merely to prevent recursive expansion.

.. automodule:: qpdk.simulation

********
 Common
********

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
