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

Geometry sweeps over the FEM solvers are embarrassingly parallel, so the plumbing for
running them on a cluster lives here: helpers that turn a layout into a solvable Palace
directory and read the result back, a loop that keeps a fixed number of trials in
flight, a command-line entry point for a single trial, and the ``sbatch`` scripts that
put those trials on Slurm, either one job per trial or a Ray cluster inside one
allocation. The transmon evaluator lives in an importable module so trial jobs can run
it; the notebook chooses the objectives and shows the analysis. Other studies can supply
their own evaluator as ``module:function``.

.. automodule:: qpdk.simulation.palace_run
    :members:
    :show-inheritance:

.. automodule:: qpdk.simulation.transmon_sweep
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
