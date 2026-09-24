"""Mesh helpers for COMSOL sheet models.

Both COMSOL notebooks build a study on the same sheet geometry and then want one
more thing from the mesh: the element count after a local refinement at the
metal plane, to check that a solved quantity has stopped moving with mesh size.
The sequence is the same in both, so it lives here once.

The study builders (:func:`~qpdk.simulation.comsol_rf.add_cpw_rf_study` and
:func:`~qpdk.simulation.comsol_capacitance.add_qubit_capacitance_study`) leave
``mesh1`` physics-controlled. Running it once realizes that sizing; a ``Refine``
feature then switches the sequence to user-controlled, and the second run meshes
the refined box.

:func:`pin_absolute_mesh_sizes` is for the other kind of study, where the mesh
must not follow the domain at all. A physics-controlled mesh scales its element
sizes with the longest dimension of the domain, so a series that varies the
distance to the boundary would vary the near-metal resolution with it.
"""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from qpdk.simulation.comsol import _format_number
from qpdk.simulation.comsol_capacitance import (
    GROUND_SELECTION,
    LEFT_PAD_SELECTION,
    RIGHT_PAD_SELECTION,
)

if TYPE_CHECKING:
    import mph

    from qpdk.simulation.comsol_layout import ComsolBoundingBox, ComsolLayout

#: Tag of the default ``Size`` feature a physics-controlled build leaves in the
#: sequence, and the one feature pinning the global sizes requires to be there.
#: The mesh chapter of the COMSOL programming reference warns that the contents
#: of a physics-controlled sequence may change between versions, so it is checked
#: at run time rather than assumed.
DEFAULT_SIZE_TAG = "size"

#: FreeTet generator COMSOL generates for a physics-controlled sequence. It does
#: not survive an edit of the sequence on a 6.3 build, but exactly when it goes is
#: undocumented, so it is swept either way. Nothing may depend on it being there.
GENERATED_FREE_TET_TAG = "ftet1"

#: Tag of the generator this module creates, the only one left in the sequence.
#: A ``Size`` feature only affects the operation features after it, so this has
#: to follow every size.
FREE_TET_TAG = "ftet_fixed"


def refine_metal_plane_mesh(
    model: mph.Model,
    layout: ComsolLayout,
    passes: int,
    *,
    z_half_um: float = 20.0,
    refine_box: ComsolBoundingBox | None = None,
) -> int:
    """Mesh a sheet model and refine the mesh around the metal plane.

    The mesh is run once at the study builder's physics-controlled size, then
    refined ``passes`` times inside a box: it spans ``refine_box`` in x and y and
    ±``z_half_um`` in z, so the metal plane and the fields just off it are refined
    while the bulk air and silicon stay coarse. Pass ``refine_box`` when the
    layout's own box is much larger than the part that matters; a few refinement
    passes over a whole 3.5 mm ground box cost elements for nothing when the
    device is 1 mm across.

    Args:
        model: A model carrying ``comp1``/``mesh1`` from
            :func:`~qpdk.simulation.comsol_rf.add_cpw_rf_study` or
            :func:`~qpdk.simulation.comsol_capacitance.add_qubit_capacitance_study`.
        layout: The layout the model was built from. Its bounding box sets the
            refine box unless ``refine_box`` is given.
        passes: Number of refinement passes, a non-negative integer. ``0`` meshes
            at the physics-controlled size without refining.
        z_half_um: Half-height in µm of the refine box above and below the metal
            plane, positive and finite.
        refine_box: x/y bounds in µm to refine instead of the layout bounding
            box. ``None`` uses the layout's own box.

    Returns:
        The number of mesh elements after meshing.

    Raises:
        ValueError: If ``passes`` is not a non-negative integer, or if
            ``z_half_um`` is not positive and finite.
    """
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
        raise ValueError(f"passes must be a non-negative integer, got {passes!r}")
    if not math.isfinite(z_half_um) or z_half_um <= 0.0:
        raise ValueError(f"z_half_um must be positive and finite, got {z_half_um!r}")

    box = layout.bbox if refine_box is None else refine_box
    mesh = model.java.component("comp1").mesh("mesh1")
    mesh.run()
    # Refine switches the sequence to user-controlled, so a generator has to
    # follow it for the refined box to have one.
    mesh.create("ftet_refine", "FreeTet")
    if passes > 0:
        refine = mesh.create("ref1", "Refine")
        # A Python int picks the numeric set() overload, which rejects it here.
        refine.set("numrefine", str(passes))
        refine.set("rmethod", "longest")
        refine.set("boxcoord", "on")
        refine.set("xmin", f"{_format_number(box.xmin)}[um]")
        refine.set("xmax", f"{_format_number(box.xmax)}[um]")
        refine.set("ymin", f"{_format_number(box.ymin)}[um]")
        refine.set("ymax", f"{_format_number(box.ymax)}[um]")
        refine.set("zmin", f"{_format_number(-z_half_um)}[um]")
        refine.set("zmax", f"{_format_number(z_half_um)}[um]")
    mesh.run()
    return int(mesh.getNumElem())


def _require_positive(name: str, value: Any) -> None:
    """Refuse a size that is not a positive, finite number.

    Args:
        name: Argument name, used in the error.
        value: The value to check.

    Raises:
        ValueError: If the value is not a real number greater than zero.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def _set_size(
    feature: Any,
    hmax_um: float,
    hmin_um: float,
    *,
    hgrad: float | None = None,
    hcurve: float | None = None,
    hnarrow: float | None = None,
) -> None:
    """Turn one ``Size`` feature custom and set its element sizes in µm.

    ``custom`` goes first: with it off the feature ignores ``hmax`` and ``hmin``
    and falls back to the predefined ``hauto`` size, which is the domain-scaled
    sizing this module exists to get rid of.

    Args:
        feature: A COMSOL ``Size`` feature.
        hmax_um: Largest element size in µm.
        hmin_um: Smallest element size in µm.
        hgrad: Maximum element growth rate, left as COMSOL has it if ``None``.
        hcurve: Curvature resolution in elements per radian, left alone if
            ``None``.
        hnarrow: Narrow region resolution, left alone if ``None``.
    """
    feature.set("custom", "on")
    feature.set("hmax", f"{_format_number(hmax_um)}[um]")
    feature.set("hmin", f"{_format_number(hmin_um)}[um]")
    for name, value in (("hgrad", hgrad), ("hcurve", hcurve), ("hnarrow", hnarrow)):
        if value is not None:
            feature.set(name, _format_number(value))


def _drop_generated_free_tet(sequence: Any) -> None:
    """Remove COMSOL's generated FreeTet generator, if the sequence holds it.

    Args:
        sequence: The ``mesh1`` sequence.
    """
    if GENERATED_FREE_TET_TAG not in sequence.feature().tags():
        return
    # A removal that fails is not fatal: the sequence order is checked after this
    # either way, and a leftover generator shows up there.
    with suppress(Exception):
        sequence.feature().remove(GENERATED_FREE_TET_TAG)


def pin_absolute_mesh_sizes(
    model: mph.Model,
    *,
    global_hmax_um: float,
    global_hmin_um: float,
    pad_hmax_um: float,
    pad_hmin_um: float,
    ground_hmax_um: float,
    ground_hmin_um: float,
    hgrad: float | None = None,
    hcurve: float | None = None,
    hnarrow: float | None = None,
) -> int:
    """Mesh a sheet model with absolute element sizes at the metal.

    A physics-controlled mesh scales its sizes with the longest dimension of the
    domain, so the near-metal resolution moves whenever the domain does. This
    helper runs that mesh once to materialise the sequence, then pins every size
    to an absolute value in µm: the default ``Size`` feature takes the global
    sizes, and one ``Size`` feature per conductor takes the pad and ground sizes
    on the existing named selections. The sequence is left user-controlled, so a
    later build or solve cannot re-derive the sizing.

    A ``Size`` feature only affects the operation features after it, so the
    conductor sizes have to sit between the default size and the generator. The
    construction builds that order and refuses to mesh if the sequence comes out
    any other way, rather than silently meshing something else. The generated
    FreeTet of the physics-controlled sequence does not survive an edit, and when
    it goes is undocumented, so it is swept before and after the explicit
    generator is created, leaving exactly one generator. The ``Size`` feature
    properties are in the COMSOL 6.3 API reference,
    https://doc.comsol.com/6.3/doc/com.comsol.help.comsol/comsol_api_mesh.49.099.html

    Args:
        model: A model carrying ``comp1``/``mesh1`` from
            :func:`~qpdk.simulation.comsol_capacitance.add_qubit_capacitance_study`,
            including the ``pad_l``, ``pad_r`` and ``gnd`` face selections that
            study creates.
        global_hmax_um: Largest element size in µm away from the metal, positive
            and finite.
        global_hmin_um: Smallest element size in µm away from the metal.
        pad_hmax_um: Largest element size in µm on the two pad faces.
        pad_hmin_um: Smallest element size in µm on the two pad faces.
        ground_hmax_um: Largest element size in µm on the ground faces.
        ground_hmin_um: Smallest element size in µm on the ground faces.
        hgrad: Maximum element growth rate for the global sizes.
        hcurve: Curvature resolution for the global sizes, in elements per radian.
        hnarrow: Narrow region resolution for the global sizes.

    Returns:
        The number of mesh elements the pinned sequence built.

    Raises:
        RuntimeError: If the physics-controlled build left no default ``size``
            feature, the sequence came out in an order the sizes would not apply
            in, the sequence is still physics-controlled afterwards, or COMSOL
            reported no usable element count.
    """
    for name, value in (
        ("global_hmax_um", global_hmax_um),
        ("global_hmin_um", global_hmin_um),
        ("pad_hmax_um", pad_hmax_um),
        ("pad_hmin_um", pad_hmin_um),
        ("ground_hmax_um", ground_hmax_um),
        ("ground_hmin_um", ground_hmin_um),
    ):
        _require_positive(name, value)
    for name, value in (("hgrad", hgrad), ("hcurve", hcurve), ("hnarrow", hnarrow)):
        if value is not None:
            _require_positive(name, value)

    sequence = model.java.component("comp1").mesh("mesh1")

    # A physics-controlled build is what writes the ordinary Size and FreeTet
    # features into the sequence; until it exists there is no editable sizing.
    sequence.run()
    features = list(sequence.feature().tags())
    if DEFAULT_SIZE_TAG not in features:
        raise RuntimeError(
            f"the physics-controlled build produced {features}, which holds no "
            f"{DEFAULT_SIZE_TAG!r}; there is no default sizing to pin the absolute "
            "sizes to"
        )

    # Editing the default size is what switches the sequence to user-controlled,
    # so the physics can no longer re-derive this sizing at the next build.
    _set_size(
        sequence.feature(DEFAULT_SIZE_TAG),
        global_hmax_um,
        global_hmin_um,
        hgrad=hgrad,
        hcurve=hcurve,
        hnarrow=hnarrow,
    )

    # A new feature is inserted after the current one and then becomes current, so
    # parking the cursor on the default size puts the conductor sizes between it
    # and everything that follows, in the order they are created.
    sequence.current(DEFAULT_SIZE_TAG)
    conductor_sizes = (
        ("size_pad_l", LEFT_PAD_SELECTION, pad_hmax_um, pad_hmin_um),
        ("size_pad_r", RIGHT_PAD_SELECTION, pad_hmax_um, pad_hmin_um),
        ("size_gnd", GROUND_SELECTION, ground_hmax_um, ground_hmin_um),
    )
    for tag, selection, hmax_um, hmin_um in conductor_sizes:
        sequence.create(tag, "Size")
        feature = sequence.feature(tag)
        feature.selection().named(selection)
        _set_size(feature, hmax_um, hmin_um)

    # The generated generator is swept first, so the one created below lands after
    # the last Size feature rather than after a feature about to disappear.
    _drop_generated_free_tet(sequence)
    sequence.current(conductor_sizes[-1][0])
    sequence.create(FREE_TET_TAG, "FreeTet")
    # Swept again: a build that kept the generated generator through the create
    # would otherwise leave two of them, and the domains would be meshed twice.
    _drop_generated_free_tet(sequence)

    desired = (DEFAULT_SIZE_TAG, *(tag for tag, *_ in conductor_sizes), FREE_TET_TAG)
    observed = tuple(sequence.feature().tags())
    if observed != desired:
        raise RuntimeError(
            f"the mesh sequence came out as {observed}, expected {desired}; a Size "
            "feature only affects the features after it, so meshing this would not "
            "apply the sizes it was asked for"
        )

    still_automatic = False
    with suppress(Exception):
        still_automatic = bool(sequence.isAutomatic())
    if still_automatic:
        raise RuntimeError(
            "the mesh sequence is still physics-controlled after editing it, so the "
            "near-metal size would follow the domain again"
        )

    sequence.run()
    try:
        count = int(sequence.getNumElem())
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"COMSOL reported no element count for the pinned mesh: {error!r}"
        ) from error
    if count <= 0:
        raise RuntimeError(f"the pinned mesh holds {count} elements")
    return count
