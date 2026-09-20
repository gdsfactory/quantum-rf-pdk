"""Private PDK/layout bindings for analytical models.

The functions in :mod:`qpdk.models` are the public scientific catalog and keep
their own port contracts (``o1``/``o2``). The PDK instead binds the same-named
wrappers defined here, which only rename the returned ports to the ones their
layout cell exposes. Nothing in this module is part of the public catalog:
``_PDK_MODEL_OVERRIDES`` is consumed by ``_build_pdk_models`` in :mod:`qpdk`
and the wrappers are reachable only through the schematic descriptors that
name them.
"""

from functools import partial

import jax
import sax

from qpdk.models.constants import DEFAULT_FREQUENCY
from qpdk.models.junction import josephson_junction as _josephson_junction
from qpdk.models.qubit import (
    double_pad_transmon_with_bbox as _double_pad_transmon_with_bbox,
    fluxonium as _fluxonium,
    fluxonium_with_bbox as _fluxonium_with_bbox,
)
from qpdk.models.unimon import unimon_coupled as _unimon_coupled


@partial(jax.jit, inline=True)
def fluxonium(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
) -> sax.SDict:
    """:func:`qpdk.models.qubit.fluxonium` with the layout cell port names.

    Returns:
        sax.SDict: S-parameters dictionary with ports left_pad and right_pad.
    """
    return sax.rename_ports(
        _fluxonium(
            f=f,
            capacitance=capacitance,
            josephson_inductance=josephson_inductance,
            superinductance=superinductance,
            ground_capacitance=ground_capacitance,
        ),
        {"o1": "left_pad", "o2": "right_pad"},
    )


@partial(jax.jit, inline=True)
def fluxonium_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 10e-15,
    josephson_inductance: float = 10e-9,
    superinductance: float = 500e-9,
    ground_capacitance: float = 0.0,
) -> sax.SType:
    """:func:`qpdk.models.qubit.fluxonium_with_bbox` with the cell port names.

    Returns:
        sax.SType: S-parameters dictionary with ports left_pad and right_pad.
    """
    return sax.rename_ports(
        _fluxonium_with_bbox(
            f=f,
            capacitance=capacitance,
            josephson_inductance=josephson_inductance,
            superinductance=superinductance,
            ground_capacitance=ground_capacitance,
        ),
        {"o1": "left_pad", "o2": "right_pad"},
    )


@partial(jax.jit, inline=True)
def double_pad_transmon_with_bbox(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    capacitance: float = 100e-15,
    inductance: float = 7e-9,
    ground_capacitance: float = 0.0,
) -> sax.SType:
    """:func:`qpdk.models.qubit.double_pad_transmon_with_bbox` with cell ports.

    Returns:
        sax.SType: S-parameters dictionary with ports left_pad and right_pad.
    """
    return sax.rename_ports(
        _double_pad_transmon_with_bbox(
            f=f,
            capacitance=capacitance,
            inductance=inductance,
            ground_capacitance=ground_capacitance,
        ),
        {"o1": "left_pad", "o2": "right_pad"},
    )


@partial(jax.jit, inline=True)
def josephson_junction(
    *,
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    ic: sax.Float = 1e-6,
    capacitance: sax.Float = 5e-15,
    resistance: sax.Float = 10e3,
    ib: sax.Float = 0.0,
) -> sax.SDict:
    """:func:`qpdk.models.junction.josephson_junction` with the cell port names.

    Returns:
        sax.SDict: S-parameters dictionary with ports left_wide and right_wide.
    """
    return sax.rename_ports(
        _josephson_junction(
            f=f,
            ic=ic,
            capacitance=capacitance,
            resistance=resistance,
            ib=ib,
        ),
        {"o1": "left_wide", "o2": "right_wide"},
    )


# Not decorated with jax.jit: the cross-section setting is a string that
# schematics pass explicitly, so callers must keep it outside the traced
# boundary by jitting their frequency array themselves.
def unimon_coupled(
    f: sax.FloatArrayLike = DEFAULT_FREQUENCY,
    arm_length: float = 3000.0,
    cross_section: str = "cpw",
    junction_capacitance: float = 5e-15,
    junction_inductance: float = -7e-9,
    coupling_capacitance: float = 10e-15,
) -> sax.SDict:
    """:func:`qpdk.models.unimon.unimon_coupled` with the cell port names.

    Returns:
        sax.SDict: S-parameters dictionary with the port coupling_o3.
    """
    return sax.rename_ports(
        _unimon_coupled(
            f=f,
            arm_length=arm_length,
            cross_section=cross_section,
            junction_capacitance=junction_capacitance,
            junction_inductance=junction_inductance,
            coupling_capacitance=coupling_capacitance,
        ),
        {"o1": "coupling_o3"},
    )


#: PDK registrations that must use the layout-facing wrapper above instead of
#: the same-named analytical function.
_PDK_MODEL_OVERRIDES = {
    "double_pad_transmon_with_bbox": double_pad_transmon_with_bbox,
    "fluxonium": fluxonium,
    "fluxonium_with_bbox": fluxonium_with_bbox,
    "josephson_junction": josephson_junction,
    "unimon_coupled": unimon_coupled,
}
