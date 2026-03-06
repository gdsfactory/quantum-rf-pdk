"""Model definitions for qpdk."""

import sax

from qpdk.models.capacitor import (
    interdigital_capacitor,
    plate_capacitor,
)
from qpdk.models.junction import josephson_junction, squid_junction

sax.set_port_naming_strategy("optical")

from qpdk.models.constants import (
    DEFAULT_FREQUENCY,
)
from qpdk.models.couplers import (
    coupler_ring,
    coupler_straight,
    cpw_cpw_coupling_capacitance,
)
from qpdk.models.generic import (
    admittance,
    capacitor,
    electrical_open,
    electrical_short,
    electrical_short_2_port,
    gamma_0_load,
    impedance,
    inductor,
    lc_resonator,
    lc_resonator_coupled,
    open,
    short,
    short_2_port,
)
from qpdk.models.media import (
    MediaCallable,
    cpw_media_skrf,
    cross_section_to_media,
)
from qpdk.models.qubit import (
    coupling_strength_to_capacitance,
    double_island_transmon,
    double_island_transmon_with_bbox,
    double_island_transmon_with_resonator,
    double_pad_transmon,
    double_pad_transmon_with_bbox,
    double_pad_transmon_with_resonator,
    ec_to_capacitance,
    ej_to_inductance,
    flipmon,
    flipmon_with_bbox,
    flipmon_with_resonator,
    qubit_with_resonator,
    shunted_transmon,
    transmon_coupled,
    transmon_with_resonator,
    xmon_transmon,
)
from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator,
    resonator_coupled,
    resonator_frequency,
    resonator_half_wave,
    resonator_quarter_wave,
)
from qpdk.models.waveguides import (
    airbridge,
    bend_circular,
    bend_euler,
    bend_s,
    indium_bump,
    launcher,
    nxn,
    rectangle,
    straight,
    straight_double_open,
    straight_open,
    straight_shorted,
    taper_cross_section,
    tee,
    tsv,
)

__all__ = [
    "DEFAULT_FREQUENCY",
    "MediaCallable",
    "admittance",
    "airbridge",
    "bend_circular",
    "bend_euler",
    "bend_s",
    "capacitor",
    "coupler_ring",
    "coupler_straight",
    "coupling_strength_to_capacitance",
    "cpw_cpw_coupling_capacitance",
    "cpw_media_skrf",
    "cross_section_to_media",
    "double_island_transmon",
    "double_island_transmon_with_bbox",
    "double_island_transmon_with_resonator",
    "double_pad_transmon",
    "double_pad_transmon_with_bbox",
    "double_pad_transmon_with_resonator",
    "ec_to_capacitance",
    "ej_to_inductance",
    "electrical_open",
    "electrical_short",
    "electrical_short_2_port",
    "flipmon",
    "flipmon_with_bbox",
    "flipmon_with_resonator",
    "gamma_0_load",
    "impedance",
    "indium_bump",
    "inductor",
    "interdigital_capacitor",
    "josephson_junction",
    "launcher",
    "lc_resonator",
    "lc_resonator_coupled",
    "models",
    "nxn",
    "open",
    "plate_capacitor",
    "quarter_wave_resonator_coupled",
    "qubit_with_resonator",
    "rectangle",
    "resonator",
    "resonator_coupled",
    "resonator_frequency",
    "resonator_half_wave",
    "resonator_quarter_wave",
    "short",
    "short_2_port",
    "shunted_transmon",
    "squid_junction",
    "straight",
    "straight_double_open",
    "straight_open",
    "straight_shorted",
    "taper_cross_section",
    "tee",
    "transmon_coupled",
    "transmon_with_resonator",
    "tsv",
    "xmon_transmon",
]


def _is_sax_model(obj: object) -> bool:
    """Check if an object is a SAX model function."""
    if not callable(obj):
        return False
    # Check if return type is sax.SType or sax.SDict
    for target in [obj, getattr(obj, "__wrapped__", None)]:
        if target is None:
            continue
        if hasattr(target, "__annotations__") and "return" in target.__annotations__:
            ret = target.__annotations__["return"]
            if hasattr(ret, "__name__") and ret.__name__ in ["SType", "SDict"]:
                return True
            # Also check identity for standard cases
            if ret in [sax.SType, sax.SDict]:
                return True
    return sax.try_into[sax.Model](obj) is not None


models = {
    k: v
    for k, v in globals().items()
    if k in __all__ and k != "models" and _is_sax_model(v)
}
