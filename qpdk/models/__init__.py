"""Model definitions for qpdk."""

import sax

sax.set_port_naming_strategy("optical")

from qpdk.models.constants import (
    DEFAULT_FREQUENCY,
)
from qpdk.models.couplers import (
    coupler_straight,
    cpw_cpw_coupling_capacitance,
)
from qpdk.models.generic import (
    admittance,
    capacitor,
    gamma_0_load,
    impedance,
    inductor,
    josephson_junction,
    open,
    short,
    short_2_port,
    tee,
)
from qpdk.models.media import (
    MediaCallable,
    cpw_media_skrf,
    cross_section_to_media,
)
from qpdk.models.resonator import (
    quarter_wave_resonator_coupled,
    resonator_frequency,
)
from qpdk.models.waveguides import (
    bend_circular,
    bend_euler,
    bend_s,
    launcher,
    rectangle,
    straight,
    straight_shorted,
    taper_cross_section,
)

__all__ = [
    "DEFAULT_FREQUENCY",
    "MediaCallable",
    "admittance",
    "bend_circular",
    "bend_euler",
    "bend_s",
    "capacitor",
    "coupler_straight",
    "cpw_cpw_coupling_capacitance",
    "cpw_media_skrf",
    "cross_section_to_media",
    "gamma_0_load",
    "impedance",
    "inductor",
    "josephson_junction",
    "launcher",
    "models",
    "open",
    "quarter_wave_resonator_coupled",
    "rectangle",
    "resonator_frequency",
    "short",
    "short_2_port",
    "straight",
    "straight_shorted",
    "taper_cross_section",
    "tee",
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
