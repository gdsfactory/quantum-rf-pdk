"""Model definitions for qpdk."""

import sax
from sax.models.rf import capacitor, inductor, gamma_0_load, admittance, impedance, tee

sax.set_port_naming_strategy("optical")

from qpdk.models.constants import (
    DEFAULT_FREQUENCY,
)
from qpdk.models.couplers import (
    coupler_straight,
    cpw_cpw_coupling_capacitance,
)
from qpdk.models.generic import (
    capacitor,
    gamma_0_load,
    inductor,
    josephson_junction,
    open,
    short,
    short_2_port,
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

models = {
    k: v
    for k, v in ((k, sax.try_into[sax.Model](v)) for k, v in globals().items())
    if v
}

__all__ = [
    "DEFAULT_FREQUENCY",
    "MediaCallable",
    "bend_circular",
    "bend_euler",
    "bend_s",
    "capacitor",
    "coupler_straight",
    "cpw_cpw_coupling_capacitance",
    "cpw_media_skrf",
    "cross_section_to_media",
    "gamma_0_load",
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
    "single_admittance_element",
    "single_impedance_element",
    "straight",
    "straight_shorted",
    "taper_cross_section",
    "tee",
]
