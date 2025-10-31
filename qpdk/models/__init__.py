"""Model definitions for qpdk."""

import sax

from .couplers import coupler_straight
from .generic import (
    capacitor,
    gamma_0_load,
    inductor,
    open,
    short,
    short_2_port,
    single_admittance_element,
    single_impedance_element,
    tee,
)
from .resonator import quarter_wave_resonator_coupled, resonator_frequency
from .waveguides import (
    bend_circular,
    bend_euler,
    bend_s,
    launcher,
    rectangle,
    straight,
    straight_shorted,
    taper_cross_section,
)

sax.set_port_naming_strategy("optical")


models = {
    func.__name__: func
    for func in (
        bend_circular,
        bend_euler,
        bend_s,
        capacitor,
        coupler_straight,
        gamma_0_load,
        inductor,
        open,
        rectangle,
        quarter_wave_resonator_coupled,
        short,
        single_admittance_element,
        single_impedance_element,
        straight,
        taper_cross_section,
        tee,
        launcher,
        short_2_port,
        straight_shorted,
    )
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
