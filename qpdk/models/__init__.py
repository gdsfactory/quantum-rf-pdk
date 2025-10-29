"""Model definitions for qpdk."""

import sax

from .couplers import coupler_straight
from .generic import (
    capacitor,
    gamma_0_load,
    inductor,
    open,
    short,
    single_admittance_element,
    single_impedance_element,
    tee,
)
from .resonator import resonator_coupled, resonator_frequency
from .waveguides import bend_circular, bend_euler, bend_s, straight

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
        resonator_coupled,
        short,
        single_admittance_element,
        single_impedance_element,
        straight,
        tee,
    )
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
