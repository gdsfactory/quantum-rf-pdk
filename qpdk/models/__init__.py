"""Model definitions for qpdk."""

import sax

import qpdk.models.skrf_duck_typing  # noqa: F401

from .generic import (
    capacitor,
    gamma_0_load,
    inductor,
    open,
    short,
    single_impedance_element,
    tee,
)
from .resonator import quarter_wave_resonator_coupled_to_probeline, resonator_frequency
from .waveguides import bend_circular, bend_euler, bend_s, straight

sax.set_port_naming_strategy("optical")


models = {
    func.__name__: func
    for func in (
        bend_circular,
        bend_euler,
        bend_s,
        capacitor,
        gamma_0_load,
        inductor,
        open,
        quarter_wave_resonator_coupled_to_probeline,
        short,
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
