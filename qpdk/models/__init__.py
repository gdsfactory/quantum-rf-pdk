"""Model definitions for qpdk."""

import sax

import qpdk.models.skrf_duck_typing  # noqa: F401

from .generic import gamma_0_load, open, short
from .resonator import quarter_wave_resonator_coupled_to_probeline, resonator_frequency
from .waveguides import bend_circular, bend_euler, bend_s, straight

sax.set_port_naming_strategy("optical")


models = {
    "bend_circular": bend_circular,
    "bend_euler": bend_euler,
    "bend_s": bend_s,
    "gamma_0_load": gamma_0_load,
    "open": open,
    "quarter_wave_resonator_coupled_to_probeline": quarter_wave_resonator_coupled_to_probeline,
    "short": short,
    "straight": straight,
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
