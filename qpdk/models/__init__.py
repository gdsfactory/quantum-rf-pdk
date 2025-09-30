"""Model definitions for qpdk."""

import sax

import qpdk.models.skrf_duck_typing  # noqa: F401

from .resonator import quarter_wave_resonator_coupled_to_probeline, resonator_frequency
from .waveguides import bend_circular, bend_euler, bend_s, straight

sax.set_port_naming_strategy("optical")


models = {
    "straight": straight,
    "bend_circular": bend_circular,
    "bend_euler": bend_euler,
    "bend_s": bend_s,
    "quarter_wave_resonator_coupled_to_probeline": quarter_wave_resonator_coupled_to_probeline,
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
