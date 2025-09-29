"""Model definitions for qpdk."""

import sax

import qpdk.models.skrf_duck_typing  # noqa: F401

from .bend_circular import bend_circular
from .resonator import quarter_wave_resonator_coupled_to_probeline, resonator_frequency
from .straight import straight

sax.set_port_naming_strategy("optical")


models = {
    "straight": straight,
    "bend_circular": bend_circular,
    "quarter_wave_resonator": quarter_wave_resonator_coupled_to_probeline,
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
