"""Model definitions for qpdk."""

import sax

from .resonator import quarter_wave_resonator_coupled_to_probeline, resonator_frequency

sax.set_port_naming_strategy("optical")

models = {
    "quarter_wave_resonator": quarter_wave_resonator_coupled_to_probeline,
}

__all__ = [
    "models",
    "resonator_frequency",
    *models.keys(),
]
