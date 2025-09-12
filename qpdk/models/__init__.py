from typing import Any, Callable

import sax

sax.set_port_naming_strategy("optical")

models: dict[str, Callable[..., Any]] = {}

from .resonator import quarter_wave_resonator

models["quarter_wave_resonator"] = quarter_wave_resonator
