"""Model constants."""

import scipy.constants

DEFAULT_FREQUENCY: float = 5e9
TEST_FREQUENCY: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (5e9, 6e9, 7e9),
    (8e9, 9e9, 10e9),
)

Φ_0 = scipy.constants.physical_constants["mag. flux quantum"][0]
c_0 = scipy.constants.speed_of_light
e = scipy.constants.e
h = scipy.constants.h
ε_0 = scipy.constants.epsilon_0
π = scipy.constants.pi
